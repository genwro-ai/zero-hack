"""Write final Industrial AI submission CSVs from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.validator import first_violated_rule
from zero_hack.models.anomaly_threshold import (
    load_threshold_examples,
    tune_anomaly_threshold_from_examples,
)
from zero_hack.models.common import pick_device
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.gpt.train import (
    _complete_greedy,
    _invalid_prediction_ids,
    predict_topk,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "final_submission"
    / "final_submission_gpt_all_families"
    / "best.pt"
)


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[GPTNextStepModel, Vocabulary]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocabulary = Vocabulary(
        token_to_id=checkpoint["vocabulary"]["token_to_id"],
        id_to_token=tuple(checkpoint["vocabulary"]["id_to_token"]),
    )
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTNextStepModel(
        len(vocabulary.id_to_token),
        config,
        pad_id=vocabulary.pad_id,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, vocabulary


def _calibration_dirs(args: argparse.Namespace) -> list[Path]:
    if args.calibration_dirs:
        return [Path(path) for path in args.calibration_dirs]
    return [
        Path(args.eval_root) / args.calibration_dataset / f"holdout_{family}" / "calibration"
        for family in ("mosfet", "igbt", "ic")
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _family_token(vocabulary: Vocabulary, family: str) -> str:
    token = FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])
    if token not in vocabulary.token_to_id:
        return FAMILY_TOKENS["unknown"]
    return token


@torch.no_grad()
def _sequence_logprob_batched(
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    family: str,
    steps: list[str] | tuple[str, ...],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Exact sequence logprob using the same prefix convention as GPT training."""

    if not steps:
        return 0.0

    family_token = _family_token(vocabulary, family)
    target_ids = [vocabulary.token_to_id.get(step, vocabulary.unk_id) for step in steps]
    total = 0.0
    max_context = model.config.max_context
    positions = list(range(len(steps)))
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        encoded_rows: list[list[int]] = []
        for position in chunk:
            tokens = ["<BOS>", family_token, *steps[:position]][-max_context:]
            encoded_rows.append(vocabulary.encode(tokens))

        max_len = max(len(row) for row in encoded_rows)
        input_ids = []
        attention_mask = []
        for row in encoded_rows:
            pad_len = max_len - len(row)
            input_ids.append(row + [vocabulary.pad_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

        logits = model(
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attention_mask, dtype=torch.bool, device=device),
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(
            [target_ids[position] for position in chunk],
            dtype=torch.long,
            device=device,
        )
        total += float(log_probs.gather(1, targets[:, None]).sum().item())

    return total


class _BatchedGPTLikelihoodAdapter:
    def __init__(
        self,
        model: GPTNextStepModel,
        vocabulary: Vocabulary,
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.batch_size = batch_size

    def score_sequence(self, family: str, steps: list[str] | tuple[str, ...]) -> float:
        return _sequence_logprob_batched(
            self.model,
            self.vocabulary,
            family,
            steps,
            device=self.device,
            batch_size=self.batch_size,
        )


def _tune_threshold(
    adapter: _BatchedGPTLikelihoodAdapter,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    if args.anomaly_threshold is not None:
        return args.anomaly_threshold, {
            "source": "manual",
            "threshold": args.anomaly_threshold,
        }

    dirs = _calibration_dirs(args)
    missing = [str(path) for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing calibration dirs for final anomaly threshold: " + ", ".join(missing)
        )

    examples = []
    for directory in dirs:
        examples.extend(load_threshold_examples(directory))
    result = tune_anomaly_threshold_from_examples(adapter, examples)
    return result.threshold, {
        "source": "threshold_calibration",
        "objective": "f1",
        "tuned_on": [str(path) for path in dirs],
        "n_examples": len(examples),
        "threshold": result.threshold,
        "val_f1": result.f1,
        "val_precision": result.precision,
        "val_recall": result.recall,
    }


def _write_nextstep_and_completion(
    *,
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    eval_input_valid: Path,
    output_dir: Path,
    device: torch.device,
    k: int,
    max_completion_steps: int,
) -> None:
    invalid_ids = _invalid_prediction_ids(vocabulary)
    rows = io.read_eval_input_valid(eval_input_valid)
    next_rows = []
    completion_rows = []
    for row in rows:
        family = row["family"]
        prefix = row["partial_sequence"]
        next_rows.append(
            {
                "example_id": row["example_id"],
                "ranks": predict_topk(
                    model,
                    vocabulary,
                    family,
                    prefix,
                    k=k,
                    device=device,
                    invalid_ids=invalid_ids,
                ),
            }
        )
        completion_rows.append(
            {
                "example_id": row["example_id"],
                "steps": _complete_greedy(
                    model,
                    vocabulary,
                    family,
                    prefix,
                    device=device,
                    invalid_ids=invalid_ids,
                    max_steps=max_completion_steps,
                ),
            }
        )

    io.write_next_step_predictions(output_dir / "nextstep.csv", next_rows)
    io.write_completion_predictions(output_dir / "completion.csv", completion_rows)


def _write_anomaly(
    *,
    adapter: _BatchedGPTLikelihoodAdapter,
    eval_input_anomaly: Path,
    output_dir: Path,
    threshold: float,
) -> None:
    rows = []
    for row in io.read_eval_input_anomaly(eval_input_anomaly):
        logprob = adapter.score_sequence(row["family"], row["sequence"])
        avg_logprob = logprob / max(1, len(row["sequence"]))
        valid = avg_logprob >= threshold
        rows.append(
            {
                "example_id": row["example_id"],
                "is_valid": int(valid),
                "score": _sigmoid(avg_logprob - threshold),
                "predicted_rule": None
                if valid
                else (first_violated_rule(row["sequence"]) or "RULE_DEP_NO_CLEAN"),
            }
        )
    io.write_anomaly_predictions(output_dir / "anomaly.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--eval-input-valid",
        default=str(PROJECT_ROOT / "data" / "industrial" / "eval_input_valid.csv"),
    )
    parser.add_argument(
        "--eval-input-anomaly",
        default=str(PROJECT_ROOT / "data" / "industrial" / "eval_input_anomaly.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "submission" / "final_submission_gpt"),
    )
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--calibration-dataset", default="valid_s100k")
    parser.add_argument("--calibration-dirs", nargs="+", default=None)
    parser.add_argument("--anomaly-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-completion-steps", type=int, default=240)
    parser.add_argument("--score-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, vocabulary = _load_checkpoint(args.checkpoint, device)
    adapter = _BatchedGPTLikelihoodAdapter(
        model,
        vocabulary,
        device=device,
        batch_size=args.score_batch_size,
    )
    threshold, threshold_meta = _tune_threshold(adapter, args)
    (output_dir / "anomaly_threshold.json").write_text(
        json.dumps(threshold_meta, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_nextstep_and_completion(
        model=model,
        vocabulary=vocabulary,
        eval_input_valid=Path(args.eval_input_valid),
        output_dir=output_dir,
        device=device,
        k=args.k,
        max_completion_steps=args.max_completion_steps,
    )
    _write_anomaly(
        adapter=adapter,
        eval_input_anomaly=Path(args.eval_input_anomaly),
        output_dir=output_dir,
        threshold=threshold,
    )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "eval_input_valid": str(args.eval_input_valid),
        "eval_input_anomaly": str(args.eval_input_anomaly),
        "output_dir": str(output_dir),
        "files": ["nextstep.csv", "completion.csv", "anomaly.csv"],
        "anomaly_threshold": threshold_meta,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
