"""A/B test: does n-gram-normalised training help, as a drop-in?

Trains two LSTMs from the *same* weight initialisation on the *same* data:
  baseline  - plain cross-entropy on the raw logits
  residual  - cross-entropy on  model_logits + log p_ngram(. | context)
The only difference is the frozen n-gram prior inside the loss, so any metric gap
is attributable to the normalisation. Reports next-step Top-1/Top-k and (if an
anomaly eval set is present) likelihood ROC-AUC, on both the in-distribution
valid split and the held-out (OOD) family.

Note: a UNIGRAM prior (n=1) is a constant vector across all contexts, which the
linear head's bias term absorbs exactly -> mathematically a no-op. Use n>=2 so
the prior is context-dependent and actually changes what the network learns.

Usage:
    uv run python scripts/compare_ngram_residual.py \
        --splits-dir data/generated/valid_s005k/splits \
        --holdout-family ic --epochs 8 --residual-ngram-n 2 \
        --eval-dir data/eval/valid_s005k/holdout_ic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zero_hack.eval.anomaly import _roc_auc
from zero_hack.eval.io import read_anomaly_truth, read_eval_input_anomaly
from zero_hack.models.common import (
    TrainConfig,
    evaluate_model,
    load_split_records,
    make_loaders,
    pick_device,
    train_model,
)
from zero_hack.models.lstm.inference import LSTMInference
from zero_hack.models.lstm.model import LSTMConfig, LSTMModel
from zero_hack.models.ngram_residual import wrap_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/generated/valid_s005k/splits")
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default="ic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=192)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--residual-ngram-n", type=int, default=2)
    parser.add_argument("--residual-ngram-alpha", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="holdout_<fam> dir with id/ and ood/ anomaly eval sets (optional).",
    )
    parser.add_argument("--out", default=None, help="Optional path to write results.json.")
    return parser.parse_args()


def _fresh_lstm(bundle, seed: int) -> LSTMModel:
    """Identical init for both arms: seed right before constructing the network."""
    torch.manual_seed(seed)
    return LSTMModel(
        vocab_size=len(bundle.vocabulary.id_to_token),
        config=LSTMConfig(),
        pad_id=bundle.vocabulary.pad_id,
    )


def _next_step(model, loader, device, k) -> dict:
    return evaluate_model(model, loader, device=device, k=k)["all"]


def _anomaly_roc(inference: LSTMInference, eval_dir: Path) -> float | None:
    anomaly_csv = eval_dir / "eval_input_anomaly.csv"
    truth_csv = eval_dir / "anomaly_truth.csv"
    if not anomaly_csv.exists() or not truth_csv.exists():
        return None
    examples = read_eval_input_anomaly(anomaly_csv)
    truth = read_anomaly_truth(truth_csv)
    scores: list[float] = []
    labels: list[int] = []
    for ex in examples:
        seq = ex["sequence"]
        eid = ex["example_id"]
        if not seq or eid not in truth:
            continue
        avg = inference.score_sequence(ex["family"], seq) / len(seq)
        scores.append(-avg)  # higher = more anomalous
        labels.append(1 if truth[eid]["is_valid"] == 0 else 0)
    return _roc_auc(scores, labels)


def _evaluate_arm(model, bundle, loaders, ood_loader, device, args) -> dict:
    result = {
        "next_step": {
            "id_valid": _next_step(model, loaders["valid"], device, args.k),
            "ood_family": _next_step(model, ood_loader, device, args.k),
        }
    }
    if args.eval_dir:
        inference = LSTMInference(
            model,
            bundle.vocabulary,
            device=device,
            max_context=args.max_context,
        )
        eval_dir = Path(args.eval_dir)
        result["anomaly_roc_auc"] = {
            "id": _anomaly_roc(inference, eval_dir / "id"),
            "ood": _anomaly_roc(inference, eval_dir / "ood"),
        }
    return result


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    bundle = load_split_records(
        args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}  device={device}")

    loaders = make_loaders(bundle, batch_size=args.batch_size, max_context=args.max_context)
    # make_loaders already builds a loader per split, including the held-out
    # family's test split (test_<fam>); reuse it as the OOD loader.
    ood_loader = loaders[f"test_{args.holdout_family}"]

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        k=args.k,
        max_train_batches=args.max_train_batches,
    )

    print("\n=== training baseline (plain cross-entropy) ===")
    baseline = _fresh_lstm(bundle, args.seed)
    baseline = train_model(
        baseline, loaders, config=train_config, device=device, pad_id=bundle.vocabulary.pad_id
    )

    print(f"\n=== training residual ({args.residual_ngram_n}-gram prior in the loss) ===")
    residual_net = _fresh_lstm(bundle, args.seed)
    residual = wrap_residual(
        residual_net,
        bundle.records["train"],
        bundle.vocabulary,
        n=args.residual_ngram_n,
        alpha=args.residual_ngram_alpha,
    )
    residual = train_model(
        residual, loaders, config=train_config, device=device, pad_id=bundle.vocabulary.pad_id
    )

    results = {
        "config": {
            "splits_dir": args.splits_dir,
            "holdout_family": args.holdout_family,
            "epochs": args.epochs,
            "residual_ngram_n": args.residual_ngram_n,
            "limit_per_family": args.limit_per_family,
            "seed": args.seed,
        },
        "baseline": _evaluate_arm(baseline, bundle, loaders, ood_loader, device, args),
        "residual": _evaluate_arm(residual, bundle, loaders, ood_loader, device, args),
    }

    _print_report(results)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


def _print_report(results: dict) -> None:
    b, r = results["baseline"], results["residual"]
    cfg = results["config"]
    print("\n" + "=" * 64)
    print(
        f"RESULT  holdout={cfg['holdout_family']}  residual={cfg['residual_ngram_n']}-gram  "
        f"epochs={cfg['epochs']}"
    )
    print("=" * 64)

    def line(label: str, bv: float, rv: float) -> None:
        delta = rv - bv
        flag = "  <-- residual better" if delta > 0 else ("  <-- worse" if delta < 0 else "")
        print(f"  {label:24s} base={bv:.4f}  residual={rv:.4f}  delta={delta:+.4f}{flag}")

    print("\nnext-step Top-1:")
    line("ID (valid)", b["next_step"]["id_valid"]["top1"], r["next_step"]["id_valid"]["top1"])
    line(
        "OOD (held-out)", b["next_step"]["ood_family"]["top1"], r["next_step"]["ood_family"]["top1"]
    )

    print("\nnext-step MRR:")
    line("ID (valid)", b["next_step"]["id_valid"]["mrr"], r["next_step"]["id_valid"]["mrr"])
    line("OOD (held-out)", b["next_step"]["ood_family"]["mrr"], r["next_step"]["ood_family"]["mrr"])

    if "anomaly_roc_auc" in b:
        print("\nanomaly ROC-AUC:")
        for view in ("id", "ood"):
            bv, rv = b["anomaly_roc_auc"][view], r["anomaly_roc_auc"][view]
            if bv is not None and rv is not None:
                line(view, bv, rv)


if __name__ == "__main__":
    main()
