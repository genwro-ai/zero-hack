"""Model-agnostic sequence-completion eval (Task 2), for before/after-GRPO comparison.

Unlike ``scripts/eval_lstm_completion.py`` (LSTM-only, via ``load_lstm_checkpoint``),
this loads *any* checkpoint through :func:`zero_hack.models.grpo.load_policy`, so it
works transparently for both the LSTM and the transformer decoder, and for both the
base (``best.pt``) and GRPO-finetuned (``best_grpo.pt``) checkpoints.

Each test sequence is cut at a fraction; the kept prefix is handed to the model and
the model **greedily** generates the rest autoregressively from its own outputs (free
running, deterministic — so base vs GRPO is a paired comparison on identical prompts).
The generated suffix is scored against the true suffix (exact-match, normalized edit
distance, token accuracy, block accuracy), and we report how often the reconstructed
full route is process-valid.

Results are broken out three ways:
  * per family,
  * per **role** — ID (a training family) vs OOD (the held-out family, which GRPO never
    saw; this is the generalization axis), and
  * per prefix fraction.

With ``--enforce-rules`` the ViolationMask is applied during generation (rule-violating
next steps masked to -inf): this is the "free validity" control — it guarantees validity
by construction, so comparing ``base + --enforce-rules`` against an unmasked GRPO run
tells you whether GRPO buys anything over simply masking at decode time.

Usage:
    uv run python scripts/eval_completion.py \
        --checkpoint outputs/models/valid_s005k/transformer_holdout_ic/best.pt \
        --holdout-family ic \
        --out outputs/metrics/valid_s005k/transformer_holdout_ic/completion_base
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from zero_hack.data import FAMILY_FILE_NAMES, load_sequence_records
from zero_hack.eval.completion import score_completion
from zero_hack.eval.io import write_completion_predictions
from zero_hack.eval.validator import validate_sequence
from zero_hack.models.common import DEFAULT_SPLITS_DIR
from zero_hack.models.grpo import SEQUENCE_TERMINATOR, StepPolicy, load_policy

FAMILIES = tuple(FAMILY_FILE_NAMES)


def _test_split_path(splits_dir: Path, family: str) -> Path:
    stem = FAMILY_FILE_NAMES[family].removesuffix(".csv")
    return splits_dir / f"{stem}_test.csv"


@torch.no_grad()
def complete(
    policy: StepPolicy,
    family: str,
    prefix: list[str],
    *,
    max_steps: int,
    enforce_rules: bool,
    temperature: float,
) -> list[str]:
    """Free-running completion.

    ``temperature <= 0`` → deterministic greedy (argmax over legal next steps).
    ``temperature > 0``  → stochastic sampling at that temperature — the regime GRPO
    was trained under (its valid_rate was measured at T=1.0, where the policy is free
    to drift; greedy hides that drift, so the sampled regime is where any GRPO effect
    should show).
    """
    if temperature > 0:
        steps, _, _ = policy.sample_completion(
            family,
            prefix,
            max_steps=max_steps,
            temperature=temperature,
            mask_sampling=enforce_rules,
        )
        return steps
    seq = list(prefix)
    out: list[str] = []
    while len(out) < max_steps:
        logits = policy.next_logits(family, seq, mask_sampling=enforce_rules)
        if not bool(torch.isfinite(logits).any()):
            break  # everything masked out — nothing legal to emit
        token = policy.vocabulary.id_to_token[int(torch.argmax(logits).item())]
        out.append(token)
        seq.append(token)
        if token == SEQUENCE_TERMINATOR:
            break
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint (.pt).")
    parser.add_argument(
        "--holdout-family",
        choices=FAMILIES,
        default=None,
        help="Family held out during training; tagged OOD in the per-role breakdown.",
    )
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.6, 0.8],
        help="Prefix fractions to keep before generating the rest.",
    )
    parser.add_argument(
        "--max-examples-per-family",
        type=int,
        default=100,
        help="Cap sequences sampled per family (per fraction). 0 = no cap.",
    )
    parser.add_argument("--max-steps", type=int, default=400, help="Generation length cap.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (deterministic). >0 = sample at this temperature (GRPO's regime).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Completions per prompt (only meaningful when --temperature > 0).",
    )
    parser.add_argument(
        "--enforce-rules",
        action="store_true",
        help="Apply the ViolationMask during generation (the 'free validity' control).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for results.{json,md} and predictions.csv.",
    )
    return parser.parse_args()


def _aggregate(
    truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    groups: dict[str, str],
    valid: dict[str, bool],
) -> dict[str, dict]:
    """score_completion per group, with validity rate folded into each group row."""
    scored = score_completion(truth, predictions, groups)
    # validity rate per group ("all" + each distinct group value)
    by_group_ids: dict[str, list[str]] = {"all": list(truth)}
    for eid, g in groups.items():
        by_group_ids.setdefault(g, []).append(eid)
    for group, ids in by_group_ids.items():
        if group in scored and ids:
            scored[group]["validity_rate"] = round(sum(valid[i] for i in ids) / len(ids), 4)
    return scored


def main() -> None:
    args = parse_args()
    device = args.device or "cpu"
    policy = load_policy(args.checkpoint, device=device)
    policy.model.eval()
    print(f"loaded {args.checkpoint} kind={policy.kind} enforce_rules={args.enforce_rules}")

    torch.manual_seed(args.seed)
    splits_dir = Path(args.splits_dir)
    rng = random.Random(args.seed)
    holdout = args.holdout_family.lower() if args.holdout_family else None

    truth: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}
    family_of: dict[str, str] = {}
    role_of: dict[str, str] = {}
    fraction_of: dict[str, float] = {}
    valid_of: dict[str, bool] = {}
    pred_rows: list[dict] = []

    for family in args.families:
        records = load_sequence_records(_test_split_path(splits_dir, family), family=family)
        records = [r for r in records if len(r.steps) >= 2]
        rng.shuffle(records)
        if args.max_examples_per_family:
            records = records[: args.max_examples_per_family]
        role = "ood" if family == holdout else "id"
        print(f"{family} ({role}): {len(records)} sequences")

        for record in records:
            steps = list(record.steps)
            for fraction in args.fractions:
                cut = max(1, int(round(fraction * len(steps))))
                if cut >= len(steps):
                    continue
                prefix, gold_suffix = steps[:cut], steps[cut:]
                n_samples = args.samples if args.temperature > 0 else 1
                for s in range(n_samples):
                    pred_suffix = complete(
                        policy,
                        family,
                        prefix,
                        max_steps=args.max_steps,
                        enforce_rules=args.enforce_rules,
                        temperature=args.temperature,
                    )
                    suffix = f"_f{int(round(fraction * 100))}" + (f"_s{s}" if n_samples > 1 else "")
                    eid = f"{family}_{record.sequence_id}{suffix}"
                    truth[eid] = gold_suffix
                    predictions[eid] = pred_suffix
                    family_of[eid] = family
                    role_of[eid] = role
                    fraction_of[eid] = fraction
                    valid_of[eid] = not validate_sequence(prefix + pred_suffix)
                    pred_rows.append({"example_id": eid, "steps": pred_suffix})

    if not truth:
        raise SystemExit("No completion examples were produced; check splits dir / fractions.")

    by_family = _aggregate(truth, predictions, family_of, valid_of)
    by_role = _aggregate(truth, predictions, role_of, valid_of)
    by_fraction = {}
    for fraction in sorted(set(fraction_of.values())):
        ids = {eid for eid, f in fraction_of.items() if f == fraction}
        sub = lambda d, ids=ids: {eid: d[eid] for eid in ids}  # noqa: E731
        by_fraction[f"{fraction:.2f}"] = _aggregate(
            sub(truth), sub(predictions), {eid: role_of[eid] for eid in ids}, sub(valid_of)
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "kind": policy.kind,
        "holdout_family": holdout,
        "enforce_rules": args.enforce_rules,
        "decoding": "greedy" if args.temperature <= 0 else f"sample@T{args.temperature}",
        "temperature": args.temperature,
        "samples_per_prompt": args.samples if args.temperature > 0 else 1,
        "n_examples": len(truth),
        "fractions": args.fractions,
        "by_family": by_family,
        "by_role": by_role,
        "by_fraction": by_fraction,
        "generated_route_validity_rate": by_family.get("all", {}).get("validity_rate", 0.0),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_completion_predictions(out_dir / "predictions.csv", pred_rows)

    md = _render_md(args.checkpoint, policy.kind, args.enforce_rules, by_family, by_role)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'results.json'}")


def _row(group: str, r: dict) -> str:
    return (
        f"| {group} | {r['n']} | {r.get('validity_rate', 0):.4f} | {r['exact_match']:.4f} | "
        f"{r['norm_edit_distance']:.4f} | {r['token_accuracy']:.4f} | {r['block_accuracy']:.4f} |"
    )


def _table(title: str, groups: dict) -> list[str]:
    head = [
        f"## {title}",
        "",
        "| Group | n | validity | exact_match | norm_edit_dist | token_acc | block_acc |",
        "|---|---|---|---|---|---|---|",
    ]
    return head + [_row(g, r) for g, r in groups.items()] + [""]


def _render_md(checkpoint, kind, enforce, by_family, by_role) -> str:
    lines = [
        f"# {kind} — sequence completion (free running, greedy)",
        "",
        f"Checkpoint: `{checkpoint}` · enforce_rules={enforce}",
        "",
    ]
    lines += _table("By role (id = train families, ood = holdout)", by_role)
    lines += _table("By family", by_family)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
