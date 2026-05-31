"""Aggregate the transformer before/after-GRPO completion matrix into a comparison.

Reads outputs/metrics/<dataset>/transformer_holdout_<F>/{sampled_base,sampled_grpo,
sampled_base_masked,greedy_base,greedy_grpo}/results.json and prints:

  E1 (validity x fidelity): base vs GRPO deltas on validity + accuracy (sampled regime)
  E2 (masked control):      GRPO vs base+ViolationMask
  E3 (ID vs OOD transfer):  the same, split by role

Writes a markdown summary to outputs/metrics/<dataset>/grpo_completion_comparison.md.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET = "valid_s005k"
ROOT = Path("outputs/metrics") / DATASET
FAMILIES = ("ic", "igbt", "mosfet")
ACC = "token_accuracy"  # headline fidelity metric


def load(holdout: str, run: str) -> dict | None:
    p = ROOT / f"transformer_holdout_{holdout}" / run / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(res: dict | None, role: str, key: str) -> float | None:
    if not res:
        return None
    return res["by_role"].get(role, {}).get(key)


def fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "  -  "
    d = b - a
    return f"{d:+.3f}"


def main() -> None:
    lines: list[str] = [
        f"# Transformer — before/after GRPO, sequence completion ({DATASET})",
        "",
        "Free-running completion on the **test** split (GRPO trained on train/valid prompts, "
        "so this is leakage-free). Sampled regime = T=1.0 (the regime GRPO optimized); "
        "greedy = deployment regime. validity = generated-route validity rate; "
        f"acc = {ACC}.",
        "",
        "## E1 — validity x fidelity (sampled T=1.0, role=all)",
        "",
        "| Holdout | validity base→grpo (Δ) | acc base→grpo (Δ) | verdict |",
        "|---|---|---|---|",
    ]
    for f in FAMILIES:
        b, g = load(f, "sampled_base"), load(f, "sampled_grpo")
        vb, vg = cell(b, "all", "validity_rate"), cell(g, "all", "validity_rate")
        ab, ag = cell(b, "all", ACC), cell(g, "all", ACC)
        verdict = "—"
        if None not in (vb, vg, ab, ag):
            dv, da = vg - vb, ag - ab
            if dv > 0.01 and da >= -0.01:
                verdict = "GRPO helps (validity↑, fidelity kept)"
            elif dv > 0.01 and da < -0.01:
                verdict = "validity↑ but fidelity↓ (degenerate?)"
            elif abs(dv) <= 0.01:
                verdict = "no validity change"
            else:
                verdict = "validity↓"
        lines.append(
            f"| {f} | {fmt(vb)}→{fmt(vg)} ({delta(vb, vg)}) | "
            f"{fmt(ab)}→{fmt(ag)} ({delta(ab, ag)}) | {verdict} |"
        )

    lines += [
        "",
        "## E2 — masked-decoding control (sampled T=1.0, role=all)",
        "",
        "Does GRPO beat just masking illegal steps at decode time? base+mask gets validity "
        "for free; GRPO only wins if it matches that validity **and** keeps higher fidelity.",
        "",
        "| Holdout | base | base+mask | grpo | mask vs grpo validity | mask vs grpo acc |",
        "|---|---|---|---|---|---|",
    ]
    for f in FAMILIES:
        b, m, g = load(f, "sampled_base"), load(f, "sampled_base_masked"), load(f, "sampled_grpo")
        vb = cell(b, "all", "validity_rate")
        vm, vg = cell(m, "all", "validity_rate"), cell(g, "all", "validity_rate")
        am, ag = cell(m, "all", ACC), cell(g, "all", ACC)
        lines.append(
            f"| {f} | val {fmt(vb)} | val {fmt(vm)} / acc {fmt(am)} | "
            f"val {fmt(vg)} / acc {fmt(ag)} | {delta(vg, vm)} | {delta(ag, am)} |"
        )

    lines += [
        "",
        "## E3 — ID vs OOD transfer (sampled T=1.0)",
        "",
        "GRPO trained only on the ID (train) families. Does the validity gain reach the "
        "held-out (OOD) family it never saw?",
        "",
        "| Holdout | role | validity base→grpo (Δ) | acc base→grpo (Δ) |",
        "|---|---|---|---|",
    ]
    for f in FAMILIES:
        b, g = load(f, "sampled_base"), load(f, "sampled_grpo")
        for role in ("id", "ood"):
            vb, vg = cell(b, role, "validity_rate"), cell(g, role, "validity_rate")
            ab, ag = cell(b, role, ACC), cell(g, role, ACC)
            lines.append(
                f"| {f} | {role} | {fmt(vb)}→{fmt(vg)} ({delta(vb, vg)}) | "
                f"{fmt(ab)}→{fmt(ag)} ({delta(ab, ag)}) |"
            )

    lines += [
        "",
        "## Deployment regime (greedy) — role=all",
        "",
        "| Holdout | validity base→grpo (Δ) | acc base→grpo (Δ) |",
        "|---|---|---|",
    ]
    for f in FAMILIES:
        b, g = load(f, "greedy_base"), load(f, "greedy_grpo")
        vb, vg = cell(b, "all", "validity_rate"), cell(g, "all", "validity_rate")
        ab, ag = cell(b, "all", ACC), cell(g, "all", ACC)
        lines.append(
            f"| {f} | {fmt(vb)}→{fmt(vg)} ({delta(vb, vg)}) | "
            f"{fmt(ab)}→{fmt(ag)} ({delta(ab, ag)}) |"
        )

    out = ROOT / "grpo_completion_comparison.md"
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
