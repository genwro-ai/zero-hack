_BLOCK_RULES: tuple[tuple[str, str, str], ...] = (
    # (block, needle, mode)  where mode in {"prefix", "substr", "exact"}
    ("LOGISTICS", "RECEIVE WAFER LOT", "exact"),
    ("LOGISTICS", "LOT IDENTIFICATION", "exact"),
    ("LOGISTICS", "LOT RELEASE", "substr"),
    ("LOGISTICS", "SHIP LOT", "exact"),
    ("LOGISTICS", "PACKAGE PREPARATION", "exact"),
    ("MEASURE", "MEASURE", "prefix"),
    ("TEST", "TEST", "substr"),
    ("TEST", "YIELD ANALYSIS", "exact"),
    ("INSPECTION", "INSPECTION", "substr"),
    ("INSPECTION", "INSPECT", "prefix"),
    ("INSPECTION", "CHECK", "substr"),
    ("LITHO", "SPIN COAT", "prefix"),
    ("LITHO", "SOFT BAKE", "exact"),
    ("LITHO", "ALIGN MASK", "prefix"),
    ("LITHO", "EXPOSE", "prefix"),
    ("LITHO", "POST EXPOSE BAKE", "exact"),
    ("LITHO", "DEVELOP", "prefix"),
    ("LITHO", "HARD BAKE", "exact"),
    ("LITHO", "PAD WINDOW LITHO", "substr"),
    ("LITHO", "OPEN PAD WINDOW", "substr"),
    ("LITHO", "OPEN BOND PAD WINDOW", "substr"),
    ("STRIP", "STRIP", "prefix"),
    ("ETCH", "ETCH", "substr"),
    ("IMPLANT", "IMPLANT", "prefix"),
    ("CMP", "CMP", "prefix"),
    ("VIA_FILL", "FILL VIA", "prefix"),
    ("ANNEAL", "ANNEAL", "substr"),
    ("ANNEAL", "DRIVE IN DIFFUSION", "exact"),
    ("ANNEAL", "DENSIFY", "prefix"),
    ("ANNEAL", "CURE PASSIVATION", "exact"),
    ("PASSIVATION", "PASSIVATION", "substr"),
    ("PASSIVATION", "PAD OPENING", "substr"),
    ("PASSIVATION", "PAD WINDOW", "substr"),
    ("BACKSIDE", "BACKSIDE", "substr"),
    ("DEPOSITION", "DEPOSIT", "prefix"),
    ("DEPOSITION", "THERMAL OXIDATION", "exact"),
    ("DEPOSITION", "EPITAXIAL DEPOSITION", "exact"),
    ("DEPOSITION", "GATE OXIDE GROWTH", "exact"),
    ("DEPOSITION", "GATE OXIDE PREP", "exact"),
    ("PREP", "EPITAXY", "substr"),
    ("PREP", "EPITAXIAL", "substr"),
    ("PREP", "SUBSTRATE CHECK", "exact"),
    ("PREP", "GRINDING WAFER", "prefix"),
    ("PREP", "GRIND", "substr"),
    ("PREP", "SURFACE PREP", "prefix"),
    ("CLEAN", "CLEAN", "substr"),
    ("CLEAN", "RCA", "substr"),
    ("CLEAN", "HF DIP", "exact"),
    ("CLEAN", "OXIDE STRIP", "exact"),
    ("CLEAN", "RINSE", "substr"),
    ("CLEAN", "DRY WAFER", "prefix"),
)

UNKNOWN_BLOCK = "OTHER"


def step_to_block(step: str) -> str:
    """Return the functional block label for a single step string."""
    s = step.strip().upper()
    for block, needle, mode in _BLOCK_RULES:
        if mode == "exact" and s == needle:
            return block
        if mode == "prefix" and s.startswith(needle):
            return block
        if mode == "substr" and needle in s:
            return block
    return UNKNOWN_BLOCK


def steps_to_blocks(steps: list[str] | tuple[str, ...]) -> list[str]:
    """Map a sequence of steps to its per-step block labels."""
    return [step_to_block(step) for step in steps]


def block_runs(steps: list[str] | tuple[str, ...]) -> list[str]:
    """Block labels with consecutive duplicates collapsed (the block *shape*)."""
    runs: list[str] = []
    for block in steps_to_blocks(steps):
        if not runs or runs[-1] != block:
            runs.append(block)
    return runs
