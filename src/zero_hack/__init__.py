"""Utilities for the Zero One Hack_01 Industrial AI track solution."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDUSTRIAL_DATA_DIR = PROJECT_ROOT / "data" / "industrial"


def main() -> None:
    """Print the local project paths used by the package."""
    print("Zero One Hack_01 Industrial AI")
    print(f"project_root={PROJECT_ROOT}")
    print(f"industrial_data_dir={INDUSTRIAL_DATA_DIR}")
