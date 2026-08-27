"""
Where each run mode writes its artifacts.

Originally every mode wrote to the same fixed filenames under data/. That
silently broke two things at once:

  1. Running the Leipzig benchmark overwrote the synthetic pipeline's
     intake_events.csv, so the dashboard's synthetic trace tab -- the one that
     shows packaging artwork -- started rendering Leipzig rows that have no
     images at all.

  2. Rebuilding mder.db for one dataset while the CSVs still described another
     left the repo holding a 1,363-row amazon-google catalog next to 1,092
     abt-buy events. Scores computed against that mixture are meaningless, and
     the mismatch is invisible because every filename looks right.

Giving each mode its own directory makes the second failure impossible and the
first obvious: if a directory is empty, that mode simply has not been run.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Datasets that can back a "leipzig" run.
LEIPZIG_DATASETS = ("abt-buy", "amazon-google")


def run_dir(mode: str = "synthetic", dataset: str | None = None) -> Path:
    """Directory holding one run's artifacts (db + all stage CSVs).

    synthetic          -> data/synthetic/
    leipzig/abt-buy    -> data/leipzig/abt-buy/
    """
    if mode == "synthetic":
        return DATA / "synthetic"
    if mode == "leipzig":
        if dataset not in LEIPZIG_DATASETS:
            raise ValueError(f"unknown leipzig dataset: {dataset!r}")
        return DATA / "leipzig" / dataset
    raise ValueError(f"unknown mode: {mode!r}")


def db_path(mode: str = "synthetic", dataset: str | None = None) -> Path:
    """Each run gets its OWN database.

    A single shared mder.db is what allowed a catalog from one dataset to be
    scored against events from another.
    """
    return run_dir(mode, dataset) / "mder.db"


def ensure(mode: str = "synthetic", dataset: str | None = None) -> Path:
    d = run_dir(mode, dataset)
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_mode_args(parser) -> None:
    """Standard --mode/--dataset pair, so every stage is pointed at one run."""
    parser.add_argument("--mode", choices=["synthetic", "leipzig"], default="synthetic")
    parser.add_argument("--dataset", choices=list(LEIPZIG_DATASETS), default="abt-buy",
                        help="which Leipzig benchmark (only used with --mode leipzig)")


def resolve(args) -> tuple[Path, Path]:
    """(run_dir, db_path) for the mode/dataset on a parsed argparse namespace."""
    dataset = args.dataset if args.mode == "leipzig" else None
    return ensure(args.mode, dataset), db_path(args.mode, dataset)
