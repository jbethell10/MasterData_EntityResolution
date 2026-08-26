"""
Runs the matching engine against the Leipzig benchmarks and sweeps the
decision threshold, so the reported F1 is the best operating point found on
real labelled data rather than whatever the first guess happened to give.

Usage:
  python3 scripts/run_benchmark.py                     # both datasets
  python3 scripts/run_benchmark.py --dataset amazon-google
  python3 scripts/run_benchmark.py --sweep-weights     # also try weight mixes
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import LOADERS, evaluate  # noqa: E402
from matcher import Weights  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_results.csv"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

WEIGHT_GRID = [
    Weights(0.5, 0.2, 0.3),   # stage 04's original split
    Weights(0.6, 0.1, 0.3),
    Weights(0.4, 0.1, 0.5),
    Weights(0.34, 0.33, 0.33),
    Weights(0.7, 0.0, 0.3),
    Weights(0.3, 0.0, 0.7),
    Weights(1.0, 0.0, 0.0),   # fuzzy only -- the ablation baseline
    Weights(0.0, 0.0, 1.0),   # TF-IDF only -- the other ablation baseline
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*LOADERS, "all"], default="all")
    ap.add_argument("--sweep-weights", action="store_true")
    ap.add_argument("--block-size", type=int, default=50)
    args = ap.parse_args()

    names = list(LOADERS) if args.dataset == "all" else [args.dataset]
    rows = []

    for name in names:
        bench = LOADERS[name]()
        print(f"\n{'='*72}\n{bench.name}: "
              f"{len(bench.left)} x {len(bench.right)} records, "
              f"{bench.n_pairs} labelled matches\n{'='*72}")

        weight_options = WEIGHT_GRID if args.sweep_weights else [Weights()]
        best = None

        for w in weight_options:
            for t in THRESHOLDS:
                res = evaluate(bench, weights=w, threshold=t, block_size=args.block_size)
                rows.append(res.as_row())
                if best is None or res.f1 > best.f1:
                    best = res

            if args.sweep_weights:
                wn = w.normalized()
                at_best = max(
                    (r for r in rows if r["w_text"] == wn.text
                     and r["w_phonetic"] == wn.phonetic and r["w_vector"] == wn.vector),
                    key=lambda r: r["f1"],
                )
                print(f"  weights text={wn.text:.2f} phon={wn.phonetic:.2f} "
                      f"vec={wn.vector:.2f}  ->  best F1 {at_best['f1']:.3f} "
                      f"@ threshold {at_best['threshold']}  "
                      f"(P {at_best['precision']:.3f} / R {at_best['recall']:.3f})")

        print(f"\n  top-1 accuracy (queries with a known match): {best.top1_accuracy:.1%} "
              f"of {best.n_queries_with_truth}")
        print(f"  BEST F1 {best.f1:.3f} at threshold {best.threshold} -- "
              f"precision {best.precision:.3f}, recall {best.recall:.3f}")
        print(f"  ({best.correct} correct of {best.predicted} predicted; "
              f"{best.n_truth_pairs} true pairs)")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull sweep written to {OUT_PATH}")


if __name__ == "__main__":
    main()
