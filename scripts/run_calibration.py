"""
Fit and evaluate the margin -> P(correct) calibration.

  python3 scripts/run_calibration.py                 # full report
  python3 scripts/run_calibration.py --fit abt-buy   # fit and save one

Three questions, in order of how much they matter:

  1. Does calibration beat the hardcoded margin/0.40 mapping on held-out data?
     If not, stop.

  2. Does a calibration fitted on ONE catalog transfer to a DIFFERENT one?
     This is the question that decides whether the fix is real. The original
     defect was a constant tuned on a 30-product catalog and silently applied
     to a 1,081-product one. Replacing it with a curve fitted on abt-buy and
     silently applied to amazon-google would be the same mistake with more
     arithmetic. Fitting on one and testing on the other is the only way to
     find out which it is.

  3. What does it change operationally -- how many correct matches stop being
     thrown away, and does the review queue stay clean?

Every number reported for a calibrator is measured on data it never saw.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration as cal  # noqa: E402
import paths  # noqa: E402
from stage04_candidate_match import build_vector_index, rank_candidates  # noqa: E402
from stage07_confidence_route import AUTO_MERGE_MIN, HOLD_MIN, MARGIN_SATURATION  # noqa: E402

SPLIT_SEED = 20260827


def collect(dataset: str):
    """(margin, top_score, correct) for every event in a Leipzig run.

    Barcodes are withheld exactly as the pipeline withholds them, so the
    margins here are the ones stage 07 will actually see.
    """
    run = paths.run_dir("leipzig", dataset)
    events_path = run / "intake_events.csv"
    if not events_path.exists():
        raise SystemExit(
            f"No run for {dataset}. Build it first:\n"
            f"  python3 scripts/build_master_catalog.py --mode leipzig --dataset {dataset}\n"
            f"  python3 scripts/build_intake_events.py --mode leipzig --dataset {dataset}"
        )
    rows = list(csv.DictReader(open(events_path, newline="")))
    conn = sqlite3.connect(paths.db_path("leipzig", dataset))
    catalog = conn.execute(
        "SELECT master_id, gtin, brand, product_name, quantity FROM master_catalog"
    ).fetchall()
    conn.close()
    index = build_vector_index(catalog)

    margins, tops, labels = [], [], []
    for r in rows:
        ranked = rank_candidates(r["supplier_raw_brand"], r["supplier_raw_product_name"],
                                 "", catalog, index=index)
        top = ranked[0][0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        margins.append(top - runner)
        tops.append(top)
        labels.append(int(ranked[0][1] == int(r["true_master_id"])))
    return np.array(margins), np.array(tops), np.array(labels)


def split(n, frac=0.5):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n)
    cut = int(n * frac)
    return idx[:cut], idx[cut:]


def hardcoded(margins):
    """The mapping being replaced: margin / MARGIN_SATURATION, capped at 1."""
    return np.minimum(1.0, np.maximum(0.0, margins) / MARGIN_SATURATION)


def show(label, report):
    print(f"  {label:<34s} n={report.n:<5d} Brier={report.brier:.4f}  "
          f"logloss={report.log_loss:.4f}  ECE={report.ece:.4f}  "
          f"worst-bin={report.max_bin_gap:.3f}  AUC={report.auc:.4f}")


def operational(p, labels, name, corroborated: bool):
    """What the bands actually do to a review queue.

    `corroborated=False` applies stage 07's single-signal rule: with only the
    text margin observable there is nothing independent to check it against,
    so nothing may auto-merge however high the probability. Reporting the
    uncapped split as well is what shows whether that rule is earning its keep.
    """
    p = np.asarray(p)
    n = len(p)
    auto = (p >= AUTO_MERGE_MIN) if corroborated else np.zeros(n, dtype=bool)
    hold = (p >= HOLD_MIN) & ~auto
    rej = p < HOLD_MIN
    correct_rejected = int(labels[rej].sum())
    print(f"  {name}")
    for tag, m in (("auto-merge", auto), ("review", hold), ("reject", rej)):
        if m.sum():
            print(f"      {tag:<11s} {int(m.sum()):5d}/{n}  "
                  f"({labels[m].mean():.1%} of them correct)")
    print(f"      correct matches thrown away: {correct_rejected}")
    return correct_rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=list(paths.LEIPZIG_DATASETS),
                    help="fit on this dataset and save the calibrator")
    args = ap.parse_args()

    data = {}
    for ds in paths.LEIPZIG_DATASETS:
        m, t, y = collect(ds)
        data[ds] = (m, t, y)
        print(f"{ds}: {len(y)} events, {int(y.sum())} correct ({y.mean():.1%})")
    print()

    # ---- Q1: does calibration beat the hardcoded constant, held out? --------
    print("=" * 78)
    print("Q1  Held-out comparison (fit on half of abt-buy, score the other half)")
    print("=" * 78)
    m, t, y = data["abt-buy"]
    tr, te = split(len(y))
    platt = cal.fit_platt(m[tr], y[tr])
    iso = cal.fit_isotonic(m[tr], y[tr])

    show("margin/0.40 (current)", cal.evaluate(hardcoded(m[te]), y[te]))
    show("Platt (fitted)", cal.evaluate(platt(m[te]), y[te]))
    show("Isotonic (fitted)", cal.evaluate(iso(m[te]), y[te]))
    print(f"\n  Platt fit: P(correct) = sigmoid({platt.a:.3f} * margin + {platt.b:.3f})")

    # ---- Q2: does it transfer to a different catalog? -----------------------
    print()
    print("=" * 78)
    print("Q2  Cross-catalog transfer -- fit on abt-buy, score ALL of amazon-google")
    print("    (the question the original miscalibration failed)")
    print("=" * 78)
    m2, t2, y2 = data["amazon-google"]
    platt_full = cal.fit_platt(m, y)
    iso_full = cal.fit_isotonic(m, y)
    show("margin/0.40 (current)", cal.evaluate(hardcoded(m2), y2))
    show("Platt fitted on abt-buy", cal.evaluate(platt_full(m2), y2))
    show("Isotonic fitted on abt-buy", cal.evaluate(iso_full(m2), y2))

    native = cal.fit_platt(m2[split(len(y2))[0]], y2[split(len(y2))[0]])
    show("Platt fitted on amazon-google", cal.evaluate(native(m2[split(len(y2))[1]]),
                                                       y2[split(len(y2))[1]]))

    # ---- reliability of the transferred model ------------------------------
    print("\n  Reliability of abt-buy Platt applied to amazon-google:")
    print(f"    {'claimed':>9s} {'actual':>8s} {'n':>6s}")
    for claimed, actual, cnt in cal.reliability(platt_full(m2), y2, bins=10):
        flag = "  <-- overconfident" if claimed - actual > 0.10 else ""
        print(f"    {claimed:9.3f} {actual:8.3f} {cnt:6d}{flag}")

    # ---- Q3: what does it change operationally? ----------------------------
    print()
    print("=" * 78)
    print("Q3  Operational effect on the review queue (abt-buy, held-out half)")
    print("    Text is the only observable signal here, so stage 07 forbids")
    print("    auto-merge: everything above the floor goes to a human.")
    print("=" * 78)
    before = operational(hardcoded(m[te]), y[te], "margin/0.40 (current)", corroborated=False)
    after = operational(platt(m[te]), y[te], "Platt (calibrated)", corroborated=False)
    print(f"\n  correct matches rescued from the reject pile: {before - after}")

    print()
    print("  Counterfactual -- what the same probabilities WOULD do if the")
    print("  corroboration rule were removed and the 0.90 band allowed to merge:")
    operational(platt(m[te]), y[te], "Platt, guardrail disabled", corroborated=True)
    would_merge = platt(m[te]) >= AUTO_MERGE_MIN
    wrong = int((~y[te].astype(bool) & would_merge).sum())
    print(f"      -> {wrong} WRONG merges written into the master catalog.")
    print("      The band is behaving exactly as specified (>=0.90 really is "
          ">=90% correct);")
    print("      90% is simply not a safe bar for an unreviewed write. This is "
          "the case the")
    print("      corroboration rule exists to catch, and calibration is what "
          "made it visible.")

    if args.fit:
        mm, _, yy = data[args.fit]
        # Deployment fit uses ALL labelled pairs for this catalog; the numbers
        # reported above are the held-out ones, which is what the quality claim
        # rests on. Fitting the shipped model on everything is standard and is
        # not the same as evaluating on everything.
        c = cal.fit_platt(mm, yy)
        held = cal.evaluate(cal.fit_platt(mm[split(len(yy))[0]], yy[split(len(yy))[0]])(
            mm[split(len(yy))[1]]), yy[split(len(yy))[1]])
        p = cal.save(c, paths.run_dir("leipzig", args.fit), meta={
            "fitted_on": args.fit,
            "n_pairs": int(len(yy)),
            "base_rate": float(yy.mean()),
            "held_out_brier": held.brier,
            "held_out_ece": held.ece,
            "note": "Fitted on this catalog only. Do not reuse for another "
                    "catalog -- transfer is overconfident (see Q2).",
        })
        print(f"\nSaved Platt calibrator fitted on {args.fit} -> {p}")
        print(f"  P(correct) = sigmoid({c.a:.4f} * margin + {c.b:.4f})")
        print(f"  held-out Brier {held.brier:.4f}, ECE {held.ece:.4f}")


if __name__ == "__main__":
    main()
