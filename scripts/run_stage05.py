"""
Runs stage 05 over the intake events, invoking the LLM only on the rows
stage 04 left genuinely ambiguous.

  python3 scripts/run_stage05.py            # offline: shows WHICH rows would escalate
  python3 scripts/run_stage05.py --live     # actually calls Claude (needs ANTHROPIC_API_KEY)
  python3 scripts/run_stage05.py --live --force-all   # ignore the lazy trigger

The offline mode is the important one for day-to-day work: it reports the
trigger decisions and the estimated call volume without spending anything, so
you can see the cost profile of this stage before enabling it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage04_candidate_match import (  # noqa: E402
    build_vector_index, detect_signal_conflict, load_catalog, rank_candidates,
)
from stage05_disambiguate import (  # noqa: E402
    Candidate, DEFAULT_MODEL, get_disambiguator, needs_disambiguation,
)

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
EVENTS_PATH = paths.run_dir("synthetic") / "intake_events.csv"
OUT_PATH = paths.run_dir("synthetic") / "stage05_results.csv"


def to_candidates(ranked, catalog_by_id) -> list[Candidate]:
    out = []
    for combined, master_id, brand, name, *_ in ranked:
        _mid, gtin, _b, _n, qty = catalog_by_id[master_id]
        out.append(Candidate(
            master_id=master_id, brand=brand, product_name=name,
            quantity=qty or "", gtin=gtin, retrieval_score=combined,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually call the Claude API (requires ANTHROPIC_API_KEY)")
    ap.add_argument("--force-all", action="store_true",
                    help="disambiguate every event, ignoring the lazy trigger")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many LLM calls to make (cost control)")
    args = ap.parse_args()

    catalog = load_catalog()
    catalog_by_id = {r[0]: r for r in catalog}
    index = build_vector_index(catalog)

    with open(EVENTS_PATH, newline="") as f:
        events = list(csv.DictReader(f))

    disambiguator = get_disambiguator(live=args.live, model=args.model)
    backend = type(disambiguator).__name__
    if args.live and backend == "NullDisambiguator":
        print("!! --live requested but ANTHROPIC_API_KEY is not set -- "
              "falling back to offline mode.\n")

    print(f"Stage 05 backend: {backend}"
          + (f" (model={args.model})" if backend == "ClaudeDisambiguator" else ""))
    print()

    results, calls = [], 0
    for e in events:
        true_id = int(e["true_master_id"])
        sup_ranked = rank_candidates(
            e["supplier_raw_brand"], e["supplier_raw_product_name"],
            e["supplier_norm_gtin"], catalog, index=index,
        )
        art_ranked = rank_candidates(
            e["artwork_brand"], e["artwork_product_name"],
            e["artwork_gtin"], catalog, index=index,
        )
        sup_c = to_candidates(sup_ranked, catalog_by_id)
        art_c = to_candidates(art_ranked, catalog_by_id)

        conflict = bool(
            detect_signal_conflict(e["supplier_raw_brand"], e["supplier_raw_product_name"],
                                   e["supplier_norm_gtin"], catalog)
            or detect_signal_conflict(e["artwork_brand"], e["artwork_product_name"],
                                      e["artwork_gtin"], catalog)
        )

        should, reason = needs_disambiguation(sup_c, art_c, signal_conflict=conflict)
        if args.force_all:
            should, reason = True, "forced (--force-all)"

        row = {
            "event_id": e["event_id"],
            "true_master_id": true_id,
            "stage04_supplier_top1": sup_c[0].master_id if sup_c else "",
            "stage04_artwork_top1": art_c[0].master_id if art_c else "",
            "ambiguous": should,
            "trigger_reason": reason,
        }

        if should and (args.limit is None or calls < args.limit):
            # merge both shortlists, de-duped, so the model sees every candidate
            # either retrieval surfaced -- not just the winning side's view
            merged, seen = [], set()
            for c in sup_c + art_c:
                if c.master_id not in seen:
                    seen.add(c.master_id)
                    merged.append(c)
            verdict = disambiguator.resolve(
                artwork={"brand": e["artwork_brand"], "product_name": e["artwork_product_name"],
                         "quantity": e["artwork_quantity"], "gtin": e["artwork_gtin"]},
                supplier={"brand": e["supplier_raw_brand"], "product_name": e["supplier_raw_product_name"],
                          "quantity": e["supplier_raw_quantity"], "gtin": e["supplier_raw_gtin"]},
                shortlist=merged,
            )
            if verdict.invoked:
                calls += 1
            row.update(verdict.as_row())
            row["llm_correct"] = (verdict.master_id == true_id) if verdict.master_id is not None else ""
        else:
            row.update({"llm_invoked": False, "llm_master_id": "", "llm_confidence": 0.0,
                        "llm_capped": False, "llm_reasoning": "", "llm_error": "",
                        "llm_correct": ""})
        results.append(row)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    n = len(results)
    ambiguous = [r for r in results if r["ambiguous"]]
    print(f"{len(ambiguous)}/{n} events flagged ambiguous "
          f"({100*len(ambiguous)/n:.0f}%) -- only these cost an LLM call")
    for r in ambiguous:
        print(f"  [{r['event_id']}] {r['trigger_reason']}")

    if calls:
        judged = [r for r in results if r["llm_correct"] != ""]
        right = sum(1 for r in judged if r["llm_correct"])
        print(f"\nLLM calls made: {calls}")
        if judged:
            print(f"  correct: {right}/{len(judged)}")
        for r in judged:
            print(f"  [{r['event_id']}] -> {r['llm_master_id']} "
                  f"(true {r['true_master_id']}, conf {r['llm_confidence']}"
                  f"{', CAPPED' if r['llm_capped'] else ''}) {r['llm_reasoning']}")
        errs = [r for r in results if r["llm_error"]]
        for r in errs:
            print(f"  [{r['event_id']}] ERROR: {r['llm_error']}")
    else:
        print("\nNo LLM calls made (offline mode or nothing ambiguous).")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
