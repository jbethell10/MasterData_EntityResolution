"""
Runs the full decision path -- stages 03 through 08 -- over every intake event
and reports what the pipeline actually DECIDED, not just how well it scored.

  python3 scripts/run_pipeline.py
  python3 scripts/run_pipeline.py --demo-alias   # show the learning loop closing

This is the stage that makes the project answer the question a real deployment
asks: of N submissions, how many can I merge without a human, how many need a
person, and for the ones that need a person -- is it the supplier's fault or
mine?
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage08_resolve as audit  # noqa: E402
from stage04_candidate_match import (  # noqa: E402
    build_vector_index, load_catalog, rank_candidates,
)
from stage06_barcode_verify import verify_three_way  # noqa: E402
from stage07_confidence_route import Route, route_event  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = ROOT / "data" / "intake_events.csv"
CROSS_PATH = ROOT / "data" / "cross_check_results.csv"
OUT_PATH = ROOT / "data" / "pipeline_decisions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-alias", action="store_true",
                    help="approve one held row, then re-run it to show the alias cache hit")
    ap.add_argument("--reset-log", action="store_true", default=True)
    args = ap.parse_args()

    catalog = load_catalog()
    catalog_by_id = {r[0]: r for r in catalog}
    index = build_vector_index(catalog)

    with open(EVENTS_PATH, newline="") as f:
        events = list(csv.DictReader(f))
    with open(CROSS_PATH, newline="") as f:
        cross = {r["event_id"]: r for r in csv.DictReader(f)}

    conn = audit.connect()
    if args.reset_log:
        conn.execute("DELETE FROM audit_log")
        conn.commit()

    rows = []
    for e in events:
        eid = e["event_id"]
        true_id = int(e["true_master_id"])
        xc = cross[eid]

        # --- signal 1: did the two independent readings agree with each other?
        agree_fields = sum(
            xc[f"{f}_artwork_vs_supplier"] == "True" for f in ("brand", "quantity", "gtin")
        )
        source_agreement = agree_fields / 3

        # --- retrieval (stage 04), keyed on the supplier submission
        ranked = rank_candidates(
            e["supplier_raw_brand"], e["supplier_raw_product_name"],
            e["supplier_norm_gtin"], catalog, index=index,
        )
        top_score, top_id = ranked[0][0], ranked[0][1]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0

        # --- signal 3: three-way barcode (stage 06), against the winning candidate
        cand_gtin = catalog_by_id[top_id][1]
        barcode = verify_three_way(e["artwork_gtin"], e["supplier_norm_gtin"], cand_gtin)

        # --- alias cache (stage 08) feeds back into routing
        hit = audit.lookup_alias(conn, e["supplier_raw_brand"], e["supplier_raw_product_name"])
        if hit:
            audit.record_alias_reuse(conn, e["supplier_raw_brand"], e["supplier_raw_product_name"])

        # --- stage 07
        decision = route_event(
            source_agreement=source_agreement,
            top_score=top_score, runner_up_score=runner_up,
            barcode=barcode, alias_hit=bool(hit),
        )

        problem_class = audit.log_decision(
            conn, event_id=eid, resolved_id=top_id, true_master_id=true_id,
            route=decision.route.value, confidence=decision.confidence,
            source_agreement=source_agreement,
            evidence={"signals": decision.signals, "overrides": decision.overrides,
                      "barcode": barcode.as_row(), "top_score": round(top_score, 3),
                      "runner_up": round(runner_up, 3)},
        )

        row = {"event_id": eid, "true_master_id": true_id, "resolved_id": top_id,
               "correct": top_id == true_id, "problem_class": problem_class,
               "alias_hit": bool(hit)}
        row.update(decision.as_row())
        row.update(barcode.as_row())
        rows.append(row)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Ran stages 03-08 over {n} intake events -> {OUT_PATH}\n")

    print("ROUTING (stage 07)")
    for r in (Route.AUTO_MERGE, Route.HOLD_FOR_REVIEW, Route.REJECT):
        got = [x for x in rows if x["route"] == r.value]
        if got:
            correct = sum(1 for x in got if x["correct"])
            print(f"  {r.value:16s} {len(got):2d}/{n}  ({correct}/{len(got)} would have been correct)")

    merged = [x for x in rows if x["route"] == Route.AUTO_MERGE.value]
    wrong_merges = [x for x in merged if not x["correct"]]
    print(f"\n  incorrect auto-merges: {len(wrong_merges)}  "
          f"<- the number that actually matters; a wrong auto-merge corrupts the catalog")

    print("\nPROBLEM CLASS (stage 08 audit log)")
    for cls, count in sorted(
        {c: sum(1 for x in rows if x["problem_class"] == c) for x in rows
         for c in [x["problem_class"]]}.items()
    ):
        print(f"  {cls:26s} {count:2d}/{n}")

    print("\nBARCODE VERDICT (stage 06)")
    for v, count in sorted(
        {x["barcode_verdict"]: sum(1 for y in rows if y["barcode_verdict"] == x["barcode_verdict"])
         for x in rows}.items()
    ):
        print(f"  {v:20s} {count:2d}/{n}")

    if args.demo_alias:
        held = [x for x in rows if x["route"] != Route.AUTO_MERGE.value and x["correct"]]
        if not held:
            print("\n(no held-but-correct row available to demo the alias cache)")
        else:
            target = held[0]
            ev = next(e for e in events if e["event_id"] == target["event_id"])
            name = catalog_by_id[target["resolved_id"]][3]
            print(f"\n--- alias-cache demo -------------------------------------")
            print(f"  {target['event_id']}: routed {target['route']} at "
                  f"{target['confidence']} — {target['rationale']}")
            audit.approve_correction(
                conn, brand=ev["supplier_raw_brand"], product_name=ev["supplier_raw_product_name"],
                resolved_id=target["resolved_id"], resolved_name=name,
                approved_by="steward@retailer.example",
            )
            print(f"  human approves: '{ev['supplier_raw_brand']} "
                  f"{ev['supplier_raw_product_name']}' -> {name}")

            ranked2 = rank_candidates(ev["supplier_raw_brand"], ev["supplier_raw_product_name"],
                                      ev["supplier_norm_gtin"], catalog, index=index)
            xc2 = cross[target["event_id"]]
            sa = sum(xc2[f"{f}_artwork_vs_supplier"] == "True"
                     for f in ("brand", "quantity", "gtin")) / 3
            bc = verify_three_way(ev["artwork_gtin"], ev["supplier_norm_gtin"],
                                  catalog_by_id[ranked2[0][1]][1])
            again = route_event(source_agreement=sa, top_score=ranked2[0][0],
                                runner_up_score=ranked2[1][0], barcode=bc, alias_hit=True)
            print(f"  same input re-submitted: {again.route.value} at {again.confidence:.3f} "
                  f"(was {target['confidence']}) — overrides: {'|'.join(again.overrides) or 'none'}")

    s = audit.summarize(conn)
    print(f"\nAudit log: {sum(s['routes'].values())} decisions | "
          f"alias cache: {s['alias_entries']} entries, {s['alias_reuses']} reuses")
    conn.close()


if __name__ == "__main__":
    main()
