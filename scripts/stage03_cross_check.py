"""
Stage 03 -- Cross-Check.

For every intake event we now have three independent readings of the same
physical product: what's printed on the pack (artwork), what the supplier
typed (supplier), and what the canonical record says (master -- normally
this third leg wouldn't be known yet, but since this is a scored prototype
we DO cross-check against it too, which is exactly what the real pipeline's
"06 Barcode Verify [3-way]" stage does).

This stage does NOT try to resolve anything or guess who's "right" -- that's
candidate matching's (stage 04) and disambiguation's (stage 05) job. It just
honestly flags, per field, which pairs of sources agree and which don't, so
a human reviewer (or the confidence/routing stage) has a clean signal to
work from.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "intake_events.csv"
OUT_PATH = ROOT / "data" / "cross_check_results.csv"

FIELD_PAIRS = [
    ("brand", "artwork_norm_brand", "supplier_norm_brand", "master_norm_brand"),
    ("quantity", "artwork_norm_quantity", "supplier_norm_quantity", "master_norm_quantity"),
    ("gtin", "artwork_gtin", "supplier_norm_gtin", "master_gtin"),
]

def eq(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a) and bool(b) and a == b

def cross_check_event(row: dict) -> dict:
    result = {"event_id": row["event_id"], "true_master_id": row["true_master_id"]}
    agree_count, compared_count = 0, 0

    for field, art_key, sup_key, mas_key in FIELD_PAIRS:
        art, sup, mas = row[art_key], row[sup_key], row[mas_key]

        art_vs_sup = eq(art, sup)
        art_vs_mas = eq(art, mas)
        sup_vs_mas = eq(sup, mas)
        all_three_agree = art_vs_sup and art_vs_mas and sup_vs_mas

        result[f"{field}_artwork_vs_supplier"] = art_vs_sup
        result[f"{field}_artwork_vs_master"] = art_vs_mas
        result[f"{field}_supplier_vs_master"] = sup_vs_mas
        result[f"{field}_all_agree"] = all_three_agree

        agree_count += all_three_agree
        compared_count += 1

    result["fields_fully_agreed"] = agree_count
    result["fields_compared"] = compared_count
    result["agreement_score"] = round(agree_count / compared_count, 2)

    if agree_count == compared_count:
        status = "full_agreement"
    elif agree_count == 0:
        status = "disagreement"
    else:
        status = "partial_agreement"
    result["cross_check_status"] = status

    # which side (artwork or supplier) tends to be the odd one out, when
    # master ground truth is available -- purely diagnostic, this is the
    # kind of thing that would drive routing/confidence weighting later
    artwork_wrong = sum(
        1 for field, art_key, sup_key, mas_key in FIELD_PAIRS
        if not eq(row[art_key], row[mas_key])
    )
    supplier_wrong = sum(
        1 for field, art_key, sup_key, mas_key in FIELD_PAIRS
        if not eq(row[sup_key], row[mas_key])
    )
    result["artwork_fields_wrong_vs_master"] = artwork_wrong
    result["supplier_fields_wrong_vs_master"] = supplier_wrong

    return result

def main():
    with open(IN_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    results = [cross_check_event(r) for r in rows]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    status_counts = {}
    for r in results:
        status_counts[r["cross_check_status"]] = status_counts.get(r["cross_check_status"], 0) + 1

    print(f"Cross-checked {n} intake events -> {OUT_PATH}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}/{n}")

    avg_artwork_wrong = sum(r["artwork_fields_wrong_vs_master"] for r in results) / n
    avg_supplier_wrong = sum(r["supplier_fields_wrong_vs_master"] for r in results) / n
    print(f"\nAvg fields wrong vs master -- artwork: {avg_artwork_wrong:.2f}/3, "
          f"supplier: {avg_supplier_wrong:.2f}/3")
    print("(supplier corruption is applied ~50-80% of the time per field by design, "
          "so heavy supplier disagreement here is expected, not a bug)")

if __name__ == "__main__":
    main()
