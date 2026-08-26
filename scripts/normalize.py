"""
Parses and normalizes the raw supplier feed: trims whitespace, folds case
for matching purposes (while keeping an original copy for display/audit),
and standardizes quantity units to a single canonical form (grams / ml)
so "0.051kg" and "51g" compare equal downstream.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "supplier_feed.csv"
OUT_PATH = ROOT / "data" / "supplier_feed_normalized.csv"

QTY_RE = re.compile(r"([\d.]+)\s*(kg|g|l|ml)", re.IGNORECASE)

def normalize_quantity(qty: str) -> str:
    """Convert to a canonical base-unit string, e.g. '51.0g' or '330.0ml'."""
    if not qty:
        return ""
    # handle multipack like "2x51g" or "8x100g" -> total grams
    multipack = re.match(r"(\d+)\s*[xX]\s*([\d.]+)\s*(kg|g|l|ml)", qty)
    if multipack:
        count, amount, unit = multipack.groups()
        amount = float(amount) * int(count)
    else:
        m = QTY_RE.search(qty)
        if not m:
            return qty.strip().lower()
        amount, unit = m.groups()
        amount = float(amount)

    unit = unit.lower()
    if unit == "kg":
        amount *= 1000
        unit = "g"
    elif unit == "l":
        amount *= 1000
        unit = "ml"
    return f"{amount:g}{unit}"

def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    value = re.sub(r"[^\w\s]", "", value)   # strip punctuation
    value = re.sub(r"\s+", " ", value)      # collapse whitespace
    return value

def main():
    with open(IN_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    normalized = []
    for row in rows:
        normalized.append(
            {
                "true_master_id": row["true_master_id"],
                "raw_brand": row["submitted_brand"],
                "raw_product_name": row["submitted_product_name"],
                "raw_quantity": row["submitted_quantity"],
                "raw_gtin": row["submitted_gtin"],
                "norm_brand": normalize_text(row["submitted_brand"]),
                "norm_product_name": normalize_text(row["submitted_product_name"]),
                "norm_quantity": normalize_quantity(row["submitted_quantity"]),
                "norm_gtin": re.sub(r"\D", "", row["submitted_gtin"]),
                "corruption_applied": row["corruption_applied"],
            }
        )

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)

    print(f"Normalized {len(normalized)} rows -> {OUT_PATH}")
    for r in normalized[:5]:
        print(f"  [{r['true_master_id']}] '{r['raw_brand']}' -> '{r['norm_brand']}'  |  "
              f"'{r['raw_quantity']}' -> '{r['norm_quantity']}'")

if __name__ == "__main__":
    main()
