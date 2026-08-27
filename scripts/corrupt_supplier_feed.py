"""
Generates a synthetic "supplier submission" feed by corrupting a sample of
master_catalog records -- standing in for messy real-world data entry.

Every corrupted row keeps its true master_id as ground truth, so downstream
matching accuracy can be scored honestly instead of eyeballed.
"""
import csv
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
DB_PATH = paths.run_dir("synthetic") / "mder.db"
OUT_PATH = paths.run_dir("synthetic") / "supplier_feed.csv"

random.seed(7)

# common real-world brand shorthand / colloquialisms suppliers actually type
BRAND_ABBREVIATIONS = {
    "Mars": ["MRS", "Mars Inc", "mars"],
    "Cadbury": ["Cadburys", "CDBRY", "cadbury"],
    "Nestle": ["Nestlé", "NESTLE", "Nestles"],
    "Heinz": ["Heinz Co", "HNZ", "heinz"],
    "Kelloggs": ["Kellogg's", "KLGS", "kelloggs"],
    "Coca-Cola": ["Coke", "Coca Cola", "CCOLA"],
    "PepsiCo": ["Pepsi Co", "PEPSICO", "pepsico"],
    "Unilever": ["Unilever PLC", "UNLVR"],
    "Danone": ["Danone SA", "DANON"],
    "Arla": ["Arla Foods", "ARLA"],
    "Warburtons": ["Warburton's", "WBTNS"],
    "Hovis": ["HOVIS LTD", "hovis"],
    "Birds Eye": ["BirdsEye", "B.Eye"],
    "McCain": ["McCain Foods", "MCCAIN"],
    "Muller": ["Müller", "MULLER"],
    "Andrex": ["ANDREX LTD", "andrex"],
    "Purina": ["Purina PetCare", "PURINA"],
    "Pedigree": ["PEDIGREE LTD", "pedigree"],
    "Whiskas": ["WHISKAS LTD", "whiskas"],
    "Dreamies": ["DREAMIES", "dreamies"],
    "Bakers": ["Baker's", "BAKERS LTD"],
}

def truncate_name(name: str) -> str:
    words = name.split()
    if len(words) > 2:
        return " ".join(words[: max(2, len(words) - 1)])
    return name

def reformat_quantity(qty: str) -> str:
    if not qty:
        return qty
    qty = qty.strip()
    if qty.endswith("g") and "x" not in qty:
        try:
            grams = float(qty[:-1])
            return f"{grams/1000:.3f}kg"
        except ValueError:
            return qty
    if qty.endswith("ml"):
        try:
            ml = float(qty[:-2])
            return f"{ml/1000:.2f}L"
        except ValueError:
            return qty
    return qty

def corrupt_barcode(gtin: str) -> str:
    """Simulate a keying error: drop or swap one digit."""
    digits = list(gtin)
    idx = random.randrange(len(digits))
    if random.random() < 0.5:
        digits[idx] = str((int(digits[idx]) + random.choice([1, -1])) % 10)
        return "".join(digits)
    else:
        return gtin[:idx] + gtin[idx + 1 :]  # dropped digit

def corrupt_row(row) -> dict:
    master_id, gtin, brand, product_name, quantity, category = row
    corrupted = {
        "true_master_id": master_id,
        "submitted_brand": brand,
        "submitted_product_name": product_name,
        "submitted_quantity": quantity,
        "submitted_gtin": gtin,
        "corruption_applied": [],
    }

    # brand corruption
    if brand in BRAND_ABBREVIATIONS and random.random() < 0.8:
        corrupted["submitted_brand"] = random.choice(BRAND_ABBREVIATIONS[brand])
        corrupted["corruption_applied"].append("brand_abbrev")

    # name truncation / case noise
    if random.random() < 0.5:
        corrupted["submitted_product_name"] = truncate_name(product_name)
        corrupted["corruption_applied"].append("name_truncated")
    if random.random() < 0.3:
        corrupted["submitted_product_name"] = corrupted["submitted_product_name"].upper()
        corrupted["corruption_applied"].append("name_upper")

    # unit format change
    if random.random() < 0.5:
        corrupted["submitted_quantity"] = reformat_quantity(quantity)
        corrupted["corruption_applied"].append("unit_reformatted")

    # barcode keying error (kept rare -- most submissions get the barcode right)
    if random.random() < 0.25:
        corrupted["submitted_gtin"] = corrupt_barcode(gtin)
        corrupted["corruption_applied"].append("barcode_error")

    corrupted["corruption_applied"] = "|".join(corrupted["corruption_applied"]) or "none"
    return corrupted

def main(n_submissions: int = 60):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT master_id, gtin, brand, product_name, quantity, category FROM master_catalog"
    ).fetchall()
    conn.close()

    # sample with replacement so some products get multiple (differently
    # corrupted) submissions, like a real supplier feed would
    submissions = [corrupt_row(random.choice(rows)) for _ in range(n_submissions)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(submissions[0].keys()))
        writer.writeheader()
        writer.writerows(submissions)

    clean = sum(1 for s in submissions if s["corruption_applied"] == "none")
    print(f"Wrote {len(submissions)} synthetic supplier submissions -> {OUT_PATH}")
    print(f"  {clean} arrived clean, {len(submissions) - clean} had at least one corruption")

if __name__ == "__main__":
    main()
