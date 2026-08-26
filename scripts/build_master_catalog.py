"""
Builds the canonical master_catalog table.

NOTE ON DATA SOURCE: this sandbox's network egress is locked to an allowlist
that does not include world.openfoodfacts.org (confirmed: the proxy returns
403 on CONNECT). Real internet access will work fine on your own laptop, so
this script ships in two modes:

  --mode seed     (default) generates a realistic, structurally-valid seed
                  catalog (real public brand/product names, properly
                  checksummed EAN-13 barcodes) so the rest of the pipeline
                  can be built and tested today.

  --mode real     calls the real Open Food Facts API. Run this on your own
                  machine (not in this sandbox) to replace the seed catalog
                  with real GTINs, brands, and image URLs before your final
                  build/demo.

Both modes write to the same SQLite table (master_catalog) so nothing
downstream needs to change when you swap seed -> real data.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mder.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS master_catalog (
    master_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    gtin          TEXT UNIQUE,
    brand         TEXT NOT NULL,
    product_name  TEXT NOT NULL,
    quantity      TEXT,
    category      TEXT,
    image_url     TEXT
);
"""

def ean13_check_digit(digits12: str) -> str:
    """Compute the correct EAN-13 check digit for a 12-digit body."""
    total = 0
    for i, d in enumerate(digits12):
        n = int(d)
        total += n * (3 if i % 2 == 1 else 1)
    check = (10 - (total % 10)) % 10
    return str(check)

def make_gtin(seq: int) -> str:
    # 500 = a real GS1 UK prefix range; the remaining digits are a
    # synthetic manufacturer/product code, not a registered real GTIN.
    body = f"500{seq:09d}"
    return body + ean13_check_digit(body)

# Real, publicly-known brand/product examples (names only -- no
# proprietary Tesco data), enough variety to exercise fuzzy matching,
# phonetic confusion, and unit-format differences downstream.
SEED_PRODUCTS = [
    ("Mars", "Mars Bar", "51g", "chocolate"),
    ("Mars", "Mars Bar Twin Pack", "2x51g", "chocolate"),
    ("Cadbury", "Dairy Milk Chocolate Bar", "45g", "chocolate"),
    ("Cadbury", "Dairy Milk Buttons", "30g", "chocolate"),
    ("Nestle", "KitKat 4 Finger", "41.5g", "chocolate"),
    ("Nestle", "Nescafe Gold Blend Instant Coffee", "200g", "coffee"),
    ("Heinz", "Tomato Ketchup", "460g", "condiments"),
    ("Heinz", "Baked Beans in Tomato Sauce", "415g", "tinned goods"),
    ("Kelloggs", "Corn Flakes", "500g", "cereal"),
    ("Kelloggs", "Coco Pops", "480g", "cereal"),
    ("Coca-Cola", "Coca-Cola Original Taste", "330ml", "soft drinks"),
    ("Coca-Cola", "Diet Coke", "330ml", "soft drinks"),
    ("PepsiCo", "Pepsi Max", "330ml", "soft drinks"),
    ("PepsiCo", "Walkers Ready Salted Crisps", "32.5g", "crisps"),
    ("Unilever", "Persil Non-Bio Washing Liquid", "1.43L", "household"),
    ("Unilever", "Marmite Yeast Extract", "250g", "spreads"),
    ("Danone", "Activia Strawberry Yogurt", "4x115g", "dairy"),
    ("Danone", "Actimel Original", "8x100g", "dairy"),
    ("Arla", "Lurpak Spreadable Butter", "500g", "dairy"),
    ("Warburtons", "Toastie White Bread", "800g", "bakery"),
    ("Hovis", "Wholemeal Bread", "800g", "bakery"),
    ("Birds Eye", "Garden Peas", "800g", "frozen"),
    ("McCain", "Oven Chips", "1kg", "frozen"),
    ("Muller", "Muller Corner Strawberry", "6x135g", "dairy"),
    ("Andrex", "Classic Clean Toilet Tissue", "9 rolls", "household"),
    ("Purina", "Felix As Good As It Looks Cat Food", "12x100g", "pet"),
    ("Pedigree", "Pedigree Adult Dog Food Chicken", "1.2kg", "pet"),
    ("Whiskas", "Whiskas Adult Cat Food Pouches", "12x85g", "pet"),
    ("Dreamies", "Dreamies Cat Treats Chicken", "60g", "pet"),
    ("Bakers", "Bakers Adult Dog Food Beef", "1.2kg", "pet"),
]

def build_seed(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("DELETE FROM master_catalog")
    # master_id is AUTOINCREMENT, whose high-water mark lives in sqlite_sequence
    # and SURVIVES a DELETE. Without resetting it, every re-run against an
    # existing mder.db shifts all 30 ids up by 30 (1-30, then 31-60, ...),
    # which silently breaks the artwork_<master_id>.png filename join and makes
    # the "fixed seeds -> exactly reproducible" guarantee false.
    cur.execute("DELETE FROM sqlite_sequence WHERE name='master_catalog'")
    for i, (brand, name, qty, cat) in enumerate(SEED_PRODUCTS, start=1):
        gtin = make_gtin(i)
        # Pin master_id explicitly rather than leaning on the counter, so seed
        # ids are 1..30 by construction on every run, on any pre-existing db.
        cur.execute(
            "INSERT INTO master_catalog (master_id, gtin, brand, product_name, quantity, category, image_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, gtin, brand, name, qty, cat, None),
        )
    conn.commit()
    return len(SEED_PRODUCTS)

def build_real(conn: sqlite3.Connection) -> int:
    import requests
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("DELETE FROM master_catalog")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='master_catalog'")
    session = requests.Session()
    session.headers.update({"User-Agent": "MDER-Prototype/1.0 (student project)"})
    categories = ["chocolates", "cereals", "soft-drinks", "dairies", "snacks"]
    total = 0
    for cat in categories:
        page = 1
        while True:
            resp = session.get(
                "https://world.openfoodfacts.org/api/v2/search",
                params={
                    "categories_tags_en": cat,
                    "countries_tags_en": "united-kingdom",
                    "fields": "code,brands,product_name,quantity,categories,image_front_url",
                    "page_size": 100,
                    "page": page,
                },
                timeout=20,
            )
            resp.raise_for_status()
            products = resp.json().get("products", [])
            if not products:
                break
            for p in products:
                if not p.get("code") or not p.get("product_name"):
                    continue
                cur.execute(
                    "INSERT OR IGNORE INTO master_catalog (gtin, brand, product_name, quantity, category, image_url) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        p.get("code"),
                        (p.get("brands") or "").split(",")[0].strip(),
                        p.get("product_name"),
                        p.get("quantity"),
                        cat,
                        p.get("image_front_url"),
                    ),
                )
                total += 1
            page += 1
            if page > 5:  # keep it to a manageable slice
                break
    conn.commit()
    return total

def build_from_leipzig(conn: sqlite3.Connection, dataset: str) -> int:
    """Build master catalog from a Leipzig benchmark (Amazon-Google or Abt-Buy)."""
    from load_leipzig_benchmark import build_master_catalog_from_benchmark

    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("DELETE FROM master_catalog")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='master_catalog'")

    master = build_master_catalog_from_benchmark(dataset)
    for master_id, gtin, brand, product_name, quantity in master:
        cur.execute(
            "INSERT INTO master_catalog (master_id, gtin, brand, product_name, quantity) "
            "VALUES (?, ?, ?, ?, ?)",
            (master_id, gtin, brand, product_name, quantity)
        )
    conn.commit()
    return len(master)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "real", "leipzig"], default="seed",
                        help="seed: synthetic 30-row catalog; "
                             "real: Open Food Facts; "
                             "leipzig: Amazon-Google or Abt-Buy benchmark")
    parser.add_argument("--dataset", choices=["amazon-google", "abt-buy"],
                        default="amazon-google",
                        help="which Leipzig benchmark (for --mode leipzig)")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if args.mode == "seed":
        n = build_seed(conn)
        print(f"Seed master catalog built: {n} products -> {DB_PATH}")
    elif args.mode == "real":
        n = build_real(conn)
        print(f"Real master catalog built from Open Food Facts: {n} products -> {DB_PATH}")
    elif args.mode == "leipzig":
        n = build_from_leipzig(conn, args.dataset)
        print(f"Master catalog built from {args.dataset}: {n} products -> {DB_PATH}")
    conn.close()

if __name__ == "__main__":
    sys.exit(main())
