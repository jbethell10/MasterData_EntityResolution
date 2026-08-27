"""
Builds "intake events" -- one real-world moment where a single product
arrives at the pipeline with THREE independent views of the truth:

  1. artwork  -- OCR'd straight off packaging photo (synthetic mode)
                 or empty for text-only benchmarks
  2. supplier -- what the supplier typed into the portal
  3. master   -- the canonical catalog record (ground truth for scoring)

Two modes:

  --mode synthetic (default): uses local artwork_*.png files and manufactures
    fresh, independent corruptions for each product. The three views come
    together in this script so stage 03 compares them.

  --mode leipzig: reads from a master_catalog built from a Leipzig benchmark
    (Amazon-Google or Abt-Buy). No artwork (text-only). Supplier is the other
    side of the benchmark. This validates stages 02-08 on real matching
    without real images (which aren't available for general e-commerce).
"""
import argparse
import csv
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from normalize import normalize_text, normalize_quantity  # noqa: E402

import paths  # noqa: E402

DB_PATH = paths.db_path("synthetic")     # repointed in main()
IMG_DIR = ROOT.parent / "images"
OUT_PATH = paths.run_dir("synthetic") / "intake_events.csv"

EVENT_SEED = 99


def build_synthetic():
    """Original mode: artwork images + fresh corruptions."""
    from corrupt_supplier_feed import corrupt_row  # noqa: E402
    from ocr_extract import ocr_fields              # noqa: E402
    from PIL import Image

    conn = sqlite3.connect(DB_PATH)
    master_rows = {
        row[0]: row for row in conn.execute(
            "SELECT master_id, gtin, brand, product_name, quantity, category FROM master_catalog"
        )
    }
    conn.close()

    events = []
    for img_path in sorted(IMG_DIR.glob("artwork_*.png")):
        master_id = int(img_path.stem.split("_")[1])
        row = master_rows[master_id]
        _, true_gtin, true_brand, true_name, true_qty, true_cat = row

        random.seed(EVENT_SEED + master_id)
        supplier = corrupt_row(row)
        artwork = ocr_fields(Image.open(img_path))

        events.append({
            "event_id": f"evt_{master_id}",
            "true_master_id": master_id,
            "image_path": str(img_path.relative_to(ROOT.parent)),
            "artwork_brand": artwork["brand"],
            "artwork_product_name": artwork["product_name"],
            "artwork_quantity": artwork["quantity"],
            "artwork_gtin": artwork["gtin"],
            "artwork_norm_brand": normalize_text(artwork["brand"]),
            "artwork_norm_quantity": normalize_quantity(artwork["quantity"]),
            "supplier_raw_brand": supplier["submitted_brand"],
            "supplier_raw_product_name": supplier["submitted_product_name"],
            "supplier_raw_quantity": supplier["submitted_quantity"],
            "supplier_raw_gtin": supplier["submitted_gtin"],
            "supplier_norm_brand": normalize_text(supplier["submitted_brand"]),
            "supplier_norm_product_name": normalize_text(supplier["submitted_product_name"]),
            "supplier_norm_quantity": normalize_quantity(supplier["submitted_quantity"]),
            "supplier_norm_gtin": re.sub(r"\D", "", supplier["submitted_gtin"]),
            "supplier_corruption_applied": supplier["corruption_applied"],
            "master_brand": true_brand,
            "master_product_name": true_name,
            "master_quantity": true_qty,
            "master_gtin": true_gtin,
            "master_norm_brand": normalize_text(true_brand),
            "master_norm_quantity": normalize_quantity(true_qty or ""),
        })

    return events


def build_leipzig(dataset: str, max_events: int | None = None):
    """Leipzig benchmark mode: text-only, no artwork."""
    from load_leipzig_benchmark import build_intake_from_benchmark  # noqa: E402

    return build_intake_from_benchmark(dataset, max_events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "leipzig"], default="synthetic",
                        help="synthetic: use local artwork images (default); "
                             "leipzig: use Leipzig benchmark (text-only)")
    parser.add_argument("--dataset", choices=["amazon-google", "abt-buy"],
                        default="amazon-google",
                        help="which Leipzig benchmark (for --mode leipzig)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of events (for testing)")
    args = parser.parse_args()

    global DB_PATH, OUT_PATH
    run_dir, DB_PATH = paths.resolve(args)
    OUT_PATH = run_dir / "intake_events.csv"

    if args.mode == "synthetic":
        events = build_synthetic()
    elif args.mode == "leipzig":
        events = build_leipzig(args.dataset, args.limit)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else [])
        writer.writeheader()
        writer.writerows(events)

    print(f"Built {len(events)} intake events ({args.mode} mode) -> {OUT_PATH}")
    for e in events[:3]:
        print(f"  [{e['event_id']}] supplier='{e['supplier_raw_brand']} {e['supplier_raw_product_name'][:40]}'  "
              f"true_id={e['true_master_id']}")


if __name__ == "__main__":
    main()
