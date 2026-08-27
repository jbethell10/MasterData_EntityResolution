"""
Load the Leipzig entity resolution benchmarks (Amazon-Google, Abt-Buy) and
expose them as a consumable format for the pipeline.

The benchmarks are real e-commerce data from 2010, with human-labelled
ground truth about which pairs refer to the same product.

  amazon-google: 1,363 × 3,226 = 1,300 labelled matches
  abt-buy:       1,081 × 1,092 = 1,081 labelled matches

They come as separate CSVs (Amazon.csv, GoogleProducts.csv, etc.) and a
"perfect mapping" file that lists which IDs refer to the same product.

We'll treat them as:
  - Master catalog: one side (e.g., Amazon)
  - Incoming feed: other side (e.g., Google), with intentional noise added
  - Gold truth: the perfect mapping file

This is text-only validation -- images are not available for general e-commerce
products, so we skip OCR and validate stages 02-08 on real matching signals.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "data" / "benchmark"


# Distinct GS1-style prefixes so a barcode carries which dataset it came from.
_DATASET_PREFIX = {"Amazon": "801", "Google": "801", "Abt": "802", "Buy": "802"}


def ean13_check_digit(body12: str) -> str:
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body12))
    return str((10 - total % 10) % 10)


def make_gtin(prefix: str, seq: int) -> str:
    """A real 13-digit EAN-13: 12 body digits plus a computed check digit."""
    body = f"{prefix}{seq:09d}"          # 3 + 9 = 12 digits
    return body + ean13_check_digit(body)


def load_csv(path: Path, encoding="latin-1") -> list[dict]:
    """Load a benchmark CSV."""
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def load_amazon_google() -> tuple[dict, dict, dict, dict]:
    """Load the Amazon-Google benchmark.

    Returns:
      amazon: {id -> row}
      google: {id -> row}
      mapping: {google_id -> amazon_id} for matched pairs (strings)
      amazon_to_master_id: {amazon_id_str -> master_id} (numeric IDs by insertion order)
    """
    amazon_csv = load_csv(BENCHMARK / "Amazon.csv")
    amazon = {r["id"]: r for r in amazon_csv}
    # Map original Amazon CSV IDs to numeric master IDs (1-based insertion order)
    amazon_to_master_id = {r["id"]: i for i, r in enumerate(amazon_csv, start=1)}

    google = {r["id"]: r for r in load_csv(BENCHMARK / "GoogleProducts.csv")}
    mapping = {}
    for m in load_csv(BENCHMARK / "Amzon_GoogleProducts_perfectMapping.csv"):
        # Store as {google_id -> amazon_id} so we can iterate supplier->master
        mapping[m["idGoogleBase"]] = m["idAmazon"]
    return amazon, google, mapping, amazon_to_master_id


def load_abt_buy() -> tuple[dict, dict, dict, dict]:
    """Load the Abt-Buy benchmark.

    Returns:
      abt: {id -> row}
      buy: {id -> row}
      mapping: {buy_id -> abt_id} for matched pairs (strings)
      abt_to_master_id: {abt_id_str -> master_id} (numeric IDs by insertion order)
    """
    abt_csv = load_csv(BENCHMARK / "Abt.csv")
    abt = {r["id"]: r for r in abt_csv}
    # Map original Abt CSV IDs to numeric master IDs (1-based insertion order)
    abt_to_master_id = {r["id"]: i for i, r in enumerate(abt_csv, start=1)}

    buy = {r["id"]: r for r in load_csv(BENCHMARK / "Buy.csv")}
    mapping = {}
    for m in load_csv(BENCHMARK / "abt_buy_perfectMapping.csv"):
        # Store as {buy_id -> abt_id} so we can iterate supplier->master
        mapping[m["idBuy"]] = m["idAbt"]
    return abt, buy, mapping, abt_to_master_id


def benchmark_to_master_catalog(
    supplier_side: dict[str, dict], source_name: str,
) -> list[tuple]:
    """Transform one side of a benchmark into master_catalog rows.

    Args:
      supplier_side: {id -> {'name': ..., 'manufacturer': ..., 'price': ...}}
      source_name: 'Amazon', 'Google', 'Abt', 'Buy'

    Returns:
      [(master_id, gtin, brand, product_name, quantity)]

    We synthesize GTINs as fake barcodes since the benchmark doesn't have
    real product codes. This is for stage 07 routing validation only.
    """
    prefix = _DATASET_PREFIX[source_name]
    rows = []
    for i, (orig_id, record) in enumerate(supplier_side.items(), start=1):
        # Use title or name field, depending on source
        title = record.get("name") or record.get("title", "Unknown")
        mfg = record.get("manufacturer", "")

        # The catalog side gets a synthetic but STRUCTURALLY VALID barcode, so
        # stage 06 can parse it rather than rejecting every row.
        #
        # The previous f"800{i:010d}7" was 14 characters, one too long for an
        # EAN-13, so is_valid_ean13() failed on all 1,092 rows and stage 06
        # reported insufficient_data everywhere -- while stage 04 was scoring
        # the very same string as a perfect GTIN match worth 0.30. Two stages
        # disagreeing about whether a field is usable is worse than either
        # answer alone.
        #
        # The prefix is per-dataset because the old index-only scheme gave
        # abt-buy row 436 and amazon-google row 436 the identical barcode,
        # so a catalog from one dataset would happily "verify" against events
        # from the other.
        rows.append((i, make_gtin(prefix, i), mfg, title, ""))
    return rows


def benchmark_to_intake(
    supplier_side: dict[str, dict],
    master_catalog: list[tuple],
    mapping: dict[str, str],
    buyer_side: dict[str, dict],
    source_name: str,
    max_events: int | None = None,
) -> list[dict]:
    """Generate intake events from a benchmark.

    Strategy: for each item in the supplier (e.g., Google), find its match
    in the master (e.g., Amazon) using the gold mapping, then construct an
    intake event as if the supplier submitted it.

    Args:
      supplier_side: the incoming data (e.g., Google), {id -> row}
      master_catalog: output of benchmark_to_master_catalog
      mapping: {supplier_id -> master_id} from gold truth
      buyer_side: the other side for comparison (for realism)
      source_name: 'amazon-google' or 'abt-buy'
      max_events: cap the number of events (for testing)

    Returns:
      list of intake event dicts
    """
    master_by_id = {r[0]: r for r in master_catalog}

    events = []
    for supplier_orig_id, master_orig_id in list(mapping.items())[:max_events or None]:
        if supplier_orig_id not in supplier_side or master_orig_id not in master_by_id:
            continue

        supplier_row = supplier_side[supplier_orig_id]
        master_row = master_by_id[int(master_orig_id)]

        title = supplier_row.get("name") or supplier_row.get("title", "")
        mfg = supplier_row.get("manufacturer", "")

        event = {
            "event_id": f"{source_name}_{len(events):04d}",
            "true_master_id": master_row[0],
            "supplier_raw_brand": mfg,
            "supplier_raw_product_name": title,
            "supplier_raw_quantity": "",
            # The supplier submits NO barcode.
            #
            # This previously copied master_row[1] -- the catalog record's own
            # GTIN -- into the submission, which handed stage 04 the answer:
            # gtin_similarity() scored an exact match of 1.0 against exactly
            # one catalog row, worth 0.30 of the combined score, for every
            # single event. That produced a "100% top-1 accuracy on real data"
            # figure that measured nothing but the leak. Withholding it, the
            # same engine scores 87.7% -- which is a real result, and matches
            # the independently-computed 85.1% top-1 the benchmark console
            # reports for this dataset.
            #
            # Empty is also the honest representation: the Leipzig benchmarks
            # are text-only catalogs. They contain no barcodes, so the pipeline
            # should run with that signal genuinely absent and report what its
            # absence costs.
            "supplier_raw_gtin": "",
            "supplier_norm_brand": mfg.lower().strip() if mfg else "",
            "supplier_norm_product_name": " ".join(title.lower().split()),
            "supplier_norm_quantity": "",
            "supplier_norm_gtin": "",
            # Since we don't have real images, leave artwork fields empty but include them
            "artwork_brand": "",
            "artwork_product_name": "",
            "artwork_quantity": "",
            "artwork_gtin": "",
            "artwork_norm_brand": "",
            "artwork_norm_quantity": "",
            # Master (ground truth)
            "master_brand": master_row[2],
            "master_product_name": master_row[3],
            "master_quantity": master_row[4],
            "master_gtin": master_row[1],
            "master_norm_brand": master_row[2].lower().strip() if master_row[2] else "",
            "master_norm_quantity": "",
        }
        events.append(event)

    return events


def build_master_catalog_from_benchmark(
    dataset: Literal["amazon-google", "abt-buy"],
) -> list[tuple]:
    """Build a master_catalog table from a Leipzig benchmark side."""
    if dataset == "amazon-google":
        amazon, google, mapping, amazon_to_master = load_amazon_google()
        return benchmark_to_master_catalog(amazon, "Amazon")
    elif dataset == "abt-buy":
        abt, buy, mapping, abt_to_master = load_abt_buy()
        return benchmark_to_master_catalog(abt, "Abt")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_intake_from_benchmark(
    dataset: Literal["amazon-google", "abt-buy"],
    max_events: int | None = None,
) -> list[dict]:
    """Build intake_events from a Leipzig benchmark."""
    if dataset == "amazon-google":
        amazon, google, mapping, amazon_to_master = load_amazon_google()
        master = benchmark_to_master_catalog(amazon, "Amazon")
        # Convert string mapping {google_id -> amazon_id} to numeric {google_id -> master_id}
        numeric_mapping = {gid: amazon_to_master[aid] for gid, aid in mapping.items()}
        return benchmark_to_intake(google, master, numeric_mapping, amazon,
                                    "amazon-google", max_events)
    elif dataset == "abt-buy":
        abt, buy, mapping, abt_to_master = load_abt_buy()
        master = benchmark_to_master_catalog(abt, "Abt")
        # Convert string mapping {buy_id -> abt_id} to numeric {buy_id -> master_id}
        numeric_mapping = {bid: abt_to_master[aid] for bid, aid in mapping.items()}
        return benchmark_to_intake(buy, master, numeric_mapping, abt,
                                    "abt-buy", max_events)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
