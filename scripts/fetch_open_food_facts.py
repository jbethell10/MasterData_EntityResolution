"""
Fetch a real product catalog from Open Food Facts.

  python3 scripts/fetch_open_food_facts.py                 # catalog only
  python3 scripts/fetch_open_food_facts.py --images 120    # + packaging photos

Why this matters more than the benchmark work: until now every number in this
project came from data the pipeline generated itself, or from text-only Leipzig
catalogs with no barcodes and no photographs. Open Food Facts gives all three
legs of the problem for the first time --

  a real product record   (brand, name, quantity, category)
  a REAL barcode          (the actual GS1 code on the pack)
  a REAL photograph       (a shopper's phone photo of the pack)

which means stage 01's OCR faces real packaging instead of a clean render, and
stage 06's three-way barcode verification does real work instead of reporting
"insufficient" on every row.

Expect the numbers to get WORSE. Synthetic renders scored 100% brand / 95% GTIN
because they were drawn by the same code that graded them: one font, dead-on
angle, flat lighting, no glare, no curved surfaces, no foreign-language panels.
A real photo has all of those. A drop is the measurement working, not a
regression.

POLITENESS
Open Food Facts is a free, volunteer-run, donation-funded service. This script
identifies itself properly, sleeps between requests, asks for only the fields it
uses, and caches every response to disk so re-running costs them nothing. Data
is ODbL-licensed; product photos are contributor-owned and used here only for a
local accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

API = "https://world.openfoodfacts.org/api/v2/search"
UA = "MDER-Prototype/1.0 (entity-resolution portfolio project; jacobbethell09@gmail.com)"

# Categories chosen to make matching genuinely hard: lots of same-brand,
# near-identical products separated only by flavour or pack size, which is
# exactly the confusion a master-data system has to survive.
CATEGORIES = [
    "chocolates", "breakfast-cereals", "sodas", "yogurts", "crisps",
    "biscuits", "teas", "coffees", "juices", "cheeses",
]

REQUEST_PAUSE = 1.2   # seconds between API calls
IMAGE_PAUSE = 0.4     # seconds between image downloads


def cache_dir() -> Path:
    d = paths.run_dir("real") / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Bumped when the requested field set changes, so cached pages from an older
# schema are not silently reused and quietly missing the new fields.
CACHE_VERSION = "v2"


def best_image_urls(product: dict) -> dict[str, str]:
    """Full-resolution URLs per view, preferring the largest stored size.

    OFF keeps each photo at 100/200/400px and 'full'. The search API's
    image_front_url hands back the 400px display copy, which is a thumbnail.
    """
    out: dict[str, str] = {}
    for view, sizes in (product.get("selected_images") or {}).items():
        # sizes looks like {"display": {...}, "small": {...}, "thumb": {...}}
        # each mapping language -> url
        for bucket in ("display", "small", "thumb"):
            urls = (sizes or {}).get(bucket) or {}
            if urls:
                url = next(iter(urls.values()))
                # promote the stored size to the original upload
                out[view] = re.sub(r"\.(100|200|400)\.jpg$", ".full.jpg", url)
                break
    return out


def fetch_page(category: str, page: int, page_size: int = 100) -> dict:
    """One page of results, served from disk if we've already asked for it."""
    cached = cache_dir() / f"{category}_p{page}_{CACHE_VERSION}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = API + "?" + urllib.parse.urlencode({
        "categories_tags_en": category,
        "countries_tags_en": "united-kingdom",
        # selected_images carries every view the contributors uploaded --
        # front, ingredients, nutrition and packaging -- at every stored size.
        # image_front_url alone is not enough for two reasons found the hard
        # way: it points at the 400px THUMBNAIL, on which whole-pack text is a
        # few pixels tall and Tesseract reads nothing; and the front of a pack
        # does not carry the barcode, which lives on the back. Measuring OCR
        # barcode accuracy against front thumbnails scored 0/160 -- not because
        # the reading is hard but because the digits were never in the frame.
        "fields": "code,brands,product_name,quantity,categories,"
                  "image_front_url,selected_images",
        "page_size": page_size,
        "page": page,
    })
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    # OFF returns 503 under load. It is a donation-funded service, so the right
    # response is to wait longer rather than hammer it -- exponential backoff,
    # and give up after a few tries instead of retrying indefinitely.
    delay = REQUEST_PAUSE
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                payload = json.load(resp)
            cached.write_text(json.dumps(payload))
            time.sleep(REQUEST_PAUSE)
            return payload
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503, 504) or attempt == 3:
                raise
            delay *= 2.5
            print(f"    {category} p{page}: HTTP {e.code}, waiting {delay:.0f}s")
            time.sleep(delay)
    raise urllib.error.URLError("retries exhausted")


def ean13_check_digit(body12: str) -> str:
    return str((10 - sum(int(d) * (3 if i % 2 else 1)
                         for i, d in enumerate(body12)) % 10) % 10)


def barcode_status(code: str) -> str:
    """Real barcodes are messy: EAN-13, EAN-8, UPC-12, and plenty of junk.

    Reported honestly rather than filtered silently -- how many real catalog
    codes fail their own checksum is itself a finding, and stage 06 has to cope
    with whatever proportion that turns out to be.
    """
    c = (code or "").strip()
    if not c.isdigit():
        return "non_numeric"
    if len(c) == 13:
        return "ean13" if ean13_check_digit(c[:12]) == c[12] else "ean13_bad_check"
    if len(c) == 12:
        return "upc12"
    if len(c) == 8:
        return "ean8"
    return "other_length"


def collect(max_pages: int = 4) -> list[dict]:
    seen: dict[str, dict] = {}
    for cat in CATEGORIES:
        for page in range(1, max_pages + 1):
            try:
                payload = fetch_page(cat, page)
            except urllib.error.URLError as e:
                print(f"  ! {cat} p{page}: {e.reason} — skipping")
                break
            products = payload.get("products", [])
            if not products:
                break
            kept = 0
            for p in products:
                code = (p.get("code") or "").strip()
                name = (p.get("product_name") or "").strip()
                brand = (p.get("brands") or "").split(",")[0].strip()
                # A record with no name or no code cannot be matched or verified.
                if not code or not name or code in seen:
                    continue
                views = best_image_urls(p)
                seen[code] = {
                    "code": code,
                    "brand": brand,
                    "product_name": name,
                    "quantity": (p.get("quantity") or "").strip(),
                    "category": cat,
                    # full-resolution URLs per view; 'packaging' is usually the
                    # back of the pack, which is where the barcode actually is
                    "image_urls": views,
                    "image_url": views.get("front", p.get("image_front_url") or ""),
                    "has_packaging_view": "packaging" in views,
                    "barcode_status": barcode_status(code),
                }
                kept += 1
            print(f"  {cat:20s} page {page}: +{kept:3d} (total {len(seen)})")
    return list(seen.values())


def _get(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 5000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 5000:          # placeholder / error page, not a photo
            return False
        dest.write_bytes(data)
        time.sleep(IMAGE_PAUSE)
        return True
    except Exception:
        return False


def download_images(records: list[dict], limit: int) -> dict:
    """Download BOTH the front and the packaging view, at full resolution.

    Two views because they answer different questions and only one of them can
    answer the barcode question: the front carries the brand and product name,
    the packaging shot is the back of the pack and is where the barcode is
    printed. The first pass of this project downloaded only fronts, at 400px,
    and then reported "0/160 GTIN accuracy" as though OCR had failed.
    """
    img_dir = paths.run_dir("real") / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    counts = {"front": 0, "packaging": 0, "products": 0}

    # Prefer products that have BOTH views -- those are the ones that can
    # exercise the whole extraction stage rather than half of it.
    ranked = sorted(records, key=lambda r: not r.get("has_packaging_view"))
    for r in ranked:
        if counts["products"] >= limit:
            break
        urls = r.get("image_urls") or {}
        if not urls.get("front"):
            continue
        got_any = False
        for view in ("front", "packaging"):
            if view not in urls:
                continue
            dest = img_dir / f"{view}_{r['code']}.jpg"
            if _get(urls[view], dest):
                counts[view] += 1
                got_any = True
        if got_any:
            counts["products"] += 1
            if counts["products"] % 25 == 0:
                print(f"  {counts['products']}/{limit} products "
                      f"({counts['front']} front, {counts['packaging']} packaging)")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=4,
                    help="pages of 100 per category")
    ap.add_argument("--images", type=int, default=0,
                    help="how many packaging photos to download")
    args = ap.parse_args()

    run = paths.ensure("real")
    print(f"Fetching Open Food Facts (UK, {len(CATEGORIES)} categories)…")
    records = collect(args.pages)
    print(f"\n{len(records)} unique products")

    status = {}
    for r in records:
        status[r["barcode_status"]] = status.get(r["barcode_status"], 0) + 1
    print("\nBarcode quality in a REAL catalog:")
    for k, v in sorted(status.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v:5d}  ({v/len(records):5.1%})")

    with_img = sum(1 for r in records if r["image_url"])
    print(f"\nwith a front photo: {with_img}/{len(records)} ({with_img/len(records):.1%})")

    have_pack = sum(1 for r in records if r.get("has_packaging_view"))
    print(f"with a packaging (back-of-pack) view: {have_pack}/{len(records)} "
          f"({have_pack/len(records):.1%})  <- the only view with a barcode on it")

    if args.images:
        print(f"\nDownloading full-resolution photos for up to {args.images} products…")
        c = download_images(records, args.images)
        print(f"  {c['products']} products: {c['front']} front + "
              f"{c['packaging']} packaging -> {run/'images'}")

    out = run / "catalog_raw.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
