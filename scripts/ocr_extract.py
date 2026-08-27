"""
Runs Tesseract OCR over each artwork image and pulls out a best-effort
brand / product name / quantity / barcode reading -- the "what does the
pack actually say" record, independent of anything the supplier typed.

This mirrors the real pipeline's vision-extraction stage exactly; the only
difference on your own machine is swapping these rendered label images for
real downloaded photos (or using a vision-LLM call instead of Tesseract for
messier real-world packaging).
"""
import csv
import re
import sqlite3
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
DB_PATH = paths.run_dir("synthetic") / "mder.db"
IMG_DIR = ROOT / "images"
OUT_PATH = paths.run_dir("synthetic") / "artwork_extracted.csv"

BARCODE_RE = re.compile(r"\d{8,14}")


def ean13_check_digit(body12: str) -> str:
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body12))
    return str((10 - total % 10) % 10)


def is_valid_ean13(digits: str) -> bool:
    return (
        len(digits) == 13
        and digits.isdigit()
        and ean13_check_digit(digits[:12]) == digits[12]
    )


def pick_barcode(raw_text: str) -> str:
    """Choose the best barcode candidate from the OCR text.

    The original implementation took the FIRST run of 12-13 digits it found.
    That silently accepts a truncated read as if it were a real barcode: when
    Tesseract drops the trailing digit of 5000000000302 it yields the 12-digit
    500000000030, which the old regex happily returned as the GTIN. Here we
    gather every digit run and prefer one that actually passes the EAN-13
    check digit -- the same validator stage 06 is specified to use -- so a
    corrupted read is reported as a miss instead of a confident wrong answer.
    """
    # Strip spaces WITHIN each line, not across the whole blob. Tesseract
    # often splits a barcode's digits with spaces ("5000 0000 00302"), so the
    # spaces have to go -- but flattening the entire text first also welds
    # genuinely separate numbers into one long run, which then matches neither
    # a valid 13-digit GTIN nor the 12-13 digit fallback, and the barcode comes
    # back empty. Per-line keeps both cases working.
    candidates = []
    for line in raw_text.splitlines():
        candidates.extend(BARCODE_RE.findall(line.replace(" ", "")))
    for c in candidates:
        if is_valid_ean13(c):
            return c
    # No checksum-valid candidate. Fall back to the longest plausible run so
    # downstream fuzzy/Levenshtein GTIN comparison still has something to work
    # with, but it will (correctly) not equal the true GTIN.
    plausible = [c for c in candidates if 12 <= len(c) <= 13]
    return max(plausible, key=len) if plausible else ""

def extract_fields(raw_text: str) -> dict:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    brand = lines[0] if lines else ""
    product_name = lines[1] if len(lines) > 1 else ""
    weight_line = next((l for l in lines if "NET WEIGHT" in l.upper()), "")
    quantity = weight_line.upper().replace("NET WEIGHT", "").strip()
    barcode = pick_barcode(raw_text)
    return {"brand": brand, "product_name": product_name, "quantity": quantity, "gtin": barcode}

def ocr_fields(img: Image.Image) -> dict:
    """Read one artwork image with two OCR passes and arbitrate between them.

    Neither pass dominates the other: the plain pass reads brand text more
    reliably, while an unsharp-masked pass resolves the small digits in the
    quantity and barcode rows that blur/rotation noise degrades. Rather than
    guess which to trust globally, we run both and merge per field.

    The GTIN merge is not a guess -- EAN-13 carries its own check digit, so a
    reading that validates is objectively correct and a reading that doesn't
    is objectively corrupt. That gives an independent arbiter for the one
    field where a confident wrong answer is most damaging downstream (stage
    04 scores GTIN closeness, and stage 06 is specified to verify it 3 ways).
    Text fields have no such validator, so they use the plain pass and only
    fall back to the sharpened one when the plain pass read nothing at all.
    """
    plain = extract_fields(pytesseract.image_to_string(img))
    sharpened = ImageOps.grayscale(img).filter(
        ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2)
    )
    sharp = extract_fields(pytesseract.image_to_string(sharpened))

    merged = dict(plain)
    for field in ("brand", "product_name", "quantity"):
        if not merged.get(field, "").strip():
            merged[field] = sharp.get(field, "")

    # checksum arbitration: a validating GTIN from either pass beats a
    # non-validating one, regardless of which pass produced it.
    if not is_valid_ean13(merged.get("gtin", "")) and is_valid_ean13(sharp.get("gtin", "")):
        merged["gtin"] = sharp["gtin"]
    return merged


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _key(s: str) -> str:
    """Comparison key with ALL punctuation and spacing removed.

    Substituting punctuation with a space (as _norm does) silently fails the
    most common real case there is: the catalog says "Sainsbury's" and OCR
    reads "Sainsburys". Under _norm those become "sainsbury s" and
    "sainsburys", which do not match -- so a perfectly correct reading was
    being scored as a miss. Retailer brands are full of apostrophes and
    ampersands, so this was not a rare edge: it accounted for most of the
    apparent brand-recall failures on the Open Food Facts fronts.
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _brand_matches(extracted: str, true_brand: str) -> bool:
    """Synthetic packs put the brand on its own line, so this stays strict."""
    return _norm(extracted) == _norm(true_brand)


def ocr_read_whole_pack(img: Image.Image) -> dict:
    """Read a REAL photograph, where there is no layout to rely on.

    The synthetic parser takes line 0 as the brand, line 1 as the product name
    and the "NET WEIGHT" line as the quantity, because the renderer put them
    there. A photograph of an actual pack has no such contract: the brand may
    be a logo rather than text, the weight may be on a side panel or rotated,
    and half the visible text is ingredients, allergens, a recycling mark and a
    French translation.

    So instead of pretending to parse structure that is not there, this returns
    the whole text blob plus the one field that IS self-identifying -- the
    barcode, which carries its own check digit. Field accuracy is then measured
    as RECALL: does the true brand appear anywhere in what OCR read? That is
    both honest and the right question, because a downstream matcher would be
    handed the whole blob rather than a fabricated structure.

    Photos are downscaled first: contributor images run to several thousand
    pixels, which is slower without being more legible to Tesseract.
    """
    if max(img.size) > 1600:
        scale = 1600 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    plain = pytesseract.image_to_string(img)
    sharpened = ImageOps.grayscale(img).filter(
        ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    sharp = pytesseract.image_to_string(sharpened)

    text = plain + "\n" + sharp
    gtin = pick_barcode(text)
    return {"text": text, "gtin": gtin,
            "brand": "", "product_name": "", "quantity": ""}


def _found_in(needle: str, haystack: str, min_len: int = 4) -> bool:
    """Did OCR capture this field anywhere in what it read?

    Compared on the punctuation-free key so "Sainsbury's" matches "Sainsburys".
    Short needles are required to be at least `min_len` characters, because a
    two- or three-letter key will appear inside some longer word by chance and
    manufacture a false hit.
    """
    n = _key(needle)
    return len(n) >= min_len and n in _key(haystack)


def _images_for(mode: str):
    """(image_path, master_id) pairs for a run.

    Synthetic images are named by master_id; real ones by barcode, because the
    barcode is what Open Food Facts keys on and it survives a catalog rebuild
    where a row index would not.
    """
    if mode == "synthetic":
        for p in sorted(IMG_DIR.glob("artwork_*.png")):
            yield p, int(p.stem.split("_")[1])
        return
    real_dir = paths.run_dir("real") / "images"
    conn = sqlite3.connect(paths.db_path("real"))
    id_by_code = dict(conn.execute("SELECT gtin, master_id FROM master_catalog"))
    conn.close()
    for p in sorted(real_dir.glob("front_*.jpg")):
        code = p.stem.replace("front_", "")
        if code in id_by_code:
            yield p, id_by_code[code]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    args = ap.parse_args()

    global DB_PATH, OUT_PATH
    DB_PATH = paths.db_path(args.mode)
    OUT_PATH = paths.run_dir(args.mode) / "artwork_extracted.csv"

    conn = sqlite3.connect(DB_PATH)
    master = {
        row[0]: row[1:] for row in conn.execute(
            "SELECT master_id, gtin, brand, product_name, quantity FROM master_catalog"
        )
    }
    conn.close()

    results = []
    correct = {"brand": 0, "quantity": 0, "gtin": 0}
    total = 0

    real = args.mode == "real"
    for img_path, master_id in _images_for(args.mode):
        extracted = (ocr_read_whole_pack if real else ocr_fields)(Image.open(img_path))

        true_gtin, true_brand, true_name, true_qty = master[master_id]
        total += 1
        if real:
            # Recall against the whole blob -- see ocr_read_whole_pack.
            blob = extracted["text"]
            brand_ok = _found_in(true_brand, blob)
            qty_ok = _found_in(true_qty, blob)
            # keep the FULL text: field-by-field recall is diagnostic, but the
            # question that decides whether vision extraction is usable is
            # whether the blob retrieves the right catalog row, and that needs
            # all of it.
            extracted = dict(extracted,
                             brand=" ".join(blob.split()),
                             product_name="", quantity="")
        else:
            brand_ok = _brand_matches(extracted["brand"], true_brand)
            qty_ok = (extracted["quantity"].replace(" ", "").lower()
                      == (true_qty or "").replace(" ", "").lower())
        gtin_ok = extracted["gtin"] == true_gtin
        correct["brand"] += brand_ok
        correct["quantity"] += qty_ok
        correct["gtin"] += gtin_ok

        results.append({
            "master_id": master_id,
            "image": img_path.name,
            "extracted_brand": extracted["brand"],
            "true_brand": true_brand,
            "brand_match": brand_ok,
            "extracted_quantity": extracted["quantity"],
            "true_quantity": true_qty,
            "quantity_match": qty_ok,
            "extracted_gtin": extracted["gtin"],
            "true_gtin": true_gtin,
            "gtin_match": gtin_ok,
        })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"OCR-extracted {total} artwork images -> {OUT_PATH}")
    print(f"  brand match:    {correct['brand']}/{total} ({100*correct['brand']/total:.0f}%)")
    print(f"  quantity match: {correct['quantity']}/{total} ({100*correct['quantity']/total:.0f}%)")
    print(f"  GTIN match:     {correct['gtin']}/{total} ({100*correct['gtin']/total:.0f}%)")
    print("\nSpot-check (first 5):")
    for r in results[:5]:
        print(f"  [{r['master_id']}] brand: '{r['extracted_brand']}' vs '{r['true_brand']}' -> {r['brand_match']}")

if __name__ == "__main__":
    main()
