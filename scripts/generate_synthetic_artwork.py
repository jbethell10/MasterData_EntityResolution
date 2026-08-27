"""
Generates synthetic packaging-label images to stand in for real product
photos (this sandbox can't reach world.openfoodfacts.org's image CDN).

Each image renders the brand, product name, and quantity as they'd appear
printed on a pack, plus the barcode digits as a human-readable string below
a drawn barcode-style pattern -- enough for the OCR step to exercise real
image-to-text extraction, with occasional print-quality noise (rotation,
blur, low contrast) so it isn't a trivial clean-text read.

On your own machine, replace this step with real downloaded photos from
Open Food Facts' image_front_url field (already captured in the real-mode
master catalog) -- the OCR script doesn't care where the image came from.
"""
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from fonts import active_font_paths, load_label_fonts

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
DB_PATH = paths.run_dir("synthetic") / "mder.db"
IMG_DIR = ROOT / "images"

random.seed(42)

def draw_barcode(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, digits: str):
    """Draw a plausible-looking (not scannable) barcode pattern."""
    n_bars = len(digits) * 3
    bar_w = w / n_bars
    for i in range(n_bars):
        # deterministic pseudo-random bar widths seeded by the digit
        seed = int(digits[i % len(digits)])
        if (i + seed) % 2 == 0:
            bx = x + i * bar_w
            draw.rectangle([bx, y, bx + bar_w * 0.6, y + h], fill="black")

# Render at 2x and keep it there. The README's documented quantity-OCR
# failure ("32.5g" -> "32.59", "4x115g" -> "4X1159") is a resolution problem:
# at 16px the lowercase g/9 and x/X glyphs are only a few pixels apart.
# Rendering the whole label at 2x is the fix the README itself proposes
# ("a larger font / higher-res render"), and it costs nothing downstream --
# ocr_extract.py doesn't care about image dimensions.
SCALE = 2


def make_label_image(brand: str, product_name: str, quantity: str, gtin: str, out_path: Path):
    S = SCALE
    W, H = 500 * S, 320 * S
    img = Image.new("RGB", (W, H), color=(250, 248, 240))
    draw = ImageDraw.Draw(img)

    font_brand, font_name, font_small = load_label_fonts(30 * S, 22 * S, 16 * S)

    draw.rectangle([0, 0, W - 1, H - 1], outline=(120, 120, 120), width=2 * S)
    draw.text((30 * S, 30 * S), brand, font=font_brand, fill=(20, 20, 20))
    draw.text((30 * S, 80 * S), product_name, font=font_name, fill=(30, 30, 30))
    draw.text((30 * S, 115 * S), f"NET WEIGHT {quantity}", font=font_small, fill=(60, 60, 60))

    # Barcode block near the bottom. The bars and the human-readable digits
    # need a real gap between them: with only ~5px of clearance Tesseract
    # merges the bar pattern and the digit row into one line, reads the bars
    # as garbage glyphs ("UAL AQLUUML UUET") and drops the digits entirely,
    # so the GTIN comes back empty. Real packs separate these too.
    draw_barcode(draw, 30 * S, 205 * S, 260 * S, 45 * S, gtin)
    draw.text((30 * S, 278 * S), gtin, font=font_small, fill=(10, 10, 10))

    # print-quality noise so OCR has to do real work
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, expand=True, fillcolor=(250, 248, 240))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.1 * SCALE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def main(sample_size: int = 20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT master_id, gtin, brand, product_name, quantity FROM master_catalog"
    ).fetchall()
    conn.close()

    # Clear stale renders first. Without this, a re-run leaves the previous
    # run's images behind; ocr_extract.py globs artwork_*.png indiscriminately
    # and will either score images that no longer correspond to any catalog
    # row or crash outright with a KeyError on a stale master_id.
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stale = list(IMG_DIR.glob("artwork_*.png"))
    for old in stale:
        old.unlink()

    regular, bold = active_font_paths()
    print(f"Rendering with font: {regular}")
    if stale:
        print(f"Cleared {len(stale)} stale artwork image(s) from a previous run")

    sample = random.sample(rows, min(sample_size, len(rows)))
    manifest = []
    for master_id, gtin, brand, product_name, quantity in sample:
        out_path = IMG_DIR / f"artwork_{master_id}.png"
        make_label_image(brand, product_name, quantity or "", gtin, out_path)
        manifest.append((master_id, str(out_path)))

    print(f"Generated {len(manifest)} synthetic artwork images -> {IMG_DIR}")
    for master_id, path in manifest[:5]:
        print(f"  master_id={master_id} -> {path}")

if __name__ == "__main__":
    main()
