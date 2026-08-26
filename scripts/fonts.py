"""
Cross-platform font resolution for the synthetic artwork renderer.

The original renderer hardcoded a single Debian/Ubuntu DejaVu path and fell
back to PIL's built-in bitmap font inside a bare `except OSError`. That
fallback is silent and catastrophic for this pipeline: the default font is a
tiny bitmap face that Tesseract cannot read, so OCR accuracy collapses from
~100% to near-zero with no error and no warning -- the numbers just quietly
get worse. This module resolves a real scalable TTF on macOS, Linux, or
Windows, and raises loudly if it genuinely can't find one, so a font problem
can never again masquerade as an OCR-accuracy problem.
"""
from pathlib import Path

from PIL import ImageFont

# (regular, bold) candidate pairs, most-preferred first. DejaVu is listed
# first so a Linux box reproduces the original renders byte-for-byte.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/Library/Fonts/Arial.ttf",
     "/Library/Fonts/Arial Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf",
     "C:/Windows/Fonts/arialbd.ttf"),
]


def _first_available() -> tuple[str, str]:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            return regular, bold
    raise RuntimeError(
        "No scalable TTF font found for artwork rendering. Tried:\n  "
        + "\n  ".join(f"{r} + {b}" for r, b in _FONT_CANDIDATES)
        + "\n\nRefusing to fall back to PIL's default bitmap font, which is "
          "unreadable to Tesseract and would silently destroy OCR accuracy. "
          "Install DejaVu (Linux: `apt install fonts-dejavu`) or point "
          "_FONT_CANDIDATES at a TTF that exists on this machine."
    )


def load_label_fonts(brand_size: int, name_size: int, small_size: int):
    """Return (font_brand, font_name, font_small) as real scalable TTFs."""
    regular, bold = _first_available()
    return (
        ImageFont.truetype(bold, brand_size),
        ImageFont.truetype(regular, name_size),
        ImageFont.truetype(regular, small_size),
    )


def active_font_paths() -> tuple[str, str]:
    """(regular, bold) actually in use -- printed by the renderer so the font
    backing a given run is recorded rather than assumed."""
    return _first_available()
