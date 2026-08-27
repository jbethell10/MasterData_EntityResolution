"""
Stage 06 -- Three-way barcode verification.

Until now the GTIN comparison lived informally inside stage 03's field loop,
which treated it as just another string field. The build guide calls it out as
its own stage for a reason: a barcode is not a fuzzy attribute like a product
name, it's a checksummed identifier, and the *pattern* of agreement across
three independent sources carries information that a pairwise string compare
throws away.

The three sources:
  - artwork  : digits OCR'd off the pack
  - supplier : digits the supplier keyed into the portal
  - master   : the GTIN already in the catalog for the candidate being tested

What this stage adds over "do the strings match":

1. CHECK-DIGIT VALIDATION. An EAN-13 carries its own checksum, so we can tell
   a corrupt reading apart from a valid barcode belonging to a different
   product. Those two cases look identical to a string compare but mean
   opposite things: the first is noise, the second is contradictory evidence.

2. AGREEMENT PATTERNS, NOT A BOOLEAN. Per the guide, two-of-three agreement is
   a meaningfully stronger signal than one external check, and it matters
   WHICH two agree -- artwork+supplier agreeing against master says something
   very different (both readings of the physical pack agree; the catalog may
   be stale) from supplier+master agreeing against artwork (probably just OCR
   noise on the pack).

3. A FULL THREE-WAY MISMATCH IS A VETO. The guide is explicit that this routes
   to human review "regardless of what the text match says" -- stage 07 honours
   that as a hard override rather than letting a high text score average it
   away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class BarcodeVerdict(str, Enum):
    ALL_THREE_AGREE = "all_three_agree"
    TWO_OF_THREE = "two_of_three"
    FULL_MISMATCH = "full_mismatch"
    INSUFFICIENT = "insufficient_data"   # fewer than 2 usable barcodes


# How much each verdict contributes as a confidence signal in stage 07.
VERDICT_SCORE = {
    BarcodeVerdict.ALL_THREE_AGREE: 1.0,
    BarcodeVerdict.TWO_OF_THREE: 0.65,
    BarcodeVerdict.FULL_MISMATCH: 0.0,
    BarcodeVerdict.INSUFFICIENT: 0.0,
}


def normalize_gtin(raw: str) -> str:
    return re.sub(r"\D", "", raw or "")


def ean13_check_digit(body12: str) -> str:
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body12))
    return str((10 - total % 10) % 10)


def is_valid_ean13(digits: str) -> bool:
    return (
        len(digits) == 13 and digits.isdigit()
        and ean13_check_digit(digits[:12]) == digits[12]
    )


def _mod10_check_digit(body: str, weights=(3, 1)) -> str:
    """GS1 mod-10, weighting from the RIGHT so it works for any even/odd length.

    EAN-13, EAN-8 and UPC-A all use the same mod-10 scheme; they differ only in
    length and therefore in which position the 3-weight lands on. Computing from
    the right handles all three without a per-format table.
    """
    total = sum(int(d) * weights[i % 2] for i, d in enumerate(reversed(body)))
    return str((10 - total % 10) % 10)


def is_valid_gtin(digits: str) -> bool:
    """Any GS1 retail barcode: EAN-13, UPC-A (12), EAN-8, or GTIN-14.

    Stage 06 originally accepted EAN-13 alone, which was fine for a seed catalog
    that only ever minted EAN-13s. Measured against a real Open Food Facts
    catalog, 14.2% of products carry an EAN-8 -- small packs, where a 13-digit
    symbol will not physically fit on the packaging. Rejecting those as
    unparseable would have discarded the barcode evidence for one product in
    seven, and silently: they would have surfaced as "insufficient_data", which
    reads like a missing barcode rather than a barcode we refused to understand.
    """
    d = (digits or "").strip()
    if not d.isdigit() or len(d) not in (8, 12, 13, 14):
        return False
    return _mod10_check_digit(d[:-1]) == d[-1]


def gtin_format(digits: str) -> str:
    d = (digits or "").strip()
    if not d.isdigit():
        return "non_numeric"
    return {8: "ean8", 12: "upc12", 13: "ean13", 14: "gtin14"}.get(
        len(d), f"len{len(d)}")


def normalize_to_gtin14(digits: str) -> str:
    """Zero-pad to 14 so formats compare correctly.

    A UPC-A and the EAN-13 that encodes the same product differ only by a
    leading zero, so comparing the raw strings reports a mismatch for two codes
    that GS1 considers identical. Padding both to 14 is the standard way to
    compare across formats.
    """
    d = (digits or "").strip()
    return d.zfill(14) if d.isdigit() else d


@dataclass
class BarcodeCheck:
    verdict: BarcodeVerdict
    score: float
    agreeing: tuple[str, ...]          # which sources agree, e.g. ("artwork", "supplier")
    dissenting: tuple[str, ...]
    valid_sources: tuple[str, ...]     # which passed the check digit
    corrupt_sources: tuple[str, ...]   # present but failed the check digit
    detail: str

    @property
    def is_veto(self) -> bool:
        """A full three-way mismatch between usable barcodes forces review."""
        return self.verdict is BarcodeVerdict.FULL_MISMATCH

    def as_row(self) -> dict:
        return {
            "barcode_verdict": self.verdict.value,
            "barcode_score": round(self.score, 3),
            "barcode_agreeing": "|".join(self.agreeing),
            "barcode_corrupt_sources": "|".join(self.corrupt_sources),
            "barcode_detail": self.detail,
        }


def verify_three_way(artwork_gtin: str, supplier_gtin: str, master_gtin: str) -> BarcodeCheck:
    raw = {
        "artwork": normalize_gtin(artwork_gtin),
        "supplier": normalize_gtin(supplier_gtin),
        "master": normalize_gtin(master_gtin),
    }
    present = {k: v for k, v in raw.items() if v}

    # A barcode that fails its own check digit is a corrupt READING, not a
    # claim about a different product. Excluding it from the agreement vote
    # stops OCR noise from manufacturing a "mismatch" that vetoes the row --
    # while still recording it, because a corrupt read is itself worth seeing.
    #
    # Accepts every GS1 retail format, not just EAN-13: 14.2% of a real Open
    # Food Facts catalog is EAN-8, and calling those unparseable would throw
    # away the barcode evidence for one product in seven.
    valid = {k: normalize_to_gtin14(v) for k, v in present.items() if is_valid_gtin(v)}
    corrupt = tuple(sorted(k for k in present if k not in valid))

    if len(valid) < 2:
        return BarcodeCheck(
            verdict=BarcodeVerdict.INSUFFICIENT,
            score=VERDICT_SCORE[BarcodeVerdict.INSUFFICIENT],
            agreeing=(), dissenting=tuple(sorted(valid)),
            valid_sources=tuple(sorted(valid)), corrupt_sources=corrupt,
            detail=(f"only {len(valid)} checksum-valid barcode(s) available"
                    + (f"; corrupt reading from {', '.join(corrupt)}" if corrupt else "")),
        )

    # group the valid sources by the value they report
    by_value: dict[str, list[str]] = {}
    for source, value in valid.items():
        by_value.setdefault(value, []).append(source)

    largest = max(by_value.values(), key=len)
    agreeing = tuple(sorted(largest))

    if len(by_value) == 1:
        verdict = (BarcodeVerdict.ALL_THREE_AGREE if len(valid) == 3
                   else BarcodeVerdict.TWO_OF_THREE)
        detail = f"{', '.join(agreeing)} agree on {valid[agreeing[0]]}"
        if len(valid) == 2:
            detail += " (third source unusable)"
    elif len(largest) >= 2:
        verdict = BarcodeVerdict.TWO_OF_THREE
        dissent = sorted(set(valid) - set(largest))
        detail = (f"{', '.join(agreeing)} agree on {valid[agreeing[0]]}; "
                  f"{', '.join(dissent)} reports {valid[dissent[0]]}")
    else:
        verdict = BarcodeVerdict.FULL_MISMATCH
        detail = "all sources report different valid barcodes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(valid.items())
        )
        agreeing = ()

    return BarcodeCheck(
        verdict=verdict,
        score=VERDICT_SCORE[verdict],
        agreeing=agreeing,
        dissenting=tuple(sorted(set(valid) - set(agreeing))),
        valid_sources=tuple(sorted(valid)),
        corrupt_sources=corrupt,
        detail=detail,
    )
