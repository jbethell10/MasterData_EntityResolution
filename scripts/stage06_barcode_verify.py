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
    valid = {k: v for k, v in present.items() if is_valid_ean13(v)}
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
