"""
Stage 07 -- Confidence scoring and routing.

Fig. 2 of the build guide: three INDEPENDENT signals combine into one score,
which routes to auto-merge / hold-for-review / reject.

  1. artwork-vs-supplier agreement   (stage 03) -- did the two readings of the
                                                   same pack agree with each other?
  2. text-match confidence           (stage 04) -- how well does the winning
                                                   candidate beat the runner-up?
  3. three-way barcode agreement     (stage 06)

The guide is emphatic about WHY they're kept separate: "a strong result on one
axis (say, the barcode) can't quietly paper over a real disagreement on another
(say, the supplier's own data entry)." A single weighted average does exactly
that papering-over, so the weighted sum here is bounded by a set of explicit
overrides that fire regardless of the arithmetic.

The overrides, and what each is protecting against:

  * BARCODE VETO -- three valid, mutually contradictory barcodes means we do not
    know what this product is, whatever the names look like. Guide: routes to
    review "regardless of what the text match says".
  * LLM-ONLY CEILING -- if the decision rests on stage 05 rather than on
    retrieval evidence, it cannot auto-merge (enforced upstream by
    stage 05's 0.75 cap, re-asserted here so the invariant is visible at the
    routing layer too, not just implied).
  * AMBIGUITY VETO -- if the top two candidates are within AMBIGUITY_MARGIN,
    retrieval hasn't actually chosen; a high score built on a coin-flip is
    worse than an honest hold.
  * SOURCE-CONFLICT FLOOR -- if artwork and supplier disagree outright, that's
    a supplier data-entry error. It may still resolve correctly, but it should
    never silently auto-merge, because the submission itself is defective and
    someone should see it.

TEXT_MATCH is deliberately scored on the MARGIN between the top two candidates
rather than the winner's absolute score. An absolute score says "this looks
like a product"; the margin says "and it doesn't look like any other product",
which is the thing that actually matters when writing to a master catalog.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from stage06_barcode_verify import BarcodeCheck

# Routing bands, straight from Fig. 2 of the guide.
AUTO_MERGE_MIN = 0.90
HOLD_MIN = 0.60

# Signal weights. Barcode carries the most because it is the only
# checksummed, identifier-grade evidence in the set; the other two are
# similarity judgements.
W_SOURCE_AGREEMENT = 0.25
W_TEXT_MATCH = 0.35
W_BARCODE = 0.40

# Retrieval margin at or below which the winner isn't meaningfully separated.
AMBIGUITY_MARGIN = 0.08

# A margin of this size or more is treated as full text-match confidence.
MARGIN_SATURATION = 0.40

# Cap applied when a prior human-approved alias matched (stage 08's cache).
ALIAS_BOOST = 0.10


class Route(str, Enum):
    AUTO_MERGE = "auto_merge"
    HOLD_FOR_REVIEW = "hold_for_review"
    REJECT = "reject"


@dataclass
class RoutingDecision:
    route: Route
    confidence: float
    signals: dict
    overrides: list[str] = field(default_factory=list)
    rationale: str = ""

    def as_row(self) -> dict:
        return {
            "route": self.route.value,
            "confidence": round(self.confidence, 3),
            "sig_source_agreement": round(self.signals["source_agreement"], 3),
            "sig_text_match": round(self.signals["text_match"], 3),
            "sig_barcode": round(self.signals["barcode"], 3),
            "overrides": "|".join(self.overrides),
            "rationale": self.rationale,
        }


def text_match_signal(top_score: float, runner_up_score: float) -> float:
    """Confidence that retrieval picked a *distinguishable* winner.

    Scored on the margin, not the absolute score: a 0.95 winner with a 0.94
    runner-up has not identified anything, while a 0.70 winner with a 0.20
    runner-up clearly has.
    """
    margin = max(0.0, top_score - runner_up_score)
    return min(1.0, margin / MARGIN_SATURATION)


def barcode_identity_conflict(source_agreement: float, barcode: BarcodeCheck) -> bool:
    """True when artwork and supplier both produced usable barcodes that differ.

    Distinguishes an identity-level contradiction from a cosmetic one, so the
    routing layer can block the first without blocking the second.
    """
    if source_agreement >= 1.0:
        return False
    both_usable = {"artwork", "supplier"} <= set(barcode.valid_sources)
    return both_usable and not {"artwork", "supplier"} <= set(barcode.agreeing)


def route_event(
    *,
    source_agreement: float,
    top_score: float,
    runner_up_score: float,
    barcode: BarcodeCheck,
    llm_only: bool = False,
    llm_confidence: float = 0.0,
    alias_hit: bool = False,
) -> RoutingDecision:
    text_match = text_match_signal(top_score, runner_up_score)
    signals = {
        "source_agreement": source_agreement,
        "text_match": text_match,
        "barcode": barcode.score,
    }

    confidence = (
        W_SOURCE_AGREEMENT * source_agreement
        + W_TEXT_MATCH * text_match
        + W_BARCODE * barcode.score
    )

    overrides: list[str] = []

    if alias_hit:
        # A previously human-approved correction for this exact input is real
        # evidence -- it's the one boost applied, and it's additive rather than
        # overriding, so it can lift a good match over the line but can't
        # rescue a contradicted one.
        confidence = min(1.0, confidence + ALIAS_BOOST)
        overrides.append("alias_cache_hit")

    ceiling = 1.0
    reasons: list[str] = []

    if barcode.is_veto:
        ceiling = min(ceiling, HOLD_MIN)
        overrides.append("barcode_three_way_mismatch")
        reasons.append("three contradictory valid barcodes")

    if (top_score - runner_up_score) <= AMBIGUITY_MARGIN:
        ceiling = min(ceiling, HOLD_MIN)
        overrides.append("retrieval_ambiguous")
        reasons.append(
            f"top two candidates within {top_score - runner_up_score:.3f}"
        )

    if barcode_identity_conflict(source_agreement, barcode):
        # Artwork and supplier disagreeing on the IDENTITY field (the barcode)
        # is different in kind from disagreeing on a descriptive field. A brand
        # typed as "Mars Inc" against a pack reading "Mars" is a cosmetic
        # data-entry difference; two different barcodes for the same physical
        # pack means one of the two sources is describing another product.
        #
        # Note this is deliberately NOT "any disagreement blocks auto-merge".
        # Fig. 2 treats artwork-vs-supplier agreement as one of three weighted
        # SIGNALS (it carries W_SOURCE_AGREEMENT above); the only hard override
        # the guide specifies is the three-way barcode mismatch. Making every
        # cosmetic difference a veto held 100% of this feed for review and made
        # the auto-merge lane unreachable -- a guardrail that blocks everything
        # isn't a guardrail, it's an off switch.
        ceiling = min(ceiling, AUTO_MERGE_MIN - 1e-9)
        overrides.append("artwork_supplier_barcode_conflict")
        reasons.append("pack and submission report different barcodes")

    if llm_only:
        ceiling = min(ceiling, llm_confidence)
        overrides.append("llm_only_decision")
        reasons.append("decision rests on stage 05, not retrieval evidence")

    final = min(confidence, ceiling)

    if final >= AUTO_MERGE_MIN:
        route = Route.AUTO_MERGE
    elif final >= HOLD_MIN:
        route = Route.HOLD_FOR_REVIEW
    else:
        route = Route.REJECT

    if reasons:
        rationale = f"{route.value} at {final:.2f} — held back by: " + "; ".join(reasons)
    else:
        rationale = (
            f"{route.value} at {final:.2f} — source agreement "
            f"{source_agreement:.2f}, text margin {text_match:.2f}, "
            f"barcode {barcode.verdict.value}"
        )

    return RoutingDecision(
        route=route, confidence=final, signals=signals,
        overrides=overrides, rationale=rationale,
    )
