"""
Stage 05 -- LLM disambiguation.

Stage 04 retrieves a ranked shortlist. Most of the time its top pick is
obviously right and nothing more is needed. This stage exists for the cases
where it isn't: where the supplier-keyed and artwork-keyed retrievals name
different products, where the top two candidates are separated by almost
nothing, or where the text evidence and the barcode evidence contradict each
other outright.

Those are exactly the cases a similarity score can't settle, because the
problem isn't "which string is closer" -- it's "which of two noisy readings of
the same physical product should I believe". That needs reasoning over the
evidence, which is what an LLM call is for.

Three design rules, all load-bearing:

1. LAZY INVOCATION. The LLM is only called when `needs_disambiguation()` says
   the case is genuinely ambiguous. On the current 20 events that is a small
   fraction of rows, so the cost of this stage scales with difficulty, not
   with volume. A pipeline that called an LLM on every row would be both
   slower and more expensive than the fuzzy matcher it was meant to assist.

2. THE LLM CANNOT INVENT A MATCH. It only ever picks from the shortlist stage
   04 already retrieved, or explicitly declines. It never sees the ground
   truth, and it cannot introduce a master_id that retrieval didn't surface.

3. AN LLM-ONLY DECISION NEVER AUTO-MERGES. Its confidence is hard-capped
   below the auto-merge threshold (see LLM_ONLY_CONFIDENCE_CAP), so a
   confident-sounding answer built on weak retrieval evidence can still only
   ever reach "hold for review". This mirrors the same rule in the sibling
   entity-resolution-agent project, and it's the difference between an LLM
   that assists a pipeline and one that can silently corrupt a master catalog.

Offline by default: with no API key configured the pipeline runs end-to-end
using NullDisambiguator, which records that a case *would* have been escalated
without calling anything. That keeps the test suite deterministic and means
this stage never becomes a hard dependency on network access.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Ambiguity triggers -----------------------------------------------------

# If the top two candidates are within this much of each other, retrieval
# hasn't actually separated them and the ranking is close to arbitrary.
AMBIGUITY_MARGIN = 0.08

# An LLM-only decision is capped here. Stage 07's auto-merge threshold is
# 0.90, so this by construction cannot auto-merge.
LLM_ONLY_CONFIDENCE_CAP = 0.75

DEFAULT_MODEL = "claude-opus-5"

_SYSTEM_PROMPT = (
    "You are a master-data entity resolution assistant for a retail product "
    "onboarding pipeline.\n\n"
    "You are given two INDEPENDENT, NOISY readings of the same physical "
    "product:\n"
    "  - ARTWORK: text extracted by OCR from a photo of the packaging. Typical "
    "errors are misread characters (lowercase 'g' read as '9'), dropped "
    "trailing digits, and missing fields where the OCR failed entirely.\n"
    "  - SUPPLIER: what a human typed into a submission form. Typical errors "
    "are brand abbreviations (MRS for Mars, KLGS for Kelloggs), truncated "
    "product names, case noise, reformatted units (0.051kg for 51g), and "
    "mis-keyed barcode digits.\n\n"
    "You are also given a SHORTLIST of candidate products retrieved from the "
    "master catalog.\n\n"
    "Your job is to decide which ONE shortlist candidate both readings refer "
    "to, or to decline if none of them plausibly fit.\n\n"
    "Rules:\n"
    "- Choose ONLY from the shortlist. Never invent a master_id.\n"
    "- A barcode that matches exactly is strong evidence, but a barcode is "
    "also the field most likely to be mis-keyed or mis-read, so it does not "
    "automatically outrank agreeing product names.\n"
    "- Where the two readings disagree, reason about WHICH reading is more "
    "likely to be corrupted given the error patterns above, rather than "
    "averaging them.\n"
    "- Distinguish product variants carefully: pack size and multipack count "
    "(a 51g single vs a 2x51g twin pack) make these DIFFERENT products, not "
    "the same product described two ways.\n"
    "- If the evidence genuinely does not separate two candidates, decline "
    "rather than guessing; a wrong auto-merge is far more costly to a master "
    "catalog than a row sent for human review.\n"
    "- confidence is your honest probability that your chosen master_id is "
    "correct: 0.0-1.0."
)


@dataclass
class Candidate:
    master_id: int
    brand: str
    product_name: str
    quantity: str
    gtin: str
    retrieval_score: float


@dataclass
class DisambiguationResult:
    invoked: bool
    master_id: Optional[int] = None
    confidence: float = 0.0
    reasoning: str = ""
    capped: bool = False
    error: str = ""

    def as_row(self) -> dict:
        return {
            "llm_invoked": self.invoked,
            "llm_master_id": self.master_id if self.master_id is not None else "",
            "llm_confidence": round(self.confidence, 3),
            "llm_capped": self.capped,
            "llm_reasoning": self.reasoning,
            "llm_error": self.error,
        }


def needs_disambiguation(supplier_top, artwork_top, signal_conflict: bool = False) -> tuple[bool, str]:
    """Decide whether this event is ambiguous enough to spend an LLM call on.

    Returns (should_invoke, human-readable reason). Kept as a pure function so
    the trigger policy is testable without any network access -- the thing you
    actually want to regression-test here is *when* you spend money, not what
    the model says.
    """
    if signal_conflict:
        return True, "text and barcode evidence name different products"

    sup_id = supplier_top[0].master_id if supplier_top else None
    art_id = artwork_top[0].master_id if artwork_top else None

    if sup_id is not None and art_id is not None and sup_id != art_id:
        return True, f"supplier-keyed picked {sup_id}, artwork-keyed picked {art_id}"

    for label, ranked in (("supplier", supplier_top), ("artwork", artwork_top)):
        if len(ranked) > 1:
            margin = ranked[0].retrieval_score - ranked[1].retrieval_score
            if margin < AMBIGUITY_MARGIN:
                return True, (
                    f"{label}-keyed top-2 separated by only {margin:.3f} "
                    f"(< {AMBIGUITY_MARGIN})"
                )

    return False, "retrieval was decisive"


class Disambiguator(Protocol):
    def resolve(self, artwork: dict, supplier: dict,
                shortlist: list[Candidate]) -> DisambiguationResult:
        ...


class NullDisambiguator:
    """Offline no-op. Records that the case is ambiguous without calling out."""

    def resolve(self, artwork, supplier, shortlist) -> DisambiguationResult:
        return DisambiguationResult(
            invoked=False,
            reasoning="LLM disambiguation not configured (no API key); "
                      "case would route to human review",
        )


def _build_prompt(artwork: dict, supplier: dict, shortlist: list[Candidate]) -> str:
    lines = [
        "ARTWORK reading (OCR from the packaging photo):",
        f"  brand:    {artwork.get('brand') or '(not read)'}",
        f"  name:     {artwork.get('product_name') or '(not read)'}",
        f"  quantity: {artwork.get('quantity') or '(not read)'}",
        f"  barcode:  {artwork.get('gtin') or '(not read)'}",
        "",
        "SUPPLIER submission (typed into the portal):",
        f"  brand:    {supplier.get('brand') or '(blank)'}",
        f"  name:     {supplier.get('product_name') or '(blank)'}",
        f"  quantity: {supplier.get('quantity') or '(blank)'}",
        f"  barcode:  {supplier.get('gtin') or '(blank)'}",
        "",
        "SHORTLIST of candidate master catalog products:",
    ]
    for c in shortlist:
        lines.append(
            f"  master_id={c.master_id} | brand={c.brand!r} | name={c.product_name!r} "
            f"| quantity={c.quantity!r} | gtin={c.gtin} | retrieval_score={c.retrieval_score:.3f}"
        )
    lines += [
        "",
        "Which ONE master_id do both readings refer to? Decline (null) if none fit.",
    ]
    return "\n".join(lines)


class ClaudeDisambiguator:
    """Disambiguation backed by the Claude Messages API.

    The client is constructed lazily, so importing or instantiating this class
    never requires credentials -- only actually calling resolve() does.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 client=None, effort: str = "high"):
        self.model = model
        self._api_key = api_key
        self._client = client          # injectable for tests
        self.effort = effort

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def resolve(self, artwork: dict, supplier: dict,
                shortlist: list[Candidate]) -> DisambiguationResult:
        if not shortlist:
            return DisambiguationResult(invoked=True, reasoning="empty shortlist")

        from pydantic import BaseModel, Field

        valid_ids = {c.master_id for c in shortlist}

        class Verdict(BaseModel):
            master_id: Optional[int] = Field(
                description="master_id of the chosen candidate, or null to decline"
            )
            confidence: float = Field(ge=0.0, le=1.0)
            reasoning: str = Field(description="one or two sentences")

        try:
            response = self._get_client().messages.parse(
                model=self.model,
                max_tokens=16000,
                system=_SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": _build_prompt(artwork, supplier, shortlist)}],
                output_format=Verdict,
            )
            if response.stop_reason == "refusal":
                return DisambiguationResult(
                    invoked=True,
                    error=f"refused: {getattr(response.stop_details, 'category', None)}",
                )
            verdict = response.parsed_output
        except Exception as exc:  # noqa: BLE001 -- degrade to "no signal", never crash the pipeline
            return DisambiguationResult(invoked=True, error=f"{type(exc).__name__}: {exc}")

        # Guardrail: the model must not introduce an id retrieval never surfaced.
        if verdict.master_id is not None and verdict.master_id not in valid_ids:
            return DisambiguationResult(
                invoked=True,
                error=f"model returned master_id={verdict.master_id} which was not in the shortlist",
            )

        confidence = float(verdict.confidence)
        capped = confidence > LLM_ONLY_CONFIDENCE_CAP
        return DisambiguationResult(
            invoked=True,
            master_id=verdict.master_id,
            confidence=min(confidence, LLM_ONLY_CONFIDENCE_CAP),
            reasoning=verdict.reasoning,
            capped=capped,
        )


def get_disambiguator(live: bool = False, model: str = DEFAULT_MODEL) -> Disambiguator:
    """Pick a backend. Offline unless explicitly asked for AND credentialed."""
    if live and os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeDisambiguator(model=model)
    return NullDisambiguator()
