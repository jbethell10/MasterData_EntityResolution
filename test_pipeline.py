"""
Automated pass/fail test suite for the MDER pipeline.

Run with: pytest test_pipeline.py -v

This is the "does the product actually work" check that doesn't depend on a
human clicking through the interactive tool -- it re-runs (or reads the
freshly-generated output of) each stage and asserts on real numbers, not
assumed ones. Every threshold below was set by first looking at what the
pipeline actually produced (see README.md), then asking "is this good
enough to trust," not by guessing a number in advance.
"""
import csv
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SYN = ROOT / "data" / "synthetic"   # the synthetic run's artifacts
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_master_catalog import ean13_check_digit, make_gtin          # noqa: E402
from normalize import normalize_quantity, normalize_text               # noqa: E402
from stage04_candidate_match import score_candidate, rank_candidates   # noqa: E402


# ---------------------------------------------------------------------------
# Stage 00 -- master catalog integrity
# ---------------------------------------------------------------------------

def test_master_catalog_gtins_are_valid_ean13():
    """Every GTIN in the catalog must carry a correct EAN-13 check digit --
    if this fails, the barcode-verify stage would be comparing against
    garbage ground truth."""
    conn = sqlite3.connect(SYN / "mder.db")
    gtins = [row[0] for row in conn.execute("SELECT gtin FROM master_catalog")]
    conn.close()
    assert len(gtins) >= 30
    for gtin in gtins:
        body, check = gtin[:-1], gtin[-1]
        assert ean13_check_digit(body) == check, f"{gtin} has an invalid check digit"

def test_master_catalog_gtins_are_unique():
    conn = sqlite3.connect(SYN / "mder.db")
    gtins = [row[0] for row in conn.execute("SELECT gtin FROM master_catalog")]
    conn.close()
    assert len(gtins) == len(set(gtins))

def test_make_gtin_roundtrips_through_check_digit():
    for seq in [1, 42, 999]:
        gtin = make_gtin(seq)
        assert len(gtin) == 13
        assert ean13_check_digit(gtin[:-1]) == gtin[-1]


# ---------------------------------------------------------------------------
# Stage 02 -- normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("0.030kg", "30g"),
    ("2x51g", "102g"),
    ("330ml", "330ml"),
    ("1.43L", "1430ml"),
    ("8x100g", "800g"),
])
def test_normalize_quantity_unit_conversion(raw, expected):
    assert normalize_quantity(raw) == expected

def test_normalize_text_strips_case_and_punctuation():
    assert normalize_text("Cadbury's DAIRY-MILK!") == "cadburys dairymilk"


# ---------------------------------------------------------------------------
# Stage 01 -- OCR extraction (spot-check against the honest, known result)
# ---------------------------------------------------------------------------

def test_ocr_extraction_ran_and_scored():
    path = SYN / "artwork_extracted.csv"
    assert path.exists(), "run scripts/ocr_extract.py first"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 20
    brand_acc = sum(r["brand_match"] == "True" for r in rows) / len(rows)
    gtin_acc = sum(r["gtin_match"] == "True" for r in rows) / len(rows)
    # brand/GTIN are short, high-contrast strings -- OCR should nail these
    assert brand_acc >= 0.95, f"brand OCR accuracy dropped to {brand_acc:.0%}"
    assert gtin_acc >= 0.95, f"GTIN OCR accuracy dropped to {gtin_acc:.0%}"
    # quantity is the KNOWN weak spot (g/9 confusion) -- document the floor,
    # don't silently let it regress further than already observed
    qty_acc = sum(r["quantity_match"] == "True" for r in rows) / len(rows)
    assert qty_acc >= 0.5, f"quantity OCR accuracy regressed below the known floor: {qty_acc:.0%}"


# ---------------------------------------------------------------------------
# Stage 03 -- cross-check
# ---------------------------------------------------------------------------

def test_cross_check_flags_known_corruptions_as_disagreements():
    """Every event with corruption_applied != 'none' in the source intake
    event should NOT be scored full_agreement -- if it is, the cross-check
    logic is silently ignoring real discrepancies."""
    with open(SYN / "intake_events.csv", newline="") as f:
        events = {r["event_id"]: r for r in csv.DictReader(f)}
    with open(SYN / "cross_check_results.csv", newline="") as f:
        checks = {r["event_id"]: r for r in csv.DictReader(f)}

    assert set(events) == set(checks)
    false_full_agreements = [
        eid for eid, ev in events.items()
        if ev["supplier_corruption_applied"] != "none"
        and checks[eid]["cross_check_status"] == "full_agreement"
        # a corruption can land on a field that still normalizes back to the
        # true value (e.g. a unit reformat that round-trips exactly) -- so we
        # only flag it as a real problem if NONE of the three per-field
        # agreement flags came back False
        and checks[eid]["fields_fully_agreed"] == checks[eid]["fields_compared"]
    ]
    # allow a small number of harmless round-trips, but most corrupted
    # events must show up as something other than full agreement
    assert len(false_full_agreements) <= 2, (
        f"cross-check missed corruption on: {false_full_agreements}"
    )

def test_cross_check_status_values_are_valid():
    with open(SYN / "cross_check_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 20
    for r in rows:
        assert r["cross_check_status"] in {"full_agreement", "partial_agreement", "disagreement"}
        assert 0.0 <= float(r["agreement_score"]) <= 1.0


# ---------------------------------------------------------------------------
# Stage 04 -- candidate matching
# ---------------------------------------------------------------------------

def test_candidate_match_recovers_true_master_top3():
    with open(SYN / "candidate_match_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 20
    supplier_top3 = sum(r["supplier_top3_hit"] == "True" for r in rows) / len(rows)
    artwork_top3 = sum(r["artwork_top3_hit"] == "True" for r in rows) / len(rows)
    # observed on this catalog: both hit 100% top-3 -- assert a safety
    # margin below that rather than hardcoding the exact figure, so a
    # future harder/bigger catalog doesn't make this test flaky
    assert supplier_top3 >= 0.9, f"supplier-keyed top-3 recall dropped to {supplier_top3:.0%}"
    assert artwork_top3 >= 0.9, f"artwork-keyed top-3 recall dropped to {artwork_top3:.0%}"

def test_candidate_match_disambiguates_near_duplicate_names_via_gtin():
    """Adversarial case the real 20-event run doesn't cover on its own:
    two catalog entries with an IDENTICAL brand+name (so text/phonetic
    score alone can't tell them apart) but different GTINs. A query that
    reuses one candidate's exact GTIN must rank that candidate first."""
    catalog = [
        (101, "5000000001019", "Acme", "Everyday Cereal 500g", "500g"),
        (102, "5000000002016", "Acme", "Everyday Cereal 500g", "500g"),
    ]
    ranked = rank_candidates("Acme", "Everyday Cereal 500g", "5000000002016", catalog)
    assert ranked[0][1] == 102, "GTIN signal failed to break a text/phonetic tie"

def test_score_candidate_rewards_phonetic_match_over_no_match():
    combined_phonetic, *_ = score_candidate("Nestle", "KitKat", "", "", "Nestle", "KitKat 4 Finger")
    combined_nomatch, *_ = score_candidate("Nestle", "KitKat", "", "", "Heinz", "Tomato Ketchup")
    assert combined_phonetic > combined_nomatch


# ---------------------------------------------------------------------------
# Regression tests for bugs found during the back-test. Each of these failed
# before the corresponding fix, so they'd catch a silent reintroduction.
# ---------------------------------------------------------------------------

def test_seed_catalog_ids_are_stable_across_rebuilds():
    """master_id must be 1..30 on EVERY run, not just the first.

    master_id is AUTOINCREMENT and the builder only DELETEs rows; SQLite keeps
    the high-water mark in sqlite_sequence, so a second run against the same
    db originally produced ids 31..60. That silently broke the
    artwork_<master_id>.png filename join and made the documented
    "fixed seeds -> exactly reproducible" guarantee false.
    """
    conn = sqlite3.connect(SYN / "mder.db")
    ids = sorted(r[0] for r in conn.execute("SELECT master_id FROM master_catalog"))
    conn.close()
    assert ids == list(range(1, 31)), f"seed ids drifted: {ids[0]}..{ids[-1]}"


def test_artwork_font_is_a_real_scalable_ttf():
    """The renderer must resolve a real TTF, never PIL's bitmap fallback.

    The original code hardcoded a Linux-only DejaVu path inside a bare
    `except OSError` that fell back to ImageFont.load_default(). On macOS that
    fallback silently produced labels Tesseract can't read -- OCR brand
    accuracy fell from 100% to 45% with no error raised anywhere.
    """
    from fonts import active_font_paths
    regular, bold = active_font_paths()
    assert Path(regular).exists() and Path(bold).exists()
    assert Path(regular).suffix.lower() in {".ttf", ".ttc", ".otf"}


def test_truncated_barcode_is_rejected_not_accepted():
    """A 12-digit OCR read of a 13-digit GTIN must not pass as valid.

    The original regex was r"\\b\\d{12,13}\\b" and returned the first hit, so
    when Tesseract dropped the trailing digit of 5000000000302 it happily
    returned 500000000030 as if it were a real barcode.
    """
    from ocr_extract import is_valid_ean13, pick_barcode
    assert is_valid_ean13("5000000000302")
    assert not is_valid_ean13("500000000030")
    # a checksum-valid run wins over a truncated one appearing earlier
    assert pick_barcode("500000000030\n5000000000302\n") == "5000000000302"
    # Tesseract commonly splits barcode digits with spaces -- still one GTIN.
    assert pick_barcode("5000 0000 00302") == "5000000000302"
    # a corrupt read with no checksum-valid candidate degrades to the longest
    # plausible run rather than inventing a match
    assert pick_barcode("500000000030") == "500000000030"


def test_valid_but_different_gtin_is_not_scored_as_near_miss():
    """Barcode proximity must not be treated as evidence.

    Catalog GTINs here are sequential, so every product sits within edit
    distance 2 of its neighbours as an artifact of the seed script. The old
    `1 - levenshtein/3` formula turned that artifact into a match signal.
    """
    from stage04_candidate_match import gtin_similarity
    assert gtin_similarity("5000000000012", "5000000000012") == 1.0
    # valid barcode belonging to a different product -> zero evidence
    assert gtin_similarity("5000000000029", "5000000000012") == 0.0
    # checksum-invalid (corrupt read) -> graded partial credit, but capped
    assert 0.0 < gtin_similarity("5000000000099", "5000000000098") <= 0.5


def test_text_vs_barcode_conflict_is_flagged_for_review():
    """The adversarial case the 20-event sample never happens to exercise:
    a supplier mis-keys a barcode into a DIFFERENT real product's GTIN, so the
    text evidence and the barcode evidence name two different rows. A weighted
    sum silently picks a winner; the pipeline must instead flag the conflict."""
    from stage04_candidate_match import detect_signal_conflict, load_catalog
    catalog = load_catalog()

    conflict = detect_signal_conflict("Mars", "Mars Bar", "5000000000029", catalog)
    assert conflict is not None, "text/barcode contradiction was not detected"
    assert conflict["text_pick"] == 1 and conflict["gtin_pick"] == 2

    # agreement is not a conflict
    assert detect_signal_conflict("Mars", "Mars Bar", "5000000000012", catalog) is None
    # a merely corrupt (checksum-invalid) read is a bad read, not a conflict
    assert detect_signal_conflict("Mars", "Mars Bar", "5000000000099", catalog) is None


# ---------------------------------------------------------------------------
# Stage 04 engine against the real Leipzig benchmarks (the independent check)
# ---------------------------------------------------------------------------

BENCH_DIR = ROOT / "data" / "benchmark"
needs_benchmark = pytest.mark.skipif(
    not (BENCH_DIR / "Amazon.csv").exists(),
    reason="Leipzig benchmark not downloaded; see scripts/run_benchmark.py",
)


def test_matcher_retrieves_across_abbreviation_without_shared_tokens():
    """'MRS' -> 'Mars' shares no whole token, so a word-level vectorizer scores
    it at exactly zero. The char-ngram analyzer is what makes the embedding
    signal useful on the abbreviation corruptions this pipeline generates."""
    from matcher import MatchEngine, Record, Weights
    targets = [
        Record("1", "Mars Bar", "Mars"),
        Record("2", "Tomato Ketchup", "Heinz"),
        Record("3", "Corn Flakes", "Kelloggs"),
    ]
    engine = MatchEngine(targets, weights=Weights(0.3, 0.0, 0.7))
    top = engine.rank(Record("q", "MARS BAR", "MRS"), top_k=1)[0]
    assert top.rec_id == "1"
    assert top.vector > 0, "TF-IDF signal contributed nothing on an abbreviation"


@needs_benchmark
def test_benchmark_engine_beats_a_trivial_baseline():
    """Guards the real accuracy claim. The floor is deliberately well below the
    ~0.62 F1 actually observed, so this fails on a genuine regression rather
    than on benchmark noise -- and it asserts the engine beats fuzzy-only
    matching, which is the whole justification for adding the TF-IDF signal."""
    from benchmark import load_amazon_google, precompute_signals, evaluate_precomputed
    from matcher import Weights

    sig = precompute_signals(load_amazon_google(), block_size=50)
    tuned = evaluate_precomputed(sig, Weights(0.3, 0.0, 0.7), threshold=0.45)
    fuzzy_only = evaluate_precomputed(sig, Weights(1.0, 0.0, 0.0), threshold=0.45)

    assert tuned.f1 >= 0.55, f"Amazon-Google F1 regressed to {tuned.f1:.3f}"
    assert tuned.f1 > fuzzy_only.f1, "TF-IDF signal is not earning its place"


@needs_benchmark
def test_original_weighting_is_documented_as_suboptimal():
    """The back-test asked whether 0.5/0.2/0.3 was justified or just a guess
    that happened to work on a 30-row catalog. On real labelled data it is
    measurably worse than a TF-IDF-led mix; this test pins that finding so the
    README's claim stays honest if the engine changes."""
    from benchmark import load_amazon_google, precompute_signals, evaluate_precomputed
    from matcher import Weights

    sig = precompute_signals(load_amazon_google(), block_size=50)
    original = evaluate_precomputed(sig, Weights(0.5, 0.2, 0.3), threshold=0.45)
    tuned = evaluate_precomputed(sig, Weights(0.3, 0.0, 0.7), threshold=0.45)
    assert tuned.f1 > original.f1


def test_stage04_weights_sum_to_one_and_include_the_vector_signal():
    """Stage 04's production weights, after retuning against the Leipzig sweep.

    The TF-IDF term is the one the build guide specified ("embedding-based
    nearest-neighbor search") and that the original implementation simply
    didn't have; fuzzy-only scored 0.511 F1 on Amazon-Google against 0.616 for
    a TF-IDF-led mix, so its absence was the single biggest gap in the scorer.
    """
    import stage04_candidate_match as s4
    total = s4.W_TEXT + s4.W_PHONETIC + s4.W_VECTOR + s4.W_GTIN
    assert abs(total - 1.0) < 1e-9, f"weights sum to {total}, not 1.0"
    assert s4.W_VECTOR > 0, "TF-IDF signal is not weighted -- stage 04 lost it"
    assert s4.W_VECTOR >= s4.W_TEXT, (
        "benchmark says the char-ngram signal should carry at least as much as "
        "plain fuzzy; if this flips, re-run run_benchmark.py --sweep-weights"
    )


def test_phonetic_on_brand_field_has_no_false_positives_on_the_feed():
    """Justifies keeping phonetic weighted despite the benchmark penalty.

    The benchmark computes metaphone on `manufacturer or name`, and most
    Amazon/Google rows have no manufacturer -- so it degrades to hashing the
    first word of a product title. Stage 04 uses a real brand field. Measured
    here: across the full corrupted feed x full catalog, every metaphone hit
    identifies the correct BRAND. It stays capped below text/vector because
    brand alone doesn't pin the SKU.
    """
    import jellyfish
    from stage04_candidate_match import load_catalog

    catalog = load_catalog()
    brand_of = {mid: brand for mid, _g, brand, _n, _q in catalog}
    with open(SYN / "supplier_feed.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    fired = wrong = 0
    for r in rows:
        qb = r["submitted_brand"].strip()
        if not qb:
            continue
        true_brand = brand_of[int(r["true_master_id"])].lower()
        qm = jellyfish.metaphone(qb)
        for _mid, _g, brand, _n, _q in catalog:
            if brand and jellyfish.metaphone(brand) == qm:
                fired += 1
                wrong += brand.lower() != true_brand
    assert fired > 0, "phonetic never fired -- the signal is dead weight"
    assert wrong == 0, f"phonetic mis-identified the brand {wrong}/{fired} times"


def test_retuned_weights_separate_top1_from_top2_at_least_as_well():
    """Retuning must not shrink the confidence gap between the winner and the
    runner-up -- that gap is what stage 07's routing thresholds will key on."""
    import csv as _csv
    import stage04_candidate_match as s4

    catalog = s4.load_catalog()
    index = s4.build_vector_index(catalog)
    with open(SYN / "intake_events.csv", newline="") as f:
        events = list(_csv.DictReader(f))

    margins = []
    for e in events:
        ranked = s4.rank_candidates(
            e["supplier_raw_brand"], e["supplier_raw_product_name"],
            e["supplier_norm_gtin"], catalog, index=index,
        )
        if len(ranked) > 1:
            margins.append(ranked[0][0] - ranked[1][0])

    mean_margin = sum(margins) / len(margins)
    assert mean_margin >= 0.45, f"top1/top2 separation regressed to {mean_margin:.3f}"


# ---------------------------------------------------------------------------
# Stage 05 -- LLM disambiguation. Every test here runs OFFLINE: the trigger
# policy is a pure function, and the Claude backend is exercised through an
# injected fake client. Nothing in the suite requires an API key or network.
# ---------------------------------------------------------------------------

def _cand(master_id, score):
    from stage05_disambiguate import Candidate
    return Candidate(master_id, "Mars", "Mars Bar", "51g", "5000000000012", score)


def test_stage05_only_fires_on_genuinely_ambiguous_cases():
    """The trigger policy is what controls this stage's cost, so it's worth
    regression-testing independently of anything the model says."""
    from stage05_disambiguate import needs_disambiguation

    # the two independent readings resolve to different products
    fires, _ = needs_disambiguation([_cand(1, 0.9)], [_cand(2, 0.9)])
    assert fires

    # retrieval didn't actually separate the top two
    fires, why = needs_disambiguation([_cand(1, 0.80), _cand(2, 0.77)],
                                      [_cand(1, 0.90), _cand(2, 0.50)])
    assert fires and "separated by only" in why

    # text and barcode contradict each other (stage 04's conflict flag)
    fires, _ = needs_disambiguation([_cand(1, 0.9), _cand(2, 0.4)],
                                    [_cand(1, 0.9), _cand(2, 0.4)], signal_conflict=True)
    assert fires

    # decisive and agreeing -> no call, no spend
    fires, _ = needs_disambiguation([_cand(1, 0.95), _cand(2, 0.40)],
                                    [_cand(1, 0.93), _cand(2, 0.38)])
    assert not fires


class _FakeClient:
    """Stands in for anthropic.Anthropic without any network access."""

    def __init__(self, verdict):
        outer = self

        class _Messages:
            def parse(self, **kwargs):
                from types import SimpleNamespace
                outer.last_request = kwargs
                return SimpleNamespace(
                    stop_reason="end_turn", parsed_output=verdict, stop_details=None
                )

        self.messages = _Messages()


def test_stage05_llm_only_verdict_cannot_reach_auto_merge():
    """Safety rule: a confident-sounding LLM answer built on weak retrieval
    evidence must still not be able to auto-merge into the master catalog.
    Stage 07's auto-merge floor is 0.90; this cap sits below it by
    construction."""
    from types import SimpleNamespace
    from stage05_disambiguate import ClaudeDisambiguator, LLM_ONLY_CONFIDENCE_CAP

    verdict = SimpleNamespace(master_id=1, confidence=0.99, reasoning="very sure")
    result = ClaudeDisambiguator(client=_FakeClient(verdict)).resolve(
        {}, {}, [_cand(1, 0.8), _cand(2, 0.79)]
    )
    assert result.confidence <= LLM_ONLY_CONFIDENCE_CAP < 0.90
    assert result.capped is True


def test_stage05_rejects_a_master_id_retrieval_never_surfaced():
    """The model must not be able to introduce a product the retrieval step
    didn't shortlist -- otherwise a hallucinated id could enter the catalog."""
    from types import SimpleNamespace
    from stage05_disambiguate import ClaudeDisambiguator

    verdict = SimpleNamespace(master_id=999, confidence=0.95, reasoning="invented")
    result = ClaudeDisambiguator(client=_FakeClient(verdict)).resolve(
        {}, {}, [_cand(1, 0.8), _cand(2, 0.7)]
    )
    assert result.master_id is None
    assert "not in the shortlist" in result.error


def test_stage05_api_failure_degrades_instead_of_crashing():
    """A network/API failure must return a no-signal result, not take the
    whole batch down -- this stage is an assist, not a dependency."""
    from stage05_disambiguate import ClaudeDisambiguator

    class _Boom:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("connection reset")

    result = ClaudeDisambiguator(client=_Boom()).resolve({}, {}, [_cand(1, 0.8)])
    assert result.master_id is None and "connection reset" in result.error


def test_stage05_defaults_to_offline_without_credentials(monkeypatch):
    """The pipeline must run end-to-end with no API key configured."""
    from stage05_disambiguate import get_disambiguator

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert type(get_disambiguator(live=True)).__name__ == "NullDisambiguator"
    assert type(get_disambiguator(live=False)).__name__ == "NullDisambiguator"


def test_stage05_request_uses_current_api_shape():
    """Pins the request shape: current model id, adaptive thinking (not the
    removed budget_tokens), and a structured output schema rather than
    free-text JSON parsing."""
    from types import SimpleNamespace
    from stage05_disambiguate import ClaudeDisambiguator

    verdict = SimpleNamespace(master_id=1, confidence=0.5, reasoning="ok")
    client = _FakeClient(verdict)
    ClaudeDisambiguator(client=client).resolve({}, {}, [_cand(1, 0.8)])

    req = client.last_request
    assert req["model"] == "claude-opus-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in str(req["thinking"])
    assert req["output_format"] is not None, "not using structured output"


# ---------------------------------------------------------------------------
# Stage 06 -- three-way barcode verification
# ---------------------------------------------------------------------------

VALID_A = "5000000000012"   # Mars Bar
VALID_B = "5000000000029"   # Mars Bar Twin Pack


def test_stage06_distinguishes_all_three_from_two_of_three():
    from stage06_barcode_verify import BarcodeVerdict, verify_three_way

    assert verify_three_way(VALID_A, VALID_A, VALID_A).verdict is BarcodeVerdict.ALL_THREE_AGREE

    two = verify_three_way(VALID_A, VALID_B, VALID_A)
    assert two.verdict is BarcodeVerdict.TWO_OF_THREE
    # it must record WHICH two agree -- artwork+master here, not supplier
    assert set(two.agreeing) == {"artwork", "master"}
    assert two.score < 1.0


def test_stage06_full_mismatch_is_a_veto():
    from stage06_barcode_verify import BarcodeVerdict, verify_three_way

    third = "5000000000036"
    check = verify_three_way(VALID_A, VALID_B, third)
    assert check.verdict is BarcodeVerdict.FULL_MISMATCH
    assert check.is_veto and check.score == 0.0


def test_stage06_corrupt_reading_is_not_treated_as_a_contradiction():
    """A barcode that fails its own check digit is a damaged READING, not a
    claim about a different product. Letting OCR noise manufacture a
    'mismatch' would veto rows for no reason."""
    from stage06_barcode_verify import BarcodeVerdict, verify_three_way

    corrupt = "5000000000011"          # valid-length, wrong check digit
    assert not corrupt == VALID_A
    check = verify_three_way(corrupt, VALID_A, VALID_A)
    assert check.verdict is not BarcodeVerdict.FULL_MISMATCH
    assert "artwork" in check.corrupt_sources
    assert set(check.agreeing) == {"supplier", "master"}


# ---------------------------------------------------------------------------
# Stage 07 -- confidence and routing
# ---------------------------------------------------------------------------

def _bc(a, s, m):
    from stage06_barcode_verify import verify_three_way
    return verify_three_way(a, s, m)


def test_stage07_routing_bands_match_the_build_guide():
    from stage07_confidence_route import AUTO_MERGE_MIN, HOLD_MIN, Route, route_event

    assert AUTO_MERGE_MIN == 0.90 and HOLD_MIN == 0.60

    strong = route_event(source_agreement=1.0, top_score=0.95, runner_up_score=0.20,
                         barcode=_bc(VALID_A, VALID_A, VALID_A))
    assert strong.route is Route.AUTO_MERGE

    weak = route_event(source_agreement=0.0, top_score=0.30, runner_up_score=0.28,
                       barcode=_bc("", "", VALID_A))
    assert weak.route is Route.REJECT


def test_stage07_barcode_mismatch_blocks_auto_merge_despite_perfect_text():
    """The guide: a full three-way mismatch routes to review 'regardless of
    what the text match says'. This is the papering-over case Fig. 2 exists to
    prevent, so it must not be reachable by a high text score."""
    from stage07_confidence_route import Route, route_event

    d = route_event(source_agreement=1.0, top_score=1.0, runner_up_score=0.0,
                    barcode=_bc(VALID_A, VALID_B, "5000000000036"))
    assert d.route is not Route.AUTO_MERGE
    assert "barcode_three_way_mismatch" in d.overrides


def test_stage07_near_tie_cannot_auto_merge():
    """A winner that didn't actually beat the runner-up hasn't identified
    anything, however high its absolute score."""
    from stage07_confidence_route import Route, route_event

    d = route_event(source_agreement=1.0, top_score=0.96, runner_up_score=0.95,
                    barcode=_bc(VALID_A, VALID_A, VALID_A))
    assert d.route is not Route.AUTO_MERGE
    assert "retrieval_ambiguous" in d.overrides


def test_stage07_cosmetic_disagreement_does_not_block_but_barcode_conflict_does():
    """The correction made after the first end-to-end run: treating ANY
    artwork/supplier difference as a veto held 100% of the feed and made the
    auto-merge lane unreachable. Only an identity-field (barcode) conflict
    should block."""
    from stage07_confidence_route import Route, route_event

    # brand typed 'Mars Inc' vs pack 'Mars' -- cosmetic; barcodes still agree
    cosmetic = route_event(source_agreement=2 / 3, top_score=0.95, runner_up_score=0.20,
                           barcode=_bc(VALID_A, VALID_A, VALID_A))
    assert cosmetic.route is Route.AUTO_MERGE

    # the two sources report different valid barcodes -- identity conflict
    identity = route_event(source_agreement=2 / 3, top_score=0.95, runner_up_score=0.20,
                           barcode=_bc(VALID_A, VALID_B, VALID_A))
    assert identity.route is not Route.AUTO_MERGE
    assert "artwork_supplier_barcode_conflict" in identity.overrides


def test_stage07_text_signal_uses_margin_not_absolute_score():
    from stage07_confidence_route import text_match_signal

    assert text_match_signal(0.95, 0.94) < text_match_signal(0.70, 0.20)


# ---------------------------------------------------------------------------
# Stage 08 -- audit log and alias cache
# ---------------------------------------------------------------------------

def _mem_conn():
    import sqlite3
    import stage08_resolve as a
    conn = sqlite3.connect(":memory:")
    conn.executescript(a.SCHEMA)
    return conn


def test_stage08_log_separates_supplier_error_from_resolution_ambiguity():
    """The guide's explicit requirement: these are two different problems with
    two different owners, so the log must not collapse them."""
    from stage08_resolve import ProblemClass, classify_problem

    assert classify_problem(1.0, "auto_merge") == ProblemClass.CLEAN
    assert classify_problem(0.5, "auto_merge") == ProblemClass.SUPPLIER_DATA_ENTRY
    assert classify_problem(1.0, "hold_for_review") == ProblemClass.RESOLUTION_AMBIGUITY
    assert classify_problem(0.5, "hold_for_review") == ProblemClass.BOTH
    assert classify_problem(1.0, "reject") == ProblemClass.UNRESOLVED


def test_stage08_alias_cache_only_stores_human_approved_corrections():
    """Learning from the pipeline's own unreviewed auto-merges is how a
    resolver drifts -- one wrong merge becomes a permanent 'fact'. The only
    write path into the cache requires an approver."""
    import inspect
    import stage08_resolve as a

    conn = _mem_conn()
    assert a.lookup_alias(conn, "MRS", "MARS BAR") is None

    a.approve_correction(conn, brand="MRS", product_name="MARS BAR",
                         resolved_id=1, resolved_name="Mars Bar",
                         approved_by="steward@example.com")
    hit = a.lookup_alias(conn, "MRS", "MARS BAR")
    assert hit["resolved_id"] == 1 and hit["approved_by"] == "steward@example.com"

    # approved_by is required -- there is no unattributed write path
    assert "approved_by" in inspect.signature(a.approve_correction).parameters


def test_stage08_alias_key_is_normalized():
    from stage08_resolve import alias_key
    assert alias_key("MRS", "Mars  Bar") == alias_key("  mrs ", "mars bar")


def test_alias_hit_can_lift_a_held_row_over_the_auto_merge_line():
    """The learning loop from Fig. 1, end to end: a row that held for review
    auto-merges on resubmission once a human has approved the correction."""
    from stage07_confidence_route import Route, route_event

    kwargs = dict(source_agreement=1 / 3, top_score=0.95, runner_up_score=0.20,
                  barcode=_bc(VALID_A, VALID_A, VALID_A))
    before = route_event(**kwargs)
    after = route_event(**kwargs, alias_hit=True)

    assert before.route is Route.HOLD_FOR_REVIEW
    assert after.route is Route.AUTO_MERGE
    assert after.confidence > before.confidence


def test_pipeline_makes_no_incorrect_auto_merges():
    """The number that actually matters: a wrong auto-merge silently corrupts
    the master catalog, and is far worse than an unnecessary hold."""
    path = SYN / "pipeline_decisions.csv"
    if not path.exists():
        pytest.skip("run scripts/run_pipeline.py first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    merged = [r for r in rows if r["route"] == "auto_merge"]
    wrong = [r for r in merged if r["correct"] != "True"]
    assert not wrong, f"{len(wrong)} incorrect auto-merge(s): {[r['event_id'] for r in wrong]}"


# ---------------------------------------------------------------------------
# Ground-truth leakage
#
# These exist because a leak shipped and was reported as a headline result.
# benchmark_to_intake() copied the catalog record's own GTIN into the supplier
# submission, so stage 04 scored an exact barcode match against exactly one
# row -- the correct one -- on every event. That produced "100% top-1 accuracy
# on real data". Withholding the GTIN, the same engine scores 87.7%.
#
# The failure was invisible from the outside: every accuracy number went UP,
# every test passed, and the dashboard looked healthier. Nothing about a leak
# announces itself, so the guard has to be an explicit invariant.
# ---------------------------------------------------------------------------

LEAKABLE_FIELDS = [
    ("supplier_norm_gtin", "master_gtin"),
    ("supplier_raw_gtin", "master_gtin"),
    ("supplier_norm_brand", "master_norm_brand"),
    ("supplier_norm_product_name", "master_product_name"),
]


def test_leipzig_events_carry_no_barcode():
    """The Leipzig benchmarks are text-only catalogs -- they contain no
    barcodes at all. Anything in the GTIN field of a Leipzig event was
    fabricated by us, and the only place it could have been copied FROM is
    the answer key."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from load_leipzig_benchmark import build_intake_from_benchmark

    events = build_intake_from_benchmark("abt-buy", max_events=50)
    assert events, "expected some events"
    leaked = [e for e in events if e["supplier_norm_gtin"] or e["supplier_raw_gtin"]]
    assert not leaked, (
        f"{len(leaked)}/{len(events)} Leipzig events carry a supplier GTIN. "
        "The source data has no barcodes, so this value came from the master record."
    )


def test_no_supplier_field_is_a_perfect_copy_of_the_master():
    """Leak canary over whatever intake set is on disk.

    A supplier submission matching the master exactly is normal for SOME rows
    -- plenty of submissions are simply correct. It matching on EVERY row is
    not a clean feed, it is the answer key wearing a supplier's name tag.
    """
    for run in (ROOT / "data").glob("**/intake_events.csv"):
        with open(run, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for sup_key, mas_key in LEAKABLE_FIELDS:
            if sup_key not in rows[0] or mas_key not in rows[0]:
                continue
            populated = [r for r in rows if r[mas_key]]
            if len(populated) < 20:
                continue
            identical = sum(1 for r in populated if r[sup_key] == r[mas_key])
            ratio = identical / len(populated)
            assert ratio < 0.99, (
                f"{run.parent.name}: {sup_key} is identical to {mas_key} on "
                f"{identical}/{len(populated)} rows ({ratio:.1%}). A supplier "
                "field that always equals the master field is leaked truth, "
                "not a matching signal."
            )


def test_synthesised_barcodes_are_structurally_valid():
    """Stage 04 and stage 06 must agree about whether a GTIN is usable.

    The original f"800{i:010d}7" was 14 characters, so is_valid_ean13()
    rejected every one and stage 06 reported insufficient_data on all 1,092
    rows -- while stage 04 scored the same strings as perfect matches worth
    0.30 each.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from load_leipzig_benchmark import build_master_catalog_from_benchmark
    from stage06_barcode_verify import is_valid_ean13

    for dataset in ("abt-buy", "amazon-google"):
        catalog = build_master_catalog_from_benchmark(dataset)
        bad = [g for _, g, *_ in catalog[:200] if not is_valid_ean13(g)]
        assert not bad, f"{dataset}: {len(bad)} invalid EAN-13s, e.g. {bad[:3]}"


def test_the_two_datasets_do_not_share_barcodes():
    """Index-derived barcodes gave abt-buy row 436 and amazon-google row 436
    the same GTIN, so a catalog from one dataset would 'verify' against events
    from the other."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from load_leipzig_benchmark import build_master_catalog_from_benchmark

    a = {g for _, g, *_ in build_master_catalog_from_benchmark("abt-buy")[:300]}
    b = {g for _, g, *_ in build_master_catalog_from_benchmark("amazon-google")[:300]}
    assert not (a & b), f"{len(a & b)} barcodes shared across datasets"


def test_each_run_mode_writes_to_its_own_directory():
    """Both modes previously wrote data/intake_events.csv, so a Leipzig run
    silently replaced the synthetic run's events -- which is why the dashboard's
    artwork tab started showing image-less Leipzig rows."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import paths

    seen = {
        paths.run_dir("synthetic"),
        paths.run_dir("leipzig", "abt-buy"),
        paths.run_dir("leipzig", "amazon-google"),
    }
    assert len(seen) == 3, f"run directories collide: {seen}"
    dbs = {
        paths.db_path("synthetic"),
        paths.db_path("leipzig", "abt-buy"),
        paths.db_path("leipzig", "amazon-google"),
    }
    assert len(dbs) == 3, f"databases collide: {dbs}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
