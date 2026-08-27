"""
Stage 04 -- Candidate Matching.

Cross-check (stage 03) tells us WHERE artwork, supplier and master disagree.
It doesn't tell us who the product actually is. That's this stage's job:
given one noisy input record, retrieve a ranked shortlist of candidate
master_catalog rows using fuzzy string matching (RapidFuzz) plus phonetic
matching (jellyfish metaphone, which survives things fuzzy matching alone
can miss -- e.g. brand nicknames that don't share many characters) plus a
barcode-closeness bonus.

We run retrieval TWICE per event -- once keyed on the supplier's typed
data, once keyed on the artwork's OCR'd data -- and score both against the
same true_master_id. That head-to-head comparison is itself a useful
finding: it tells you which of your two noisy sources is the better search
key, which a real system would use to weight the two inputs during
disambiguation (stage 05).
"""
import argparse
import csv
from pathlib import Path

import sys

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_extract import is_valid_ean13  # noqa: E402

import paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Default to the synthetic run; main() repoints these from --mode/--dataset.
EVENTS_PATH = paths.run_dir("synthetic") / "intake_events.csv"
DB_PATH = paths.db_path("synthetic")
OUT_PATH = paths.run_dir("synthetic") / "candidate_match_results.csv"

TOP_K = 3

def load_catalog(db_path=None):
    import sqlite3
    conn = sqlite3.connect(db_path or DB_PATH)
    rows = conn.execute(
        "SELECT master_id, gtin, brand, product_name, quantity FROM master_catalog"
    ).fetchall()
    conn.close()
    return rows

def gtin_similarity(query_gtin: str, cand_gtin: str) -> float:
    """How much barcode evidence links this query to this candidate.

    The original formula was a flat `1 - levenshtein/3` on the raw digits,
    which treats "numerically adjacent" as "probably the same product". That
    is wrong for barcodes in two ways:

    1. A checksum-VALID GTIN that differs from the candidate's is not a near
       miss -- it is a correct barcode belonging to a *different* product, so
       the right amount of evidence for this candidate is zero. Giving it
       partial credit is what let a supplier `barcode_error` corruption (which
       transposes digits into a neighbouring real product's GTIN) outscore the
       true product: the true "Mars Bar" lost to "Mars Bar Twin Pack" 0.80 to
       0.86 purely on spurious digit proximity.
    2. This catalog assigns GTINs sequentially, so every product sits within
       edit distance 2 of its neighbours as an artifact of the seed script.
       Graded proximity therefore encodes "adjacent row in build_master_catalog"
       rather than any real-world signal, and would not survive contact with a
       real catalog where GTINs are effectively random.

    So: exact match is full evidence. A valid-but-different barcode is
    counter-evidence, scored 0. Only a checksum-INVALID query (a genuinely
    corrupt OCR or keying read, where the digits are damaged rather than
    pointing elsewhere) gets graded partial credit, capped well below an
    exact match so it can inform ranking without dominating it.
    """
    if not query_gtin or not cand_gtin:
        return 0.0
    if query_gtin == cand_gtin:
        return 1.0
    if is_valid_ean13(query_gtin):
        # a valid barcode for some other product -- no evidence for this one
        return 0.0
    dist = Levenshtein.distance(query_gtin, cand_gtin)
    return max(0.0, 0.5 * (1.0 - dist / 3.0))


# ---------------------------------------------------------------------------
# Signal weights.
#
# These replace the original 0.5 text / 0.2 phonetic / 0.3 GTIN split, which
# was never tuned against labelled data. Swept over the Leipzig benchmarks
# (scripts/run_benchmark.py --sweep-weights) that split came second-to-last of
# eight configurations on Amazon-Google, F1 0.546 against 0.616 for a
# TF-IDF-led mix -- and fuzzy-only scored 0.511, so the character-ngram signal
# is worth +0.105 F1 on its own.
#
# Two changes follow from that sweep:
#
# 1. A TF-IDF char-ngram signal is added, which is also what the build guide
#    specified for this stage ("embedding-based nearest-neighbor search") and
#    was simply missing. It carries the largest share.
# 2. Phonetic KEEPS a real share, deliberately, even though the benchmark
#    sweep showed global phonetic weighting hurting there. That penalty does
#    not transfer: in the benchmark, metaphone is computed on
#    `manufacturer or name`, and most Amazon/Google rows have an empty
#    manufacturer, so it silently fell back to hashing the first word of the
#    product title -- matching on random title words. Stage 04 computes it on
#    a real, always-populated brand field instead.
#
#    Measured directly on this pipeline's own 60-row corrupted feed against
#    the full catalog, metaphone-on-brand fires 69 times and is right about
#    the brand 69 of 69 times -- zero false positives, at both short (MRS,
#    KLGS) and long (Warburtons) brand lengths. It is capped below the text
#    and vector signals only because identifying the BRAND still leaves the
#    SKU ambiguous ("Mars Bar" vs "Mars Bar Twin Pack").
#
# GTIN keeps its 0.3 share: unlike the benchmark datasets this catalog has
# barcodes, and post-fix gtin_similarity() is authoritative (exact = 1.0,
# valid-but-different = 0.0) rather than a fuzzy proximity guess.
W_TEXT = 0.20
W_PHONETIC = 0.15
W_VECTOR = 0.35
W_GTIN = 0.30


def phonetic_score(query_brand: str, cand_brand: str) -> float:
    """Metaphone agreement on the brand field."""
    q = (query_brand or "").strip()
    c = (cand_brand or "").strip()
    if not q or not c:
        return 0.0
    return 1.0 if jellyfish.metaphone(q) == jellyfish.metaphone(c) else 0.0


def score_candidate(query_brand, query_name, query_gtin, cand_gtin, cand_brand, cand_name,
                    vector_score: float = 0.0):
    text_score = fuzz.token_sort_ratio(
        f"{query_brand} {query_name}".lower(), f"{cand_brand} {cand_name}".lower()
    ) / 100.0

    phon_bonus = phonetic_score(query_brand, cand_brand)
    gtin_score = gtin_similarity(query_gtin, cand_gtin)

    combined = (W_TEXT * text_score + W_PHONETIC * phon_bonus
                + W_VECTOR * vector_score + W_GTIN * gtin_score)
    return combined, text_score, phon_bonus, gtin_score


def build_vector_index(catalog):
    """TF-IDF char-ngram index over the master catalog.

    Same analyzer settings as scripts/matcher.py, so the signal stage 04 uses
    in production is the identical one the Leipzig benchmark measured -- the
    tuning transfers only because the implementation does.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    blobs = [f"{brand} {name}".strip().lower() for _, _, brand, name, _ in catalog]
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
        matrix = vec.fit_transform(blobs)
    except ValueError as e:
        # Empty vocabulary on edge cases. Retry with more permissive settings.
        if "empty vocabulary" in str(e):
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                                  sublinear_tf=True, stop_words=[], max_features=5000)
            matrix = vec.fit_transform(blobs)
        else:
            raise
    return vec, matrix


def rank_candidates(query_brand, query_name, query_gtin, catalog, index=None):
    if index is None:
        index = build_vector_index(catalog)
    vec, matrix = index

    qv = vec.transform([f"{query_brand} {query_name}".strip().lower()])
    # cosine similarity; both sides are L2-normalized by TfidfVectorizer
    sims = (matrix @ qv.T).toarray().ravel()

    scored = []
    for i, (master_id, gtin, brand, name, qty) in enumerate(catalog):
        combined, text_s, phon_s, gtin_s = score_candidate(
            query_brand, query_name, query_gtin, gtin, brand, name,
            vector_score=float(sims[i]),
        )
        scored.append((combined, master_id, brand, name, text_s, phon_s, gtin_s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:TOP_K]


def detect_signal_conflict(query_brand, query_name, query_gtin, catalog):
    """Flag the case where the text evidence and the barcode evidence point at
    two DIFFERENT catalog rows.

    Fig. 2 of the build guide is explicit that the signals are kept separate
    so "a strong result on one axis (say, the barcode) can't quietly paper
    over a real disagreement on another". A single weighted sum does exactly
    the papering-over it warns about: when a supplier mis-keys a barcode into
    some other real product's GTIN, the barcode term is a clean 1.0 for the
    wrong row while the text term is a clean 1.0 for the right one, and
    whichever weight happens to be larger silently wins with no indication
    that the two sources ever disagreed.

    So we detect it explicitly and return it alongside the ranking. This is
    the signal stage 07 routes on (conflict -> hold for review rather than
    auto-merge) and it is why the conflict is surfaced here rather than
    resolved here -- stage 04's job is retrieval, not adjudication.
    """
    if not query_gtin or not is_valid_ean13(query_gtin):
        return None

    gtin_owner = next((mid for mid, g, *_ in catalog if g == query_gtin), None)
    if gtin_owner is None:
        return None

    text_only = sorted(
        (
            (fuzz.token_sort_ratio(
                f"{query_brand} {query_name}".lower(), f"{brand} {name}".lower()
            ) / 100.0, mid)
            for mid, g, brand, name, qty in catalog
        ),
        reverse=True,
    )
    text_best_score, text_best_id = text_only[0]

    # Only a genuine conflict if the text evidence is actually strong. A weak
    # text read disagreeing with the barcode is just a bad OCR/typo, not two
    # credible sources contradicting each other.
    if text_best_id != gtin_owner and text_best_score >= 0.85:
        return {
            "text_pick": text_best_id,
            "text_score": round(text_best_score, 3),
            "gtin_pick": gtin_owner,
        }
    return None

def main():
    global EVENTS_PATH, DB_PATH, OUT_PATH
    ap = argparse.ArgumentParser()
    paths.add_mode_args(ap)
    run_dir, db = paths.resolve(ap.parse_args())
    EVENTS_PATH = run_dir / "intake_events.csv"
    DB_PATH = db
    OUT_PATH = run_dir / "candidate_match_results.csv"

    catalog = load_catalog()
    index = build_vector_index(catalog)   # fit once, reuse for every query
    with open(EVENTS_PATH, newline="") as f:
        events = list(csv.DictReader(f))

    results = []
    supplier_top1 = supplier_top3 = artwork_top1 = artwork_top3 = 0

    for e in events:
        true_id = int(e["true_master_id"])

        sup_ranked = rank_candidates(
            e["supplier_raw_brand"], e["supplier_raw_product_name"],
            e["supplier_norm_gtin"], catalog, index=index
        )
        art_ranked = rank_candidates(
            e["artwork_brand"], e["artwork_product_name"],
            e["artwork_gtin"], catalog, index=index
        )

        sup_conflict = detect_signal_conflict(
            e["supplier_raw_brand"], e["supplier_raw_product_name"],
            e["supplier_norm_gtin"], catalog
        )
        art_conflict = detect_signal_conflict(
            e["artwork_brand"], e["artwork_product_name"],
            e["artwork_gtin"], catalog
        )

        sup_ids = [c[1] for c in sup_ranked]
        art_ids = [c[1] for c in art_ranked]

        sup_top1_hit = sup_ids[0] == true_id if sup_ids else False
        art_top1_hit = art_ids[0] == true_id if art_ids else False
        sup_top3_hit = true_id in sup_ids
        art_top3_hit = true_id in art_ids

        supplier_top1 += sup_top1_hit
        artwork_top1 += art_top1_hit
        supplier_top3 += sup_top3_hit
        artwork_top3 += art_top3_hit

        results.append({
            "event_id": e["event_id"],
            "true_master_id": true_id,
            "supplier_top1_id": sup_ids[0] if sup_ids else "",
            "supplier_top1_hit": sup_top1_hit,
            "supplier_top3_hit": sup_top3_hit,
            "supplier_candidates": "|".join(f"{c[1]}:{c[0]:.2f}" for c in sup_ranked),
            "artwork_top1_id": art_ids[0] if art_ids else "",
            "artwork_top1_hit": art_top1_hit,
            "artwork_top3_hit": art_top3_hit,
            "artwork_candidates": "|".join(f"{c[1]}:{c[0]:.2f}" for c in art_ranked),
            # text-vs-barcode contradiction, for stage 07 routing
            "supplier_signal_conflict": bool(sup_conflict),
            "supplier_conflict_detail": (
                f"text->{sup_conflict['text_pick']}({sup_conflict['text_score']}) "
                f"vs gtin->{sup_conflict['gtin_pick']}" if sup_conflict else ""
            ),
            "artwork_signal_conflict": bool(art_conflict),
            "artwork_conflict_detail": (
                f"text->{art_conflict['text_pick']}({art_conflict['text_score']}) "
                f"vs gtin->{art_conflict['gtin_pick']}" if art_conflict else ""
            ),
        })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(events)
    print(f"Candidate-matched {n} intake events against {len(catalog)} master products -> {OUT_PATH}")
    print(f"  supplier-keyed:  top-1 {supplier_top1}/{n} ({100*supplier_top1/n:.0f}%)   "
          f"top-3 {supplier_top3}/{n} ({100*supplier_top3/n:.0f}%)")
    print(f"  artwork-keyed:   top-1 {artwork_top1}/{n} ({100*artwork_top1/n:.0f}%)   "
          f"top-3 {artwork_top3}/{n} ({100*artwork_top3/n:.0f}%)")

    conflicts = [r for r in results
                 if r["supplier_signal_conflict"] or r["artwork_signal_conflict"]]
    print(f"  text-vs-barcode signal conflicts: {len(conflicts)}/{n} "
          f"(route to review in stage 07, don't auto-merge)")
    for r in conflicts[:5]:
        detail = r["supplier_conflict_detail"] or r["artwork_conflict_detail"]
        print(f"    [{r['event_id']}] {detail}")

    misses = [r for r in results if not r["supplier_top3_hit"] or not r["artwork_top3_hit"]]
    if misses:
        print(f"\n{len(misses)} event(s) missed top-3 on at least one side (spot-check these):")
        for r in misses[:5]:
            print(f"  [{r['event_id']}] true={r['true_master_id']}  "
                  f"supplier_top1={r['supplier_top1_id']}  artwork_top1={r['artwork_top1_id']}")

if __name__ == "__main__":
    main()
