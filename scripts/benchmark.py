"""
Leipzig entity-resolution benchmark: loader + precision/recall/F1 harness.

This is the independent accuracy check the build guide asks for. The project's
own "100% top-1 on 20 events against 30 products" number is graded on data the
project corrupted itself, on a catalog small enough that retrieval is nearly
trivial -- it demonstrates the logic runs, not that it works. These are real,
peer-reviewed, human-labelled product-matching pairs from the Leipzig DB group,
so the number that comes out is comparable to published work.

Datasets (data/benchmark/):
  Amazon-Google : 1,363 x 3,226 records, 1,300 labelled matches
  Abt-Buy       : 1,081 x 1,092 records, 1,097 labelled matches

Both files are latin-1, not UTF-8 -- decoding them as UTF-8 raises, and
decoding with errors="ignore" silently corrupts exactly the accented product
names that make matching hard, so the encoding is pinned explicitly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from matcher import MatchEngine, Record, Weights

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "data" / "benchmark"
ENCODING = "latin-1"


@dataclass
class Benchmark:
    name: str
    left: list[Record]
    right: list[Record]
    truth: dict[str, set[str]]      # left_id -> {right_id, ...}

    @property
    def n_pairs(self) -> int:
        return sum(len(v) for v in self.truth.values())


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding=ENCODING) as f:
        return list(csv.DictReader(f))


def _records(rows: list[dict], name_col: str) -> list[Record]:
    out = []
    for r in rows:
        out.append(Record(
            rec_id=r["id"],
            name=(r.get(name_col) or "").strip(),
            manufacturer=(r.get("manufacturer") or "").strip(),
            description=(r.get("description") or "").strip(),
        ))
    return out


def load_amazon_google() -> Benchmark:
    left = _records(_read_csv(BENCH_DIR / "Amazon.csv"), "title")
    right = _records(_read_csv(BENCH_DIR / "GoogleProducts.csv"), "name")
    truth: dict[str, set[str]] = {}
    for row in _read_csv(BENCH_DIR / "Amzon_GoogleProducts_perfectMapping.csv"):
        truth.setdefault(row["idAmazon"], set()).add(row["idGoogleBase"])
    return Benchmark("Amazon-Google", left, right, truth)


def load_abt_buy() -> Benchmark:
    left = _records(_read_csv(BENCH_DIR / "Abt.csv"), "name")
    right = _records(_read_csv(BENCH_DIR / "Buy.csv"), "name")
    truth: dict[str, set[str]] = {}
    for row in _read_csv(BENCH_DIR / "abt_buy_perfectMapping.csv"):
        truth.setdefault(row["idAbt"], set()).add(row["idBuy"])
    return Benchmark("Abt-Buy", left, right, truth)


LOADERS = {"amazon-google": load_amazon_google, "abt-buy": load_abt_buy}


@dataclass
class EvalResult:
    dataset: str
    threshold: float
    weights: Weights
    precision: float
    recall: float
    f1: float
    top1_accuracy: float      # of queries WITH a known match, how often top-1 is right
    predicted: int
    correct: int
    n_truth_pairs: int
    n_queries_with_truth: int

    def as_row(self) -> dict:
        w = self.weights
        return {
            "dataset": self.dataset, "threshold": round(self.threshold, 3),
            "w_text": w.text, "w_phonetic": w.phonetic, "w_vector": w.vector,
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "top1_accuracy": round(self.top1_accuracy, 4),
            "predicted": self.predicted, "correct": self.correct,
            "truth_pairs": self.n_truth_pairs,
        }


@dataclass
class PrecomputedSignals:
    """Per-signal scores for every (query, blocked-candidate) pair.

    Blocking and the three signal computations don't depend on the weights --
    only the final weighted sum does. Separating them means the interactive
    app can re-score the entire benchmark on a slider drag with one numpy
    multiply, instead of re-running a ~10s TF-IDF + fuzzy pass per frame.
    """
    dataset: str
    text: "np.ndarray"        # (n_queries, block_size)
    phonetic: "np.ndarray"
    vector: "np.ndarray"
    cand_ids: list[list[str]]
    query_ids: list[str]
    truth: dict[str, set[str]]
    n_truth_pairs: int


def precompute_signals(bench: Benchmark, block_size: int = 50) -> PrecomputedSignals:
    import numpy as np
    from rapidfuzz import fuzz

    engine = MatchEngine(bench.right, weights=Weights())
    qvs = engine._vec.transform([q.blob() for q in bench.left])
    k = min(block_size, len(bench.right))
    dists, idxs = engine._nn.kneighbors(qvs, n_neighbors=k)
    sims = 1.0 - dists

    n = len(bench.left)
    text = np.zeros((n, k), dtype=np.float32)
    phon = np.zeros((n, k), dtype=np.float32)
    cand_ids: list[list[str]] = []

    for i, (q, idx_row) in enumerate(zip(bench.left, idxs)):
        q_blob = q.blob()
        q_phon = engine._metaphone(q.manufacturer or q.name)
        ids_row = []
        for j, pos in enumerate(idx_row):
            text[i, j] = fuzz.token_sort_ratio(q_blob, engine._blobs[pos]) / 100.0
            phon[i, j] = 1.0 if (q_phon and engine._phon[pos] and q_phon == engine._phon[pos]) else 0.0
            ids_row.append(bench.right[pos].rec_id)
        cand_ids.append(ids_row)

    return PrecomputedSignals(
        dataset=bench.name, text=text, phonetic=phon, vector=sims.astype(np.float32),
        cand_ids=cand_ids, query_ids=[q.rec_id for q in bench.left],
        truth=bench.truth, n_truth_pairs=bench.n_pairs,
    )


def evaluate_precomputed(sig: PrecomputedSignals, weights: Weights,
                         threshold: float) -> EvalResult:
    """Re-score cached signals under new weights. Pure arithmetic -- fast
    enough to drive a live slider over the full 1,363-query benchmark."""
    import numpy as np

    w = weights.normalized()
    combined = w.text * sig.text + w.phonetic * sig.phonetic + w.vector * sig.vector
    best_pos = combined.argmax(axis=1)
    best_score = combined[np.arange(len(best_pos)), best_pos]

    predicted = correct = top1_hits = top1_eligible = 0
    for i, qid in enumerate(sig.query_ids):
        gold = sig.truth.get(qid, set())
        pick = sig.cand_ids[i][best_pos[i]]
        if gold:
            top1_eligible += 1
            if pick in gold:
                top1_hits += 1
        if best_score[i] >= threshold:
            predicted += 1
            if pick in gold:
                correct += 1

    precision = correct / predicted if predicted else 0.0
    recall = correct / sig.n_truth_pairs if sig.n_truth_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvalResult(
        dataset=sig.dataset, threshold=threshold, weights=w,
        precision=precision, recall=recall, f1=f1,
        top1_accuracy=top1_hits / top1_eligible if top1_eligible else 0.0,
        predicted=predicted, correct=correct,
        n_truth_pairs=sig.n_truth_pairs, n_queries_with_truth=top1_eligible,
    )


def evaluate(bench: Benchmark, weights: Weights | None = None,
             threshold: float = 0.5, block_size: int = 50) -> EvalResult:
    """Score every left-hand record against the right-hand catalog.

    A prediction is emitted only when the top candidate clears `threshold`;
    that is what makes precision meaningful (an engine that always guesses
    would score 100% recall and near-zero precision). Recall is measured
    against the full labelled pair set, including pairs the blocking step
    never surfaced -- so blocking misses are honestly counted as recall loss
    rather than quietly excluded from the denominator.
    """
    engine = MatchEngine(bench.right, weights=weights or Weights())
    ranked = engine.rank_many(bench.left, top_k=1, block_size=block_size)

    predicted = correct = 0
    top1_hits = top1_eligible = 0

    for query, cands in zip(bench.left, ranked):
        gold = bench.truth.get(query.rec_id, set())
        if gold:
            top1_eligible += 1
            if cands and cands[0].rec_id in gold:
                top1_hits += 1
        if cands and cands[0].combined >= threshold:
            predicted += 1
            if cands[0].rec_id in gold:
                correct += 1

    precision = correct / predicted if predicted else 0.0
    recall = correct / bench.n_pairs if bench.n_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalResult(
        dataset=bench.name, threshold=threshold, weights=(weights or Weights()).normalized(),
        precision=precision, recall=recall, f1=f1,
        top1_accuracy=top1_hits / top1_eligible if top1_eligible else 0.0,
        predicted=predicted, correct=correct,
        n_truth_pairs=bench.n_pairs, n_queries_with_truth=top1_eligible,
    )
