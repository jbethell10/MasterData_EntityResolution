"""
The matching engine, factored out so ONE implementation serves both datasets.

Stage 04 originally hardcoded its scorer against the 30-row seed catalog. That
made the headline "100% top-1" number untestable anywhere else -- it could not
be pointed at the Leipzig benchmark without a rewrite, so the accuracy claim
had no independent check. This module is the same three-signal idea written
source-agnostically:

    fuzzy (RapidFuzz)  +  phonetic (metaphone)  +  TF-IDF char-ngram cosine

The third signal is the "embedding-based nearest-neighbor search" the build
guide specifies for stage 04. It is a character-ngram TF-IDF vector space
rather than a neural sentence embedding: on short, typo-heavy product strings
char-ngrams handle exactly the corruption classes this pipeline cares about
(truncation, transposition, missing letters), they need no model download, and
they are fully deterministic -- which matters when the whole project's premise
is "fixed seeds, exactly reproducible".

It also doubles as the BLOCKING step. Amazon x Google is 1,363 x 3,226 = 4.4M
candidate pairs; scoring all of them pairwise is the thing the sibling
entity-resolution-agent README calls out as its own known limitation ("no
blocking/indexing"). Here the TF-IDF index retrieves the top-N neighbours
first, and the expensive per-pair fuzzy/phonetic scoring only ever runs on
that shortlist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jellyfish
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class Weights:
    """Per-signal weights. Defaults mirror stage 04's original 0.5/0.2/0.3
    text/phonetic/GTIN split, with the GTIN share reassigned to the TF-IDF
    signal for datasets that have no barcode column (Amazon-Google doesn't).
    Exposed as a dataclass specifically so the interactive app can tune them
    live against a real F1 number instead of leaving them as a guess."""
    text: float = 0.5
    phonetic: float = 0.2
    vector: float = 0.3

    def normalized(self) -> "Weights":
        total = self.text + self.phonetic + self.vector
        if total <= 0:
            return Weights(1.0, 0.0, 0.0)
        return Weights(self.text / total, self.phonetic / total, self.vector / total)


@dataclass
class Record:
    """One side of a match: an id plus the free-text fields to compare."""
    rec_id: str
    name: str
    manufacturer: str = ""
    description: str = ""

    def blob(self) -> str:
        parts = [self.manufacturer, self.name]
        return " ".join(p for p in parts if p).strip().lower()


@dataclass
class Candidate:
    rec_id: str
    name: str
    combined: float
    text: float
    phonetic: float
    vector: float


@dataclass
class MatchEngine:
    """Fit once against a target catalog, then query it many times."""
    targets: list[Record]
    weights: Weights = field(default_factory=Weights)
    ngram_range: tuple[int, int] = (2, 4)

    def __post_init__(self):
        self._blobs = [t.blob() for t in self.targets]
        # char_wb keeps ngrams inside word boundaries, which is what makes this
        # robust to the truncation/typo corruptions rather than just to word
        # reordering (a word-level vectorizer would score "MRS" against "Mars"
        # at exactly zero -- no shared token).
        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, min_df=1, sublinear_tf=True
        )
        self._matrix = self._vec.fit_transform(self._blobs)
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self._matrix)
        self._phon = [self._metaphone(t.manufacturer or t.name) for t in self.targets]

    @staticmethod
    def _metaphone(s: str) -> str:
        token = (s or "").strip().split(" ")[0] if s else ""
        return jellyfish.metaphone(token) if token else ""

    def _block(self, query: Record, n_neighbors: int):
        """TF-IDF nearest-neighbour blocking: return (indices, cosine_sims)."""
        qv = self._vec.transform([query.blob()])
        k = min(n_neighbors, len(self.targets))
        dist, idx = self._nn.kneighbors(qv, n_neighbors=k)
        return idx[0], 1.0 - dist[0]  # cosine distance -> similarity

    def rank(self, query: Record, top_k: int = 3, block_size: int = 50) -> list[Candidate]:
        idx, sims = self._block(query, block_size)
        w = self.weights.normalized()
        q_blob = query.blob()
        q_phon = self._metaphone(query.manufacturer or query.name)

        scored: list[Candidate] = []
        for pos, vec_score in zip(idx, sims):
            target = self.targets[pos]
            text_score = fuzz.token_sort_ratio(q_blob, self._blobs[pos]) / 100.0
            phon_score = 1.0 if (q_phon and self._phon[pos] and q_phon == self._phon[pos]) else 0.0
            combined = w.text * text_score + w.phonetic * phon_score + w.vector * float(vec_score)
            scored.append(Candidate(
                rec_id=target.rec_id, name=target.name, combined=combined,
                text=text_score, phonetic=phon_score, vector=float(vec_score),
            ))
        scored.sort(key=lambda c: c.combined, reverse=True)
        return scored[:top_k]

    def rank_many(self, queries: list[Record], top_k: int = 3, block_size: int = 50):
        """Batched version -- vectorizes and blocks all queries in one pass,
        which is what makes a full 1,363-query benchmark run take seconds
        rather than minutes."""
        qvs = self._vec.transform([q.blob() for q in queries])
        k = min(block_size, len(self.targets))
        dists, idxs = self._nn.kneighbors(qvs, n_neighbors=k)
        sims = 1.0 - dists
        w = self.weights.normalized()

        out = []
        for q, idx_row, sim_row in zip(queries, idxs, sims):
            q_blob = q.blob()
            q_phon = self._metaphone(q.manufacturer or q.name)
            scored = []
            for pos, vec_score in zip(idx_row, sim_row):
                target = self.targets[pos]
                text_score = fuzz.token_sort_ratio(q_blob, self._blobs[pos]) / 100.0
                phon_score = 1.0 if (q_phon and self._phon[pos] and q_phon == self._phon[pos]) else 0.0
                combined = (w.text * text_score + w.phonetic * phon_score
                            + w.vector * float(vec_score))
                scored.append(Candidate(
                    rec_id=target.rec_id, name=target.name, combined=combined,
                    text=text_score, phonetic=phon_score, vector=float(vec_score),
                ))
            scored.sort(key=lambda c: c.combined, reverse=True)
            out.append(scored[:top_k])
        return out
