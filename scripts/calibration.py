"""
Turning the match margin into an actual probability.

Stage 07 routes on bands the build guide states in the language of confidence:
auto-merge at >=0.90, review at 0.60-0.90, reject below. For those numbers to
mean anything, 0.90 has to mean "90% of the submissions scored this way are
correct". It did not. Confidence was `margin / MARGIN_SATURATION`, where
MARGIN_SATURATION was a constant picked against a 30-product catalog.

On 1,081 products the median margin for a CORRECT match is 0.078, so that
mapping squashed almost everything into the bottom of the range: only 4% of
rows cleared the review threshold and 914 correct matches were rejected rather
than queued for a human. The signal itself is fine -- it ranks correct above
wrong 91% of the time -- but the function mapping it onto a confidence band was
calibrated for a different world.

So: fit the mapping on labelled data instead of guessing it.

  margin  --calibrator-->  P(top candidate is correct)

Two standard choices, both fitted here so the comparison is on evidence rather
than preference:

  PLATT SCALING     a logistic curve, 2 parameters. Strong prior that the
                    relationship is monotone and smooth; hard to overfit;
                    extrapolates sanely past the edges of the training data.

  ISOTONIC          a monotone step function, non-parametric. Can follow a
                    kink Platt would smooth over, but needs more data and
                    will happily memorise noise in sparse regions.

WHAT THIS CANNOT TELL YOU
Every labelled event in the Leipzig benchmarks is a positive pair -- the gold
mapping only lists submissions that DO have a catalog match. So the fitted
probability is P(correct | a match exists), and the reject band is not
exercised at all. Production feeds contain submissions for products that are
genuinely not in the catalog, and nothing here measures those. See
`synthesise_absent_matches()` for a partial answer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "data" / "calibration"


# ---------------------------------------------------------------------------
# Fitted calibrators
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    """P(correct) = sigmoid(a * margin + b)."""
    a: float
    b: float
    kind: str = "platt"

    def __call__(self, margin):
        m = np.asarray(margin, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.a * m + self.b)))


@dataclass
class IsotonicCalibrator:
    """Monotone step function through (margin, P) knots, linearly interpolated."""
    x: list
    y: list
    kind: str = "isotonic"

    def __call__(self, margin):
        return np.interp(np.asarray(margin, dtype=float), self.x, self.y)


def fit_platt(margins, labels, iters: int = 200) -> PlattCalibrator:
    """Newton-Raphson on the logistic log-likelihood.

    Hand-rolled rather than pulled from sklearn so the two parameters that
    decide every routing threshold are visible and auditable in this file --
    the previous magic constant is exactly what went wrong.
    """
    x = np.asarray(margins, dtype=float)
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = np.array([np.sum((y - p) * x), np.sum(y - p)])
        h = np.array([[-np.sum(w * x * x), -np.sum(w * x)],
                      [-np.sum(w * x),     -np.sum(w)]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return PlattCalibrator(a=float(a), b=float(b))


def fit_isotonic(margins, labels) -> IsotonicCalibrator:
    """Pool-adjacent-violators: the monotone step function minimising squared
    error against the labels."""
    order = np.argsort(np.asarray(margins, dtype=float))
    x = np.asarray(margins, dtype=float)[order]
    y = np.asarray(labels, dtype=float)[order]

    # PAVA: repeatedly merge adjacent blocks that violate monotonicity.
    vals = list(y)
    weights = [1.0] * len(y)
    idx = [[i] for i in range(len(y))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1]:
            i += 1
            continue
        tw = weights[i] + weights[i + 1]
        vals[i] = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / tw
        weights[i] = tw
        idx[i] = idx[i] + idx[i + 1]
        del vals[i + 1], weights[i + 1], idx[i + 1]
        if i > 0:
            i -= 1

    knot_x, knot_y = [], []
    for v, block in zip(vals, idx):
        knot_x.append(float(x[block[0]]))
        knot_y.append(float(v))
        knot_x.append(float(x[block[-1]]))
        knot_y.append(float(v))
    return IsotonicCalibrator(x=knot_x, y=knot_y)


# ---------------------------------------------------------------------------
# Scoring a calibrator
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    n: int
    brier: float          # mean squared error of the probability -- lower better
    log_loss: float       # penalises confident-and-wrong hardest
    ece: float            # expected calibration error: |claimed - actual|, binned
    max_bin_gap: float    # worst single bin -- ECE can hide one bad band
    auc: float

    def as_row(self):
        return asdict(self)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc_score(scores, labels):
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def reliability(p, y, bins: int = 10):
    """Per-bin (claimed probability, observed frequency, count)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def evaluate(p, y, bins: int = 10) -> CalibrationReport:
    rel = reliability(p, y, bins)
    n = len(y)
    ece = sum(cnt / n * abs(claimed - actual) for claimed, actual, cnt in rel)
    gap = max((abs(c - a) for c, a, _ in rel), default=0.0)
    return CalibrationReport(
        n=n, brier=brier(p, y), log_loss=log_loss(p, y),
        ece=float(ece), max_bin_gap=float(gap), auc=auc_score(p, y),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

CALIBRATION_FILE = "calibration.json"


def save(calibrator, run_dir: Path, meta: dict | None = None) -> Path:
    """Store a calibrator ALONGSIDE the run it was fitted on.

    Deliberately not a single global calibration file. The measured transfer
    result (see run_calibration.py, Q2) is that a curve fitted on abt-buy is
    systematically overconfident when applied to amazon-google -- it claims
    0.65 where the true rate is 0.49. Better than the hardcoded constant, but
    still wrong, and shipping one global curve would be the same error that
    produced MARGIN_SATURATION: a number fitted against one catalog and quietly
    applied to every other.

    So each catalog carries its own. A run with no calibration.json falls back
    to the uncalibrated mapping rather than borrowing someone else's.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CALIBRATION_FILE
    payload = asdict(calibrator)
    if meta:
        payload["_meta"] = meta
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(run_dir: Path):
    """The calibrator fitted for this run, or None to fall back uncalibrated."""
    path = Path(run_dir) / CALIBRATION_FILE
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d.pop("_meta", None)
    kind = d.pop("kind")
    return PlattCalibrator(**d) if kind == "platt" else IsotonicCalibrator(**d)
