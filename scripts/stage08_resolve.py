"""
Stage 08 -- Resolve, log, and learn.

Two persistent artifacts, both in SQLite alongside the master catalog.

AUDIT LOG
  Every decision written with the evidence behind it. The guide is specific
  about one thing here: the log must distinguish a SUPPLIER DATA-ENTRY ERROR
  (caught at stage 03 -- the pack and the form disagree) from a RESOLUTION
  AMBIGUITY (caught at stage 07 -- we can't tell which catalog product this
  is). Those are two different problems with two different owners: the first
  goes back to the supplier, the second goes to a data steward. A log that
  collapses them into "failed" tells you how many problems you have but not
  who should fix any of them, which is the difference between a metric and an
  operational signal.

ALIAS CACHE
  When a human approves a correction ("MRS" really is Mars Bar), that decision
  is stored keyed on the normalized input. Next time the identical input
  arrives, stage 07 gets an alias hit and the same correction doesn't have to
  be re-derived or re-reviewed.

  The cache stores only HUMAN-APPROVED corrections, never the pipeline's own
  auto-merges. Learning from your own unreviewed output is how a resolution
  system drifts: one wrong auto-merge becomes a permanent "fact" that raises
  confidence on the next identical wrong input. Requiring a human in the loop
  before anything enters the cache keeps the feedback loop grounded.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

# The audit log and alias cache live in the same database as the catalog the
# run resolved against, so a run's decisions can never be read back against a
# different dataset's products.
DB_PATH = paths.db_path("synthetic")

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    resolved_id     INTEGER,
    true_master_id  INTEGER,
    route           TEXT NOT NULL,
    confidence      REAL NOT NULL,
    problem_class   TEXT NOT NULL,
    decision_maker  TEXT NOT NULL,
    approved_by     TEXT,
    evidence        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alias_cache (
    alias_key       TEXT PRIMARY KEY,
    resolved_id     INTEGER NOT NULL,
    resolved_name   TEXT NOT NULL,
    approved_by     TEXT NOT NULL,
    approved_at     TEXT NOT NULL,
    times_reused    INTEGER NOT NULL DEFAULT 0
);
"""


# Problem taxonomy -- the distinction the guide asks the log to preserve.
class ProblemClass:
    CLEAN = "clean"                                  # nothing wrong
    SUPPLIER_DATA_ENTRY = "supplier_data_entry_error"  # stage 03: pack vs form
    RESOLUTION_AMBIGUITY = "resolution_ambiguity"      # stage 07: which product?
    BOTH = "both"                                      # defective AND ambiguous
    UNRESOLVED = "unresolved"                          # rejected outright


def classify_problem(source_agreement: float, route: str) -> str:
    defective = source_agreement < 1.0
    unresolved = route == "reject"
    ambiguous = route == "hold_for_review"

    if unresolved:
        return ProblemClass.UNRESOLVED
    if defective and ambiguous:
        return ProblemClass.BOTH
    if defective:
        return ProblemClass.SUPPLIER_DATA_ENTRY
    if ambiguous:
        return ProblemClass.RESOLUTION_AMBIGUITY
    return ProblemClass.CLEAN


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def alias_key(brand: str, product_name: str) -> str:
    """Cache key -- normalized so trivial case/spacing variation still hits."""
    return " ".join(f"{brand} {product_name}".lower().split())


def lookup_alias(conn: sqlite3.Connection, brand: str, product_name: str):
    row = conn.execute(
        "SELECT resolved_id, resolved_name, approved_by FROM alias_cache WHERE alias_key = ?",
        (alias_key(brand, product_name),),
    ).fetchone()
    return None if row is None else {"resolved_id": row[0], "resolved_name": row[1],
                                     "approved_by": row[2]}


def record_alias_reuse(conn: sqlite3.Connection, brand: str, product_name: str) -> None:
    conn.execute(
        "UPDATE alias_cache SET times_reused = times_reused + 1 WHERE alias_key = ?",
        (alias_key(brand, product_name),),
    )
    conn.commit()


def log_decision(
    conn: sqlite3.Connection, *, event_id: str, resolved_id, true_master_id,
    route: str, confidence: float, source_agreement: float, evidence: dict,
    decision_maker: str = "agent", approved_by: str | None = None,
) -> str:
    problem_class = classify_problem(source_agreement, route)
    conn.execute(
        "INSERT INTO audit_log (ts, event_id, resolved_id, true_master_id, route, "
        "confidence, problem_class, decision_maker, approved_by, evidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), event_id,
         resolved_id, true_master_id, route, confidence, problem_class,
         decision_maker, approved_by, json.dumps(evidence, sort_keys=True)),
    )
    conn.commit()
    return problem_class


def approve_correction(
    conn: sqlite3.Connection, *, brand: str, product_name: str,
    resolved_id: int, resolved_name: str, approved_by: str,
) -> None:
    """Record a HUMAN-approved correction into the alias cache.

    Deliberately the only way anything enters the cache -- the pipeline never
    writes its own auto-merges here. See the module docstring.
    """
    conn.execute(
        "INSERT INTO alias_cache (alias_key, resolved_id, resolved_name, approved_by, approved_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(alias_key) DO UPDATE SET resolved_id=excluded.resolved_id, "
        "resolved_name=excluded.resolved_name, approved_by=excluded.approved_by, "
        "approved_at=excluded.approved_at",
        (alias_key(brand, product_name), resolved_id, resolved_name, approved_by,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()


def summarize(conn: sqlite3.Connection) -> dict:
    routes = dict(conn.execute(
        "SELECT route, COUNT(*) FROM audit_log GROUP BY route").fetchall())
    problems = dict(conn.execute(
        "SELECT problem_class, COUNT(*) FROM audit_log GROUP BY problem_class").fetchall())
    aliases = conn.execute("SELECT COUNT(*), COALESCE(SUM(times_reused),0) FROM alias_cache").fetchone()
    return {"routes": routes, "problems": problems,
            "alias_entries": aliases[0], "alias_reuses": aliases[1]}
