from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CachedTeacherScore:
    key: str
    seq_hash: str
    seq_len: int
    global_score: float
    junction_scores: list[float]
    meta: dict


class ScoreCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS teacher_scores (
                  key TEXT PRIMARY KEY,
                  seq_hash TEXT NOT NULL,
                  seq_len INTEGER NOT NULL,
                  global_score REAL NOT NULL,
                  junction_scores_json TEXT NOT NULL,
                  meta_json TEXT NOT NULL,
                  created_at REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[CachedTeacherScore]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT key, seq_hash, seq_len, global_score, junction_scores_json, meta_json FROM teacher_scores WHERE key=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return CachedTeacherScore(
            key=row[0],
            seq_hash=row[1],
            seq_len=int(row[2]),
            global_score=float(row[3]),
            junction_scores=json.loads(row[4]),
            meta=json.loads(row[5]),
        )

    def set(
        self,
        *,
        key: str,
        seq_hash: str,
        seq_len: int,
        global_score: float,
        junction_scores: list[float],
        meta: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO teacher_scores
                (key, seq_hash, seq_len, global_score, junction_scores_json, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    seq_hash,
                    int(seq_len),
                    float(global_score),
                    json.dumps(list(junction_scores)),
                    json.dumps(meta),
                    time.time(),
                ),
            )

