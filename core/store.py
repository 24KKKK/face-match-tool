# -*- coding: utf-8 -*-
"""SQLite 状态库。

存在的意义：几万张图可能要跑一两个小时，中途崩了不必重来。
已处理过的文件会记录 status，重跑时自动跳过；同时也给归档阶段提供数据来源。
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, Iterable, List

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_result (
    path        TEXT PRIMARY KEY,
    kind        TEXT,
    size        INTEGER,
    mtime       REAL,
    cos         REAL,
    score       REAL,
    faces       INTEGER DEFAULT 0,
    frames      INTEGER DEFAULT 0,
    best_sec    REAL,
    ts          TEXT,
    ts_source   TEXT,
    status      TEXT,
    message     TEXT,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_score ON scan_result(score);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def done_paths(self) -> set:
        """已处理的路径集合，用于断点续跑。"""
        rows = self.conn.execute("SELECT path FROM scan_result").fetchall()
        return {r[0] for r in rows}

    def upsert_many(self, rows: Iterable[Dict]) -> None:
        payload = []
        for row in rows:
            payload.append((
                row.get("path"), row.get("kind"), row.get("size"), row.get("mtime"),
                row.get("cos"), row.get("score"),
                int(row.get("faces", 0) or 0), int(row.get("frames", 0) or 0),
                row.get("best_sec"), row.get("ts"), row.get("ts_source"),
                row.get("status"), row.get("message"), time.strftime("%Y-%m-%d %H:%M:%S"),
            ))
        self.conn.executemany(
            "INSERT INTO scan_result(path, kind, size, mtime, cos, score, faces, frames,"
            " best_sec, ts, ts_source, status, message, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, size=excluded.size,"
            " mtime=excluded.mtime, cos=excluded.cos, score=excluded.score,"
            " faces=excluded.faces, frames=excluded.frames, best_sec=excluded.best_sec,"
            " ts=excluded.ts, ts_source=excluded.ts_source, status=excluded.status,"
            " message=excluded.message, updated_at=excluded.updated_at",
            payload,
        )
        self.conn.commit()

    def fetch_all(self) -> List[Dict]:
        cursor = self.conn.execute(
            "SELECT path, kind, size, mtime, cos, score, faces, frames, best_sec,"
            " ts, ts_source, status, message FROM scan_result"
        )
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def summary_by_status(self) -> Dict[str, int]:
        cursor = self.conn.execute("SELECT status, COUNT(*) FROM scan_result GROUP BY status")
        return {row[0]: row[1] for row in cursor.fetchall()}
