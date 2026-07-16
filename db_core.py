"""
db_core.py — lowest-level SQLite plumbing.

Kept separate from backend.py so that `data_sources/*` (the external data
layer) can share the same on-disk cache/observability tables without
creating a circular import with backend.py (which imports data_sources).
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager

DB_PATH = os.getenv("FINBOT_DB_PATH", os.path.join(os.path.dirname(__file__), "finbot_memory.db"))


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # safer for concurrent threads
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT DEFAULT 'chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,               -- 'user' | 'assistant' | 'system'
                content TEXT NOT NULL,
                agent_used TEXT,
                image_path TEXT,
                is_regeneration INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observability (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                agent_name TEXT,
                event TEXT,
                latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                error TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            );

            -- Generic TTL cache for every external data source (SEC, Finnhub,
            -- NewsAPI, yfinance). Cuts down on repeated network + SEC calls.
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON data_cache(expires_at);
            """
        )
        # Backward-compatible upgrade for older DBs created before `metadata` existed.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(observability)").fetchall()]
        if "metadata" not in cols:
            conn.execute("ALTER TABLE observability ADD COLUMN metadata TEXT")
