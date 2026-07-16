"""
cache.py — reusable TTL cache on top of the shared `data_cache` SQLite table.

Every external data client (SEC, Finnhub, NewsAPI, yfinance) should check
this cache before making a network call, and populate it after a successful
call. This cuts down on latency, third-party API usage, and — most
importantly for SEC — the number of outbound requests that have to pass
through the rate limiter.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from db_core import get_conn, new_id

logger = logging.getLogger("finbot.cache")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_cached_data(cache_key: str) -> Optional[Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT payload, expires_at FROM data_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
    if not row:
        return None
    try:
        expires_at = datetime.fromisoformat(row["expires_at"])
    except ValueError:
        return None
    if expires_at < _now():
        return None  # stale — caller will refetch and overwrite
    try:
        return json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        logger.warning("Cache payload for %s was not valid JSON; ignoring", cache_key)
        return None


def set_cached_data(cache_key: str, source: str, payload: Any, ttl_seconds: int) -> None:
    now = _now()
    expires_at = now + timedelta(seconds=ttl_seconds)
    try:
        serialized = json.dumps(payload, default=str)
    except TypeError as e:
        logger.warning("Could not cache payload for %s: %s", cache_key, e)
        return
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO data_cache (cache_key, source, payload, fetched_at, expires_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 source=excluded.source, payload=excluded.payload,
                 fetched_at=excluded.fetched_at, expires_at=excluded.expires_at""",
            (cache_key, source, serialized, now.isoformat(), expires_at.isoformat()),
        )


def cache_key(*parts: str) -> str:
    return ":".join(str(p) for p in parts if p is not None)


# Suggested TTLs (seconds), centralized so callers stay consistent.
TTL_TICKER_MAP = 30 * 24 * 3600
TTL_10K = 24 * 3600
TTL_10Q = 12 * 3600
TTL_8K = 4 * 3600
TTL_COMPANY_FACTS = 12 * 3600
TTL_HISTORICAL_MARKET = 3600
TTL_QUOTE = 180
TTL_NEWS = 30 * 60
TTL_PROFILE = 24 * 3600
TTL_PEERS = 24 * 3600
