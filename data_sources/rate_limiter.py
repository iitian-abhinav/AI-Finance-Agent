"""
rate_limiter.py — a single, process-wide, thread-safe rate limiter for SEC EDGAR.

SEC EDGAR's fair-access policy allows a higher ceiling, but this application
deliberately never exceeds SEC_MAX_REQUESTS_PER_SECOND (5), regardless of how
many threads or agents are issuing requests concurrently.

Implementation: a token bucket with capacity == rate. `acquire()` blocks the
calling thread (sleeping, not busy-waiting) until a token is available. A
single `threading.Lock` protects the token count, so the limiter is safe to
share across an arbitrary number of worker threads (e.g. the ThreadPoolExecutor
used by run_full_research).

Only SECClient is meant to call `.acquire()` on this limiter — see sec_client.py.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("finbot.rate_limiter")

SEC_MAX_REQUESTS_PER_SECOND = 5.0


class TokenBucketRateLimiter:
    """Thread-safe token-bucket limiter. One instance is shared process-wide."""

    def __init__(self, max_per_second: float):
        # Hard ceiling: no configuration can push this above the SEC max.
        self.max_per_second = min(max_per_second, SEC_MAX_REQUESTS_PER_SECOND)
        self.capacity = self.max_per_second
        self._tokens = self.max_per_second
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()

    def acquire(self) -> float:
        """Blocks (sleeping) until a token is available. Returns seconds waited."""
        total_wait = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._last_refill = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.max_per_second)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    if total_wait > 0:
                        logger.info("SEC rate limiter: waited %.3fs before request", total_wait)
                    return total_wait
                deficit = 1.0 - self._tokens
                sleep_for = deficit / self.max_per_second
            time.sleep(sleep_for)
            total_wait += sleep_for


def _configured_rate() -> float:
    try:
        configured = float(os.getenv("SEC_MAX_REQUESTS_PER_SECOND", SEC_MAX_REQUESTS_PER_SECOND))
    except ValueError:
        configured = SEC_MAX_REQUESTS_PER_SECOND
    effective_rate = min(configured, SEC_MAX_REQUESTS_PER_SECOND)
    if configured > SEC_MAX_REQUESTS_PER_SECOND:
        logger.warning(
            "Configured SEC rate %.2f/s exceeds the hard cap; clamping to %.2f/s",
            configured, SEC_MAX_REQUESTS_PER_SECOND,
        )
    return effective_rate


# Single process-wide instance. Every SECClient (even if multiple are
# instantiated) shares this same limiter object.
SEC_RATE_LIMITER = TokenBucketRateLimiter(_configured_rate())
