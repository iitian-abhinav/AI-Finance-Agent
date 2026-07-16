"""
finnhub_client.py — quote, company profile, news, and peers via Finnhub.

Finnhub is optional: if FINNHUB_API_KEY is not set, every method returns
None/[] and the caller (research_engine / agents) falls back to yfinance
and/or NewsAPI. No SEC traffic goes through this module.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from .cache import TTL_NEWS, TTL_PEERS, TTL_PROFILE, TTL_QUOTE, cache_key, get_cached_data, set_cached_data
from .models import NewsArticle

logger = logging.getLogger("finbot.finnhub")

BASE_URL = "https://finnhub.io/api/v1"
TIMEOUT = 10


class FinnhubClient:
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict) -> Optional[dict | list]:
        if not self.enabled:
            return None
        params = {**params, "token": self.api_key}
        try:
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("finnhub_request failed path=%s err=%s", path, e)
            return None

    def quote(self, symbol: str) -> Optional[dict]:
        key = cache_key("finnhub", "quote", symbol)
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        data = self._get("/quote", {"symbol": symbol})
        if data and data.get("c"):  # 'c' == current price
            set_cached_data(key, "Finnhub", data, TTL_QUOTE)
        return data

    def company_profile(self, symbol: str) -> Optional[dict]:
        key = cache_key("finnhub", "profile", symbol)
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        data = self._get("/stock/profile2", {"symbol": symbol})
        if data:
            set_cached_data(key, "Finnhub", data, TTL_PROFILE)
        return data

    def company_news(self, symbol: str, from_date: str, to_date: str) -> list[NewsArticle]:
        key = cache_key("finnhub", "news", symbol, from_date, to_date)
        cached = get_cached_data(key)
        if cached is not None:
            return [NewsArticle(**a) for a in cached]
        data = self._get("/company-news", {"symbol": symbol, "from": from_date, "to": to_date}) or []
        articles = [
            NewsArticle(
                title=a.get("headline", ""),
                source=a.get("source", "Finnhub"),
                published_at=a.get("datetime"),
                url=a.get("url"),
                description=a.get("summary"),
                provider="finnhub",
            )
            for a in data if a.get("headline")
        ]
        set_cached_data(key, "Finnhub", [a.to_dict() for a in articles], TTL_NEWS)
        return articles

    def company_peers(self, symbol: str) -> list[str]:
        key = cache_key("finnhub", "peers", symbol)
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        data = self._get("/stock/peers", {"symbol": symbol}) or []
        peers = [p for p in data if isinstance(p, str) and p.upper() != symbol.upper()]
        set_cached_data(key, "Finnhub", peers, TTL_PEERS)
        return peers


finnhub_client = FinnhubClient()
