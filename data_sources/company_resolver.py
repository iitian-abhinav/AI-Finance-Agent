"""
company_resolver.py — turns "AAPL" / "Apple" / "Apple Inc." into a normalized
CompanyIdentity, preferring deterministic SEC lookups over any LLM guess.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import CompanyIdentity
from .sec_client import sec_client

logger = logging.getLogger("finbot.resolver")

_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")
_EMBEDDED_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

# Common all-caps acronyms that show up in financial questions but are not
# themselves tickers — excluded so "What is NVDA's RSI?" doesn't try RSI
# before (or instead of) NVDA. This is a heuristic guard, not a full deny
# list; a genuine ticker collision is still resolved correctly since we
# verify every candidate against SEC's real ticker list before accepting it.
_ACRONYM_STOPWORDS = {
    "RSI", "SEC", "CEO", "CFO", "COO", "EPS", "ROE", "ROI", "ROA", "ROIC",
    "MACD", "SMA", "EMA", "GDP", "ETF", "IPO", "LLC", "INC", "USD", "EUR",
    "GBP", "API", "FAQ", "ATH", "YOY", "QOQ", "EBIT", "PE", "DCF", "SWOT",
    "10K", "10Q", "8K", "XBRL", "MDA", "OK", "US", "USA", "AI", "IT", "ID",
    "RSA", "AM", "PM",
}

# Trailing/leading noise words to strip before treating remaining text as a
# candidate company name for fuzzy SEC title search.
_NAME_QUERY_STOPWORDS = re.compile(
    r"^(what|whats|show|tell|give|is|are|the|me|please|for)\s+",
    re.IGNORECASE,
)


class CompanyResolver:
    def resolve(self, query: str) -> CompanyIdentity:
        query = query.strip()
        if not query:
            return CompanyIdentity(company_name="", ticker="", resolution_method="unresolved")

        # 1) Exact ticker match (cheapest, most common case: "NVDA", "chart AAPL")
        if _TICKER_RE.match(query):
            identity = self._try_ticker(query.upper(), "exact_ticker", 1.0)
            if identity:
                return identity

        # 2) Ticker embedded in a longer sentence, e.g. "What is NVDA's RSI?"
        # Try each all-caps token in the order it appears, skipping common
        # financial acronyms, and accept the first one that's a real SEC
        # ticker (verified via ticker_to_cik — never assumed).
        for token in _EMBEDDED_TICKER_RE.findall(query):
            if token in _ACRONYM_STOPWORDS:
                continue
            identity = self._try_ticker(token, "embedded_ticker", 0.9)
            if identity:
                return identity

        # 3) Fuzzy company-name match against SEC's title list. Strip common
        # leading question words first so "What is Apple's revenue" searches
        # for "Apple's revenue" -> still matches "Apple Inc." via substring.
        cleaned = _NAME_QUERY_STOPWORDS.sub("", query).strip(" ?.!'\"")
        for candidate in (cleaned, query):
            try:
                matches = sec_client.find_by_company_name(candidate, limit=1)
            except Exception as e:
                logger.warning("SEC company-name lookup failed for %r: %s", candidate, e)
                matches = []
            if matches:
                m = matches[0]
                return CompanyIdentity(
                    company_name=m["company_name"], ticker=m["ticker"] or "", cik=m["cik"],
                    resolution_method="fuzzy_name", confidence=0.7,
                )

        # 4) Unresolved — caller should treat this as "SEC data unavailable"
        # rather than fabricate a CIK/ticker.
        return CompanyIdentity(company_name=query, ticker="", resolution_method="unresolved", confidence=0.0)

    def _try_ticker(self, ticker: str, method: str, confidence: float) -> Optional[CompanyIdentity]:
        try:
            cik = sec_client.ticker_to_cik(ticker)
        except Exception as e:
            logger.warning("SEC ticker lookup failed for %r: %s", ticker, e)
            return None
        if not cik:
            return None
        name = self._name_for_ticker(ticker)
        return CompanyIdentity(
            company_name=name or ticker, ticker=ticker, cik=cik,
            resolution_method=method, confidence=confidence,
        )

    def _name_for_ticker(self, ticker: str) -> str:
        try:
            tickers = sec_client.get_company_tickers()
        except Exception:
            return ""
        for entry in tickers.values():
            if entry.get("ticker", "").upper() == ticker:
                return entry.get("title", "")
        return ""


company_resolver = CompanyResolver()