"""
models.py — shared, serializable data models for the research pipeline.

Every externally-retrieved fact flows through one of these dataclasses so
that provenance (source, URL, filing date, retrieval time) travels with the
value all the way into the agent prompts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


@dataclass
class SourceReference:
    source: str                      # "SEC EDGAR" | "Finnhub" | "NewsAPI" | "yfinance"
    description: str = ""
    url: Optional[str] = None
    retrieved_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class FinancialMetric:
    metric: str
    value: Optional[float]
    unit: str = "USD"
    period: Optional[str] = None          # e.g. "FY2025" or "Q3 2025"
    fiscal_period: Optional[str] = None   # e.g. "FY" | "Q1".."Q4"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source: str = "SEC EDGAR"
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class SECFilingMetadata:
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    cik: str
    report_date: Optional[str] = None
    items: Optional[str] = None            # relevant for 8-K ("Item 2.02" etc.)
    source_url: str = ""
    extracted_sections: dict = field(default_factory=dict)   # section_name -> text
    extraction_note: str = ""
    content_truncated: bool = False

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class NewsArticle:
    title: str
    source: str
    published_at: Optional[str]
    url: Optional[str]
    description: Optional[str] = None
    provider: str = ""   # "finnhub" | "newsapi"

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class CompanyIdentity:
    company_name: str
    ticker: str
    cik: Optional[str] = None
    exchange: Optional[str] = None
    resolution_method: str = ""   # "exact_ticker" | "fuzzy_name" | "unresolved"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class MarketDataSnapshot:
    ticker: str
    latest_price: Optional[float] = None
    currency: Optional[str] = None
    as_of: Optional[str] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    return_1y: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    historical_volatility_annualized: Optional[float] = None
    max_drawdown_1y: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    source: str = ""
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class ResearchDataBundle:
    company_name: str
    ticker: str
    cik: Optional[str] = None

    latest_10k: Optional[SECFilingMetadata] = None
    latest_10q: Optional[SECFilingMetadata] = None
    recent_8ks: list = field(default_factory=list)         # list[SECFilingMetadata]
    sec_company_facts_available: bool = False

    normalized_financials: dict = field(default_factory=dict)   # metric -> list[FinancialMetric]
    financial_ratios: dict = field(default_factory=dict)        # period -> {ratio: value}

    market_data: Optional[MarketDataSnapshot] = None

    news: list = field(default_factory=list)                # list[NewsArticle]

    peers: list = field(default_factory=list)                # list[str tickers]
    peer_metrics: dict = field(default_factory=dict)         # ticker -> {metric: value}

    sources: list = field(default_factory=list)              # list[SourceReference]
    fetched_at: str = field(default_factory=now_iso)
    errors: dict = field(default_factory=lambda: {"sec": [], "finnhub": [], "newsapi": [], "yfinance": []})

    def to_dict(self) -> dict:
        return _asdict(self)

    def add_error(self, source: str, message: str) -> None:
        self.errors.setdefault(source, []).append(message)
