"""
research_engine.py — orchestrates real, evidence-grounded data retrieval for
the financial research agents.

Pipeline (see refactor_full_research / lazy per-agent helpers below):
  1. Resolve company identity (SEC-backed, deterministic).
  2. Fetch independent external sources concurrently: SEC (rate-limited
     internally), Finnhub, NewsAPI, yfinance.
  3. Assemble one ResearchDataBundle.
  4. Run deterministic Python analytics: XBRL normalization, ratio
     calculations, technical indicators, news dedup.
  5. Build compact, source-grounded context blocks for each specialist agent.

Nothing here calls the LLM — this module only gathers and computes facts.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Prompt/context builders — turn the bundle into compact, evidence-only text
# for each specialist agent. Kept short and structured so agents cite facts
# rather than reasoning over a wall of raw JSON.
# --------------------------------------------------------------------------

# Conservative LLM-facing SEC evidence budgets. Raw retrieved filings remain
# unchanged; these limits apply only to the evidence sent to the model.
SEC_CONTEXT_MAX_CHARS = 24_000
SEC_10K_MAX_CHARS = 12_000
SEC_10Q_MAX_CHARS = 7_000
SEC_8K_COMBINED_MAX_CHARS = 3_000
SEC_8K_PER_FILING_MAX_CHARS = 1_500

from data_sources import (
    ResearchDataBundle, SourceReference,
    calculate_ratios, normalize_xbrl_facts,
    company_resolver, finnhub_client, market_client, news_client, sec_client,
)

logger = logging.getLogger("finbot.research_engine")

ProgressCB = Optional[Callable[[str, str], None]]


def build_research_bundle(query: str, progress_cb: ProgressCB = None,
                           observe: Optional[Callable[..., None]] = None) -> ResearchDataBundle:
    """query may be a ticker or a company name. Fetches everything concurrently
    (SEC calls are internally rate-limited to 5 req/s regardless of thread count)."""

    def _log(agent: str, event: str, latency_ms: float = 0.0, error: Optional[str] = None):
        if observe:
            observe(agent, event, latency_ms, error=error)

    if progress_cb:
        progress_cb("resolve", "running")
    identity = company_resolver.resolve(query)
    if progress_cb:
        progress_cb("resolve", "done")

    bundle = ResearchDataBundle(
        company_name=identity.company_name or query,
        ticker=identity.ticker or query.upper(),
        cik=identity.cik,
    )

    if identity.resolution_method == "unresolved":
        bundle.add_error("sec", f"Could not resolve '{query}' to a known SEC-registered company/ticker.")

    def _fetch_sec():
        if identity.resolution_method == "unresolved":
            return
        if progress_cb:
            progress_cb("sec_filing", "running")
        start = time.time()
        try:
            result = sec_client.get_company_sec_bundle(identity.ticker, cik=identity.cik)
            bundle.latest_10k = result["latest_10k"]
            bundle.latest_10q = result["latest_10q"]
            bundle.recent_8ks = result["recent_8ks"]
            bundle.sources.extend(result["sources"])
            for e in result["errors"]:
                bundle.add_error("sec", e)
            facts = result.get("company_facts")
            bundle.sec_company_facts_available = bool(facts)
            if facts:
                bundle.normalized_financials = normalize_xbrl_facts(facts)
                bundle.financial_ratios = calculate_ratios(bundle.normalized_financials)
            _log("SECClient", "sec_bundle_fetch", (time.time() - start) * 1000)
        except Exception as e:
            bundle.add_error("sec", str(e))
            _log("SECClient", "sec_bundle_fetch", (time.time() - start) * 1000, error=str(e))
        if progress_cb:
            progress_cb("sec_filing", "done")

    def _fetch_market():
        if progress_cb:
            progress_cb("market_data", "running")
        start = time.time()
        try:
            bundle.market_data = market_client.get_snapshot(identity.ticker or query)
            bundle.sources.append(SourceReference("yfinance / Finnhub", "Market data & technicals"))
            for e in (bundle.market_data.errors or []):
                bundle.add_error("yfinance", e)
            _log("MarketDataClient", "market_data_fetch", (time.time() - start) * 1000)
        except Exception as e:
            bundle.add_error("yfinance", str(e))
            _log("MarketDataClient", "market_data_fetch", (time.time() - start) * 1000, error=str(e))
        if progress_cb:
            progress_cb("market_data", "done")

    def _fetch_news():
        if progress_cb:
            progress_cb("news_intelligence", "running")
        start = time.time()
        try:
            articles, errors = news_client.get_company_news(identity.company_name, identity.ticker)
            bundle.news = articles
            for e in errors:
                bundle.add_error("newsapi", e)
            if articles:
                bundle.sources.append(SourceReference("NewsAPI / Finnhub", f"{len(articles)} recent articles"))
            _log("NewsClient", "news_fetch", (time.time() - start) * 1000)
        except Exception as e:
            bundle.add_error("newsapi", str(e))
            _log("NewsClient", "news_fetch", (time.time() - start) * 1000, error=str(e))
        if progress_cb:
            progress_cb("news_intelligence", "done")

    def _fetch_peers():
        if progress_cb:
            progress_cb("comparative_analysis", "running")
        start = time.time()
        try:
            peers = finnhub_client.company_peers(identity.ticker) if identity.ticker else []
            bundle.peers = peers[:5]
            for p in bundle.peers:
                try:
                    q = finnhub_client.quote(p)
                    profile = finnhub_client.company_profile(p)
                    if q or profile:
                        bundle.peer_metrics[p] = {
                            "price": (q or {}).get("c"),
                            "market_cap": (profile or {}).get("marketCapitalization"),
                            "industry": (profile or {}).get("finnhubIndustry"),
                        }
                except Exception as e:
                    bundle.add_error("finnhub", f"Peer {p} metrics failed: {e}")
            if not bundle.peers:
                bundle.add_error("finnhub", "Peer comparison unavailable (Finnhub peers not available).")
            _log("FinnhubClient", "peer_fetch", (time.time() - start) * 1000)
        except Exception as e:
            bundle.add_error("finnhub", str(e))
            _log("FinnhubClient", "peer_fetch", (time.time() - start) * 1000, error=str(e))
        if progress_cb:
            progress_cb("comparative_analysis", "done")

    # Independent sources fetched concurrently. SEC calls inside _fetch_sec
    # still funnel through the single shared, thread-safe SEC_RATE_LIMITER.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(f) for f in (_fetch_sec, _fetch_market, _fetch_news, _fetch_peers)]
        for fut in futures:
            fut.result()

    return bundle


_SEC_SECTION_PRIORITY = {
    "10-K": ["item_1a_risk_factors", "item_7_mdna", "item_1_business",
             "item_3_legal_proceedings", "item_7a_market_risk",
             "item_8_financial_statements"],
    "10-Q": ["mdna", "risk_factors", "controls_and_procedures",
             "financial_statements"],
}

def _clip_text(text: str, max_chars: int,
               marker: str = "\n[TRUNCATED TO FIT LLM EVIDENCE BUDGET]") -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker

def _ordered_sections(filing) -> list[tuple[str, str]]:
    sections = list((filing.extracted_sections or {}).items())
    priority = _SEC_SECTION_PRIORITY.get(filing.form_type, [])
    rank = {name: i for i, name in enumerate(priority)}
    return sorted(sections, key=lambda item: (rank.get(item[0], len(priority)), item[0]))

def _filing_context(filing, max_chars: int) -> str:
    if not filing:
        return "(not available)"
    header_lines = [
        f"Form: {filing.form_type} | Filed: {filing.filing_date} | Report period: {filing.report_date}",
        f"Source URL: {filing.source_url}",
    ]
    if filing.items:
        header_lines.append(f"8-K Items: {filing.items}")
    if filing.content_truncated:
        header_lines.append("NOTE: source extraction was already truncated; this is not the full filing.")
    output = "\n".join(header_lines)
    remaining = max_chars - len(output)
    if remaining <= 0:
        return _clip_text(output, max_chars)
    sections = _ordered_sections(filing)
    if not sections:
        return output + "\n(No filing sections were extracted.)"
    for idx, (name, section_text) in enumerate(sections):
        prefix = f"\n\n--- {name} ---\n"
        if remaining <= len(prefix) + 80:
            break
        output += prefix
        remaining -= len(prefix)
        sections_left = max(1, len(sections) - idx)
        section_budget = max(500, remaining // sections_left)
        clipped = _clip_text(section_text or "", min(section_budget, remaining))
        output += clipped
        remaining -= len(clipped)
        if remaining <= 100:
            break
    return _clip_text(output, max_chars)

def build_sec_filing_context(bundle: ResearchDataBundle) -> str:
    parts = [f"Company: {bundle.company_name} ({bundle.ticker}), CIK {bundle.cik}"]
    parts.append("### Latest 10-K\n" + _filing_context(bundle.latest_10k, SEC_10K_MAX_CHARS))
    parts.append("### Latest 10-Q\n" + _filing_context(bundle.latest_10q, SEC_10Q_MAX_CHARS))
    if bundle.recent_8ks:
        eight_k_parts = []
        remaining_8k = SEC_8K_COMBINED_MAX_CHARS
        for filing in bundle.recent_8ks:
            if remaining_8k <= 200:
                break
            per_filing = min(SEC_8K_PER_FILING_MAX_CHARS, remaining_8k)
            ctx = _filing_context(filing, per_filing)
            eight_k_parts.append(ctx)
            remaining_8k -= len(ctx)
        parts.append("### Recent 8-Ks\n" + "\n\n".join(eight_k_parts))
    else:
        parts.append("### Recent 8-Ks\n(none retrieved)")
    if bundle.errors.get("sec"):
        parts.append("### Retrieval issues\n" + _clip_text("\n".join(bundle.errors["sec"]), 1_000))
    context = "\n\n".join(parts)
    context = _clip_text(context, SEC_CONTEXT_MAX_CHARS)
    context += (f"\n\n[LLM EVIDENCE BUDGET: {len(context):,} characters supplied. "
                "Truncation markers indicate prompt-size budgeting only.]")
    return context


def build_financial_statement_context(bundle: ResearchDataBundle) -> str:
    if not bundle.normalized_financials:
        return (
            f"No SEC XBRL Company Facts could be retrieved for {bundle.company_name} "
            f"({bundle.ticker}). Errors: {'; '.join(bundle.errors.get('sec', [])) or 'unknown'}. "
            "Do not fabricate figures — state that financial data is unavailable."
        )
    lines = [f"Company: {bundle.company_name} ({bundle.ticker})\n", "### Normalized financials (most recent periods first)"]
    for metric, entries in bundle.normalized_financials.items():
        vals = "; ".join(f"{e.period}: {e.value:,.0f} {e.unit}" if e.value is not None else f"{e.period}: N/A" for e in entries[:4])
        lines.append(f"- {metric}: {vals}")
    lines.append("\n### Python-calculated ratios (most recent period)")
    for k, v in bundle.financial_ratios.items():
        lines.append(f"- {k}: {v if v is not None else 'N/A (insufficient data)'}")
    return "\n".join(lines)


def build_market_data_context(bundle: ResearchDataBundle) -> str:
    md = bundle.market_data
    if not md or md.source == "unavailable":
        return f"Live market data unavailable for {bundle.ticker}. Errors: {'; '.join(bundle.errors.get('yfinance', []))}"
    lines = [
        f"Ticker: {md.ticker} | Source: {md.source} | As of: {md.as_of}",
        f"Latest price: {md.latest_price}",
        f"Returns — 1m: {md.return_1m}% | 3m: {md.return_3m}% | 6m: {md.return_6m}% | 1y: {md.return_1y}%",
        f"SMA20: {md.sma_20} | SMA50: {md.sma_50} | SMA200: {md.sma_200}",
        f"RSI(14): {md.rsi_14} | MACD: {md.macd} (signal {md.macd_signal})",
        f"Annualized historical volatility: {md.historical_volatility_annualized}%",
        f"Max drawdown (1y): {md.max_drawdown_1y}%",
        f"Avg volume (30d): {md.avg_volume_30d}",
        f"52-week high/low: {md.week52_high} / {md.week52_low}",
    ]
    if md.errors:
        lines.append("Partial data issues: " + "; ".join(md.errors))
    return "\n".join(lines)


def build_news_context(bundle: ResearchDataBundle) -> str:
    if not bundle.news:
        return f"No recent news articles retrieved for {bundle.company_name}. Errors: {'; '.join(bundle.errors.get('newsapi', []))}"
    lines = [f"{len(bundle.news)} recent articles (deduplicated, most recent first):\n"]
    for a in bundle.news:
        lines.append(f"- [{a.published_at}] {a.title} ({a.source}) — {a.url}")
    if bundle.latest_10k and bundle.latest_10k.items:
        lines.append(f"\nMost recent 8-K items: {bundle.latest_10k.items}")
    return "\n".join(lines)


def build_comparative_context(bundle: ResearchDataBundle) -> str:
    if not bundle.peers:
        return (
            f"Peer comparison unavailable for {bundle.ticker} "
            f"({'; '.join(bundle.errors.get('finnhub', [])) or 'no peer data source available'}). "
            "Do not invent peer companies or metrics."
        )
    lines = [f"Target: {bundle.company_name} ({bundle.ticker})", f"Target ratios: {bundle.financial_ratios}\n", "Peers:"]
    for p, m in bundle.peer_metrics.items():
        lines.append(f"- {p}: {m}")
    return "\n".join(lines)


def quick_context_for_agent(agent_key: str, query: str,
                             observe: Optional[Callable[..., None]] = None) -> tuple[str, ResearchDataBundle]:
    """Lazy, intent-specific retrieval for single-agent chat mode (requirement
    #19): fetch only what that one specialist actually needs, not the full
    5-source bundle. Returns (context_text, partial_bundle)."""

    def _log(agent: str, event: str, latency_ms: float = 0.0, error: Optional[str] = None):
        if observe:
            observe(agent, event, latency_ms, error=error)

    start = time.time()
    identity = company_resolver.resolve(query)
    _log("CompanyResolver", "resolve", (time.time() - start) * 1000)

    bundle = ResearchDataBundle(
        company_name=identity.company_name or query,
        ticker=identity.ticker or (query.upper() if len(query) <= 5 else ""),
        cik=identity.cik,
    )

    if identity.resolution_method == "unresolved":
        bundle.add_error("sec", f"Could not resolve '{query}' to a known SEC-registered company/ticker.")
        return (
            f"No company/ticker could be confidently identified from '{query}'. "
            "Tell the user you need a valid company name or ticker symbol; do not guess.",
            bundle,
        )

    try:
        if agent_key == "sec_filing":
            t0 = time.time()
            result = sec_client.get_company_sec_bundle(identity.ticker, cik=identity.cik, fetch_8k=2)
            bundle.latest_10k, bundle.latest_10q, bundle.recent_8ks = (
                result["latest_10k"], result["latest_10q"], result["recent_8ks"]
            )
            for e in result["errors"]:
                bundle.add_error("sec", e)
            _log("SECClient", "sec_bundle_fetch", (time.time() - t0) * 1000)
            return build_sec_filing_context(bundle), bundle

        if agent_key == "financial_statement":
            t0 = time.time()
            facts = sec_client.get_company_facts(identity.cik) if identity.cik else None
            if facts:
                bundle.sec_company_facts_available = True
                bundle.normalized_financials = normalize_xbrl_facts(facts)
                bundle.financial_ratios = calculate_ratios(bundle.normalized_financials)
            else:
                bundle.add_error("sec", "XBRL Company Facts unavailable.")
            _log("SECClient", "company_facts_fetch", (time.time() - t0) * 1000)
            return build_financial_statement_context(bundle), bundle

        if agent_key == "market_data":
            t0 = time.time()
            bundle.market_data = market_client.get_snapshot(identity.ticker or query)
            for e in (bundle.market_data.errors or []):
                bundle.add_error("yfinance", e)
            _log("MarketDataClient", "market_data_fetch", (time.time() - t0) * 1000)
            return build_market_data_context(bundle), bundle

        if agent_key == "news_intelligence":
            t0 = time.time()
            articles, errors = news_client.get_company_news(identity.company_name, identity.ticker)
            bundle.news = articles
            for e in errors:
                bundle.add_error("newsapi", e)
            _log("NewsClient", "news_fetch", (time.time() - t0) * 1000)
            return build_news_context(bundle), bundle

        if agent_key == "comparative_analysis":
            t0 = time.time()
            facts = sec_client.get_company_facts(identity.cik) if identity.cik else None
            if facts:
                bundle.normalized_financials = normalize_xbrl_facts(facts)
                bundle.financial_ratios = calculate_ratios(bundle.normalized_financials)
            peers = finnhub_client.company_peers(identity.ticker) if identity.ticker else []
            bundle.peers = peers[:5]
            for p in bundle.peers:
                try:
                    q = finnhub_client.quote(p)
                    profile = finnhub_client.company_profile(p)
                    bundle.peer_metrics[p] = {
                        "price": (q or {}).get("c"),
                        "market_cap": (profile or {}).get("marketCapitalization"),
                        "industry": (profile or {}).get("finnhubIndustry"),
                    }
                except Exception as e:
                    bundle.add_error("finnhub", f"Peer {p} metrics failed: {e}")
            if not bundle.peers:
                bundle.add_error("finnhub", "Peer comparison unavailable.")
            _log("FinnhubClient", "peer_fetch", (time.time() - t0) * 1000)
            return build_comparative_context(bundle), bundle

        if agent_key == "investment_decision":
            t0 = time.time()
            facts = sec_client.get_company_facts(identity.cik) if identity.cik else None
            if facts:
                bundle.normalized_financials = normalize_xbrl_facts(facts)
                bundle.financial_ratios = calculate_ratios(bundle.normalized_financials)
            bundle.market_data = market_client.get_snapshot(identity.ticker or query)
            articles, errors = news_client.get_company_news(identity.company_name, identity.ticker)
            bundle.news = articles[:10]
            for e in errors:
                bundle.add_error("newsapi", e)
            _log("ResearchEngine", "lightweight_synthesis_fetch", (time.time() - t0) * 1000)
            parts = [
                build_financial_statement_context(bundle),
                build_market_data_context(bundle),
                build_news_context(bundle),
            ]
            return "\n\n".join(parts), bundle
    except Exception as e:
        bundle.add_error("sec", str(e))
        _log(agent_key, "quick_context_error", 0.0, error=str(e))
        return f"(Data retrieval failed: {e}. State that data is unavailable rather than fabricating it.)", bundle

    return "", bundle


def build_sources_summary(bundle: ResearchDataBundle) -> str:
    lines = [f"Data fetched at: {bundle.fetched_at}", "Sources used:"]
    for s in bundle.sources:
        lines.append(f"- {s.source}: {s.description}" + (f" ({s.url})" if s.url else ""))
    all_errors = [f"{src}: {e}" for src, errs in bundle.errors.items() for e in errs]
    if all_errors:
        lines.append("\nUnavailable / failed sources:")
        lines.extend(f"- {e}" for e in all_errors)
    return "\n".join(lines)
