"""
sec_client.py — the ONLY place in this application that is allowed to make
HTTP requests to data.sec.gov / www.sec.gov / SEC archives.

Every request goes through `SECClient._request()`, which:
  1. Acquires a token from the shared, process-wide SEC_RATE_LIMITER
     (max 5 requests/second across the whole application, all threads).
  2. Uses a shared `requests.Session` with a compliant User-Agent built
     from SEC_USER_AGENT / SEC_CONTACT_EMAIL env vars.
  3. Retries transient failures (429/500/502/503/504) with exponential
     backoff + jitter, honoring `Retry-After` when SEC provides it.
  4. Never retries permanent 4xx errors (e.g. 404) blindly.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .cache import (
    TTL_10K, TTL_10Q, TTL_8K, TTL_COMPANY_FACTS, TTL_TICKER_MAP,
    cache_key, get_cached_data, set_cached_data,
)
from .models import SECFilingMetadata, SourceReference
from .rate_limiter import SEC_RATE_LIMITER

logger = logging.getLogger("finbot.sec")

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
REQUEST_TIMEOUT = 15  # seconds

# ~ how much filing text we'll hand to an LLM per section, in characters
MAX_SECTION_CHARS = 12_000
MAX_TOTAL_FILING_CHARS = 45_000

TEN_K_SECTIONS = {
    "item_1_business": r"item\s*1\.?\s*business",
    "item_1a_risk_factors": r"item\s*1a\.?\s*risk\s*factors",
    "item_3_legal_proceedings": r"item\s*3\.?\s*legal\s*proceedings",
    "item_7_mdna": r"item\s*7\.?\s*management.?s\s*discussion",
    "item_7a_market_risk": r"item\s*7a\.?\s*quantitative\s*and\s*qualitative",
    "item_8_financial_statements": r"item\s*8\.?\s*financial\s*statements",
}
TEN_Q_SECTIONS = {
    "financial_statements": r"financial\s*statements",
    "mdna": r"management.?s\s*discussion\s*and\s*analysis",
    "risk_factors": r"risk\s*factors",
    "controls_and_procedures": r"controls\s*and\s*procedures",
}


class SECClient:
    """Thin, rate-limited, cached wrapper around SEC EDGAR's public JSON + archive endpoints."""

    def __init__(self):
        user_agent_app = os.getenv("SEC_USER_AGENT", "FinBot-Research-App")
        contact_email = os.getenv("SEC_CONTACT_EMAIL", "")
        if not contact_email:
            logger.warning(
                "SEC_CONTACT_EMAIL is not set. SEC EDGAR requires a real contact "
                "in the User-Agent header; requests may be rejected."
            )
        self.user_agent = f"{user_agent_app} ({contact_email})".strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    # ------------------------------------------------------------------
    # Centralized, rate-limited request method — nothing else should call
    # requests.get() for an SEC resource.
    # ------------------------------------------------------------------
    def _request(self, url: str, *, as_json: bool = True):
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            wait = SEC_RATE_LIMITER.acquire()
            if wait > 0:
                logger.info("sec_rate_limit_wait: %.3fs before %s", wait, url)
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last_exc = e
                logger.warning("sec_request failed (network) attempt=%d url=%s err=%s", attempt, url, e)
                self._backoff_sleep(attempt)
                continue

            if resp.status_code == 200:
                if attempt > 0:
                    logger.info("sec_retry succeeded after %d attempt(s): %s", attempt, url)
                return resp.json() if as_json else resp.text

            if resp.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self._backoff_delay(attempt)
                else:
                    delay = self._backoff_delay(attempt)
                logger.warning(
                    "sec_retry status=%d attempt=%d delay=%.2fs url=%s",
                    resp.status_code, attempt, delay, url,
                )
                time.sleep(delay)
                continue

            # Permanent failure (e.g. 404) or retries exhausted — don't retry blindly.
            logger.warning("sec_failure status=%d url=%s", resp.status_code, url)
            resp.raise_for_status()

        logger.error("sec_failure: exhausted retries for %s (%s)", url, last_exc)
        if last_exc:
            raise last_exc
        raise RuntimeError(f"SEC request failed after {MAX_RETRIES} retries: {url}")

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        base = min(1.5 ** attempt, 10)
        return base + random.uniform(0, 0.5)

    def _backoff_sleep(self, attempt: int) -> None:
        time.sleep(self._backoff_delay(attempt))

    # ------------------------------------------------------------------
    # Ticker <-> CIK resolution
    # ------------------------------------------------------------------
    def get_company_tickers(self) -> dict:
        key = cache_key("sec", "company_tickers")
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        data = self._request(f"{SEC_BASE}/files/company_tickers.json")
        set_cached_data(key, "SEC EDGAR", data, TTL_TICKER_MAP)
        return data

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper().strip()
        tickers = self.get_company_tickers()
        for entry in tickers.values():
            if entry.get("ticker", "").upper() == ticker:
                return str(entry["cik_str"]).zfill(10)
        return None

    def find_by_company_name(self, name: str, limit: int = 5) -> list[dict]:
        """Fuzzy-ish substring search over the SEC ticker/company title list."""
        name_l = name.lower().strip()
        tickers = self.get_company_tickers()
        matches = []
        for entry in tickers.values():
            title = entry.get("title", "")
            if name_l in title.lower():
                matches.append({
                    "company_name": title,
                    "ticker": entry.get("ticker"),
                    "cik": str(entry["cik_str"]).zfill(10),
                })
        return matches[:limit]

    # ------------------------------------------------------------------
    # Submissions / filing history
    # ------------------------------------------------------------------
    def get_submissions(self, cik: str) -> dict:
        cik = str(cik).zfill(10)
        key = cache_key("sec", "submissions", cik)
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        data = self._request(f"{SEC_DATA_BASE}/submissions/CIK{cik}.json")
        set_cached_data(key, "SEC EDGAR", data, TTL_10Q)
        return data

    def get_company_facts(self, cik: str) -> Optional[dict]:
        cik = str(cik).zfill(10)
        key = cache_key("sec", "company_facts", cik)
        cached = get_cached_data(key)
        if cached is not None:
            return cached
        try:
            data = self._request(f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        except requests.HTTPError:
            return None
        set_cached_data(key, "SEC EDGAR", data, TTL_COMPANY_FACTS)
        return data

    def _iter_recent_filings(self, submissions: dict):
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i in range(len(forms)):
            yield {
                "form": forms[i],
                "filingDate": recent.get("filingDate", [None] * len(forms))[i],
                "reportDate": recent.get("reportDate", [None] * len(forms))[i],
                "accessionNumber": recent.get("accessionNumber", [None] * len(forms))[i],
                "primaryDocument": recent.get("primaryDocument", [None] * len(forms))[i],
                "items": recent.get("items", [None] * len(forms))[i] if "items" in recent else None,
            }

    def get_recent_filings(self, cik: str, form_type: str, limit: int = 5) -> list[SECFilingMetadata]:
        submissions = self.get_submissions(cik)
        out = []
        for f in self._iter_recent_filings(submissions):
            if f["form"] == form_type:
                out.append(self._to_filing_metadata(cik, f))
            if len(out) >= limit:
                break
        return out

    def get_latest_filing(self, cik: str, form_type: str) -> Optional[SECFilingMetadata]:
        results = self.get_recent_filings(cik, form_type, limit=1)
        return results[0] if results else None

    def _to_filing_metadata(self, cik: str, f: dict) -> SECFilingMetadata:
        accession_nodash = f["accessionNumber"].replace("-", "")
        source_url = (
            f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accession_nodash}/{f['primaryDocument']}"
        )
        return SECFilingMetadata(
            form_type=f["form"],
            filing_date=f["filingDate"],
            accession_number=f["accessionNumber"],
            primary_document=f["primaryDocument"],
            cik=str(cik),
            report_date=f.get("reportDate"),
            items=f.get("items"),
            source_url=source_url,
        )

    # ------------------------------------------------------------------
    # Filing download + text extraction
    # ------------------------------------------------------------------
    def download_filing(self, filing: SECFilingMetadata) -> SECFilingMetadata:
        """Downloads the primary document, extracts prioritized sections, and
        fills in `filing.extracted_sections` (bounded in size)."""
        ttl = {"10-K": TTL_10K, "10-Q": TTL_10Q, "8-K": TTL_8K}.get(filing.form_type, TTL_10Q)
        key = cache_key("sec", "filing_text", filing.accession_number, filing.primary_document)
        cached = get_cached_data(key)
        if cached is not None:
            filing.extracted_sections = cached.get("sections", {})
            filing.extraction_note = cached.get("note", "")
            filing.content_truncated = cached.get("truncated", False)
            return filing

        try:
            raw_html = self._request(filing.source_url, as_json=False)
        except requests.HTTPError as e:
            filing.extraction_note = f"Could not download filing document: {e}"
            return filing

        text = self._clean_html(raw_html)

        if filing.form_type == "10-K":
            sections = self._split_sections(text, TEN_K_SECTIONS)
        elif filing.form_type == "10-Q":
            sections = self._split_sections(text, TEN_Q_SECTIONS)
        else:  # 8-K and others — the whole (short) document is the "event" text
            sections = {"event_text": text[:MAX_SECTION_CHARS]}

        truncated = False
        total = 0
        bounded_sections = {}
        for name, content in sections.items():
            if not content:
                continue
            content = content[:MAX_SECTION_CHARS]
            if total + len(content) > MAX_TOTAL_FILING_CHARS:
                content = content[: max(0, MAX_TOTAL_FILING_CHARS - total)]
                truncated = True
            bounded_sections[name] = content
            total += len(content)
            if total >= MAX_TOTAL_FILING_CHARS:
                truncated = True
                break

        filing.extracted_sections = bounded_sections
        filing.content_truncated = truncated
        filing.extraction_note = (
            "Sections extracted via heuristic Item-header matching; content "
            "bounded to control prompt size."
            + (" NOTE: filing was large and has been truncated — not all "
               "sections/content are included." if truncated else "")
        )
        set_cached_data(key, "SEC EDGAR", {
            "sections": bounded_sections, "note": filing.extraction_note, "truncated": truncated,
        }, ttl)
        return filing

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, features="xml")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def _split_sections(text: str, section_patterns: dict[str, str]) -> dict[str, str]:
        """Heuristic section splitter: finds each 'Item N.' heading and grabs the
        text up to the next recognized heading. Not a full SEC-filing parser —
        real filings vary a lot in formatting — but it reliably narrows a
        multi-hundred-KB filing down to the sections we actually care about."""
        lower = text.lower()
        matches = []
        for name, pattern in section_patterns.items():
            for m in re.finditer(pattern, lower):
                matches.append((m.start(), name))
        matches.sort()
        if not matches:
            return {}

        # De-dupe: keep the LAST occurrence of each heading (filings often
        # repeat "Item 1A" once in the table of contents and once for real).
        last_occurrence: dict[str, int] = {}
        for pos, name in matches:
            last_occurrence[name] = pos
        ordered = sorted(last_occurrence.items(), key=lambda kv: kv[1])

        sections = {}
        for i, (name, pos) in enumerate(ordered):
            end = ordered[i + 1][1] if i + 1 < len(ordered) else min(len(text), pos + MAX_SECTION_CHARS * 2)
            sections[name] = text[pos:end].strip()
        return sections

    # ------------------------------------------------------------------
    # High-level bundle
    # ------------------------------------------------------------------
    def get_company_sec_bundle(self, ticker: str, cik: Optional[str] = None,
                                fetch_8k: int = 3, download_filings: bool = True) -> dict:
        """Returns {latest_10k, latest_10q, recent_8ks, company_facts, sources, errors}."""
        errors: list[str] = []
        sources: list[SourceReference] = []
        result = {"latest_10k": None, "latest_10q": None, "recent_8ks": [],
                   "company_facts": None, "sources": sources, "errors": errors}

        if not cik:
            try:
                cik = self.ticker_to_cik(ticker)
            except Exception as e:
                errors.append(f"Ticker/CIK lookup failed: {e}")
                return result
        if not cik:
            errors.append(f"No SEC CIK found for ticker '{ticker}'.")
            return result

        try:
            latest_10k = self.get_latest_filing(cik, "10-K")
            if latest_10k and download_filings:
                latest_10k = self.download_filing(latest_10k)
            result["latest_10k"] = latest_10k
            if latest_10k:
                sources.append(SourceReference("SEC EDGAR", "Latest 10-K", latest_10k.source_url))
        except Exception as e:
            errors.append(f"10-K retrieval failed: {e}")

        try:
            latest_10q = self.get_latest_filing(cik, "10-Q")
            if latest_10q and download_filings:
                latest_10q = self.download_filing(latest_10q)
            result["latest_10q"] = latest_10q
            if latest_10q:
                sources.append(SourceReference("SEC EDGAR", "Latest 10-Q", latest_10q.source_url))
        except Exception as e:
            errors.append(f"10-Q retrieval failed: {e}")

        try:
            recent_8ks = self.get_recent_filings(cik, "8-K", limit=fetch_8k)
            if download_filings:
                recent_8ks = [self.download_filing(f) for f in recent_8ks]
            result["recent_8ks"] = recent_8ks
            for f in recent_8ks:
                sources.append(SourceReference("SEC EDGAR", f"8-K ({f.filing_date})", f.source_url))
        except Exception as e:
            errors.append(f"8-K retrieval failed: {e}")

        try:
            facts = self.get_company_facts(cik)
            result["company_facts"] = facts
            if facts:
                sources.append(SourceReference(
                    "SEC EDGAR", "XBRL Company Facts",
                    f"{SEC_DATA_BASE}/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
                ))
            else:
                errors.append("XBRL Company Facts unavailable for this CIK.")
        except Exception as e:
            errors.append(f"Company Facts retrieval failed: {e}")

        result["cik"] = cik
        return result


# Single shared client instance (the Session + rate limiter usage is thread-safe).
sec_client = SECClient()
