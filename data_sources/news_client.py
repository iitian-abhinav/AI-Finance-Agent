"""
news_client.py — recent company news from NewsAPI + Finnhub, deduplicated
and capped so we don't dump hundreds of articles into an LLM prompt.

NewsAPI and Finnhub use different publication-date formats:
- NewsAPI commonly returns ISO-8601 strings.
- Finnhub commonly returns Unix timestamps.

All publication dates are normalized to Unix timestamps for sorting and
comparison so mixed provider data never causes str/int comparison errors.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional

import requests

from .cache import (
    TTL_NEWS,
    cache_key,
    get_cached_data,
    set_cached_data,
)
from .finnhub_client import finnhub_client
from .models import NewsArticle


logger = logging.getLogger("finbot.news")

NEWSAPI_URL = "https://newsapi.org/v2/everything"
TIMEOUT = 10
MAX_ARTICLES = 25


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """
    Normalize a title for fuzzy duplicate detection.
    """
    return re.sub(
        r"[^a-z0-9]+",
        "",
        (title or "").lower(),
    )


def _to_unix_timestamp(value) -> Optional[int]:
    """
    Normalize a publication date to a Unix timestamp in seconds.

    Supports:
    - Unix timestamps as int or float
    - Unix timestamps stored as strings
    - ISO-8601 strings such as:
        2026-07-15T12:30:00Z
        2026-07-15T12:30:00+00:00

    Returns None when the value cannot be parsed.
    """
    if value is None:
        return None

    # Already numeric.
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        # Numeric timestamp stored as text.
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            pass

        # ISO-8601 timestamp.
        try:
            parsed = datetime.fromisoformat(
                value.replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return int(
                parsed.timestamp()
            )

        except (TypeError, ValueError, OverflowError):
            return None

    return None


def _publication_sort_key(
    article: NewsArticle,
) -> int:
    """
    Return a consistent integer sort key for a NewsArticle.

    Articles with unavailable or invalid dates sort last.
    """
    timestamp = _to_unix_timestamp(
        article.published_at
    )

    return (
        timestamp
        if timestamp is not None
        else 0
    )


# ---------------------------------------------------------------------------
# News client
# ---------------------------------------------------------------------------

class NewsClient:

    def __init__(self):
        self.api_key = os.getenv(
            "NEWS_API_KEY",
            "",
        )

        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(
            self.api_key
        )

    # -----------------------------------------------------------------------
    # NewsAPI
    # -----------------------------------------------------------------------

    def _newsapi_search(
        self,
        query: str,
        days: int = 21,
    ) -> list[NewsArticle]:
        """
        Retrieve recent English-language articles from NewsAPI.

        Returns an empty list when NewsAPI is disabled or the request fails.
        """
        if not self.enabled:
            return []

        key = cache_key(
            "newsapi",
            "everything",
            query,
            str(days),
        )

        cached = get_cached_data(
            key
        )

        if cached is not None:
            try:
                return [
                    NewsArticle(**article)
                    for article in cached
                ]

            except Exception as exc:
                logger.warning(
                    "Failed to restore cached NewsAPI articles "
                    "query=%s err=%s",
                    query,
                    exc,
                )

        from_date = (
            datetime.now(timezone.utc)
            - timedelta(days=days)
        ).strftime(
            "%Y-%m-%d"
        )

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.api_key,
        }

        try:
            response = self.session.get(
                NEWSAPI_URL,
                params=params,
                timeout=TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as exc:
            logger.warning(
                "newsapi_request failed query=%s err=%s",
                query,
                exc,
            )

            return []

        except ValueError as exc:
            logger.warning(
                "newsapi_invalid_json query=%s err=%s",
                query,
                exc,
            )

            return []

        articles: list[NewsArticle] = []

        for article in data.get(
            "articles",
            [],
        ):
            title = article.get(
                "title",
                "",
            )

            if not title:
                continue

            articles.append(
                NewsArticle(
                    title=title,
                    source=(
                        article.get("source")
                        or {}
                    ).get(
                        "name",
                        "NewsAPI",
                    ),
                    published_at=article.get(
                        "publishedAt"
                    ),
                    url=article.get(
                        "url"
                    ),
                    description=article.get(
                        "description"
                    ),
                    provider="newsapi",
                )
            )

        set_cached_data(
            key,
            "NewsAPI",
            [
                article.to_dict()
                for article in articles
            ],
            TTL_NEWS,
        )

        return articles

    # -----------------------------------------------------------------------
    # Combined company news
    # -----------------------------------------------------------------------

    def get_company_news(
        self,
        company_name: str,
        ticker: str,
    ) -> tuple[list[NewsArticle], list[str]]:
        """
        Return:
            (
                deduplicated_articles,
                error_messages,
            )

        Articles are:
        1. Retrieved independently from NewsAPI and Finnhub.
        2. Combined.
        3. Deduplicated by URL and highly similar titles.
        4. Sorted newest-first using normalized timestamps.
        5. Capped at MAX_ARTICLES.
        """
        errors: list[str] = []
        combined: list[NewsArticle] = []

        # -------------------------------------------------------------------
        # NewsAPI
        # -------------------------------------------------------------------

        try:
            combined.extend(
                self._newsapi_search(
                    company_name or ticker
                )
            )

        except Exception as exc:
            logger.warning(
                "NewsAPI processing failed "
                "company=%s ticker=%s err=%s",
                company_name,
                ticker,
                exc,
            )

            errors.append(
                f"NewsAPI failed: {exc}"
            )

        # -------------------------------------------------------------------
        # Finnhub
        # -------------------------------------------------------------------

        try:
            if ticker:
                now = datetime.now(
                    timezone.utc
                )

                today = now.strftime(
                    "%Y-%m-%d"
                )

                from_date = (
                    now
                    - timedelta(days=21)
                ).strftime(
                    "%Y-%m-%d"
                )

                finnhub_articles = (
                    finnhub_client.company_news(
                        ticker,
                        from_date,
                        today,
                    )
                )

                if finnhub_articles:
                    combined.extend(
                        finnhub_articles
                    )

        except Exception as exc:
            logger.warning(
                "Finnhub news processing failed "
                "ticker=%s err=%s",
                ticker,
                exc,
            )

            errors.append(
                f"Finnhub news failed: {exc}"
            )

        # -------------------------------------------------------------------
        # Deduplicate
        # -------------------------------------------------------------------

        deduped = self._deduplicate(
            combined
        )

        # -------------------------------------------------------------------
        # Sort newest first using one consistent timestamp type.
        #
        # This fixes:
        # TypeError:
        # '<' not supported between instances of 'str' and 'int'
        # -------------------------------------------------------------------

        deduped.sort(
            key=_publication_sort_key,
            reverse=True,
        )

        return (
            deduped[:MAX_ARTICLES],
            errors,
        )

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        articles: list[NewsArticle],
    ) -> list[NewsArticle]:
        """
        Remove duplicate articles using:
        1. Canonicalized URL equality.
        2. Highly similar normalized titles.

        Articles without URLs may still be retained if their titles are unique.
        """
        seen_urls: set[str] = set()
        seen_titles: list[str] = []

        result: list[NewsArticle] = []

        for article in articles:
            if not article or not article.title:
                continue

            url_key = (
                (article.url or "")
                .split("?")[0]
                .rstrip("/")
            )

            title_key = _normalize_title(
                article.title
            )

            if not title_key:
                continue

            # Exact canonical URL duplicate.
            if (
                url_key
                and url_key in seen_urls
            ):
                continue

            # Near-duplicate headline.
            is_duplicate_title = any(
                SequenceMatcher(
                    None,
                    title_key,
                    seen_title,
                ).ratio() > 0.9
                for seen_title in seen_titles
            )

            if is_duplicate_title:
                continue

            if url_key:
                seen_urls.add(
                    url_key
                )

            seen_titles.append(
                title_key
            )

            result.append(
                article
            )

        return result


news_client = NewsClient()
