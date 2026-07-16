"""
FinBot Backend
==============
Multi-agent financial research backend built on `agno` + Groq, grounded in
REAL retrieved data (SEC EDGAR, Finnhub, NewsAPI, yfinance) rather than an
LLM's internal knowledge.

Architecture (see research_engine.py + data_sources/ for the data layer):
  - data_sources/sec_client.py     : sole gateway for all SEC HTTP traffic,
                                      rate-limited to <=5 req/s process-wide
  - data_sources/finnhub_client.py : quotes, profile, news, peers
  - data_sources/news_client.py    : NewsAPI + Finnhub news, deduplicated
  - data_sources/market_client.py  : yfinance history + Python-computed
                                      technical indicators
  - data_sources/analytics.py      : XBRL normalization + ratio calculations
                                      (all deterministic, no LLM involved)
  - research_engine.py             : orchestrates the above into a
                                      ResearchDataBundle and builds compact,
                                      source-grounded context for each agent

This file keeps the same public functions the Streamlit frontend already
depends on: create_conversation, list_conversations, save_message,
get_messages, stream_chat, regenerate_last, run_full_research, etc.
Persistence (conversations/messages/observability/cache tables) lives in
db_core.py so data_sources/* can share it without a circular import.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Callable, Iterator, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.message import Message

import db_core
from data_sources import company_resolver, market_client, sec_client
import research_engine as re_engine

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is required. Add it to a .env file.")

TEXT_MODEL_ID = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL_ID = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Optional integrations — the app degrades gracefully without them.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finbot")

if not FINNHUB_API_KEY:
    logger.info("FINNHUB_API_KEY not set — Finnhub quotes/news/peers will be skipped (yfinance/NewsAPI fallback).")
if not NEWS_API_KEY:
    logger.info("NEWS_API_KEY not set — NewsAPI will be skipped (Finnhub company news fallback only).")

# --------------------------------------------------------------------------
# Persistence — thin wrappers around db_core so the public interface used by
# the Streamlit frontend (create_conversation, get_messages, ...) is unchanged.
# --------------------------------------------------------------------------

get_conn = db_core.get_conn
new_id = db_core.new_id


def create_conversation(title: str = "New chat", mode: str = "chat") -> str:
    cid = new_id()
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) VALUES (?,?,?,?,?)",
            (cid, title, mode, now, now),
        )
    return cid


def list_conversations() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()


def rename_conversation(cid: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                      (title, datetime.utcnow().isoformat(), cid))


def touch_conversation(cid: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?",
                      (datetime.utcnow().isoformat(), cid))


def delete_conversation(cid: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))


def save_message(conversation_id: str, role: str, content: str,
                  agent_used: Optional[str] = None, image_path: Optional[str] = None,
                  is_regeneration: bool = False) -> str:
    mid = new_id()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, agent_used, image_path, is_regeneration, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (mid, conversation_id, role, content, agent_used, image_path,
             int(is_regeneration), datetime.utcnow().isoformat()),
        )
    touch_conversation(conversation_id)
    return mid


def get_messages(conversation_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC", (conversation_id,)
        ).fetchall()


def delete_last_assistant_message(conversation_id: str) -> None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM messages WHERE conversation_id=? AND role='assistant' ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM messages WHERE id=?", (row["id"],))


# Never log secrets — only structural info (agent, event, latency, token counts).
def log_observability(conversation_id: Optional[str], agent_name: str, event: str,
                       latency_ms: float = 0.0, input_tokens: int = 0, output_tokens: int = 0,
                       error: Optional[str] = None, metadata: Optional[dict] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO observability
               (id, conversation_id, agent_name, event, latency_ms, input_tokens, output_tokens, error, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (new_id(), conversation_id, agent_name, event, latency_ms, input_tokens, output_tokens,
             error, json.dumps(metadata) if metadata else None, datetime.utcnow().isoformat()),
        )
    logger.info("event=%s agent=%s latency_ms=%.0f in_tok=%s out_tok=%s error=%s",
                event, agent_name, latency_ms, input_tokens, output_tokens, error)


def get_observability(conversation_id: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if conversation_id:
            return conn.execute(
                "SELECT * FROM observability WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return conn.execute("SELECT * FROM observability ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()


def _observe(conversation_id: Optional[str] = None):
    """Returns a callback matching research_engine's observe(agent, event, latency_ms, error=...) shape."""
    def _cb(agent_name: str, event: str, latency_ms: float = 0.0, error: Optional[str] = None, **meta):
        log_observability(conversation_id, agent_name, event, latency_ms, error=error, metadata=meta or None)
    return _cb


# --------------------------------------------------------------------------
# Agents — prompts now instruct strict evidence-grounding. Agents analyze and
# interpret retrieved data; they never compute figures or invent facts.
# --------------------------------------------------------------------------

def _groq(vision: bool = False) -> Groq:
    return Groq(id=VISION_MODEL_ID if vision else TEXT_MODEL_ID, api_key=GROQ_API_KEY)


_GROUNDING_RULE = dedent("""
    You will be given retrieved evidence (SEC filing excerpts, XBRL-derived
    financials, Python-calculated ratios/technical indicators, or news
    articles) directly in the prompt. You must:
    - Base your analysis ONLY on the supplied evidence, not on general
      knowledge of the company.
    - Explicitly say so if a figure, filing, or article was not supplied or
      retrieval failed — never invent a number, filing, headline, or quote.
    - Distinguish reported facts from your own interpretation.
    - Cite source URLs / filing dates / article dates when given, so the
      user can verify the underlying evidence.
""")


def build_agents() -> dict[str, Agent]:
    sec_filing_agent = Agent(
        model=_groq(), name="SEC Filing Agent",
        description="You are a specialized SEC Filing analyst expert (10-K, 10-Q, 8-K, XBRL).",
        instructions=_GROUNDING_RULE + dedent("""
    You are an SEC filing analyst. Analyze ONLY the retrieved SEC evidence
    supplied in the prompt.

    EVIDENCE DISCIPLINE:
    - Never use pretrained knowledge to fill gaps in the retrieved filings.
    - Every material factual claim must be attributable to a specific filing:
      form type, filing date, and report period.
    - Preserve SEC source URLs exactly as supplied in the evidence.
    - Never combine facts from different filings without identifying each source.
    - Never describe an 8-K generically when its specific event or Item number
      is available.
    - If an 8-K's extracted content is insufficient to identify the material
      event, explicitly say that the event could not be determined from the
      retrieved evidence.
    - Do not infer the contents of unavailable or truncated sections.

    FACT VS INTERPRETATION:
    For important findings, clearly separate:
    **Filing fact:** What the retrieved SEC evidence explicitly states.
    **Analyst interpretation:** What the fact may imply for the business or
    investors.

    The interpretation must be clearly labeled and must not introduce new facts.

    REQUIRED COVERAGE:
    1. Latest 10-K:
       - filing date and report period
       - business developments
       - risk factors
       - management discussion, if retrieved
       - legal/regulatory matters, if retrieved

    2. Latest 10-Q:
       - filing date and report period
       - material developments since the annual filing
       - management discussion, if retrieved
       - risk-factor changes, if retrieved
       - market-risk or controls disclosures, if retrieved

    3. Recent 8-Ks:
       For each filing separately state:
       - filing date
       - report/event date
       - SEC URL
       - Item number(s), if supplied
       - exact material event supported by the retrieved evidence

    RETRIEVAL COMPLETENESS:
    End with a section titled "Evidence Availability and Limitations".
    Explicitly list:
    - filings successfully retrieved
    - sections unavailable
    - sections truncated by source extraction
    - evidence truncated only because of the LLM prompt budget
    - any retrieval errors

    Do not say a filing or section was fully reviewed if the evidence indicates
    that it was truncated.

    OUTPUT STRUCTURE:

    # SEC Filing Analysis — [Company] ([Ticker])

    ## Executive Takeaways

    ## Latest 10-K
    ### Filing Metadata
    ### Business Developments
    ### Risk Factors
    ### Management Discussion
    ### Legal and Regulatory Matters

    ## Latest 10-Q
    ### Filing Metadata
    ### Developments Since the 10-K
    ### Management Discussion
    ### Risk and Market-Risk Updates

    ## Recent 8-K Filings
    ### [Filing date]
    Repeat separately for each retrieved 8-K.

    ## Cross-Filing Developments
    Identify only changes that can actually be supported by comparing the
    retrieved filings.

    ## Evidence Availability and Limitations

    Keep filing facts and analyst interpretation explicitly separated.
"""),
        markdown=True, add_datetime_to_context=True,
    )

    financial_statement_agent = Agent(
        model=_groq(), name="Financial Statement Agent",
        description="You are a specialized financial statement and valuation analyst.",
        instructions=_GROUNDING_RULE + dedent("""
        Analyze ONLY the SEC XBRL financial data, deterministic Python-calculated
        ratios, and period metadata supplied in the retrieved evidence.

        Never use pretrained knowledge to fill gaps or introduce company-specific
        facts that are not present in the supplied financial evidence.

        DATA RULES:
        - Never independently calculate or recompute a missing financial metric.
        - Use only figures and ratios explicitly supplied by the system.
        - Treat N/A, None, missing, and unavailable values as data gaps.
        - Never treat missing data as zero.
        - Do not annualize quarterly or year-to-date figures unless the
          deterministic analytics explicitly supply an annualized value.

        PERIOD METADATA IS MANDATORY:
        - Use the supplied period metadata in the answer.
        - Never say only "the most recent period" when exact dates, period labels,
          or period types are available.
        - For every growth metric, explicitly state the current comparison period
          and prior comparison period supplied by the deterministic analytics.
        - State whether each growth comparison is annual YoY, quarterly YoY,
          six-month YTD YoY, or nine-month YTD YoY.
        - For operating cash flow, capital expenditure, and free cash flow,
          explicitly state the shared reporting period when supplied.
        - Distinguish annual, standalone quarterly, six-month YTD, and
          nine-month YTD values.
        - Never compare a full fiscal year with a quarter or YTD period.
        - Never compare a standalone quarter with a cumulative YTD period.
        - If comparison periods are missing, ambiguous, or non-comparable,
          state that the metric cannot be reliably interpreted.

        STRICT EVIDENCE BOUNDARY:
        - Do not introduce company-specific products, geographic exposures,
          geopolitical risks, competitors, industry conditions, macroeconomic
          factors, or other facts unless they are explicitly present in the
          supplied financial evidence.
        - Never introduce an unsupported claim and then qualify it by saying
          that it is "not directly reflected in the supplied data."
        - If a weakness is not supported by the supplied evidence, do not invent
          one.
        - Separate financial weaknesses from data limitations.

        PROFITABILITY:
        - Analyze only the supplied gross margin, operating margin, net margin,
          ROA, ROE, and other profitability metrics.
        - Faster net income growth than revenue growth may be described as
          consistent with improved earnings conversion for the compared periods,
          but do not claim that one comparison establishes a durable trend.
        - Do not automatically interpret high ROE as purely stronger operating
          performance. High ROE may also be affected by the equity base.
        - If the supplied methodology notes identify a limitation, state it.

        LIQUIDITY:
        - Analyze only supplied liquidity metrics.
        - A current ratio only slightly above 1 may indicate relatively limited
          short-term liquidity headroom.
        - Do not automatically describe a current ratio near 1 as financial
          distress.

        LEVERAGE:
        - Analyze only supplied leverage metrics.
        - Debt-to-equity alone is insufficient to conclude that leverage is
          definitively manageable, excessive, or conservative.
        - Explain what the supplied ratio indicates while acknowledging the
          limitation of interpreting it alone.

        CASH FLOW:
        - State the exact reporting period for operating cash flow, capital
          expenditure, and free cash flow when metadata is available.
        - Discuss free cash flow quality only when operating cash flow and
          capital expenditure cover compatible reporting periods.
        - When comparing CapEx with operating cash flow, confirm that both
          figures cover the same reporting interval.
        - Do not describe a YTD cash-flow figure as a standalone quarterly figure.

        REQUIRED OUTPUT STRUCTURE:

        # Financial Statement Analysis — [Company] ([Ticker])

        ## Reporting Periods and Data Coverage
        Explicitly state:
        - Revenue period
        - Revenue growth comparison periods
        - Net income growth comparison periods
        - Balance-sheet date
        - Cash-flow period

        ## Revenue and Earnings Trends
        Analyze only valid comparable-period growth.

        ## Profitability

        ## Liquidity

        ## Leverage

        ## Operating Cash Flow and Capital Expenditure

        ## Free Cash Flow Quality

        ## Evidence-Based Financial Strengths

        ## Evidence-Based Financial Weaknesses

        ## Data Limitations
        Explicitly identify unavailable metrics, N/A values, missing metadata,
        and methodological limitations supplied by the deterministic analytics.

        Keep financial facts separate from interpretation.
        Do not present conclusions with greater certainty than the supplied
        evidence supports.
    """),
        markdown=True, add_datetime_to_context=True,
    )

    market_data_agent = Agent(
        model=_groq(), name="Market Data Agent",
        description="You are a specialized market data and technical analysis expert.",
        instructions=_GROUNDING_RULE + dedent("""
            MARKET-DATA INTERPRETATION RULES:

- Treat the supplied "as of" timestamp as the data retrieval timestamp unless
  the evidence explicitly identifies it as the exchange trade timestamp.
  Do not imply that the latest quote was executed at the retrieval timestamp.

- Describe RSI objectively. RSI below 70 means it is below the conventional
  overbought threshold; do not say this means the stock "has room to grow."

- Describe MACD above its signal line as positive or bullish momentum at the
  observation time. Do not claim that it predicts continuation of the uptrend.

- Do not classify historical volatility as "low", "moderate", "high", or
  "extreme" unless the supplied evidence includes a benchmark or comparison.

- Do not classify maximum drawdown as "manageable" or "severe" without a
  supplied benchmark.

- Do not characterize average trading volume as "high", "low", "significant",
  or as confirming price movements unless the supplied evidence provides a
  comparison benchmark such as historical average volume.

- When the latest price is near the 52-week high, state the exact position
  using the supplied values. Do not treat proximity to the high by itself as
  proof that the trend will continue.

- Technical outlooks must be conditional assessments of the current indicator
  configuration, not predictions of future price movement.

- Avoid phrases such as:
  "has room to grow"
  "supports continuation of the uptrend"
  "has the potential to continue its growth trend"
  unless a probabilistic forecasting model explicitly supplied that conclusion.

- Use clean Markdown formatting for the 52-week high and low. Never emit broken
  formatting artifacts or raw markup characters.

REQUIRED OUTLOOK STYLE:

For the short-term outlook:
- Evaluate latest price versus SMA20 and SMA50.
- Evaluate RSI.
- Evaluate MACD versus its signal line.
- Identify any conflicting signals.

For the medium-term outlook:
- Evaluate latest price versus SMA200.
- Use 6-month and 1-year returns where available.
- Consider position within the 52-week range.
- Consider historical volatility and maximum drawdown as risk context.
- State what evidence would weaken the current technical configuration.

Clearly distinguish:
1. Observed data.
2. Indicator interpretation.
3. Conditional technical outlook.
        """),
        markdown=True, add_datetime_to_context=True,
    )

    news_intelligence_agent = Agent(
        model=_groq(), name="News Intelligence Agent",
        description="You are a specialized news intelligence and sentiment analysis expert.",
        instructions=_GROUNDING_RULE + dedent("""
            STRICT ARTICLE RELEVANCE:

- Analyze only articles whose supplied title, description, or retrieved content
  contains a clear and material connection to the target company.

- The mere presence of the company name, ticker, product name, or an ambiguous
  headline is not sufficient to establish that the article reports a material
  development about the company.

- Exclude articles that are primarily about another company, person, industry,
  technology trend, country, commodity, or macroeconomic development unless the
  supplied article evidence explicitly explains the connection to the target
  company.

- Do not invent indirect effects. For example, do not claim that a semiconductor
  policy, supplier development, geopolitical event, AI trend, or industry event
  affects the target company's supply chain, revenue, costs, competition, or
  valuation unless that connection is explicitly stated in the supplied
  evidence.

- If an article's relevance is ambiguous, omit it from the main thematic
  analysis or place it in a separate "Ambiguous or Low-Relevance Items" section.
  Do not use ambiguous items to determine the overall news-flow classification.

FACT VERSUS INTERPRETATION:

For each major theme, clearly separate:

CONFIRMED REPORTED DEVELOPMENTS:
- State only what is explicitly supported by the supplied article title,
  description, date, source, and other retrieved evidence.

POSSIBLE INVESTOR SIGNIFICANCE:
- Explain a cautious, conditional interpretation of why the confirmed
  development may matter.
- Label interpretation as interpretation.
- Do not present possible effects as established facts.

LEGAL AND REGULATORY DISCIPLINE:

- Do not state that the target company is involved in a lawsuit, investigation,
  legal fight, regulatory action, or dispute unless the supplied evidence
  explicitly identifies the company as a party or subject.

- An ambiguous headline mentioning the company alongside a legal dispute is not
  sufficient evidence that the company itself faces legal or reputational risk.

- If the parties to a legal matter cannot be determined from the supplied
  evidence, state that the article is too ambiguous to classify as a
  company-specific legal development.

MARKET PERFORMANCE VERSUS FINANCIAL PERFORMANCE:

- Stock-price movements, record highs, analyst commentary, and investment
  opinions are market developments, not evidence of operating or financial
  performance.

- Never describe a stock-price record as evidence of stronger revenue,
  earnings, profitability, cash flow, or other financial performance unless
  those financial results are explicitly supplied in the article evidence.

SOURCE AND ARTICLE QUALITY:

- Preserve the supplied article date, source, and URL when available.
- Do not omit URLs when the system supplied them.
- Distinguish factual corporate announcements and reported developments from
  opinion articles, analyst commentary, television commentary, and investment
  recommendations.
- Do not give an opinion article the same evidentiary weight as a confirmed
  corporate event.
- When multiple articles report the same underlying event, group them into one
  theme rather than treating them as separate independent developments.

SENTIMENT CLASSIFICATION:

- Classify each relevant development as positive, negative, mixed, or neutral
  based only on the supplied evidence.
- Do not classify an ambiguous or indirectly related article as positive or
  negative for the target company.
- Determine the overall news flow using only clearly relevant company-specific
  developments.
- Article count alone must not determine overall sentiment.
- Explain briefly why the aggregate news flow is positive, negative, neutral,
  or mixed.

REQUIRED OUTPUT STRUCTURE:

# Recent News Analysis — [Company] ([Ticker])

## Evidence Coverage
- Number of articles retrieved
- Number considered directly relevant
- Number excluded as ambiguous, duplicate, or low relevance
- Date range of relevant articles

## Major Theme 1 — [Theme Name]

### Confirmed Reported Developments
For each article:
- Date
- Source
- Headline or concise description
- URL, if supplied

### Possible Investor Significance
Provide cautious interpretation separately from the reported facts.

### Theme Classification
Positive / Negative / Mixed / Neutral

Repeat for each material theme.

## Ambiguous or Low-Relevance Items
Briefly identify important retrieved items that were excluded and explain why,
without speculating about their effect on the company.

## Overall News-Flow Assessment
Classify the relevant news flow as:
- Predominantly positive
- Predominantly negative
- Mixed
- Neutral

Explain the classification using only directly relevant supplied evidence.

Do not introduce any headline, event, causal relationship, legal exposure,
supply-chain effect, market impact, or company-specific claim that is not
supported by the retrieved evidence.
        """),
        markdown=True, add_datetime_to_context=True,
    )

    comparative_analysis_agent = Agent(
        model=_groq(), name="Comparative Analysis Agent",
        description="You are a specialized comparative and competitive analysis expert.",
        instructions=_GROUNDING_RULE + dedent("""
            STRICT COMPARATIVE EVIDENCE RULES:

- A company being returned by the peer-retrieval system does not by itself prove
  that it is a close operational competitor or that competition is intense.

- Compare the target company with a peer only when the same metric is available
  for both companies.

- Never compare different metrics across companies.

- Never infer that the target company is stronger or weaker on a metric when
  the corresponding peer metric is unavailable.

- If only peer identities, prices, market capitalizations, or industry labels
  are available, explicitly state that meaningful financial comparison is
  limited to those supplied fields.

- Do not introduce additional competitors, industries, market trends,
  regulatory risks, geographic opportunities, product opportunities, or
  strategic initiatives that are not present in the supplied evidence.

SWOT GROUNDING RULES:

- Every SWOT item must be traceable to a specific supplied company financial
  metric, peer metric, or other explicitly supplied comparative fact.

- Strengths and weaknesses may use the target company's supplied financial
  evidence, but clearly distinguish:
  1. an absolute company characteristic; and
  2. a demonstrated peer-relative advantage or disadvantage.

- Never describe a company metric as a peer-relative strength or weakness
  unless comparable peer data exists.

- Opportunities and threats must also be grounded in supplied evidence.

- Do not invent generic SWOT items such as:
  "growing demand for technology"
  "expansion into new markets"
  "intense competition"
  "regulatory changes"
  "changing consumer preferences"
  unless those developments are explicitly present in the supplied evidence.

- If the supplied evidence contains no supportable opportunities or threats,
  explicitly state:
  "No evidence-grounded opportunities were identified from the supplied data."
  or:
  "No evidence-grounded threats were identified from the supplied data."

INTERPRETATION DISCIPLINE:

- A debt-to-equity ratio alone does not establish that leverage is high,
  excessive, manageable, or conservative without a supplied benchmark.

- A current ratio slightly above 1 may indicate limited short-term liquidity
  headroom. Do not automatically characterize it as a liquidity problem or
  distress.

- An industry classification alone does not establish close business-model
  comparability.

- Do not characterize a retrieved peer set as evidence of intense competition.

REQUIRED OUTPUT STRUCTURE:

# Comparative Analysis — [Company] ([Ticker])

## Successfully Retrieved Peers
List exactly and only the peers successfully retrieved by the system.

## Available Comparison Coverage
For each metric, state:
- Whether the target-company value is available
- Which peers have the same metric available
- Whether a valid comparison can be made

## Valid Peer Comparisons
Compare only metrics available for both the target company and at least one
retrieved peer.

## Target-Company Financial Characteristics
Discuss supplied target-company metrics that cannot be compared with peers.
Clearly label these as absolute observations, not peer-relative conclusions.

## Where the Company Appears Stronger
Include only demonstrated peer-relative advantages.

If none:
"No demonstrated peer-relative strengths can be established from the supplied
data."

## Where the Company Appears Weaker
Include only demonstrated peer-relative disadvantages.

If none:
"No demonstrated peer-relative weaknesses can be established from the supplied
data."

## Evidence-Grounded SWOT

### Strengths
Only evidence-grounded items.

### Weaknesses
Only evidence-grounded items.

### Opportunities
Only evidence-grounded items. If none, explicitly state that none can be
identified from the supplied evidence.

### Threats
Only evidence-grounded items. If none, explicitly state that none can be
identified from the supplied evidence.

## Comparison Limitations
Identify missing peer metrics and comparisons that cannot be made.

Never manufacture a comparison to make the report appear more complete.
        """),
        markdown=True, add_datetime_to_context=True,
    )

    investment_decision_agent = Agent(
        model=_groq(), name="Investment Decision Agent",
        description="You are a specialized investment recommendation analyst.",
        instructions=_GROUNDING_RULE + dedent("""
            STRICT EVIDENCE BOUNDARY:

- Use only facts, metrics, news, SEC filing evidence, market data, technical
  indicators, peer data, and deterministic calculations explicitly supplied
  in the system evidence.

- Never use pretrained knowledge about the company, its products, historical
  business model, competition, industry, supply chain, regulation, geographic
  exposure, market share, or strategy unless that information is explicitly
  present in the supplied evidence.

- Never write phrases such as:
  "although not explicitly stated in the evidence"
  "historically"
  "typically"
  "generally"
  "like other companies in the industry"
  as a way to introduce unsupported company-specific claims.

- If a potential bull point, bear point, catalyst, or risk is not supported by
  the supplied evidence, omit it.

- Do not create generic investment risks or catalysts merely to fill every
  section.

PERIOD DISCIPLINE:

- Use the supplied period metadata.
- If a metric is six-month YTD, describe it as:
  "for the six months ended [date]"
  or:
  "FY2026 Q2 YTD (six months)"
  and never simply as a standalone "FY2026 Q2" result.

- Do not compare incompatible reporting periods.

FINANCIAL INTERPRETATION:

- High margins demonstrate the supplied level of profitability. They do not by
  themselves prove pricing power, competitive advantage, brand strength, or
  operational efficiency unless those conclusions are supported by additional
  supplied evidence.

- Debt-to-equity alone does not establish that leverage is manageable,
  conservative, excessive, or risky without a supplied benchmark.

- A current ratio slightly above 1 may indicate limited short-term liquidity
  headroom but does not by itself establish financial distress.

- Faster net income growth than revenue growth may be described as consistent
  with stronger earnings conversion over the compared periods, but not as proof
  of a durable trend.

BULL CASE:

- Include only favorable evidence actually supplied by the system.
- Distinguish current financial strength from future catalysts.
- Do not convert a current financial metric into an unsupported future-growth
  claim.

BEAR CASE:

- Include only adverse evidence, weaknesses, or negative developments actually
  supplied by the system.
- Do not invent product concentration, competition, regulation, macroeconomic
  risk, supply-chain risk, market saturation, or other generic risks.
- If the evidence does not support a substantial bear case, explicitly state
  that the supplied evidence contains limited evidence for a bear case.

CATALYSTS:

- A catalyst must be a specific supplied event or development that could
  plausibly change investor expectations.
- Do not list generic possibilities such as:
  "new products"
  "innovation"
  "expansion into new markets"
  "AI opportunities"
  unless a specific supplied development supports them.
- Current financial ratios are not automatically catalysts.
- Technical momentum is not a corporate catalyst; if relevant, label it
  separately as a market/technical factor.

RISKS:

- Include only risks explicitly supported by the supplied evidence.
- Clearly distinguish:
  1. company-specific risks;
  2. financial weaknesses;
  3. market/technical risks;
  4. uncertainty caused by missing data.
- Missing information is not itself evidence that a risk exists.

VALUATION AND PRICE TARGET:

- Provide a price target only when the supplied evidence contains sufficient
  valuation inputs for a clearly stated and defensible methodology.
- Do not invent P/E, EV/EBITDA, P/S, discount rates, terminal growth rates,
  forecast earnings, forecast cash flows, or peer valuation multiples.
- If sufficient valuation evidence is unavailable, explicitly state:
  "A reliable price target cannot be calculated from the supplied evidence."

RATING DISCIPLINE:

- BUY, HOLD, SELL, and AVOID are investment conclusions, not summaries of
  company quality.

- Strong financial performance alone is not sufficient for a BUY rating if the
  supplied evidence does not establish whether the current market price is
  attractive relative to fundamentals or a defensible valuation framework.

- Positive technical momentum alone is not sufficient for a BUY rating.

- A high-quality company can still be unattractively valued; therefore do not
  infer investment attractiveness solely from profitability, growth, or cash
  generation.

- When valuation evidence is materially insufficient, explicitly reduce
  confidence in the rating.

- Prefer HOLD over BUY or SELL when:
  * the company evidence is fundamentally positive,
  * but valuation evidence is insufficient to establish attractive upside or
    downside from the current price.

- Use AVOID only when the supplied evidence is too incomplete or unreliable to
  support a meaningful investment assessment, or when evidence quality is
  materially compromised.

- Use SELL only when supplied evidence supports a sufficiently negative
  investment case. Do not use SELL merely because valuation data is missing.

- State a confidence level:
  High / Medium / Low

- Explain exactly which missing evidence prevents a higher-confidence rating.

CONFLICTING SIGNALS:

- Explicitly identify conflicts between evidence categories.

Examples:
- Strong financial performance but limited valuation evidence.
- Positive technical momentum but price near the 52-week high.
- Positive news flow but weak peer-comparison coverage.
- Strong cash generation but limited liquidity headroom.

- Do not resolve conflicting evidence by simply selecting the most positive
  signals.

REQUIRED OUTPUT STRUCTURE:

# Investment Decision — [Company] ([Ticker])

## Evidence Used
Summarize only the evidence categories actually available:
- SEC filings
- Financial statements and deterministic ratios
- Market and technical data
- Recent news
- Peer comparison
- Valuation evidence

Clearly identify unavailable or materially incomplete categories.

## Bull Case
List only evidence-supported positive factors.

For each factor:
- Evidence
- Interpretation

## Bear Case
List only evidence-supported negative factors or weaknesses.

If limited:
"The supplied evidence provides only a limited evidence-grounded bear case."

## Major Catalysts
List only specific supplied developments that qualify as potential catalysts.

If none:
"No clearly identifiable evidence-grounded catalysts were supplied."

## Key Risks
List only evidence-supported risks.

Do not use missing information to invent a risk.

## Conflicting Signals
Explain material tensions across the supplied evidence.

## Uncertainty and Missing Data
Identify missing information that materially limits the investment decision,
especially:
- valuation inputs;
- comparable peer metrics;
- forecast data;
- incomplete filing, news, or market evidence.

## Price Target
Either:
- provide a defensible range with methodology based entirely on supplied
  inputs;
or:
- state that a reliable price target cannot be calculated from the supplied
  evidence.

## Final Rating
Rating: BUY / HOLD / SELL / AVOID
Confidence: High / Medium / Low

Explain:
1. why the rating follows from the supplied evidence;
2. what evidence argues against the rating;
3. what missing information limits confidence;
4. what additional evidence could materially change the conclusion.

Do not include generic investment disclaimers unless specifically requested.
Do not invent information to make the report appear complete.
FINAL GROUNDING RULES:

- If a statement would require wording such as:
  "though not directly mentioned in the evidence"
  "not explicitly stated in the evidence"
  "historically"
  "generally"
  "for companies like this"
  then omit the statement entirely.

- Never include an unsupported claim merely to acknowledge that it is
  unsupported.

- Key risks must be tied to specific supplied evidence. Generic statements such
  as "market volatility", "investor sentiment shifts", "competition",
  "regulatory risk", "supply-chain risk", or "product concentration" must be
  omitted unless supported by the supplied evidence.

- When supplied, use specific market-risk evidence such as historical
  volatility, maximum drawdown, proximity to the 52-week high, RSI, or other
  retrieved indicators rather than inventing generic market risks.

- Keep evidence categories separate:
  * Financial fundamentals: revenue, earnings, margins, liquidity, leverage,
    operating cash flow, free cash flow.
  * Market/technical evidence: price returns, moving averages, RSI, MACD,
    historical volatility, maximum drawdown, 52-week range.
  * News/catalysts: specific retrieved developments.
  * Peer evidence: only valid same-metric comparisons.

- Never describe operating cash flow or free cash flow as technical momentum.

- In the bear case and key risks, consider all supplied adverse or cautionary
  evidence, including limited liquidity headroom, drawdown, volatility, price
  position within the 52-week range, negative news, and filing risks, but only
  when those items are actually present in the supplied evidence.

- The absence of identifiable catalysts is an uncertainty or limitation on the
  bullish case; it is not by itself evidence of a negative corporate
  development.
        """),
        markdown=True, add_datetime_to_context=True,
    )

    supervisor = Agent(
        model=_groq(), name="Financial Research Supervisor",
        description="You are the Financial Research Supervisor, synthesizing specialist analyses into one report.",
        instructions=_GROUNDING_RULE + dedent("""
            Synthesize the specialist sections you're given into one
            coherent report. Preserve citations/source URLs the specialists
            mentioned. If specialists disagree or report conflicting
            figures, FLAG the conflict explicitly rather than silently
            picking one. State how fresh the underlying data is. End with a
            "Sources / Data Provenance" section listing exactly what was
            retrieved and what was unavailable.
        """),
        expected_output=dedent("""
            # Financial Research Report
            ## Executive Summary
            ## SEC Filing Analysis
            ## Financial Analysis
            ## Market Data & Technicals
            ## News Intelligence
            ## Competitive Analysis
            ## Investment Thesis (Bull / Bear / Risk)
            ## Recommendation (Rating, Price Target or N/A, Action Plan)
            ## Sources / Data Provenance
        """),
        markdown=True, add_datetime_to_context=True,
    )

    general_agent = Agent(
        model=_groq(), name="General Research Assistant",
        description="A helpful financial research assistant for day-to-day questions that don't need a data lookup.",
        instructions="Answer conversationally and concisely unless asked for depth.",
        markdown=True, add_datetime_to_context=True,
    )

    vision_agent = Agent(
        model=_groq(vision=True), name="Vision Analyst",
        description="Analyzes uploaded images (charts, screenshots, tables) using a Groq vision model.",
        instructions="Describe and analyze the financial content of the image the user attached.",
        markdown=True,
    )

    return {
        "sec_filing": sec_filing_agent,
        "financial_statement": financial_statement_agent,
        "market_data": market_data_agent,
        "news_intelligence": news_intelligence_agent,
        "comparative_analysis": comparative_analysis_agent,
        "investment_decision": investment_decision_agent,
        "supervisor": supervisor,
        "general": general_agent,
        "vision": vision_agent,
    }


AGENTS = build_agents()

ROUTER_KEYWORDS = {
    "sec_filing": ["10-k", "10-q", "8-k", "xbrl", "sec filing", "annual report", "quarterly report", "mda", "risk factors"],
    "financial_statement": ["ratio", "margin", "cash flow", "valuation", "balance sheet", "eps", "roe", "roic", "current ratio", "debt to equity", "debt-to-equity"],
    "market_data": ["price", "technical", "rsi", "macd", "moving average", "volume", "dividend", "analyst rating", "price target", "chart", "volatility"],
    "news_intelligence": ["news", "headline", "sentiment", "press release", "announcement"],
    "comparative_analysis": ["compare", " vs ", "versus", "swot", "competitor", "peer"],
    "investment_decision": ["should i buy", "should i sell", "recommend", "buy or sell", "investment thesis", "bull case", "bear case", "rating"],
}


def route_query(query: str) -> str:
    """Cheap keyword router so a single quick question doesn't trigger every data source."""
    q = f" {query.lower()} "
    scores = {k: sum(1 for kw in kws if kw in q) for k, kws in ROUTER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# --------------------------------------------------------------------------
# Ticker extraction + chart generation (real yfinance data via market_client)
# --------------------------------------------------------------------------

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
COMMON_WORDS = {"I", "A", "THE", "IS", "IT", "TO", "AND", "OR", "FOR", "OF", "ON", "IN", "RSI", "SEC", "EPS", "ROE", "CEO", "SWOT"}


def extract_ticker(text: str) -> Optional[str]:
    candidates = [t for t in TICKER_RE.findall(text) if t not in COMMON_WORDS]
    return candidates[0] if candidates else None


def extract_company_query(text: str) -> Optional[str]:
    """Best-effort: prefer an explicit ticker; otherwise let CompanyResolver try
    fuzzy name matching against the full message (cheap, SEC-side only)."""
    ticker = extract_ticker(text)
    if ticker:
        return ticker
    return text.strip() or None


def wants_chart(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["chart", "graph", "plot", "picture", "image", "show me the price", "visualize"])


def generate_price_chart(ticker: str, period: str = "6mo") -> Optional[str]:
    """Renders a PNG chart from real yfinance data (via market_client, shares
    the same historical-data cache used by the Market Data Agent)."""
    try:
        hist = market_client.get_history(ticker, period=period)
        if hist is None or hist.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hist.index, hist["Close"], color="#2563eb", linewidth=1.6)
        ax.fill_between(hist.index, hist["Close"], hist["Close"].min(), alpha=0.08, color="#2563eb")
        ax.set_title(f"{ticker.upper()} — Close Price ({period})")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        path = os.path.join(CHART_DIR, f"{ticker.upper()}_{uuid.uuid4().hex[:8]}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning("Chart generation failed for %s: %s", ticker, e)
        return None


# --------------------------------------------------------------------------
# History helpers
# --------------------------------------------------------------------------

def build_history_messages(conversation_id: str, max_turns: int = 12) -> list[Message]:
    rows = get_messages(conversation_id)[-max_turns * 2:]
    return [Message(role=r["role"], content=r["content"]) for r in rows if r["role"] in ("user", "assistant")]


# --------------------------------------------------------------------------
# Streaming chat — lazy, intent-specific real-data retrieval per message
# --------------------------------------------------------------------------

@dataclass
class StreamResult:
    text: str = ""
    agent_used: str = ""
    image_path: Optional[str] = None


def _event_name(event) -> str:
    """Normalize Agno event identifiers across versions (str or Enum)."""
    raw = getattr(event, "event", "")
    if hasattr(raw, "value"):
        raw = raw.value
    elif hasattr(raw, "name"):
        raw = raw.name
    text = str(raw or "")
    return text.rsplit(".", 1)[-1]


def _response_text(obj) -> str:
    """Best-effort text extraction from Agno RunResponse / stream events."""
    if obj is None:
        return ""
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return str(content)
    for attr in ("text", "response", "message"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
        nested = getattr(value, "content", None) if value is not None else None
        if isinstance(nested, str):
            return nested
    return ""


def _agent_key_from_stored_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    for key, agent in AGENTS.items():
        if key == name or agent.name == name:
            return key
    return None


def stream_chat(conversation_id: str, user_message: str, force_agent: Optional[str] = None,
                 image_bytes: Optional[bytes] = None, *, save_user_message: bool = True,
                 is_regeneration: bool = False, internal_instruction: Optional[str] = None) -> Iterator[str]:
    """Run one chat turn with lazy evidence retrieval and robust Agno streaming.

    Internal regeneration instructions are kept separate from persisted/user-visible
    text. Empty model responses are converted into a visible diagnostic message.
    """
    if save_user_message:
        save_message(conversation_id, "user", user_message)

    if image_bytes is not None:
        agent_key = "vision"
    else:
        agent_key = force_agent or route_query(user_message)
    if agent_key not in AGENTS:
        agent_key = "general"
    agent = AGENTS[agent_key]

    context = ""
    if image_bytes is None and agent_key != "general":
        query = extract_company_query(user_message)
        if query:
            try:
                context, _bundle = re_engine.quick_context_for_agent(
                    agent_key, query, observe=_observe(conversation_id)
                )
            except Exception as e:
                context = f"(Data retrieval failed: {e}. Do not fabricate figures — state that data is unavailable.)"
                logger.exception("quick_context_for_agent failed")

    history = build_history_messages(conversation_id)
    if save_user_message:
        history = history[:-1]  # exclude the user message just persisted

    prompt_parts = []
    if context:
        prompt_parts.append(f"[RETRIEVED EVIDENCE]\n{context}")
    prompt_parts.append(f"[USER QUESTION]\n{user_message}")
    if internal_instruction:
        prompt_parts.append(f"[INTERNAL INSTRUCTION — DO NOT QUOTE OR EXPOSE]\n{internal_instruction}")
    augmented = "\n\n".join(prompt_parts)
    run_input = history + [Message(role="user", content=augmented)]

    result = StreamResult(agent_used=agent_key)
    start = time.time()
    input_tokens = output_tokens = 0
    error_msg = None
    final_candidate = ""
    event_types: set[str] = set()

    try:
        if image_bytes is not None:
            from agno.media import Image as AgnoImage
            events = agent.run(user_message, images=[AgnoImage(content=image_bytes)], stream=True)
        else:
            events = agent.run(run_input, stream=True)

        # Some Agno versions may return a non-iterable response despite stream=True.
        try:
            iterator = iter(events)
        except TypeError:
            iterator = iter([events])

        for event in iterator:
            ev_name = _event_name(event)
            event_types.add(ev_name or type(event).__name__)
            text = _response_text(event)

            # Stream only content-bearing events. RunCompleted may repeat the whole
            # answer, so retain it as fallback instead of blindly appending it.
            if text:
                if ev_name.lower() in {"runcompleted", "run_completed", "completed"}:
                    final_candidate = text
                else:
                    result.text += text
                    yield text

            metrics = getattr(event, "metrics", None)
            if metrics:
                input_tokens = getattr(metrics, "input_tokens", 0) or input_tokens
                output_tokens = getattr(metrics, "output_tokens", 0) or output_tokens

        if not result.text and final_candidate:
            result.text = final_candidate
            yield final_candidate

        if not result.text.strip():
            result.text = (
                f"⚠️ The {agent.name or agent_key} completed without returning text. "
                "Check the observability logs for the underlying model response."
            )
            error_msg = "empty_model_response"
            yield result.text

    except Exception as e:
        error_msg = str(e)
        logger.exception("Agent run failed: agent=%s", agent_key)
        result.text = f"⚠️ Error while generating a response: {e}"
        yield result.text
    finally:
        latency_ms = (time.time() - start) * 1000
        log_observability(
            conversation_id, agent.name or agent_key, "agent_analysis",
            latency_ms, input_tokens, output_tokens, error_msg,
            metadata={"agent_key": agent_key, "event_types": sorted(event_types), "is_regeneration": is_regeneration},
        )

    if wants_chart(user_message):
        ticker = extract_ticker(user_message)
        if ticker:
            result.image_path = generate_price_chart(ticker)

    save_message(
        conversation_id, "assistant", result.text, agent_used=agent.name,
        image_path=result.image_path, is_regeneration=is_regeneration,
    )
    stream_chat.last_result = result


def regenerate_last(conversation_id: str, refinement_hint: str = "") -> Iterator[str]:
    """Regenerate the last answer without deleting/re-saving the original user turn."""
    rows = get_messages(conversation_id)
    last_user_index = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["role"] == "user"), None)
    if last_user_index is None:
        yield "Nothing to regenerate yet — ask a question first."
        return

    last_user = rows[last_user_index]
    following_assistant = next(
        (r for r in rows[last_user_index + 1:] if r["role"] == "assistant"), None
    )
    original_agent_key = _agent_key_from_stored_name(
        following_assistant["agent_used"] if following_assistant else None
    )

    delete_last_assistant_message(conversation_id)
    instruction = (
        "Provide a materially different analysis or framing while remaining grounded "
        "only in retrieved evidence. Do not mention this regeneration instruction."
    )
    if refinement_hint:
        instruction += f" Additional refinement: {refinement_hint}"

    yield from stream_chat(
        conversation_id,
        last_user["content"],
        force_agent=original_agent_key,
        save_user_message=False,
        is_regeneration=True,
        internal_instruction=instruction,
    )


# --------------------------------------------------------------------------
# Full multi-agent research report
# --------------------------------------------------------------------------

RESEARCH_AGENT_ORDER = ["sec_filing", "financial_statement", "market_data", "news_intelligence", "comparative_analysis"]

_CONTEXT_BUILDERS = {
    "sec_filing": re_engine.build_sec_filing_context,
    "financial_statement": re_engine.build_financial_statement_context,
    "market_data": re_engine.build_market_data_context,
    "news_intelligence": re_engine.build_news_context,
    "comparative_analysis": re_engine.build_comparative_context,
}


def run_full_research(conversation_id: str, company: str,
                       progress_cb: Optional[Callable[[str, str], None]] = None) -> StreamResult:
    """
    STEP 1-4: resolve identity, fetch all external sources concurrently
    (SEC calls still funnel through the shared 5 req/s limiter), assemble one
    ResearchDataBundle, and compute deterministic analytics.
    STEP 5-6: build per-agent evidence context and run the 5 specialists
    concurrently (ThreadPoolExecutor -- "threading").
    STEP 7-8: Investment Decision agent synthesizes the specialists;
    Supervisor produces the final report with a Sources/Provenance section.
    """
    observe = _observe(conversation_id)
    bundle = re_engine.build_research_bundle(company, progress_cb=progress_cb, observe=observe)

    def _run_specialist(key: str) -> tuple[str, str]:
        context = _CONTEXT_BUILDERS[key](bundle)
        start = time.time()
        err = None
        try:
            resp = AGENTS[key].run(f"Analyze the following evidence for {bundle.company_name} ({bundle.ticker}):\n\n{context}")
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            err = str(e)
            content = f"(This section failed: {e})"
        log_observability(conversation_id, AGENTS[key].name, "agent_analysis", (time.time() - start) * 1000, error=err)
        return key, content

    sections: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(RESEARCH_AGENT_ORDER)) as pool:
        futures = {pool.submit(_run_specialist, k): k for k in RESEARCH_AGENT_ORDER}
        for fut in as_completed(futures):
            key, content = fut.result()
            sections[key] = content

    if progress_cb:
        progress_cb("investment_decision", "running")
    decision_start = time.time()
    decision_resp = AGENTS["investment_decision"].run(
        f"Company: {bundle.company_name} ({bundle.ticker})\n\n"
        "Specialist sections (grounded in retrieved evidence):\n\n"
        + "\n\n".join(f"### {k}\n{v}" for k, v in sections.items())
        + f"\n\n### Evidence/source summary\n{re_engine.build_sources_summary(bundle)}"
    )
    decision_content = decision_resp.content if hasattr(decision_resp, "content") else str(decision_resp)
    log_observability(conversation_id, AGENTS["investment_decision"].name, "agent_analysis", (time.time() - decision_start) * 1000)
    if progress_cb:
        progress_cb("investment_decision", "done")

    if progress_cb:
        progress_cb("supervisor", "running")
    synth_start = time.time()
    synthesis_input = "\n\n".join(f"### {k}\n{v}" for k, v in sections.items())
    synthesis_input += f"\n\n### investment_decision\n{decision_content}"
    synthesis_input += f"\n\n### sources_and_provenance\n{re_engine.build_sources_summary(bundle)}"
    supervisor_resp = AGENTS["supervisor"].run(
        f"Company: {bundle.company_name} ({bundle.ticker})\n\nSynthesize the following specialist research into the final report:\n\n{synthesis_input}"
    )
    final_text = supervisor_resp.content if hasattr(supervisor_resp, "content") else str(supervisor_resp)
    log_observability(conversation_id, AGENTS["supervisor"].name, "research_synthesis", (time.time() - synth_start) * 1000)
    if progress_cb:
        progress_cb("supervisor", "done")

    chart_path = generate_price_chart(bundle.ticker) if bundle.ticker else None

    save_message(conversation_id, "assistant", final_text, agent_used="Financial Research Supervisor", image_path=chart_path)
    return StreamResult(text=final_text, agent_used="supervisor", image_path=chart_path)


db_core.init_db()
