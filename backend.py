"""
FinBot Backend
==============
Multi-agent financial research backend built on the `agno` framework + Groq.

Covers every agent defined in the original notebook (SEC Filing, Financial
Statement, Market Data, News Intelligence, Comparative Analysis, Investment
Decision, and the Supervisor that orchestrates them) and adds everything
needed to run it as a persistent, multi-chat, streaming chatbot:

- SQLite persistence (conversations + messages survive app restarts)
- Multiple independent chat threads (used by the Streamlit sidebar)
- Streaming token-by-token responses
- Parallel ("threaded") execution of the specialist agents for full reports
- Chart/image output (Groq LLMs are text-only -> no image *generation*, so
  real price charts are rendered locally from live yfinance data; Groq
  vision models are used when the user uploads an image)
- "Regenerate" / iteration support for when a recommendation isn't wanted
- Lightweight observability: every run's latency + token usage is logged
  to SQLite and can be inspected from the UI
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEXT_MODEL_ID = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
# Vision-capable Groq model, used only when the user attaches an image.
VISION_MODEL_ID = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

DB_PATH = os.getenv("FINBOT_DB_PATH", os.path.join(os.path.dirname(__file__), "finbot_memory.db"))
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finbot")

# --------------------------------------------------------------------------
# Persistence layer (SQLite) — chats survive closing/reopening the app
# --------------------------------------------------------------------------

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
                role TEXT NOT NULL,              
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
                created_at TEXT NOT NULL
            );
            """
        )


def new_id() -> str:
    return uuid.uuid4().hex[:12]


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
        return conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()


def rename_conversation(cid: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, datetime.utcnow().isoformat(), cid),
        )


def touch_conversation(cid: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), cid),
        )


def delete_conversation(cid: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    agent_used: Optional[str] = None,
    image_path: Optional[str] = None,
    is_regeneration: bool = False,
) -> str:
    mid = new_id()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, agent_used, image_path, is_regeneration, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                mid, conversation_id, role, content, agent_used, image_path,
                int(is_regeneration), datetime.utcnow().isoformat(),
            ),
        )
    touch_conversation(conversation_id)
    return mid


def get_messages(conversation_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()


def delete_last_assistant_message(conversation_id: str) -> None:
    """Used before regenerating: drop the most recent assistant reply."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM messages WHERE conversation_id=? AND role='assistant' ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM messages WHERE id=?", (row["id"],))


def log_observability(
    conversation_id: Optional[str],
    agent_name: str,
    event: str,
    latency_ms: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    error: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO observability
               (id, conversation_id, agent_name, event, latency_ms, input_tokens, output_tokens, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                new_id(), conversation_id, agent_name, event, latency_ms,
                input_tokens, output_tokens, error, datetime.utcnow().isoformat(),
            ),
        )
    logger.info(
        "event=%s agent=%s latency_ms=%.0f in_tok=%s out_tok=%s error=%s",
        event, agent_name, latency_ms, input_tokens, output_tokens, error,
    )


def get_observability(conversation_id: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if conversation_id:
            return conn.execute(
                "SELECT * FROM observability WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM observability ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


# --------------------------------------------------------------------------
# Agents — one per cell of the original notebook, unchanged in spirit
# --------------------------------------------------------------------------

def _groq(vision: bool = False) -> Groq:
    return Groq(id=VISION_MODEL_ID if vision else TEXT_MODEL_ID, api_key=GROQ_API_KEY)


def build_agents() -> dict[str, Agent]:
    sec_filing_agent = Agent(
        model=_groq(),
        name="SEC Filing Agent",
        description=dedent("""
            You are a specialized SEC Filing analyst expert.
            Expertise: 10-K, 10-Q, 8-K, XBRL data extraction, risk factor
            identification, MD&A interpretation, change detection across filings.
        """),
        instructions=dedent("""
            Based on your knowledge, analyze the most recent SEC filings for the
            requested company: key financial data, risk factors, management
            changes, period-over-period metric comparisons, material events,
            and regulatory concerns. Focus on performance changes, business
            risk, accounting policy changes, executive comp, debt/liquidity,
            and forward guidance. If you are not certain of a specific recent
            figure, say so explicitly rather than inventing one.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    financial_statement_agent = Agent(
        model=_groq(),
        name="Financial Statement Agent",
        description=dedent("""
            You are a specialized financial statement and valuation analyst.
            Expertise: ratio analysis, growth metrics, cash flow analysis,
            DCF/comparable/asset-based valuation, benchmarking.
        """),
        instructions=dedent("""
            Provide financial ratios (profitability, liquidity, efficiency,
            leverage), growth trends, cash flow quality, a valuation
            assessment via multiple methods, industry benchmarks, balance
            sheet health, and clearly labelled strengths/weaknesses. Be
            explicit about assumptions and flag anything you're uncertain of.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    market_data_agent = Agent(
        model=_groq(),
        name="Market Data Agent",
        description=dedent("""
            You are a specialized market data and technical analysis expert.
            Expertise: price/volume trends, RSI/MACD/Bollinger/MAs, dividend
            history, analyst consensus, institutional ownership, sentiment.
        """),
        instructions=dedent("""
            Summarize price action, technical indicators, volume/liquidity,
            analyst consensus and price targets, institutional/insider
            trends, relative performance vs. market/sector, and dividend
            yield. State clearly that live numbers should be verified against
            a market data source, since your knowledge is not real-time.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    news_intelligence_agent = Agent(
        model=_groq(),
        name="News Intelligence Agent",
        description=dedent("""
            You are a specialized news intelligence and sentiment analysis expert.
            Expertise: financial news synthesis, filings/press releases,
            leadership changes, M&A, earnings surprises, sentiment/tone.
        """),
        instructions=dedent("""
            Summarize likely recent news themes, sentiment, catalysts,
            earnings/guidance changes, leadership/regulatory news, and
            market impact, based on your knowledge. Be explicit that you
            cannot browse live headlines and recommend the user verify
            against a live news feed for anything time-sensitive.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    comparative_analysis_agent = Agent(
        model=_groq(),
        name="Comparative Analysis Agent",
        description=dedent("""
            You are a specialized comparative and competitive analysis expert.
            Expertise: multi-company comparison, SWOT, relative valuation,
            competitive moats, market share, peer ranking.
        """),
        instructions=dedent("""
            Identify relevant peers, compare financial metrics and
            valuation multiples, produce a SWOT analysis, assess competitive
            positioning/moats, and rank the company vs. peers with reasoning.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    investment_decision_agent = Agent(
        model=_groq(),
        name="Investment Decision Agent",
        description=dedent("""
            You are a specialized investment recommendation analyst.
            Expertise: scoring models, risk assessment, bull/bear cases,
            thesis development, scenario planning.
        """),
        instructions=dedent("""
            Synthesize financial health, valuation, sentiment, growth
            catalysts, and risk into a bull case, a bear case, a rating
            (BUY/HOLD/SELL/AVOID), a rough price-target range, a risk
            level, and a short action plan. Always include a one-line
            disclaimer that this is not personalized financial advice.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    supervisor = Agent(
        model=_groq(),
        name="Financial Research Supervisor",
        description=dedent("""
            You are the Financial Research Supervisor, an elite investment
            research director who synthesizes findings from six specialist
            analysts into one coherent report.
        """),
        instructions=dedent("""
            You will be given the raw outputs of the SEC Filing, Financial
            Statement, Market Data, News Intelligence, Comparative Analysis
            and Investment Decision agents for one company. Synthesize them
            into a single, well-organized report. Resolve/flag any
            inconsistencies across sources instead of silently picking one.
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
            ## Recommendation (Rating, Price Target, Action Plan)
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    # General-purpose conversational agent for everyday, non-report chat.
    general_agent = Agent(
        model=_groq(),
        name="General Research Assistant",
        description="A helpful, knowledgeable financial research assistant for day-to-day questions.",
        instructions=dedent("""
            Answer conversationally and concisely unless the user asks for
            depth. You are part of a financial research toolkit, so lean on
            that expertise, but you can also just chat normally.
        """),
        markdown=True,
        add_datetime_to_context=True,
    )

    vision_agent = Agent(
        model=_groq(vision=True),
        name="Vision Analyst",
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
    "sec_filing": ["10-k", "10-q", "8-k", "xbrl", "sec filing", "annual report", "quarterly report", "mda"],
    "financial_statement": ["ratio", "margin", "cash flow", "valuation", "dcf", "balance sheet", "earnings per share", "eps", "roe", "roic"],
    "market_data": ["price", "technical", "rsi", "macd", "moving average", "volume", "dividend", "analyst rating", "price target"],
    "news_intelligence": ["news", "headline", "sentiment", "press release", "announcement"],
    "comparative_analysis": ["compare", "vs", "versus", "swot", "competitor", "peer"],
    "investment_decision": ["should i buy", "should i sell", "recommend", "buy or sell", "investment thesis", "bull case", "bear case"],
}


def route_query(query: str) -> str:
    """Cheap keyword router so a single quick question doesn't spin up all 6 agents."""
    q = query.lower()
    scores = {k: sum(1 for kw in kws if kw in q) for k, kws in ROUTER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# --------------------------------------------------------------------------
# Ticker / chart helpers — this is the "picture output" the app can produce.
# Groq's LLMs are text-only (no image generation), so real charts are
# rendered locally from live market data instead.
# --------------------------------------------------------------------------

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
COMMON_WORDS = {"I", "A", "THE", "IS", "IT", "TO", "AND", "OR", "FOR", "OF", "ON", "IN"}


def extract_ticker(text: str) -> Optional[str]:
    candidates = [t for t in TICKER_RE.findall(text) if t not in COMMON_WORDS]
    return candidates[0] if candidates else None


def generate_price_chart(ticker: str, period: str = "6mo") -> Optional[str]:
    """Fetch live data via yfinance and render a PNG chart. Returns file path or None."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; skipping chart generation")
        return None

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
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


def wants_chart(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["chart", "graph", "plot", "picture", "image", "show me the price", "visualize"])


# --------------------------------------------------------------------------
# History helpers — build agno Message list from persisted SQLite rows
# --------------------------------------------------------------------------

def build_history_messages(conversation_id: str, max_turns: int = 12) -> list[Message]:
    rows = get_messages(conversation_id)[-max_turns * 2:]
    return [Message(role=r["role"], content=r["content"]) for r in rows if r["role"] in ("user", "assistant")]


# --------------------------------------------------------------------------
# Streaming chat — the main entry point used by the Streamlit frontend
# --------------------------------------------------------------------------

@dataclass
class StreamResult:
    text: str = ""
    agent_used: str = ""
    image_path: Optional[str] = None


def stream_chat(
    conversation_id: str,
    user_message: str,
    force_agent: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
) -> Iterator[str]:
    """Yields text deltas as they arrive, then persists the final message.

    Usage (Streamlit):
        placeholder = st.empty()
        for chunk in stream_chat(cid, prompt):
            placeholder.markdown(acc)
    The final StreamResult (with image_path, agent_used) is attached as
    `stream_chat.last_result` after the generator is exhausted.
    """
    save_message(conversation_id, "user", user_message)

    if image_bytes is not None:
        agent_key = "vision"
    else:
        agent_key = force_agent or route_query(user_message)
    agent = AGENTS[agent_key]

    history = build_history_messages(conversation_id)[:-1]  # exclude the message we just saved
    run_input = history + [Message(role="user", content=user_message)]

    result = StreamResult(agent_used=agent_key)
    start = time.time()
    input_tokens = output_tokens = 0
    error_msg = None

    try:
        if image_bytes is not None:
            from agno.media import Image as AgnoImage
            events = agent.run(user_message, images=[AgnoImage(content=image_bytes)], stream=True)
        else:
            events = agent.run(run_input, stream=True)

        for event in events:
            ev_name = getattr(event, "event", "")
            if ev_name == "RunContent" and getattr(event, "content", None):
                delta = event.content if isinstance(event.content, str) else str(event.content)
                result.text += delta
                yield delta
            elif ev_name == "RunCompleted":
                metrics = getattr(event, "metrics", None)
                if metrics:
                    input_tokens = getattr(metrics, "input_tokens", 0) or 0
                    output_tokens = getattr(metrics, "output_tokens", 0) or 0
    except Exception as e:
        error_msg = str(e)
        result.text = result.text or f"⚠️ Error while generating a response: {e}"
        yield result.text
    finally:
        latency_ms = (time.time() - start) * 1000
        log_observability(
            conversation_id, AGENTS[agent_key].name or agent_key, "chat_run",
            latency_ms, input_tokens, output_tokens, error_msg,
        )

    if wants_chart(user_message):
        ticker = extract_ticker(user_message)
        if ticker:
            chart_path = generate_price_chart(ticker)
            result.image_path = chart_path

    save_message(
        conversation_id, "assistant", result.text,
        agent_used=agent.name, image_path=result.image_path,
    )
    stream_chat.last_result = result


def regenerate_last(conversation_id: str, refinement_hint: str = "") -> Iterator[str]:
    """Iteration support: re-ask for a different take on the last user question."""
    rows = get_messages(conversation_id)
    last_user = next((r for r in reversed(rows) if r["role"] == "user"), None)
    if not last_user:
        yield "Nothing to regenerate yet — ask a question first."
        return
    delete_last_assistant_message(conversation_id)
    prompt = (
        f"{last_user['content']}\n\n"
        "(The user didn't like the previous answer/recommendation — "
        "reconsider your assumptions and provide a genuinely different "
        f"angle or a more cautious/alternative take. {refinement_hint})"
    )
    # Reuse stream_chat but don't re-save the user's original question again.
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE id=?", (last_user["id"],))
    yield from stream_chat(conversation_id, prompt)


# --------------------------------------------------------------------------
# Full multi-agent research report — runs the 5 specialists concurrently
# ("threading") then synthesizes with the supervisor agent.
# --------------------------------------------------------------------------

RESEARCH_AGENT_ORDER = [
    "sec_filing", "financial_statement", "market_data",
    "news_intelligence", "comparative_analysis",
]


def run_full_research(
    conversation_id: str,
    company: str,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> StreamResult:
    """Runs the 5 specialist agents in parallel threads, then the supervisor
    synthesizes their outputs plus the investment decision agent's verdict.

    progress_cb(agent_key, status) is called as each specialist starts/finishes,
    so the Streamlit UI can show live progress.
    """
    query = f"Provide your specialized analysis of {company} for an investment research report."

    def _run_one(key: str) -> tuple[str, str]:
        if progress_cb:
            progress_cb(key, "running")
        start = time.time()
        err = None
        try:
            resp = AGENTS[key].run(query)
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            err = str(e)
            content = f"(This section failed: {e})"
        log_observability(conversation_id, AGENTS[key].name, "research_section",
                           (time.time() - start) * 1000, error=err)
        if progress_cb:
            progress_cb(key, "done")
        return key, content

    sections: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(RESEARCH_AGENT_ORDER)) as pool:
        futures = {pool.submit(_run_one, k): k for k in RESEARCH_AGENT_ORDER}
        for fut in as_completed(futures):
            key, content = fut.result()
            sections[key] = content

    if progress_cb:
        progress_cb("investment_decision", "running")
    decision_start = time.time()
    decision_resp = AGENTS["investment_decision"].run(
        f"Based on the following research sections about {company}, provide the final "
        f"bull/bear case, rating, price-target range, and action plan.\n\n"
        + "\n\n".join(f"### {k}\n{v}" for k, v in sections.items())
    )
    decision_content = decision_resp.content if hasattr(decision_resp, "content") else str(decision_resp)
    log_observability(conversation_id, AGENTS["investment_decision"].name, "research_section",
                       (time.time() - decision_start) * 1000)
    if progress_cb:
        progress_cb("investment_decision", "done")

    if progress_cb:
        progress_cb("supervisor", "running")
    synth_start = time.time()
    synthesis_input = "\n\n".join(f"### {k}\n{v}" for k, v in sections.items())
    synthesis_input += f"\n\n### investment_decision\n{decision_content}"
    supervisor_resp = AGENTS["supervisor"].run(
        f"Company: {company}\n\nSynthesize the following specialist research into the final report:\n\n{synthesis_input}"
    )
    final_text = supervisor_resp.content if hasattr(supervisor_resp, "content") else str(supervisor_resp)
    log_observability(conversation_id, AGENTS["supervisor"].name, "research_synthesis",
                       (time.time() - synth_start) * 1000)
    if progress_cb:
        progress_cb("supervisor", "done")

    chart_path = generate_price_chart(company) if re.fullmatch(r"[A-Za-z.]{1,5}", company.strip()) else None

    save_message(conversation_id, "assistant", final_text, agent_used="Financial Research Supervisor", image_path=chart_path)
    return StreamResult(text=final_text, agent_used="supervisor", image_path=chart_path)


init_db()
