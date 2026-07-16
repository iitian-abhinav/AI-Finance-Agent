"""
progress_utils.py — renders live multi-agent progress updates in the
Streamlit UI while `run_full_research()` is running.

Why this needs to exist: `run_full_research()` calls `progress_cb(agent_key,
status)` from worker threads (it uses a ThreadPoolExecutor to run the
specialist agents concurrently). Streamlit's `st.*` calls / placeholder
updates are only safe from a thread that has the app's ScriptRunContext
attached — plain background threads don't have one by default, which
causes "missing ScriptRunContext" warnings or silently-dropped UI updates.

`render_progress_update` attaches the main thread's context to whatever
thread it's called from (once) before touching the placeholder, and keeps a
lock around the shared status dict since multiple worker threads call this
concurrently.
"""

from __future__ import annotations

import threading

from streamlit.runtime.scriptrunner_utils.script_run_context import (
    add_script_run_ctx,
    get_script_run_ctx,
)

AGENT_LABELS = {
    "resolve": "🔎 Resolving company",
    "sec_filing": "📄 SEC Filing Agent",
    "financial_statement": "📊 Financial Statement Agent",
    "market_data": "📈 Market Data Agent",
    "news_intelligence": "📰 News Intelligence Agent",
    "comparative_analysis": "⚖️ Comparative Analysis Agent",
    "investment_decision": "🎯 Investment Decision Agent",
    "supervisor": "🧭 Supervisor (final synthesis)",
}

STATUS_ICONS = {"running": "⏳", "done": "✅", "error": "⚠️"}

_lock = threading.Lock()


def render_progress_update(placeholder, status_lines: dict, agent_key: str,
                            status: str, script_ctx=None) -> None:
    """Update `status_lines[agent_key] = status` and re-render `placeholder`.

    Safe to call from any thread, including ThreadPoolExecutor workers,
    PROVIDED `script_ctx` is the context captured on the main Streamlit
    thread via `get_script_run_ctx()` before the workers were started.
    """
    current_thread = threading.current_thread()
    if script_ctx is not None and get_script_run_ctx(suppress_warning=True) is None:
        add_script_run_ctx(current_thread, script_ctx)

    with _lock:
        status_lines[agent_key] = status
        lines = []
        for key, label in AGENT_LABELS.items():
            if key in status_lines:
                icon = STATUS_ICONS.get(status_lines[key], "•")
                lines.append(f"{icon} {label}")
        try:
            placeholder.markdown("\n\n".join(lines) if lines else "Starting…")
        except Exception:
            # If the context attach still didn't take (e.g. very old Streamlit
            # version with a different internal API), fail quietly rather than
            # crashing the research run — the final report still completes.
            pass
