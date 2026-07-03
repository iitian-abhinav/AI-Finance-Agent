from __future__ import annotations

from typing import Any, MutableMapping


def render_progress_update(
    progress: Any,
    status_lines: MutableMapping[str, str],
    agent_key: str,
    status: str,
    *,
    script_ctx: Any = None,
) -> None:
    """Safely update the research progress UI from a worker thread.

    Streamlit widgets must be touched from the script thread. When a background
    thread calls this helper, we attach the captured script context to the
    current thread before rendering and quietly skip the UI update if no
    session context is available.
    """

    status_lines[agent_key] = "✅" if status == "done" else "⏳"
    if progress is None:
        return

    if script_ctx is None:
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

            script_ctx = get_script_run_ctx()
        except Exception:
            script_ctx = None

    if script_ctx is None:
        return

    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx, remove_script_run_ctx

        add_script_run_ctx(script_ctx)
        progress.markdown("\n".join(f"{v} {k}" for k, v in status_lines.items()))
    except Exception:
        # Preserve the research run even if the UI update races with Streamlit.
        pass
    finally:
        try:
            remove_script_run_ctx()
        except Exception:
            pass
