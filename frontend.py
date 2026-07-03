
import os
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

import backend as bk
from progress_utils import render_progress_update

st.set_page_config(page_title="FinBot — Financial Research Assistant", page_icon="📈", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to a .env file next to these scripts.")
    st.stop()

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    convs = bk.list_conversations()
    st.session_state.conversation_id = convs[0]["id"] if convs else bk.create_conversation()

AGENT_LABELS = {
    "general": "General Chat (fast)",
    "sec_filing": "SEC Filing Agent",
    "financial_statement": "Financial Statement Agent",
    "market_data": "Market Data Agent",
    "news_intelligence": "News Intelligence Agent",
    "comparative_analysis": "Comparative Analysis Agent",
    "investment_decision": "🎯 Investment Decision Agent",
}

# --------------------------------------------------------------------------
# Sidebar — multi-chat management
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 FinBot")
    st.caption("Multi-agent financial research, powered by Groq")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.conversation_id = bk.create_conversation()
        st.rerun()

    st.divider()
    st.subheader("Chats")
    for conv in bk.list_conversations():
        cols = st.columns([5, 1])
        label = conv["title"] or "New chat"
        active = conv["id"] == st.session_state.conversation_id
        if cols[0].button(("🟢 " if active else "") + label, key=f"sel_{conv['id']}", use_container_width=True):
            st.session_state.conversation_id = conv["id"]
            st.rerun()
        if cols[1].button("🗑️", key=f"del_{conv['id']}"):
            bk.delete_conversation(conv["id"])
            if st.session_state.conversation_id == conv["id"]:
                remaining = bk.list_conversations()
                st.session_state.conversation_id = remaining[0]["id"] if remaining else bk.create_conversation()
            st.rerun()

    st.divider()
    st.subheader("Mode")
    mode = st.radio(
        "How should this message be handled?",
        ["Auto-route (recommended)", "Force a specialist agent", "Full research report (all 6 agents)"],
        label_visibility="collapsed",
    )
    forced_agent = None
    if mode == "Force a specialist agent":
        forced_agent = st.selectbox("Specialist", list(AGENT_LABELS.keys()), format_func=lambda k: AGENT_LABELS[k])

    st.divider()
    uploaded_image = st.file_uploader("Attach an image for analysis (Groq vision model)", type=["png", "jpg", "jpeg"])

    with st.expander("📊 Observability"):
        logs = bk.get_observability(st.session_state.conversation_id, limit=15)
        if not logs:
            st.caption("No runs logged yet.")
        for l in logs:
            st.caption(
                f"**{l['agent_name']}** · {l['event']} · {l['latency_ms']:.0f} ms"
                + (f" · {l['input_tokens']}→{l['output_tokens']} tok" if l['input_tokens'] else "")
                + (f" · ⚠️ {l['error']}" if l["error"] else "")
            )

# --------------------------------------------------------------------------
# Rename the chat automatically from the first user message
# --------------------------------------------------------------------------
def maybe_autotitle(cid: str, first_message: str) -> None:
    rows = bk.get_messages(cid)
    if len(rows) <= 1:
        title = (first_message[:40] + "…") if len(first_message) > 40 else first_message
        bk.rename_conversation(cid, title)


# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
cid = st.session_state.conversation_id
st.title("Financial Research Assistant")
st.caption(
    "Auto-route asks the best single specialist. Full research report runs all 6 "
    "agents in parallel threads and synthesizes a complete report. Ask for a "
    "'chart' or 'price graph' with a ticker to get a real, live price chart."
)

messages = bk.get_messages(cid)
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["image_path"] and os.path.exists(m["image_path"]):
            st.image(m["image_path"])
        if m["role"] == "assistant":
            if st.button("🔄 Regenerate / I don't like this", key=f"regen_{m['id']}"):
                st.session_state["_regenerate"] = True
                st.rerun()

prompt = st.chat_input("Ask about a company, request a report, or say 'chart NVDA'...")

# Handle an explicit regenerate click (regenerates the LAST assistant message)
if st.session_state.pop("_regenerate", False):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        for chunk in bk.regenerate_last(cid):
            acc += chunk
            placeholder.markdown(acc + "▌")
        placeholder.markdown(acc)
        result = getattr(bk.stream_chat, "last_result", None)
        if result and result.image_path:
            st.image(result.image_path)
    st.rerun()

if prompt:
    image_bytes = uploaded_image.getvalue() if uploaded_image else None

    with st.chat_message("user"):
        st.markdown(prompt)

    maybe_autotitle(cid, prompt)

    if mode == "Full research report (all 6 agents)":
        with st.chat_message("assistant"):
            st.info("Running SEC Filing, Financial Statement, Market Data, News "
                    "Intelligence and Comparative Analysis agents in parallel, "
                    "then synthesizing with the Investment Decision + Supervisor agents…")
            progress = st.empty()
            status_lines: dict[str, str] = {}
            script_ctx = get_script_run_ctx()

            def cb(agent_key: str, status: str) -> None:
                render_progress_update(progress, status_lines, agent_key, status, script_ctx=script_ctx)

            bk.save_message(cid, "user", prompt)
            result = bk.run_full_research(cid, prompt, progress_cb=cb)
            st.markdown(result.text)
            if result.image_path:
                st.image(result.image_path)
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc = ""
            for chunk in bk.stream_chat(cid, prompt, force_agent=forced_agent, image_bytes=image_bytes):
                acc += chunk
                placeholder.markdown(acc + "▌")
            placeholder.markdown(acc or "_(no response)_")
            result = getattr(bk.stream_chat, "last_result", None)
            if result and result.image_path:
                st.image(result.image_path)
            if result:
                st.caption(f"Answered by: {result.agent_used}")

    st.rerun()
