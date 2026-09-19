import os
import time
import uuid

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables explicitly from the repository root so the UI
# works with a plain `streamlit run` call (not just `uv run`, which loads .env).
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


def _get_secret(name: str, default: str) -> str:
    """Read a Streamlit secret, falling back to the default when no
    secrets.toml exists (st.secrets.get raises instead of returning its default)."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# Initialize Logfire
try:
    logfire.configure(token=_get_secret("LOGFIRE_TOKEN", os.getenv("LOGFIRE_TOKEN", "")))
    logfire.instrument_requests()   # propagates trace context to the FastAPI backend
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception:
    LOGFIRE_STATUS = "Standby (No Token)"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    base_url = "http://localhost:8000"

    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    # --- Gateway (Portkey) switcher ---
    # Lets you fall back to a different provider slug/model per conversation.
    st.markdown("### 🔌 Gateway (Portkey)")
    default_slug = _get_secret("PORTKEY_PRIMARY_SLUG", os.getenv("PORTKEY_PRIMARY_SLUG", "groq"))
    default_model = _get_secret("PORTKEY_PRIMARY_MODEL", os.getenv("PORTKEY_PRIMARY_MODEL", "openai/gpt-oss-20b"))
    gateway_slug = st.text_input("Provider slug", value=default_slug, key="gateway_slug")
    gateway_model = st.text_input("Model", value=default_model, key="gateway_model")
    st.caption(f"Routing as `@{gateway_slug}/{gateway_model}`")

    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warning(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        url = f"{base_url}/query"
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id,
                            "slug": st.session_state.get("gateway_slug", "groq"),
                            "model": st.session_state.get("gateway_model", "openai/gpt-oss-20b"),
                        }
                        response = requests.post(url, json=payload, timeout=60)

                        if response.status_code != 200:
                            st.error(f"Backend Error: {response.status_code} - {response.text}")
                            st.stop()

                        data = response.json()

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.markdown(f"⚙️ {step}", unsafe_allow_html=False)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Answer streaming — outside status so it's always visible
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)

            # Sources — outside status so they're visible after it collapses
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 Retrieved Context ({len(sources)} chunks)"):
                    for i, source in enumerate(sources):
                        content = source.get("content", "") if isinstance(source, dict) else str(source)
                        src = source.get("source", "unknown") if isinstance(source, dict) else "unknown"
                        st.caption(f"Chunk {i + 1} [{src}]")
                        st.info(content)
            else:
                st.caption("ℹ️ No context retrieved — conversational response.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")