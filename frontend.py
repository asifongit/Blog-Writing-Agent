"""
Blog Writing Agent – Streamlit Frontend
Isolated from the backend: imports only generate_blog / list_blogs / get_blog from main.py.
"""

import streamlit as st
from main import generate_blog, list_blogs, get_blog

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

# ── Minimal custom style ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 280px; max-width: 320px; }
        .blog-btn > button {
            text-align: left !important;
            white-space: normal !important;
            height: auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "topic_used" not in st.session_state:
    st.session_state.topic_used = ""
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# ── Sidebar – blog history ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Blog History")

    if st.button("✏️ New Blog", type="primary", use_container_width=True):
        st.session_state.result      = None
        st.session_state.topic_used  = ""
        st.session_state.topic_input = ""   # ← clears the text box
        st.rerun()

    st.divider()
    st.caption("Click a title to re-read it")

    blogs = list_blogs()
    if blogs:
        for blog in blogs:
            label = blog["blog_title"]
            date  = blog["created_at"][:10]          # YYYY-MM-DD
            with st.container():
                st.markdown(f"<small style='color:grey'>{date}</small>", unsafe_allow_html=True)
                if st.button(label, key=blog["thread_id"], use_container_width=True):
                    saved = get_blog(blog["thread_id"])
                    if saved:
                        st.session_state.result    = saved
                        st.session_state.topic_used = saved["topic"]
                        st.rerun()
            st.markdown("---")
    else:
        st.info("No blogs yet. Generate your first one!")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("✍️ Blog Writing Agent")
st.caption("Powered by LangGraph")
st.divider()

# ── Input row ─────────────────────────────────────────────────────────────────
topic = st.text_input(
    "Blog topic",
    placeholder="e.g. Docker, LangGraph, Python async, Kubernetes networking …",
    key="topic_input",   # bound to session state so New Blog can reset it
)

generate_btn = st.button("Generate Blog", type="primary")

# ── Generation ────────────────────────────────────────────────────────────────
if generate_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a blog topic first.")
        st.stop()

    with st.spinner("🤖 Planning & writing sections in parallel …"):
        try:
            st.session_state.result    = generate_blog(topic.strip())
            st.session_state.topic_used = topic.strip()
        except Exception as exc:
            st.error(f"❌ Generation failed: {exc}")
            st.stop()

    st.rerun()   # refresh sidebar to show the new entry immediately

# ── Display result (persists across any re-run / download click) ──────────────
if st.session_state.result:
    result     = st.session_state.result
    used_topic = st.session_state.topic_used
    final_md: str = result["final"]
    plan          = result["plan"]

    # plan summary
    with st.expander("📋 Blog plan", expanded=False):
        st.markdown(f"**Title:** {plan.blog_title}")
        st.markdown(f"**Audience:** {plan.audience}")
        st.markdown(f"**Tone:** {plan.tone}")
        for task in plan.tasks:
            st.markdown(
                f"- **{task.title}** *(~{task.target_words} words · {task.section_type})*"
            )

    st.divider()

    # rendered blog
    st.markdown(final_md)

    st.divider()

    # download — clicking this no longer wipes the page
    st.download_button(
        label="⬇️ Download .md",
        data=final_md.encode("utf-8"),
        file_name=f"{used_topic.lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )
