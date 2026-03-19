"""
app.py — Curriculum Radar: Streamlit Frontend (Mobile-Friendly)
"""

import os
import tempfile
import streamlit as st
import networkx as nx

st.set_page_config(
    page_title="Curriculum Radar",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #0a0c14;
        color: #d4d8e8;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }
    .cr-hero {
        background: linear-gradient(135deg, #0f1221 0%, #141830 60%, #0a0c14 100%);
        border: 1px solid #1e2646;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .cr-hero h1 {
        font-family: 'Space Mono', monospace;
        font-size: 1.7rem;
        color: #e8ecff;
        margin: 0 0 0.4rem 0;
    }
    .cr-hero p {
        font-size: 0.9rem;
        color: #8890b0;
        margin: 0;
        font-weight: 300;
    }
    .metric-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1 1 80px;
        background: #111428;
        border: 1px solid #1e2646;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #4CC9F0;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: #5a6080;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .node-card {
        background: #111428;
        border: 1px solid #252b50;
        border-left: 4px solid #4CC9F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .node-card h4 {
        font-family: 'Space Mono', monospace;
        color: #c8d0f0;
        font-size: 0.85rem;
        margin: 0 0 0.4rem 0;
    }
    .node-card p {
        font-size: 0.82rem;
        color: #7080a0;
        margin: 0;
        line-height: 1.6;
    }
    .concept-badge {
        display: inline-block;
        background: #1a2040;
        border: 1px solid #2a3060;
        border-radius: 20px;
        padding: 2px 9px;
        font-size: 0.7rem;
        color: #7090d0;
        margin: 2px;
        font-family: 'Space Mono', monospace;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4361EE, #7209B7);
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        padding: 0.65rem 1.4rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stProgress > div > div { background-color: #4361EE; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "graph": None, "html_path": None, "analytics": {},
        "sections": [], "concepts": [], "processed": False, "pdf_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='cr-hero'>"
    "<h1>🧭 Curriculum Radar</h1>"
    "<p>Upload an academic PDF to visualise how topics and chapters interconnect</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Upload (in main page, no sidebar needed) ──────────────────────────────────
uploaded = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"],
    help="Syllabus, textbook, lecture notes, or research paper",
)

# ── Settings (collapsible) ────────────────────────────────────────────────────
edge_kw_threshold  = 3
edge_sim_threshold = 0.45

with st.expander("⚙️ Settings (optional)"):
    edge_kw_threshold = st.slider(
        "Min shared keywords for prerequisite edge",
        min_value=1, max_value=10, value=3,
    )
    edge_sim_threshold = st.slider(
        "Semantic similarity threshold",
        min_value=0.20, max_value=0.90, value=0.45, step=0.05,
    )

# ── Analyse Button ────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Analyse PDF", disabled=uploaded is None)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn and uploaded is not None:
    import logic as _logic
    _logic.EDGE_KEYWORD_THRESHOLD    = edge_kw_threshold
    _logic.EDGE_SIMILARITY_THRESHOLD = edge_sim_threshold

    pdf_bytes = uploaded.read()
    st.session_state["pdf_name"] = uploaded.name

    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    def on_progress(step, total, msg):
        progress_bar.progress(int(step / total * 100), text=msg)
        status_text.markdown(
            f"<p style='font-size:0.78rem;color:#5a6080;text-align:center'>{msg}</p>",
            unsafe_allow_html=True,
        )

    try:
        from logic import (run_pipeline, detect_sections,
                           extract_text_from_pdf, extract_concepts, load_nlp_model)

        G, html_path, analytics = run_pipeline(pdf_bytes, progress_callback=on_progress)

        pages    = extract_text_from_pdf(pdf_bytes)
        sections = detect_sections(pages)
        nlp      = load_nlp_model()
        concepts = [extract_concepts(s, nlp) for s in sections]

        st.session_state.update({
            "graph": G, "html_path": html_path, "analytics": analytics,
            "sections": sections, "concepts": concepts, "processed": True,
        })
        progress_bar.progress(100, text="✅ Done!")
        status_text.empty()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {exc}")
        st.exception(exc)

# ── Display Results ───────────────────────────────────────────────────────────
if st.session_state["processed"] and st.session_state["html_path"]:
    G             = st.session_state["graph"]
    html_path     = st.session_state["html_path"]
    sections      = st.session_state["sections"]
    concepts_list = st.session_state["concepts"]
    a             = st.session_state["analytics"]

    # Stats row
    st.markdown(
        f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='value'>{a.get('nodes','—')}</div><div class='label'>Nodes</div></div>
          <div class='metric-card'><div class='value'>{a.get('edges','—')}</div><div class='label'>Edges</div></div>
          <div class='metric-card'><div class='value'>{a.get('components','—')}</div><div class='label'>Parts</div></div>
          <div class='metric-card'><div class='value'>{a.get('density','—')}</div><div class='label'>Density</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_graph, tab_sections, tab_export = st.tabs([
        "🌐 Graph", "📚 Sections", "📤 Export",
    ])

    # ── Graph Tab ─────────────────────────────────────────────────────────
    with tab_graph:
        st.markdown(
            "<p style='font-size:0.78rem;color:#5a6080;'>"
            "🔴 Solid = prerequisite &nbsp;|&nbsp; 🔵 Dashed = related. "
            "Tap nodes to see details.</p>",
            unsafe_allow_html=True,
        )
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=False)

    # ── Sections Tab ──────────────────────────────────────────────────────
    with tab_sections:
        titles = [s["title"] for s in sections]
        selected_title = st.selectbox("Select a section", titles)
        idx      = titles.index(selected_title)
        section  = sections[idx]
        concepts = concepts_list[idx] if idx < len(concepts_list) else []
        n_attrs  = G.nodes.get(idx, {})
        color    = n_attrs.get("color", "#4CC9F0")
        summary  = n_attrs.get("summary", section["text"][:400])

        st.markdown(
            f"<div class='node-card' style='border-left-color:{color}'>"
            f"<h4>{section['title']}</h4>"
            f"<p style='color:#4a5080;font-size:0.72rem;margin-bottom:0.5rem;'>"
            f"📄 {n_attrs.get('page_range','—')}</p>"
            f"<p>{summary}</p></div>",
            unsafe_allow_html=True,
        )

        if concepts:
            badges = " ".join(f"<span class='concept-badge'>{c}</span>" for c in concepts[:10])
            st.markdown(
                f"<p style='font-size:0.75rem;color:#5a6080;margin:0.5rem 0 0.3rem;'>"
                f"🔑 Key Concepts</p>{badges}",
                unsafe_allow_html=True,
            )

        out_edges = list(G.out_edges(idx, data=True))
        in_edges  = list(G.in_edges(idx, data=True))
        if out_edges or in_edges:
            st.markdown(
                "<p style='font-size:0.75rem;color:#5a6080;margin-top:0.8rem;'>🔗 Connections</p>",
                unsafe_allow_html=True,
            )
            for src, tgt, edata in out_edges:
                lbl  = G.nodes[tgt].get("label", f"Section {tgt}")
                icon = "🔴" if edata.get("type") == "prerequisite" else "🔵"
                st.markdown(
                    f"<span style='font-size:0.8rem;color:#8890b0'>"
                    f"{icon} → <b style='color:#c8d0f0'>{lbl}</b></span>",
                    unsafe_allow_html=True,
                )
            for src, tgt, edata in in_edges:
                lbl  = G.nodes[src].get("label", f"Section {src}")
                icon = "🔴" if edata.get("type") == "prerequisite" else "🔵"
                st.markdown(
                    f"<span style='font-size:0.8rem;color:#8890b0'>"
                    f"{icon} ← <b style='color:#c8d0f0'>{lbl}</b></span>",
                    unsafe_allow_html=True,
                )

    # ── Export Tab ────────────────────────────────────────────────────────
    with tab_export:
        with open(html_path, "rb") as f:
            st.download_button(
                "⬇️ Download Graph (HTML)", data=f.read(),
                file_name="curriculum_radar_graph.html", mime="text/html",
                use_container_width=True,
            )

        import pandas as pd
        rows = []
        for src, tgt, edata in G.edges(data=True):
            rows.append({
                "source": G.nodes[src].get("label", src),
                "target": G.nodes[tgt].get("label", tgt),
                "type":   edata.get("type", ""),
                "weight": edata.get("weight", ""),
            })
        df = pd.DataFrame(rows)
        st.download_button(
            "⬇️ Download Edge List (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name="edges.csv", mime="text/csv",
            use_container_width=True,
        )

else:
    if not st.session_state["processed"]:
        st.markdown(
            "<div style='text-align:center;padding:2.5rem 1rem;'>"
            "<div style='font-size:3.5rem;margin-bottom:1rem;'>📄 → 🌐</div>"
            "<p style='color:#3a4060;font-size:0.9rem;'>"
            "Upload a PDF above and tap <b style='color:#4361EE'>Analyse PDF</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )
