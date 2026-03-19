"""
app.py — Curriculum Radar: Streamlit Frontend
==============================================
Run with:  streamlit run app.py
"""

import io
import os
import base64
import tempfile
import textwrap

import streamlit as st
import networkx as nx

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Curriculum Radar",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark, refined, academic-tech aesthetic) ─────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background: #0a0c14;
        color: #d4d8e8;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f1221;
        border-right: 1px solid #1e2340;
    }

    /* ── Hero header ── */
    .cr-hero {
        background: linear-gradient(135deg, #0f1221 0%, #141830 60%, #0a0c14 100%);
        border: 1px solid #1e2646;
        border-radius: 16px;
        padding: 2.4rem 2.8rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .cr-hero::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, #4361ee33 0%, transparent 70%);
        border-radius: 50%;
    }
    .cr-hero h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.1rem;
        letter-spacing: -0.02em;
        color: #e8ecff;
        margin: 0 0 0.3rem 0;
    }
    .cr-hero p {
        font-size: 1.05rem;
        color: #8890b0;
        margin: 0;
        max-width: 580px;
        font-weight: 300;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.2rem 0;
    }
    .metric-card {
        flex: 1 1 120px;
        background: #111428;
        border: 1px solid #1e2646;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #4CC9F0;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #5a6080;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* ── Section node inspector ── */
    .node-card {
        background: #111428;
        border: 1px solid #252b50;
        border-left: 4px solid #4CC9F0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .node-card h4 {
        font-family: 'Space Mono', monospace;
        color: #c8d0f0;
        font-size: 0.9rem;
        margin: 0 0 0.4rem 0;
    }
    .node-card p {
        font-size: 0.85rem;
        color: #7080a0;
        margin: 0;
        line-height: 1.6;
    }
    .concept-badge {
        display: inline-block;
        background: #1a2040;
        border: 1px solid #2a3060;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        color: #7090d0;
        margin: 2px;
        font-family: 'Space Mono', monospace;
    }

    /* ── Legend ── */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #7080a0;
        margin-bottom: 6px;
    }
    .legend-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .legend-line-prereq {
        width: 24px; height: 2px;
        background: #F72585;
        flex-shrink: 0;
    }
    .legend-line-related {
        width: 24px; height: 2px;
        background: #4CC9F0;
        border-top: 2px dashed #4CC9F0;
        flex-shrink: 0;
    }

    /* ── File uploader override ── */
    [data-testid="stFileUploader"] {
        background: #0f1221;
        border: 1px dashed #2a3060;
        border-radius: 12px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4361EE, #7209B7);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Progress ── */
    .stProgress > div > div { background-color: #4361EE; }

    /* ── Selectbox ── */
    .stSelectbox label { color: #8890b0 !important; }

    /* ── Scrollable graph frame ── */
    .graph-frame {
        border: 1px solid #1e2646;
        border-radius: 14px;
        overflow: hidden;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "graph": None,
        "html_path": None,
        "analytics": {},
        "sections": [],
        "concepts": [],
        "processed": False,
        "pdf_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:Space Mono,monospace;font-size:1.1rem;"
        "color:#4CC9F0;margin-bottom:0.2rem;'>🧭 Curriculum Radar</p>"
        "<p style='font-size:0.75rem;color:#3a4060;margin-top:0;'>Syllabus Knowledge Graph</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload Academic PDF",
        type=["pdf"],
        help="Syllabus, textbook chapter, or academic paper",
    )

    st.markdown("#### ⚙️ Settings")
    edge_kw_threshold = st.slider(
        "Min shared keywords for prerequisite edge",
        min_value=1, max_value=10, value=3,
        help="Lower = more edges (noisier). Higher = stricter links.",
    )
    edge_sim_threshold = st.slider(
        "Semantic similarity threshold",
        min_value=0.20, max_value=0.90, value=0.45, step=0.05,
        help="Cosine similarity required to draw a 'related' edge.",
    )

    run_btn = st.button("🚀 Analyse PDF", use_container_width=True, disabled=uploaded is None)

    st.divider()

    # Legend
    st.markdown("#### 🗺️ Legend")
    st.markdown(
        "<div class='legend-item'>"
        "<div class='legend-line-prereq'></div>"
        "Prerequisite edge</div>"
        "<div class='legend-item'>"
        "<div class='legend-line-related'></div>"
        "Semantically related</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state["analytics"]:
        a = st.session_state["analytics"]
        st.markdown("#### 📊 Graph Stats")
        st.markdown(
            f"""
            <div class='metric-row'>
              <div class='metric-card'><div class='value'>{a.get('nodes','—')}</div><div class='label'>Nodes</div></div>
              <div class='metric-card'><div class='value'>{a.get('edges','—')}</div><div class='label'>Edges</div></div>
            </div>
            <div class='metric-row'>
              <div class='metric-card'><div class='value'>{a.get('components','—')}</div><div class='label'>Components</div></div>
              <div class='metric-card'><div class='value'>{a.get('density','—')}</div><div class='label'>Density</div></div>
            </div>
            <p style='font-size:0.75rem;color:#5a6080;margin-top:0.4rem;'>
              🏆 Most central: <b style='color:#c8d0f0'>{a.get('most_central','—')}</b><br>
              🔗 Top hub: <b style='color:#c8d0f0'>{a.get('top_hub','—')}</b>
            </p>
            """,
            unsafe_allow_html=True,
        )


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='cr-hero'>"
    "<h1>🧭 Curriculum Radar</h1>"
    "<p>Upload an academic PDF to extract core concepts and visualise "
    "how chapters, topics, and ideas interconnect as a live knowledge graph.</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ── RUN PIPELINE on button press ─────────────────────────────────────────────
if run_btn and uploaded is not None:
    # Inject runtime thresholds into logic module
    import logic as _logic
    _logic.EDGE_KEYWORD_THRESHOLD  = edge_kw_threshold
    _logic.EDGE_SIMILARITY_THRESHOLD = edge_sim_threshold

    pdf_bytes = uploaded.read()
    st.session_state["pdf_name"] = uploaded.name

    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    def on_progress(step, total, msg):
        pct = int(step / total * 100)
        progress_bar.progress(pct, text=msg)
        status_text.markdown(
            f"<p style='font-size:0.8rem;color:#5a6080;text-align:center'>{msg}</p>",
            unsafe_allow_html=True,
        )

    try:
        from logic import run_pipeline, detect_sections, extract_text_from_pdf, extract_concepts, load_nlp_model

        G, html_path, analytics = run_pipeline(pdf_bytes, progress_callback=on_progress)

        # Cache intermediate artefacts for node inspector
        pages    = extract_text_from_pdf(pdf_bytes)
        sections = detect_sections(pages)
        nlp      = load_nlp_model()
        concepts = [extract_concepts(s, nlp) for s in sections]

        st.session_state.update({
            "graph":     G,
            "html_path": html_path,
            "analytics": analytics,
            "sections":  sections,
            "concepts":  concepts,
            "processed": True,
        })
        progress_bar.progress(100, text="✅ Done!")
        status_text.empty()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Pipeline error: {exc}")
        st.exception(exc)


# ── DISPLAY RESULTS ───────────────────────────────────────────────────────────
if st.session_state["processed"] and st.session_state["html_path"]:
    G: nx.DiGraph      = st.session_state["graph"]
    html_path: str     = st.session_state["html_path"]
    sections: list     = st.session_state["sections"]
    concepts_list: list = st.session_state["concepts"]

    tab_graph, tab_sections, tab_export = st.tabs([
        "🌐 Knowledge Graph", "📚 Section Explorer", "📤 Export",
    ])

    # ── TAB 1: Interactive Graph ──────────────────────────────────────────
    with tab_graph:
        st.markdown(
            "<p style='font-size:0.83rem;color:#5a6080;margin-bottom:0.8rem;'>"
            "Hover over nodes to see summaries. Drag to rearrange. "
            "<b style='color:#F72585'>Red solid</b> = prerequisite. "
            "<b style='color:#4CC9F0'>Blue dashed</b> = related topic."
            "</p>",
            unsafe_allow_html=True,
        )
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=700, scrolling=False)

    # ── TAB 2: Section Explorer ───────────────────────────────────────────
    with tab_sections:
        col_list, col_detail = st.columns([1, 2])

        with col_list:
            st.markdown("#### Sections")
            titles = [s["title"] for s in sections]
            selected_title = st.radio(
                "Select a section",
                options=titles,
                label_visibility="collapsed",
            )

        with col_detail:
            idx = titles.index(selected_title)
            section = sections[idx]
            concepts = concepts_list[idx] if idx < len(concepts_list) else []

            node_attrs = G.nodes.get(idx, {})
            color = node_attrs.get("color", "#4CC9F0")
            page_range = node_attrs.get("page_range", "—")
            summary    = node_attrs.get("summary", section["text"][:400])

            st.markdown(
                f"<div class='node-card' style='border-left-color:{color}'>"
                f"<h4>{section['title']}</h4>"
                f"<p style='color:#4a5080;font-size:0.75rem;margin-bottom:0.6rem;'>📄 {page_range}</p>"
                f"<p>{summary}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Key concepts badges
            if concepts:
                badges = " ".join(
                    f"<span class='concept-badge'>{c}</span>"
                    for c in concepts[:12]
                )
                st.markdown(
                    f"<p style='font-size:0.78rem;color:#5a6080;"
                    f"margin:0.2rem 0 0.3rem;'>🔑 Key Concepts</p>{badges}",
                    unsafe_allow_html=True,
                )

            # Edges
            out_edges = list(G.out_edges(idx, data=True))
            in_edges  = list(G.in_edges(idx, data=True))

            if out_edges or in_edges:
                st.markdown(
                    "<p style='font-size:0.78rem;color:#5a6080;"
                    "margin-top:0.8rem;'>🔗 Connections</p>",
                    unsafe_allow_html=True,
                )
                for src, tgt, edata in out_edges:
                    tgt_label = G.nodes[tgt].get("label", f"Section {tgt}")
                    icon = "🔴" if edata.get("type") == "prerequisite" else "🔵"
                    shared = edata.get("shared", "")
                    tip = f" — *{shared}*" if shared else ""
                    st.markdown(
                        f"<span style='font-size:0.8rem;color:#8890b0'>"
                        f"{icon} → <b style='color:#c8d0f0'>{tgt_label}</b> "
                        f"({edata.get('type','link')}){tip}</span>",
                        unsafe_allow_html=True,
                    )
                for src, tgt, edata in in_edges:
                    src_label = G.nodes[src].get("label", f"Section {src}")
                    icon = "🔴" if edata.get("type") == "prerequisite" else "🔵"
                    st.markdown(
                        f"<span style='font-size:0.8rem;color:#8890b0'>"
                        f"{icon} ← <b style='color:#c8d0f0'>{src_label}</b> "
                        f"({edata.get('type','link')})</span>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 3: Export ─────────────────────────────────────────────────────
    with tab_export:
        st.markdown("#### 📤 Download Artefacts")

        col_a, col_b, col_c = st.columns(3)

        # HTML graph
        with col_a:
            with open(html_path, "rb") as f:
                graph_bytes = f.read()
            st.download_button(
                "⬇️ Knowledge Graph (HTML)",
                data=graph_bytes,
                file_name="curriculum_radar_graph.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Open in any browser — no server needed.")

        # GraphML
        with col_b:
            tmp_gml = tempfile.NamedTemporaryFile(suffix=".graphml", delete=False)
            # Strip non-serialisable attrs
            export_G = nx.DiGraph()
            for n, d in G.nodes(data=True):
                export_G.add_node(n, **{k: str(v) for k, v in d.items()})
            for s, t, d in G.edges(data=True):
                export_G.add_edge(s, t, **{k: str(v) for k, v in d.items()})
            nx.write_graphml(export_G, tmp_gml.name)
            with open(tmp_gml.name, "rb") as f:
                gml_bytes = f.read()
            st.download_button(
                "⬇️ Graph Data (GraphML)",
                data=gml_bytes,
                file_name="curriculum_radar.graphml",
                mime="application/xml",
                use_container_width=True,
            )
            st.caption("Import into Gephi, yEd, or Cytoscape.")

        # CSV edge list
        with col_c:
            import pandas as pd
            rows = []
            for src, tgt, edata in G.edges(data=True):
                rows.append({
                    "source": G.nodes[src].get("label", src),
                    "target": G.nodes[tgt].get("label", tgt),
                    "type":   edata.get("type", ""),
                    "weight": edata.get("weight", ""),
                    "shared_concepts": edata.get("shared", ""),
                })
            df = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Edge List (CSV)",
                data=df.to_csv(index=False).encode(),
                file_name="curriculum_radar_edges.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Use in Excel, Python, or R.")

        # Section summary table
        st.divider()
        st.markdown("#### 📋 Section Summary Table")
        summary_rows = []
        for idx, section in enumerate(sections):
            c_list = concepts_list[idx] if idx < len(concepts_list) else []
            n_attrs = G.nodes.get(idx, {})
            summary_rows.append({
                "Section": section["title"],
                "Pages": n_attrs.get("page_range", "—"),
                "Key Concepts": ", ".join(c_list[:8]),
                "Out-Degree": G.out_degree(idx),
                "In-Degree":  G.in_degree(idx),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center;padding:4rem 2rem;'>
          <div style='font-size:4rem;margin-bottom:1rem;'>📄→🌐</div>
          <p style='color:#3a4060;font-size:1rem;max-width:420px;margin:0 auto;'>
            Upload an academic PDF in the sidebar and click
            <b style='color:#4361EE'>Analyse PDF</b> to generate your
            interactive knowledge graph.
          </p>
          <br>
          <p style='color:#2a3050;font-size:0.8rem;max-width:420px;margin:0 auto;'>
            Works with syllabi, textbooks, lecture notes, and research papers.
            Concepts and prerequisite relationships are extracted automatically.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
