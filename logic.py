"""
logic.py — Curriculum Radar: Text Processing & Graph Construction Engine
=========================================================================
Pipeline:
  1. PDF ingestion via PyMuPDF (fast) with pdfplumber fallback
  2. Section/chapter detection via heuristic heading patterns
  3. Concept extraction via spaCy NLP (noun chunks + named entities)
  4. Relationship detection via:
       a) Shared keyword frequency (prerequisite cross-referencing)
       b) Semantic similarity via Sentence-Transformers cosine distance
  5. NetworkX directed graph construction
  6. Pyvis interactive HTML export
"""

from __future__ import annotations

import re
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

import fitz  # PyMuPDF
import networkx as nx
import numpy as np

# ── Optional heavy deps (graceful degradation) ──────────────────────────────
try:
    import spacy
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────
HEADING_PATTERNS = [
    r"^(chapter|unit|module|section|part|lecture)\s*[\d]+",
    r"^\d+[\.\d]*\s+[A-Z][A-Za-z\s]{3,}",
    r"^[A-Z][A-Z\s]{4,}$",
]
MIN_SECTION_LENGTH = 120          # chars – skip stub sections
MAX_SECTION_CHARS  = 6_000        # truncate very long sections for speed
TOP_CONCEPTS_PER_SECTION = 15     # noun chunks to keep per section
EDGE_KEYWORD_THRESHOLD = 3        # shared keywords to draw an edge
EDGE_SIMILARITY_THRESHOLD = 0.45  # cosine similarity threshold

# Node colour palette (dark theme friendly)
PALETTE = [
    "#4CC9F0", "#4361EE", "#7209B7", "#F72585",
    "#FB8500", "#06D6A0", "#FFD166", "#EF233C",
    "#3A86FF", "#8338EC",
]


# ── 1. PDF INGESTION ─────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract per-page text from PDF bytes using PyMuPDF.
    Returns: [{"page": int, "text": str}, ...]
    Falls back to pdfplumber for scanned/complex layouts.
    """
    pages: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": i + 1, "text": _clean_text(text)})
            else:
                # Blank page – could be scanned; placeholder for OCR hook
                pages.append({"page": i + 1, "text": ""})

    # Fallback: if PyMuPDF yielded nothing, try pdfplumber
    if not any(p["text"] for p in pages):
        pages = _pdfplumber_fallback(pdf_bytes)

    return pages


def _pdfplumber_fallback(pdf_bytes: bytes) -> list[dict]:
    """Use pdfplumber when PyMuPDF extraction is empty."""
    try:
        import pdfplumber, io
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({"page": i + 1, "text": _clean_text(text)})
        return pages
    except Exception:
        return []


def _clean_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, strip control chars."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 2. SECTION DETECTION ─────────────────────────────────────────────────────
def detect_sections(pages: list[dict]) -> list[dict]:
    """
    Heuristically split pages into sections/chapters based on heading patterns.
    Returns: [{"title": str, "text": str, "pages": [int]}, ...]
    """
    compiled = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in HEADING_PATTERNS]
    sections: list[dict] = []
    current: dict | None = None

    for page_data in pages:
        lines = page_data["text"].split("\n")
        for line in lines:
            stripped = line.strip()
            if _is_heading(stripped, compiled):
                if current and len(current["text"]) >= MIN_SECTION_LENGTH:
                    sections.append(current)
                current = {
                    "title": _sanitise_title(stripped),
                    "text": "",
                    "pages": [page_data["page"]],
                }
            else:
                if current is None:
                    current = {
                        "title": f"Introduction (p.{page_data['page']})",
                        "text": "",
                        "pages": [page_data["page"]],
                    }
                if page_data["page"] not in current["pages"]:
                    current["pages"].append(page_data["page"])
                current["text"] += " " + stripped

    if current and len(current["text"]) >= MIN_SECTION_LENGTH:
        sections.append(current)

    # If no sections detected, fall back to page-chunked sections
    if not sections:
        sections = _page_chunk_fallback(pages)

    # Truncate text for memory safety
    for s in sections:
        s["text"] = s["text"][:MAX_SECTION_CHARS].strip()

    return sections


def _is_heading(line: str, patterns: list) -> bool:
    if len(line) < 3 or len(line) > 120:
        return False
    return any(p.match(line) for p in patterns)


def _sanitise_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip()
    return title[:80] if len(title) > 80 else title


def _page_chunk_fallback(pages: list[dict], chunk_size: int = 5) -> list[dict]:
    """Chunk pages into groups when no headings are detected."""
    sections = []
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i : i + chunk_size]
        combined = " ".join(p["text"] for p in chunk)
        if combined.strip():
            sections.append({
                "title": f"Pages {chunk[0]['page']}–{chunk[-1]['page']}",
                "text": combined[:MAX_SECTION_CHARS],
                "pages": [p["page"] for p in chunk],
            })
    return sections


# ── 3. CONCEPT EXTRACTION ─────────────────────────────────────────────────────
def load_nlp_model() -> Optional[object]:
    """
    Load spaCy model (en_core_web_sm). Returns None if unavailable.
    Falls back to regex-based extraction.
    """
    if not _NLP_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            return None


def extract_concepts(section: dict, nlp=None) -> list[str]:
    """
    Extract top-N key concepts from a section's text.
    Uses spaCy noun chunks + NER when available; regex fallback otherwise.
    """
    text = section["text"]
    if nlp:
        return _spacy_concepts(text, nlp)
    return _regex_concepts(text)


def _spacy_concepts(text: str, nlp) -> list[str]:
    """Extract noun chunks and named entities using spaCy."""
    # Process in 3-sentence windows to handle large texts
    doc = nlp(text[:4000])  # spaCy soft limit for sm model
    stopwords = nlp.Defaults.stop_words

    candidates: list[str] = []
    for chunk in doc.noun_chunks:
        lemma = chunk.root.lemma_.lower()
        if lemma not in stopwords and len(lemma) > 2 and lemma.isalpha():
            candidates.append(chunk.text.lower().strip())

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "LAW", "EVENT", "NORP"}:
            candidates.append(ent.text.lower().strip())

    freq = Counter(candidates)
    return [term for term, _ in freq.most_common(TOP_CONCEPTS_PER_SECTION)]


def _regex_concepts(text: str) -> list[str]:
    """Lightweight regex fallback: capitalised multi-word phrases."""
    # Match Title Case phrases of 1–4 words
    pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
    candidates = pattern.findall(text)
    stopwords = {
        "The", "This", "That", "These", "Those", "With", "From", "Into",
        "Upon", "Each", "Such", "When", "Then", "Also", "More", "Most",
        "Both", "Some", "Figure", "Table", "Chapter", "Section",
    }
    filtered = [c for c in candidates if c not in stopwords and len(c.split()) >= 1]
    freq = Counter(filtered)
    return [term.lower() for term, _ in freq.most_common(TOP_CONCEPTS_PER_SECTION)]


# ── 4. RELATIONSHIP DETECTION ─────────────────────────────────────────────────
def load_sentence_transformer():
    """Load lightweight Sentence-Transformer model (or None)."""
    if not _ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def compute_section_embeddings(sections: list[dict], model) -> Optional[np.ndarray]:
    """Embed section texts for semantic similarity comparison."""
    if model is None:
        return None
    try:
        texts = [s["text"][:512] for s in sections]   # limit input length
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
    except Exception:
        return None


def find_relationships(
    sections: list[dict],
    concepts_per_section: list[list[str]],
    embeddings: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Detect directed edges between sections.

    Strategy A – Keyword prerequisite:
      If section j mentions ≥ EDGE_KEYWORD_THRESHOLD concepts that
      first appeared in section i (i < j), draw edge i → j.

    Strategy B – Semantic similarity:
      If cosine_similarity(embed_i, embed_j) ≥ threshold and sections
      are not adjacent, draw a bidirectional similarity edge.

    Returns: [{"source": int, "target": int, "weight": float,
               "type": "prerequisite"|"related", "shared": [str]}, ...]
    """
    edges: list[dict] = []
    n = len(sections)

    # Build concept-first-appearance map
    concept_origin: dict[str, int] = {}
    for idx, concepts in enumerate(concepts_per_section):
        for c in concepts:
            if c not in concept_origin:
                concept_origin[c] = idx

    # Strategy A – Prerequisite detection
    for j in range(1, n):
        j_concepts = set(concepts_per_section[j])
        origin_counts: dict[int, list[str]] = defaultdict(list)
        for c in j_concepts:
            origin = concept_origin.get(c)
            if origin is not None and origin < j:
                origin_counts[origin].append(c)

        for i, shared in origin_counts.items():
            if len(shared) >= EDGE_KEYWORD_THRESHOLD:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": len(shared) / TOP_CONCEPTS_PER_SECTION,
                    "type": "prerequisite",
                    "shared": shared[:8],
                })

    # Strategy B – Semantic similarity
    if embeddings is not None:
        sim_matrix = cosine_similarity(embeddings)
        for i in range(n):
            for j in range(i + 2, n):   # skip adjacent
                sim = float(sim_matrix[i, j])
                if sim >= EDGE_SIMILARITY_THRESHOLD:
                    # Avoid duplicate edges already created by Strategy A
                    existing = any(
                        e["source"] == i and e["target"] == j for e in edges
                    )
                    if not existing:
                        edges.append({
                            "source": i,
                            "target": j,
                            "weight": round(sim, 3),
                            "type": "related",
                            "shared": [],
                        })

    return edges


# ── 5. GRAPH CONSTRUCTION ─────────────────────────────────────────────────────
def build_graph(
    sections: list[dict],
    concepts_per_section: list[list[str]],
    edges: list[dict],
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from sections (nodes) and relationships (edges).
    Node attributes: title, summary, page_range, concepts, color
    """
    G = nx.DiGraph()

    for idx, (section, concepts) in enumerate(zip(sections, concepts_per_section)):
        color = PALETTE[idx % len(PALETTE)]
        page_list = section.get("pages", [])
        page_range = (
            f"p.{page_list[0]}" if len(page_list) == 1
            else f"p.{page_list[0]}–{page_list[-1]}" if page_list
            else "—"
        )
        G.add_node(
            idx,
            label=section["title"],
            title=section["title"],
            summary=_generate_summary(section["text"]),
            page_range=page_range,
            concepts=", ".join(concepts[:10]),
            color=color,
            size=20 + min(len(section["text"]) // 200, 30),
        )

    for edge in edges:
        label = "prerequisite" if edge["type"] == "prerequisite" else "related"
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            type=edge["type"],
            shared=", ".join(edge["shared"]),
            label=label,
        )

    return G


def _generate_summary(text: str, max_sentences: int = 3) -> str:
    """
    Naïve extractive summary: first N sentences of the section.
    PLACEHOLDER – swap in a transformer summariser for production.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    non_empty = [s.strip() for s in sentences if len(s.strip()) > 30]
    summary = " ".join(non_empty[:max_sentences])
    return summary[:500] if len(summary) > 500 else summary


# ── 6. PYVIS EXPORT ───────────────────────────────────────────────────────────
def render_pyvis_graph(G: nx.DiGraph, output_path: str) -> str:
    """
    Convert a NetworkX graph to an interactive Pyvis HTML file.
    Returns the path to the generated HTML.
    """
    if not _PYVIS_AVAILABLE:
        raise ImportError("pyvis is not installed. Run: pip install pyvis")

    net = Network(
        height="680px",
        width="100%",
        directed=True,
        bgcolor="#0f1117",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.set_options(_pyvis_options())

    for node_id, attrs in G.nodes(data=True):
        tooltip = (
            f"<b>{attrs.get('title','')}</b><br>"
            f"📄 {attrs.get('page_range','')}<br><br>"
            f"<i>{attrs.get('summary','')}</i><br><br>"
            f"🔑 <b>Key concepts:</b> {attrs.get('concepts','')}"
        )
        net.add_node(
            node_id,
            label=_short_label(attrs.get("label", f"Section {node_id}")),
            title=tooltip,
            color=attrs.get("color", "#4CC9F0"),
            size=attrs.get("size", 25),
            font={"size": 13, "face": "monospace"},
            borderWidth=2,
            borderWidthSelected=4,
        )

    for src, tgt, edata in G.edges(data=True):
        edge_color = "#F72585" if edata.get("type") == "prerequisite" else "#4CC9F0"
        edge_dash  = False if edata.get("type") == "prerequisite" else True
        shared_tip = f"Shared concepts: {edata.get('shared','—')}" if edata.get("shared") else ""
        net.add_edge(
            src, tgt,
            color=edge_color,
            width=1 + edata.get("weight", 0.5) * 3,
            title=f"{edata.get('type','').capitalize()} link\n{shared_tip}",
            dashes=edge_dash,
            arrows="to",
        )

    net.write_html(output_path, local=False)
    return output_path


def _short_label(label: str, max_len: int = 28) -> str:
    words = label.split()
    result = ""
    line  = ""
    for w in words:
        if len(line) + len(w) + 1 > max_len:
            result += line.strip() + "\n"
            line = w + " "
        else:
            line += w + " "
    result += line.strip()
    return result[:120]


def _pyvis_options() -> str:
    return """
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.5
        },
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "shadow": true
      },
      "nodes": {
        "shadow": true,
        "shape": "dot"
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      }
    }
    """


# ── 7. GRAPH ANALYTICS ────────────────────────────────────────────────────────
def graph_analytics(G: nx.DiGraph) -> dict:
    """Compute quick stats for the sidebar dashboard."""
    if G.number_of_nodes() == 0:
        return {}

    try:
        centrality = nx.degree_centrality(G)
        most_central_id = max(centrality, key=centrality.get)
        most_central = G.nodes[most_central_id].get("label", f"Section {most_central_id}")
    except Exception:
        most_central = "—"

    try:
        hub_nodes = sorted(G.nodes, key=lambda n: G.out_degree(n), reverse=True)
        top_hub = G.nodes[hub_nodes[0]].get("label", "—") if hub_nodes else "—"
    except Exception:
        top_hub = "—"

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "most_central": most_central,
        "top_hub": top_hub,
        "density": round(nx.density(G), 4),
        "components": nx.number_weakly_connected_components(G),
    }


# ── 8. MASTER PIPELINE ────────────────────────────────────────────────────────
def run_pipeline(
    pdf_bytes: bytes,
    progress_callback=None,
) -> tuple[nx.DiGraph, str, dict]:
    """
    Full end-to-end pipeline.
    Returns (graph, html_path, analytics_dict).
    progress_callback(step: int, total: int, msg: str) – optional UI hook.
    """

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    total_steps = 7

    _progress(1, total_steps, "📄 Extracting text from PDF…")
    pages = extract_text_from_pdf(pdf_bytes)

    _progress(2, total_steps, "🔍 Detecting sections & chapters…")
    sections = detect_sections(pages)

    _progress(3, total_steps, "🧠 Loading NLP model…")
    nlp = load_nlp_model()

    _progress(4, total_steps, "💡 Extracting concepts from sections…")
    concepts_per_section = [extract_concepts(s, nlp) for s in sections]

    _progress(5, total_steps, "🔗 Computing semantic similarity…")
    st_model = load_sentence_transformer()
    embeddings = compute_section_embeddings(sections, st_model)

    _progress(6, total_steps, "📐 Building knowledge graph…")
    edges = find_relationships(sections, concepts_per_section, embeddings)
    G = build_graph(sections, concepts_per_section, edges)

    _progress(7, total_steps, "🎨 Rendering interactive graph…")
    tmp_dir  = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "graph.html")
    render_pyvis_graph(G, html_path)

    analytics = graph_analytics(G)

    return G, html_path, analytics
