# 🧭 Curriculum Radar — Syllabus Knowledge Graph

> Upload an academic PDF. Get an interactive, AI-powered knowledge graph showing how topics, chapters, and concepts interconnect.

---

## Project Structure

```
curriculum_radar/
├── app.py            # Streamlit UI
├── logic.py          # PDF processing + graph engine
├── requirements.txt  # Python dependencies
└── .streamlit/
    └── config.toml   # Streamlit dark theme config
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## How It Works

| Step | What happens |
|------|-------------|
| **1. Upload PDF** | PyMuPDF extracts per-page text; pdfplumber is used as fallback |
| **2. Section Detection** | Heuristic regex patterns identify chapter/section headings |
| **3. Concept Extraction** | spaCy noun chunks + NER find key terms per section |
| **4. Relationship Detection** | Two strategies run in parallel: |
| | **A) Keyword prerequisite** — if section J reuses ≥ N terms introduced in section I, draw edge I→J |
| | **B) Semantic similarity** — Sentence-Transformers cosine distance between section embeddings |
| **5. Graph Construction** | NetworkX DiGraph with node metadata (summary, page range, concepts) |
| **6. Visualisation** | Pyvis renders an interactive HTML graph (hover, drag, zoom) |

---

## AI/NLP Stack

| Component | Library | Notes |
|-----------|---------|-------|
| PDF extraction | `PyMuPDF` + `pdfplumber` | Handles text & layout |
| Concept extraction | `spaCy en_core_web_sm` | Noun chunks + NER |
| Semantic similarity | `sentence-transformers` | `all-MiniLM-L6-v2` |
| Graph logic | `NetworkX` | DiGraph with attrs |
| Graph rendering | `Pyvis` | Self-contained HTML |

---

## Extending the AI Logic

`logic.py` has clearly marked **PLACEHOLDER** comments for upgrading:

- **Summarisation** → swap `_generate_summary()` with a transformer model (e.g., `facebook/bart-large-cnn`)
- **Concept extraction** → replace `_spacy_concepts()` with a fine-tuned NER model
- **Relationship detection** → add LLM-based inference (GPT / Claude) for deeper prerequisite reasoning
- **OCR support** → `pytesseract` hook in `_page_chunk_fallback()` for scanned PDFs

---

## Adjustable Settings (Sidebar)

| Setting | Default | Effect |
|---------|---------|--------|
| Min shared keywords | 3 | Lower = more edges drawn |
| Similarity threshold | 0.45 | Lower = more "related" edges |

---

## Export Formats

- **HTML** — self-contained interactive graph (no server required)
- **GraphML** — import into Gephi, yEd, Cytoscape
- **CSV** — edge list for Excel / R / Python analysis
