---
title: MovieMind
emoji: 🎬
sdk: docker
app_port: 7860
colorFrom: red
colorTo: gray
pinned: true
---

# 🎬 MovieMind — Conversational Movie Recommender

**Hybrid retrieval + LLM‑enhanced NLU + constraint‑aware ranking over 1.1 million movies.**

[![Hugging Face](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-red)](https://huggingface.co/spaces/PavanBoorla/movie-mind)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://github.com/pavanramkalyanboorla-debug/movie-recommender/blob/main/Dockerfile)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 What It Does

MovieMind lets you search for movies using **natural language** — the way you'd talk to a friend. Type *"Christopher Nolan movies without Oppenheimer"*, *"marvel movies after 2020"*, *"rom‑com movies"*, or *"suggest awesome heist movies"* and get relevant, ranked recommendations with AI‑generated explanations and streaming availability badges.

| Query | What Happens |
|---|---|
| `Christopher Nolan movies` | Director filmography pull → ranked by popularity |
| `Tom Cruise movies` | Actor filmography pull → filtered by vote count |
| `marvel movies after 2020` | Franchise keyword pull → year‑filtered |
| `movies like Dune` | Genre‑similarity boost on semantic search |
| `horror movies after 2020` | Semantic search + year constraint + genre filter |
| `rom com movies` | Semantic search + genre filter + documentary penalty |
| `Christopher Nolan movies without Oppenheimer` | Director pull + title exclusion |

---

## 🖼️ Demo

<p align="center">
  <img src="assets/demo.gif" alt="MovieMind Demo" width="80%">
</p>

---

## 🏗️ Architecture

```text
User Query
   │
   ▼
┌──────────────────────────────────┐
│ Structured NLU (Groq JSON)        │ ← intent, entities, exclusions
│ + JSON Schema Validation          │
└──────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│ Heuristic Validation              │ ← entity counts, confidence
└──────────────────────────────────┘
   │
   ├── director intent → _pull_director_films()
   ├── actor intent    → _pull_actor_films()
   ├── franchise intent→ _pull_franchise_films()
   └── semantic        → FAISS hybrid retrieval (top‑200)
   │
   ▼
┌──────────────────────────────────┐
│ Constraint Filters                │ ← year / genre / exclusion / vote guards
└──────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│ Hybrid Scoring + Reranking        │ ← similarity + rating + popularity + boosts
└──────────────────────────────────┘
   │
   ▼
Final Results (+ streaming badges)
```

### Key Design Decisions

| Component | Approach | Why |
|---|---|---|
| **Retrieval** | FAISS (IndexFlatIP) over Sentence‑Transformer embeddings | Fast inner‑product search over 1 M vectors |
| **Embedding** | `all-MiniLM-L6-v2` (80 MB) | CPU‑friendly, good for short text |
| **NLU** | Groq `llama-3.1-8b-instant` + RobustParser fallback | LLM gives best intent detection; rule‑based parser works without API key |
| **Person queries** | Exact filmography pull (director/actor columns) | Guarantees recall for named‑person queries |
| **Franchise queries** | Keyword pull over title/overview/genres | Direct retrieval without relying on embedding similarity |
| **Scoring** | Weighted hybrid: similarity + rating + popularity + 7 boost/penalty factors | Balances relevance with quality |
| **Filtering** | Year range, genre, title/overview exclusions, vote‑count guards | Removes noise before ranking |
| **Frontend** | Streamlit (Netflix‑inspired dark theme) | Fast to build, looks professional |
| **Backend** | FastAPI + Uvicorn | Fast, async, well‑suited for ML serving |
| **Deployment** | Single‑container Docker → Hugging Face Spaces | Zero‑infra, free tier |

### Boost & Penalty System

| Signal | Weight / Effect |
|---|---|
| Similarity (cosine via FAISS) | User‑tunable (`w_sim`, default 0.6) |
| Rating | User‑tunable (`w_rating`, default 0.2) |
| Popularity | User‑tunable (`w_pop`, default 0.2) |
| Keyword boost (non‑pull queries) | +0.10 per word hit in soup |
| Franchise keyword boost | +0.05 per franchise word in soup |
| Director boost (FAISS fallback) | +0.50 per matching director entity |
| Actor boost (FAISS fallback) | +0.50 per matching actor entity |
| Genre‑similarity boost (movies‑like‑X) | +0.50 for shared genres |
| Year proximity boost | +0.30 if within ±2 years of target |
| Recency boost | +0.15 (≤2 yr), +0.05 (≤5 yr) |
| Documentary penalty (no doc intent) | −0.30 for documentaries |
| Movie priority (no doc intent) | +0.10 for non‑documentaries |

---

## 📦 Project Structure

```
movie-recommender/
├── app/
│   ├── main.py                  # FastAPI backend (/recommend, /health)
│   └── streamlit_app.py         # Streamlit frontend (Netflix‑style, streaming badges)
├── src/
│   ├── constants.py              # Paths, column mappings, model name
│   ├── pipeline/
│   │   ├── build_pipeline.py     # Orchestrates artifact creation
│   │   └── predict_pipeline.py   # Core recommendation engine
│   ├── components/
│   │   ├── data_loader.py        # Loads raw CSV
│   │   ├── data_preprocessor.py  # Cleans, extracts director, cast, genres, builds soup
│   │   └── index_builder.py      # Builds FAISS index + TF‑IDF vectorizer
│   └── utils/
│       ├── exceptions.py
│       └── logger.py
├── benchmark/
│   ├── get_titles.py             # Interactive query runner
│   ├── run_eval.py               # Recall@k / Precision@k evaluation
│   ├── test_queries.json         # Curated ground‑truth queries (movie IDs)
│   ├── test_queries_large.json   # 50 diverse test queries
│   ├── capture_gt.py             # Captures pipeline output as ground truth
│   └── diagnose.py               # Cast/director column diagnostics
├── scripts/
│   └── build_streaming_lookup.py # Creates streaming_lookup.json from Ruchi798 dataset
├── notebooks/
│   ├── artifact-builder-alan-vourch-clean-dataset.ipynb
│   ├── 01_EDA_and_Data_Preparation_MR_.ipynb
│   ├── 02_Retrieval_Engine_&_Semantic_Search_MR_.ipynb
│   └── 03_Ranking_+_NDCG_+_Ablation_MR_.ipynb
├── artifacts/                    # Git‑ignored; built offline, uploaded separately
│   ├── movies_processed_final.parquet
│   ├── movies_faiss.index
│   ├── tfidf_vectorizer.pkl
│   ├── model_name.txt
│   └── streaming_lookup.json
├── Dockerfile                    # Multi‑stage, CPU‑only PyTorch
├── pyproject.toml                # Dependencies (uv)
├── uv.lock
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 🚀 Quick Start (Local Docker)

```powershell
# 1. Clone the repo
git clone https://github.com/pavanramkalyanboorla-debug/movie-recommender.git
cd movie-recommender

# 2. Build the Docker image (CPU‑only, ~2.3 GB after build)
docker build -t moviemind .

# 3. Run (with Groq API key for LLM explanations)
docker run -d -p 7860:7860 --name movie-mind -e GROQ_API_KEY="gsk_your_key_here" moviemind

# 4. Open your browser
#    http://localhost:7860

# 5. Stop when done
docker rm -f movie-mind
```

> ⚡ **Without a Groq API key?** Run without `-e GROQ_API_KEY` — the robust parser still handles director, actor, genre, and year queries correctly.

---

## 🧠 Dataset & Artifact Building

| Component | Source | Size |
|---|---|---|
| Movie metadata | [Alan Vourch TMDB Daily Updates](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) | ~1.2 M movies |
| Streaming availability | [Ruchi798 Movies on Streaming Platforms](https://www.kaggle.com/datasets/ruchi798/movies-on-netflix-prime-video-hulu-and-disney) | ~9,500 movies |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 80 MB |

**Preprocessing steps (run in Kaggle notebook):**

1. Lowercase column names & clean titles
2. Extract director (plain string)
3. Extract top‑5 cast members (JSON‑aware parser)
4. Normalize genres (JSON → space‑separated)
5. Build semantic "soup": `director + title + cast + genres + overview + keywords`
6. Deduplicate by `title` + `year`
7. Encode soups → FAISS index → TF‑IDF vectorizer

Artifacts are built offline on Kaggle and uploaded to Hugging Face Spaces via `hf upload` (stored via Xet, not Git).

---

## 📊 Evaluation

### Benchmark Suite (`benchmark/`)

| Script | Purpose |
|---|---|
| `run_eval.py` | Compute Recall@10 & Precision@10 against ground‑truth IDs |
| `get_titles.py` | Interactive query runner (supports `--use_llm` and `--queries`) |
| `capture_gt.py` | Captures pipeline output as ground truth for alignment |
| `diagnose.py` | Inspects cast/director columns for debugging |

### Results (Full Capacity — with Groq LLM)

| Query | Recall@10 | Precision@10 |
|---|---|---|
| Christopher Nolan movies | 1.00 | 1.00 |
| Tom Cruise movies | 1.00 | 0.30¹ |
| movies like Dune | 1.00 | 1.00 |
| rom com movies | 1.00 | 1.00 |
| horror movies after 2020 | 1.00 | 1.00 |
| **Average** | **1.000** | **0.860** |

> ¹ Tom Cruise precision is limited by cast coverage in the Alan Vourch dataset (only 23 of his films have his name in the top‑5 cast). Increasing `top_n` in `clean_cast()` from 5 to 15 in the Kaggle notebook would resolve this.

---

## 🔬 Key Features Evolved Through Iteration

1. **Director‑first heuristic** — prevents "Martin Scorsese movies" from being treated as an actor query (checks dataset counts).
2. **JSON‑aware cast parser** — correctly extracts actor names from JSON arrays when present.
3. **Vote‑count guards** — filters out low‑quality metadata entries (`vote_count > 50` for directors, `> 5` for actors).
4. **Extended regex for person detection** — catches *"Christopher Nolan movies without Oppenheimer"*.
5. **Franchise pull (`_pull_franchise_films`)** — directly retrieves Marvel/DC/Pixar films.
6. **Recency boost** — favours recent movies for generic queries.
7. **Documentary penalty** — prevents non‑documentary queries from returning documentaries.
8. **LLM JSON validation** — repairs malformed Groq output.
9. **Hybrid scoring** — weighted combination of 10+ signals.
10. **Streaming badges** — green chips showing Netflix/Hulu/Prime/Disney+ availability.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Embeddings** | Sentence‑Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (IndexFlatIP, L2‑normalized) |
| **NLU** | Groq API (`llama-3.1-8b-instant`) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (Netflix‑inspired dark theme) |
| **Container** | Multi‑stage Docker (CPU‑only PyTorch) |
| **Package Manager** | `uv` (fast, lock‑file based) |
| **Deployment** | Hugging Face Spaces (Docker SDK) |
| **Artifact Building** | Kaggle (CPU, ~1‑2 hours) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🔮 Future Projects (After MovieMind)

- **Cost‑Aware Decision Intelligence System** — profit‑maximizing loan/pricing engine with asymmetric cost evaluation.
- **Self‑Evaluating Agentic LLM System** — multi‑agent system that researches, writes, critiques itself, and self‑corrects.

---

## 🙋‍♂️ About the Author

**Pavan Ram Kalyan Boorla** — Civil‑engineering graduate transitioning into ML Engineering. This is portfolio project #3 of 5.

- [GitHub](https://github.com/pavanramkalyanboorla-debug)
- [Hugging Face](https://huggingface.co/PavanBoorla)
- [LinkedIn](https://linkedin.com/in/pavanboorla)

---

<p align="center">
  <b>Built with ❤️ and a lot of debugging in Docker containers.</b>
</p>