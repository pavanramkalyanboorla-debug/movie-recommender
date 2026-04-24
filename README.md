---
title: MovieMind
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 MovieMind — Conversational Movie Recommender with AI Explanations

**Understands what you *mean*, not just what you *type.***  
**Tells you *why* each movie is right for you.**

<p align="center">
  <a href="https://huggingface.co/spaces/PavanBoorla/movie-mind"><strong>🚀 Live Demo</strong></a>
  ·
  <a href="#-quick-start"><strong>⚡ Quick Start</strong></a>
  ·
  <a href="#-project-structure"><strong>📁 Project Structure</strong></a>
  ·
  <a href="#-how-it-works"><strong>🧠 Architecture</strong></a>
</p>

---

## ✨ What is MovieMind?

Most movie recommenders force you into filters and dropdowns. MovieMind lets you **talk naturally**:

> *"A mind‑bending sci‑fi movie like Inception, but not too violent"*  
> *"Light‑hearted romantic comedies from the 90s"*  
> *"Batman movies without the Joker"*

It **retrieves** the most relevant movies from a large catalog, **ranks** them by quality and popularity, and **explains** each recommendation with an AI‑generated sentence that tells you **why** it was chosen.

This isn't just a notebook — it's a **deployed, production‑style recommendation system**.

---

## 🎯 Why This Project Stands Out

| Problem with typical recommenders | How MovieMind solves it |
|-----------------------------------|--------------------------|
| Keyword‑based — fail on vague queries | **Semantic search** with Sentence‑Transformers |
| Black‑box — no justification | **Groq LLM** generates a human‑readable explanation for every recommendation |
| Academic only — never deployed | **Live on Hugging Face Spaces**, Docker‑containerised |
| Quality‑blind retrieval | **Two‑stage architecture** with hybrid scoring (similarity + rating + popularity) |
| One‑size‑fits‑all ranking | **User‑adjustable weights** via the Streamlit sidebar |

---

## 🧠 How It Works (Two‑Stage Architecture)
User Query (free text)\
│\
▼\
┌─────────────────────────────────────────┐\
│ Groq LLM (Query Parser) │ ← Extracts genre, year, exclusions, etc.
└─────────────────────────────────────────┘\
│\
▼\
┌─────────────────────────────────────────┐\
│ FAISS Semantic Retrieval (Stage 1) │ ← Finds 200 most relevant candidates
└─────────────────────────────────────────┘\
│\
▼\
┌─────────────────────────────────────────┐\
│ Hybrid Ranking (Stage 2) │ ← Scores by similarity + rating + popularity
└─────────────────────────────────────────┘\
│\
▼\
┌─────────────────────────────────────────┐\
│ Groq LLM (Explanation Generator) │ ← "Because it shares the mind‑bending plot..."
└─────────────────────────────────────────┘\
│\
▼\
Ranked list + explanations returned to UI


---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Natural Language Understanding** | Groq Llama-3 extracts filters like genre, year, "not", and "similar to" from free‑text queries |
| **Semantic Search** | Sentence‑Transformers (`all-MiniLM-L6-v2`) + FAISS index for sub‑second retrieval |
| **Hybrid Scoring** | Combines semantic similarity, average rating, and log‑scaled popularity into a single score |
| **Explainable AI** | Every recommendation includes a concise sentence explaining the match |
| **Interactive UI** | Streamlit interface with sliders to adjust ranking weights in real‑time |
| **Docker‑Ready** | Single `Dockerfile` builds the entire app — no separate API or database needed |
| **Free‑Tier Deployed** | Live on Hugging Face Spaces (Docker runtime) — no cost, always on |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Embeddings** | Sentence‑Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **LLM Integration** | Groq (Llama 3 8B) |
| **Data Processing** | Pandas, NumPy, Scikit‑learn |
| **Containerisation** | Docker |
| **Package Management** | `uv` (fast Python package & project manager) |
| **Deployment** | Hugging Face Spaces |

---

## 📁 Project Structure
movie-recommender/\
├── .env.example # Template for API keys\
├── .gitignore\
├── pyproject.toml # uv dependencies\
├── uv.lock # Locked dependency versions\
├── Dockerfile # Multi‑stage Docker build\
├── recommender.py # Core logic: loading artifacts, FAISS search, hybrid scoring\
├── streamlit_app.py # Streamlit UI\
├── pipeline/\
│ └── build_artifacts.py # Data preprocessing + FAISS index + TF‑IDF vectorizer\
├── data/ # (gitignored) raw CSV files\
└── artifacts/ # (gitignored) generated models & index

---

## ⚡ Quick Start (Local)

### Prerequisites

- Python ≥ 3.11
- `uv` installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- Docker (optional, for containerised run)

### Step 1 — Clone and add data

```bash
git clone https://github.com/pavanramkalyanboorla-debug/movie-recommender.git
cd movie-recommender
```
Place these CSV files in the data/ folder:

|File|	Source|
|-----|-------|
|ratings.csv|	MovieLens 20M|
|movies.csv|	MovieLens 20M|
|links.csv|	MovieLens 20M|
|tmdb_5000_movies.csv|	TMDB 5000 on Kaggle|

### Step 2 — Set your Groq API key
```bash
cp .env.example .env
# Edit .env with your Groq key from https://console.groq.com
```

### Step 3 — Run the pipeline
This generates the processed data, FAISS index, and TF‑IDF vectorizer:
```bash
uv run python pipeline/build_artifacts.py
```
⏳ This downloads the Sentence‑Transformer model and encodes all movies — may take a few minutes.
### Step 4 — Launch the app
Option A — Docker (recommended):
```bash
docker build -t moviemind .
docker run -p 7860:7860 --env-file .env moviemind
```
Option B — Direct with uv:
```bash
uv run streamlit run streamlit_app.py
```
Open http://localhost:7860 and start exploring movies.

----
| Example Queries to Try Query | What It Tests |
|------|--------|
|"Mind‑bending sci‑fi like Inception"	|Semantic similarity + genre understanding|
| "Romantic comedy but not cheesy" | Exclusion logic|
|"Batman movies"	|Cast/character matching|
|"Space movies before 2000"	|Year filtering|
----
### 📊 Evaluation & Research Background
This project originated from three detailed Jupyter notebooks that explored:

- EDA — Ratings sparsity (99%+), long‑tail distribution, popularity bias

- Retrieval — TF‑IDF baseline vs. Sentence‑Transformer embeddings

- Ranking — Hybrid scoring with ablation studies on similarity, rating, and popularity

---
### Key findings from the research phase:
- Semantic embeddings outperform TF‑IDF on descriptive queries like "mind‑bending psychological movies"

- Logical constraints ("NOT", director, year range) require structured filtering on top of vector search

- NDCG@10 ≈ 0.98 was achieved with the three‑signal hybrid scorer

- Ablation study revealed that within a strong candidate pool, any reasonable combination of similarity, rating, and popularity yields high‑quality rankings

- Robustness confirmed — varying weights from (0.6,0.2,0.2) to (0.4,0.3,0.3) changed NDCG by only ±0.005

>The complete notebooks are available in the notebooks/ folder of this repository.

---
### 🚢 Deployment
Live demo: https://huggingface.co/spaces/PavanBoorla/movie-mind

The app is deployed as a Docker Space on Hugging Face. The artifacts (movies_processed_final.parquet, movies_faiss.index, tfidf_vectorizer.pkl) are bundled directly in the Space, so no external storage or network calls are needed at runtime.

---
### Deploy your own instance:
1. Fork this repository

2. Upload your artifacts/ folder to a Hugging Face Space

3. Add the Dockerfile, recommender.py, streamlit_app.py, pyproject.toml, uv.lock

4. Add GROQ_API_KEY as a Space secret

5. The Space builds and deploys automatically
---
### 🔮 Future Roadmap
- User session history and personalised profiles

- Collaborative filtering integration for cold‑start users

- Poster images from TMDB API

- Advanced query parsing: runtime, awards, language

- A/B testing framework for ranking weight optimisation
---

### 🙏 Acknowledgements
- Data: MovieLens 20M (GroupLens Research) and TMDB 5000

- Embeddings: Sentence‑Transformers (all-MiniLM-L6-v2)

- Vector Search: FAISS by Meta

- LLM: Groq for fast, free‑tier inference

- Architecture inspiration: Two‑stage retrieval‑ranking patterns from Netflix, Spotify, and industry literature

---
### 📄 License
MIT — use, modify, and deploy freely.

---
### 👤 Author
Boorla Pavan Ram Kalyan

 https://github.com/pavanramkalyanboorla-debug • www.linkedin.com/in/pavan-ram-kalyan-boorla-0a3402405

 ---
 If you found this useful, a ⭐ on the repo means a lot! 🍿