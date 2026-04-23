---
title: Movie Mind API
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
#  MovieMind — Conversational Movie Recommender with AI Explanations

**Understands what you *mean*, not just what you *type*.  
Tells you *why* each movie is right for you.**

---

##  What is MovieMind?

MovieMind is a **production‑ready conversational recommendation system** that takes natural language queries like:

> *"A mind‑bending sci‑fi movie like Inception, but not too violent"*  
> *"Light‑hearted romantic comedies from the 90s"*  
> *"Batman movies without the Joker"*

It **retrieves** relevant movies from a large catalog, **ranks** them by quality and popularity, and **explains** each recommendation with a short, human‑readable sentence – powered by a large language model.

---

##  Why This Project Stands Out

Most movie recommenders are either:

- **Keyword‑based** – fail on vague or descriptive queries  
- **Black‑box** – give no justification for their choices  
- **Academic only** – never leave the notebook  

MovieMind bridges the gap between **research insights** and **deployable software**:

-  Two‑stage retrieval + ranking pipeline (industry best practice)  
-  Semantic search with FAISS and Sentence Transformers  
-  Hybrid scoring that balances relevance, rating, and popularity  
-  **Explainable AI** – Groq‑powered natural language justifications  
-  Clean web interface built with Streamlit  
-  Fully containerised with Docker, ready for cloud deployment  

---

##  How It Works (Two‑Stage Architecture)
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

##  Features

| Feature | Description |
|---------|-------------|
| **Natural Language Understanding** | Groq parses user intent – genre, year, "not", "similar to", etc. |
| **Semantic Search** | Sentence‑Transformers + FAISS index for sub‑second retrieval. |
| **Hybrid Scoring** | Combines semantic similarity, average rating, and log‑scaled popularity. |
| **Explainable Recommendations** | Each result includes a concise AI‑generated *why*. |
| **Interactive Web UI** | Built with Streamlit – sliders for ranking weights, live results. |
| **REST API** | FastAPI backend, ready for integration with other apps. |
| **Docker Support** | Run the whole stack with one command. |
| **Free‑tier Cloud Ready** | Deploy on Render, Hugging Face Spaces, or Railway. |

---

##  Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend API** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Embeddings** | Sentence‑Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **LLM Integration** | Groq (Llama 3 / Mixtral) |
| **Data Processing** | Pandas, NumPy, Scikit‑learn |
| **Containerisation** | Docker, Docker Compose |
| **Package Management** | `uv` (fast Python package installer) |

---

##  Project Structure
movie-recommender/\
├── .env.example # Template for API keys\
├── pyproject.toml # uv dependencies\
├── docker-compose.yml\
├── Dockerfile.api\
├── Dockerfile.frontend\
├── pipeline/\
│ └── build_artifacts.py # Data preprocessing + FAISS index builder\
├── api/\
│ ├── init.py\
│ └── main.py # FastAPI app with Groq integration\
├── frontend/\
│ └── app.py # Streamlit UI\
├── data/ # (gitignored) raw CSV files\
└── artifacts/ # (gitignored) generated models & index

---

##  Quick Start (Local with Docker)

### 1. Clone and prepare data

```bash
git clone https://github.com/pavanramkalyanboorla-debug/movie-recommender
cd movie-recommender
```
Place the following CSV files in ./data/:

- ratings.csv (MovieLens)

- movies.csv (MovieLens)

- links.csv (MovieLens)

- tmdb_5000_movies.csv (TMDB)

### 2. Set your Groq API key
```bash
cp .env.example .env\
Edit .env and add your Groq key\
```
### 3. Run the pipeline (generates FAISS index and processed data)
``` bash
uv run python pipeline/build_artifacts.py
```
### 4. Start the services
```bash
docker-compose up --build
```
### 5. Open the app
👉 http://localhost:8501

Running Without Docker (Development)
If you prefer to run services directly:

```bash
# Terminal 1 – Backend
export GROQ_API_KEY=your_key_here
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 – Frontend
export API_URL=http://localhost:8000
uv run streamlit run frontend/app.py
```
| Example Queries to Try Query | What It Tests |
|------|--------|
|"Mind‑bending sci‑fi like Inception"	|Semantic similarity + genre understanding|
| "Romantic comedy but not cheesy" | Exclusion logic|
|"Batman movies"	|Cast/character matching|
|"Space movies before 2000"	|Year filtering|
----
## Research Background: From Notebooks to Production
### This production system is built on three detailed Jupyter notebooks that performed:

- Exploratory Data Analysis – sparsity, long‑tail, popularity bias

- Retrieval Comparison – TF‑IDF vs. Sentence Transformers

- Failure Mode Analysis – why embeddings alone can't handle "NOT"

- Ablation Studies – proving each ranking signal matters

- NDCG@10 Evaluation – achieving ~0.98 ranking quality

### Key insights from the notebooks:

- Semantic search dramatically outperforms TF‑IDF on vague or descriptive queries.

- Within a strong candidate set, hybrid scoring with quality signals lifts the best movies to the top.

- The system is robust to weight choices – retrieval quality is the main driver.

- Logical constraints ("NOT", director, year) require structured filtering on top of embeddings.

### The production code preserves all the logic from the notebooks while adding:

- FAISS for millisecond‑level retrieval

- Groq LLM for query parsing and explanation generation

- FastAPI + Streamlit for a clean, interactive interface

## Deployment on Free Cloud Tiers
### MovieMind is designed to run on free cloud offerings:

- Render – Deploy API and frontend as separate web services using the provided Dockerfiles.

- Hugging Face Spaces – Host the Streamlit frontend for free; point it to a Render‑hosted API.

- Railway / Fly.io – Single‑command deploys with generous free allowances.

## Future Roadmap
- User profiles and session history

- Collaborative filtering integration

- Poster images from TMDB

- More advanced query parsing (director, actor, runtime)

- A/B testing framework for ranking weights

- Monitoring dashboard with Prometheus metrics

## Acknowledgements
This project builds upon the MovieLens 20M dataset (GroupLens Research) and TMDB 5000 metadata.
The two‑stage architecture and hybrid ranking approach were inspired by industry best practices from companies like Netflix and Spotify.
Special thanks to the open‑source community behind Sentence‑Transformers, FAISS, and Groq.

## License
MIT – feel free to use, modify, and deploy this project for your own portfolio or production use.

## Author
Boorla Pavan Ram Kalyan\
 https://github.com/pavanramkalyanboorla-debug • www.linkedin.com/in/pavan-ram-kalyan-boorla-0a3402405

If you found this useful, a ⭐ on the repo means a lot!

