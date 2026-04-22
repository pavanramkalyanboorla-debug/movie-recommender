"""
api/main.py
-----------
FastAPI backend that loads the FAISS index and movie data once at startup,
then serves a /recommend endpoint with optional Groq LLM query understanding
and explanation generation.
"""

import os
import json
import pickle
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Global variables for models and data
df = None
index = None
model = None
tfidf_vectorizer = None
groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized")
else:
    logger.warning("GROQ_API_KEY not set – LLM features disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts on startup and clean up on shutdown."""
    global df, index, model, tfidf_vectorizer
    logger.info("Loading artifacts...")

    # Load processed movie dataframe
    df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies_processed_final.parquet"))

    # Load FAISS index
    index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "movies_faiss.index"))

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load TF-IDF vectorizer (optional)
    tfidf_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)

    logger.info(" All artifacts loaded successfully")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="MovieMind API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    query: str
    top_n: int = 10
    w_sim: float = 0.6
    w_rating: float = 0.2
    w_pop: float = 0.2
    use_llm_parse: bool = True
    generate_explanations: bool = True


class MovieResponse(BaseModel):
    title: str
    year: float
    genres: str | None = None
    overview: str | None = None
    avg_rating: float
    rating_count: int
    similarity: float
    hybrid_score: float
    explanation: str | None = None


def parse_query_with_llm(query: str) -> dict:
    """Use Groq to extract structured filters from natural language."""
    if not groq_client:
        return {"raw_query": query}

    system_prompt = """You are a movie recommendation query parser.
Extract structured information from the user's natural language query.
Return a JSON object with the following optional fields:
- genre: string (e.g., "sci-fi", "comedy")
- year_min: integer (earliest release year)
- year_max: integer (latest release year)
- must_include: list of keywords that MUST be in the movie description
- exclude: list of keywords to avoid (e.g., "not horror", "no space")
- similar_to: string (movie title to find similar movies)

If a field is not present, omit it.
Example: "A sci-fi movie like Inception but not too long" → 
{"genre": "sci-fi", "similar_to": "Inception", "exclude": ["long"]}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=200
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"LLM query parsing failed: {e}, falling back to raw query")
        return {"raw_query": query}


def generate_explanation(movie_row, user_query: str) -> str:
    """Generate a short, personalized explanation for a recommended movie."""
    if not groq_client:
        return f"Matches your interest in '{user_query[:30]}...'"

    prompt = f"""User asked: "{user_query}"
We recommended: "{movie_row['title']}" ({int(movie_row['year'])})
Genres: {movie_row.get('genres', '')}
Overview: {movie_row.get('overview', '')[:200]}
Rating: {movie_row['avg_rating']:.1f}/10 from {movie_row['rating_count']} votes.

Write a single engaging sentence (max 20 words) that connects this movie to the user's request. Be specific."""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Explanation generation failed: {e}")
        return f"Matches your interest in {user_query[:30]}..."


@app.post("/recommend", response_model=list[MovieResponse])
async def recommend(req: RecommendRequest):
    if df is None or index is None or model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # 1. Optional LLM query parsing
    parsed = {}
    if req.use_llm_parse and groq_client:
        parsed = parse_query_with_llm(req.query)

    # 2. Build enhanced query for embedding
    enhanced_query = req.query
    if parsed.get("similar_to"):
        enhanced_query += f" similar to {parsed['similar_to']}"
    if parsed.get("must_include"):
        enhanced_query += " " + " ".join(parsed["must_include"])

    # 3. FAISS retrieval (top 200 candidates)
    query_vec = model.encode([enhanced_query]).astype('float32')
    query_vec = np.ascontiguousarray(query_vec)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, 200)
    scores = scores[0]
    indices = indices[0]

    candidates = df.iloc[indices].copy()
    candidates['similarity'] = scores

    # 4. Apply filters from LLM (post-retrieval)
    if parsed:
        if "genre" in parsed:
            genre_filter = parsed["genre"].lower()
            candidates = candidates[candidates['genres'].str.lower().str.contains(genre_filter, na=False)]
        if "year_min" in parsed:
            candidates = candidates[candidates['year'] >= parsed["year_min"]]
        if "year_max" in parsed:
            candidates = candidates[candidates['year'] <= parsed["year_max"]]
        if "exclude" in parsed:
            for term in parsed["exclude"]:
                candidates = candidates[~candidates['overview'].str.lower().str.contains(term.lower(), na=False)]

    # 5. Hybrid scoring (as in Notebook 3)
    w_sim, w_rating, w_pop = req.w_sim, req.w_rating, req.w_pop
    rating_norm = candidates['avg_rating'] / 10.0
    pop_max = candidates['popularity_log'].max() + 1e-8
    pop_norm = candidates['popularity_log'] / pop_max

    candidates['hybrid_score'] = (
        w_sim * candidates['similarity'] +
        w_rating * rating_norm +
        w_pop * pop_norm
    )

    candidates = candidates.sort_values('hybrid_score', ascending=False).head(req.top_n)

    # 6. Generate explanations (optional)
    results = []
    for _, row in candidates.iterrows():
        item = {
            "title": row['title'],
            "year": row['year'],
            "genres": row.get('genres', ''),
            "overview": row.get('overview', '')[:300] + "..." if len(str(row.get('overview', ''))) > 300 else row.get('overview', ''),
            "avg_rating": row['avg_rating'],
            "rating_count": int(row['rating_count']),
            "similarity": float(row['similarity']),
            "hybrid_score": float(row['hybrid_score']),
        }
        if req.generate_explanations:
            item['explanation'] = generate_explanation(row, req.query)
        else:
            item['explanation'] = None
        results.append(item)

    return results


@app.get("/health")
async def health():
    return {"status": "ok"}