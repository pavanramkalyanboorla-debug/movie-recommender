"""
recommender.py - Core logic for the MovieMind recommender.
Loads artifacts and provides a recommend() function.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Load artifacts once (module‑level)
# ----------------------------------------------------------------------
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")

df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies_processed_final.parquet"))
index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "movies_faiss.index"))
model = SentenceTransformer('all-MiniLM-L6-v2')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
logger.info(" All artifacts loaded")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_query_with_llm(query: str) -> dict:
    if not groq_client:
        return {"raw_query": query}
    system_prompt = """You are a movie recommendation query parser.
Extract structured information from the user's natural language query.
Return a JSON object with the following optional fields:
- genre: string
- year_min: integer
- year_max: integer
- must_include: list of keywords
- exclude: list of keywords to avoid
- similar_to: string (movie title)
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
        logger.warning(f"LLM parse failed: {e}")
        return {"raw_query": query}

def generate_explanation(movie_row, user_query: str) -> str:
    if not groq_client:
        return f"Matches your interest in {user_query[:30]}..."
    prompt = f"""User asked: "{user_query}"
Recommended: "{movie_row['title']}" ({int(movie_row['year'])})
Genres: {movie_row.get('genres', '')}
Overview: {movie_row.get('overview', '')[:200]}
Rating: {movie_row['avg_rating']:.1f}/10
Write a single engaging sentence (max 20 words) that connects this movie to the user's request."""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except:
        return f"Matches your interest in {user_query[:30]}..."

def recommend(query, top_n=10, w_sim=0.6, w_rating=0.2, w_pop=0.2,
              use_llm_parse=True, generate_explanations=True):
    """
    Main entry point – returns a list of dicts with movie recommendations.
    """
    parsed = {}
    if use_llm_parse and groq_client:
        parsed = parse_query_with_llm(query)

    # Enhance query for embedding
    enhanced_query = query
    if parsed.get("similar_to"):
        enhanced_query += f" similar to {parsed['similar_to']}"
    if parsed.get("must_include"):
        enhanced_query += " " + " ".join(parsed["must_include"])

    # FAISS search
    query_vec = model.encode([enhanced_query]).astype('float32')
    query_vec = np.ascontiguousarray(query_vec)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, 200)
    scores = scores[0]
    indices = indices[0]

    candidates = df.iloc[indices].copy()
    candidates['similarity'] = scores

    # Apply LLM filters
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

    # Hybrid scoring
    rating_norm = candidates['avg_rating'] / 10.0
    pop_max = candidates['popularity_log'].max() + 1e-8
    pop_norm = candidates['popularity_log'] / pop_max

    candidates['hybrid_score'] = (
        w_sim * candidates['similarity'] +
        w_rating * rating_norm +
        w_pop * pop_norm
    )
    candidates = candidates.sort_values('hybrid_score', ascending=False).head(top_n)

    results = []
    for _, row in candidates.iterrows():
        item = {
            "title": row['title'],
            "year": row['year'],
            "genres": row.get('genres', ''),
            "overview": row.get('overview', '')[:300] + "..."
                       if len(str(row.get('overview', ''))) > 300
                       else row.get('overview', ''),
            "avg_rating": row['avg_rating'],
            "rating_count": int(row['rating_count']),
            "similarity": float(row['similarity']),
            "hybrid_score": float(row['hybrid_score']),
        }
        if generate_explanations:
            item['explanation'] = generate_explanation(row, query)
        else:
            item['explanation'] = None
        results.append(item)
    return results