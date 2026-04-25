"""
recommender.py - Core logic for the MovieMind recommender.
Loads artifacts, processes queries, and returns ranked results with explanations.
"""
import os
import json
import re
import time
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
logger.info("✅ All artifacts loaded")

# ----------------------------------------------------------------------
# Robust rule‑based exclusion — catches hyphens, slashes, etc.
# ----------------------------------------------------------------------
def rule_based_exclude(query: str) -> list:
    """
    Extracts exclusion keywords from the query.
    Captures: without ..., not ..., no ...
    The phrase may contain letters, spaces, hyphens, slashes, and digits.
    Returns a list of lowercase phrases to exclude.
    """
    # List of trigger words (all lowercase for matching)
    triggers = ['without', 'not', 'no']
    query_lower = query.lower()
    exclude_terms = []

    # Find positions of trigger words
    for trig in triggers:
        # We'll find all occurrences of trigger followed by a phrase
        idx = 0
        while True:
            pos = query_lower.find(trig, idx)
            if pos == -1:
                break
            # Make sure it's a whole word (preceded by space or start, followed by space)
            start_ok = (pos == 0 or not query_lower[pos-1].isalpha())
            end_pos = pos + len(trig)
            end_ok = (end_pos == len(query_lower) or not query_lower[end_pos].isalpha())
            if start_ok and end_ok:
                # Extract the phrase after the trigger, until a conjunction or ending punctuation
                rest = query_lower[end_pos:].strip()
                # Stop at words: and, or, but, or punctuation: . , ! ? ; :
                stop_pattern = r'\b(?:and|or|but)\b|[.,!?;:]'
                match = re.search(r'([^.,!?;:]+?)(?:\s*\b(?:and|or|but)\b|[.,!?;:])', rest)
                if match:
                    phrase = match.group(1).strip()
                else:
                    # No stop present, take whole rest of string
                    phrase = rest
                if phrase:
                    exclude_terms.append(phrase)
            idx = end_pos  # move past this trigger

    # Remove duplicates while preserving order (last seen wins, but doesn't matter)
    return list(set(exclude_terms))

# ----------------------------------------------------------------------
# LLM query parsing (optional, adds structured filters)
# ----------------------------------------------------------------------
def parse_query_with_llm(query: str) -> dict:
    if not groq_client:
        return {"raw_query": query}

    system_prompt = """You are a movie recommendation query parser.
Extract structured information from the user's natural language query.
Return a JSON object with the following optional fields:
- genre: string (e.g., "sci-fi", "comedy")
- year_min: integer (earliest release year)
- year_max: integer (latest release year)
- must_include: list of keywords that MUST be in the movie description
- exclude: list of keywords to avoid (e.g., "horror", "space")
- similar_to: string (movie title to find similar movies)

Examples:
"a sci‑fi movie like Inception but not too violent" → {"genre": "sci-fi", "similar_to": "Inception", "exclude": ["violent"]}
"movies like Avatar without aliens" → {"similar_to": "Avatar", "exclude": ["aliens"]}
If a field is not present, omit it.
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

# ----------------------------------------------------------------------
# Explanation generation with retry + rate limit handling
# ----------------------------------------------------------------------
def generate_explanation(movie_row, user_query: str, max_retries=2) -> str:
    if not groq_client:
        return f"Matches your interest in {user_query[:30]}..."

    prompt = f"""User asked: "{user_query}"
Recommended: "{movie_row['title']}" ({int(movie_row['year'])})
Genres: {movie_row.get('genres', '')}
Overview: {movie_row.get('overview', '')[:200]}
Rating: {movie_row['avg_rating']:.1f}/10
Write a single engaging sentence (max 20 words) that connects this movie to the user's request."""

    for attempt in range(max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60
            )
            explanation = response.choices[0].message.content.strip()
            if explanation:
                return explanation
        except Exception as e:
            logger.warning(f"Explanation attempt {attempt+1} failed: {e}")
            if attempt < max_retries:
                time.sleep(1)  # brief pause before retry
            else:
                logger.error(f"All explanation retries exhausted for {movie_row['title']}")

    return f"Matches your interest in {user_query[:30]}..."

# ----------------------------------------------------------------------
# Main recommendation function
# ----------------------------------------------------------------------
def recommend(query, top_n=10, w_sim=0.6, w_rating=0.2, w_pop=0.2,
              use_llm_parse=True, generate_explanations=True):
    """
    Returns a list of dicts with movie recommendations.
    """
    # 1. Always apply rule‑based exclusion (fast and reliable)
    rule_excludes = rule_based_exclude(query)

    # 2. Optionally enrich with LLM parsing (genre, year, etc.)
    parsed = {}
    if use_llm_parse and groq_client:
        parsed = parse_query_with_llm(query)

    # 3. Merge exclusions
    if rule_excludes:
        if 'exclude' in parsed:
            parsed['exclude'] = list(set(parsed['exclude'] + rule_excludes))
        else:
            parsed['exclude'] = rule_excludes

    # 4. Enhance query for embedding
    enhanced_query = query
    if parsed.get("similar_to"):
        enhanced_query += f" similar to {parsed['similar_to']}"
    if parsed.get("must_include"):
        enhanced_query += " " + " ".join(parsed["must_include"])

    # 5. FAISS retrieval
    query_vec = model.encode([enhanced_query]).astype('float32')
    query_vec = np.ascontiguousarray(query_vec)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, 200)
    scores = scores[0]
    indices = indices[0]

    candidates = df.iloc[indices].copy()
    candidates['similarity'] = scores

    # 6. Apply structured filters from parsed intent
    if parsed and "raw_query" not in parsed:
        if "genre" in parsed:
            genre_filter = parsed["genre"].lower()
            candidates = candidates[candidates['genres'].str.lower().str.contains(genre_filter, na=False)]
        if "year_min" in parsed:
            candidates = candidates[candidates['year'] >= parsed["year_min"]]
        if "year_max" in parsed:
            candidates = candidates[candidates['year'] <= parsed["year_max"]]
        if "exclude" in parsed:
            for term in parsed["exclude"]:
                # Convert to lowercase once
                term_lower = term.lower()
                # Split into words and filter if ANY word appears
                words = term_lower.split()
                for word in words:
                    if word:
                        candidates = candidates[
                            ~candidates['overview'].str.lower().str.contains(word, na=False) &
                            ~candidates['title'].str.lower().str.contains(word, na=False)
                        ]
                # Also filter ignoring all spaces/hyphens/special characters
                clean_term = re.sub(r'[^a-z0-9]', '', term_lower)
                if clean_term:
                    # Remove all non-alphanumeric from candidate strings for comparison
                    candidates['_clean_overview'] = candidates['overview'].fillna('').str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                    candidates['_clean_title'] = candidates['title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                    candidates = candidates[
                        ~candidates['_clean_overview'].str.contains(re.escape(clean_term)) &
                        ~candidates['_clean_title'].str.contains(re.escape(clean_term))
                    ]
                    # Drop temporary columns
                    candidates.drop(['_clean_overview', '_clean_title'], axis=1, inplace=True)

    # 7. Hybrid scoring
    rating_norm = candidates['avg_rating'] / 10.0
    pop_max = candidates['popularity_log'].max() + 1e-8
    pop_norm = candidates['popularity_log'] / pop_max

    candidates['hybrid_score'] = (
        w_sim * candidates['similarity'] +
        w_rating * rating_norm +
        w_pop * pop_norm
    )
    candidates = candidates.sort_values('hybrid_score', ascending=False).head(top_n)

    # 8. Build results (with explanations if requested)
    results = []
    for i, (_, row) in enumerate(candidates.iterrows()):
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
            if i > 0:
                time.sleep(0.6)
            item['explanation'] = generate_explanation(row, query)
        else:
            item['explanation'] = None

        results.append(item)

    return results