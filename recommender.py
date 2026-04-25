"""
recommender.py - MovieMind core logic.
Reliable rule‑based parsing + Groq enhancement, robust explanations, keyword‑aware ranking.
"""
import os
import re
import time
import logging
import concurrent.futures
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")

# Load artifacts
df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies_processed_final.parquet"))
index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "movies_faiss.index"))
model = SentenceTransformer('all-MiniLM-L6-v2')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

logger.info("✅ Artifacts loaded")

# ----------------------------------------------------------------------
# 1. Robust Rule‑based Parser (always active, never fails)
# ----------------------------------------------------------------------
class RobustParser:
    """Parses a free‑text movie query using token rules and common patterns."""
    def __init__(self, query: str):
        self.original = query
        self.lower = query.lower()
        self.tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", self.lower)  # keep hyphens
        self.result = {
            'genre': None,
            'year_min': None,
            'year_max': None,
            'exclude': [],
            'must_include': [],
            'similar_to': None,
            'raw_query': query
        }
        self._parse()

    def _parse(self):
        # ----- Exclusions: "without X", "not X", "no X" -----
        triggers = ['without', 'not', 'no']
        for trig in triggers:
            # find trigger followed by something until conjunctions or end
            # Use a token walker
            for i, tok in enumerate(self.tokens):
                if tok == trig:
                    # collect until we hit: and, or, but, or a new trigger word
                    phrase_parts = []
                    j = i + 1
                    while j < len(self.tokens) and self.tokens[j] not in ('and', 'or', 'but', 'without', 'not', 'no'):
                        phrase_parts.append(self.tokens[j])
                        j += 1
                    if phrase_parts:
                        # reconstruct original casing from original query slice? just use lower
                        phrase = ' '.join(phrase_parts)
                        self.result['exclude'].append(phrase)

        # ----- "like X" or "similar to X" -----
        for pattern in [r'like\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)', 
                        r'similar\s+to\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)']:
            m = re.search(pattern, self.lower)
            if m:
                candidate = m.group(1).strip()
                if candidate and not any(trig in candidate for trig in triggers):
                    self.result['similar_to'] = candidate
                    break

        # ----- "with X", "must include X" -----
        for pattern in [r'must\s+include\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)', 
                        r'with\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)']:
            m = re.search(pattern, self.lower)
            if m:
                phrase = m.group(1).strip()
                if phrase and not any(trig in phrase for trig in triggers):
                    self.result['must_include'] = [phrase]
                break

        # ----- Genre detection (simple list) -----
        known_genres = {
            'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
            'drama', 'fantasy', 'horror', 'mystery', 'romance', 'sci-fi', 'sci fi',
            'thriller', 'war', 'western'
        }
        for genre in known_genres:
            if genre in self.lower:
                self.result['genre'] = genre
                break

        # ----- Year filters -----
        m = re.search(r'(before|after|from|in the)\s+(\d{4})', self.lower)
        if m:
            rel, year = m.groups()
            year = int(year)
            if rel == 'before':
                self.result['year_max'] = year - 1
            elif rel == 'after':
                self.result['year_min'] = year + 1
            elif rel == 'from':
                self.result['year_min'] = year
            elif rel == 'in the':
                decade = year - (year % 10)
                self.result['year_min'] = decade
                self.result['year_max'] = decade + 9

        # Deduplicate excludes
        self.result['exclude'] = list(set(self.result['exclude']))

    def get_result(self):
        return self.result

# ----------------------------------------------------------------------
# 2. Groq enhancer (optional, runs after rules)
# ----------------------------------------------------------------------
def enhance_with_groq(rule_result: dict) -> dict:
    """Call Groq to fill missing fields, but never override rule‑based exclusions."""
    if not groq_client:
        return rule_result

    system_prompt = """You are a movie recommendation query parser.
Given the user's query, return a JSON object with any of these optional fields:
- genre (string)
- year_min (integer)
- year_max (integer)
- must_include (list of strings)
- exclude (list of strings) – only if explicitly mentioned
- similar_to (string)
Do NOT overwrite exclusions already known."""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rule_result['raw_query']}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=150,
            timeout=3.0
        )
        llm = json.loads(response.choices[0].message.content)
        # Merge: LLM can add new fields but cannot override existing non‑empty values
        for key in ['genre', 'year_min', 'year_max', 'similar_to']:
            if not rule_result.get(key) and key in llm:
                rule_result[key] = llm[key]
        # For must_include and exclude, merge if LLM provides additional
        if 'must_include' in llm:
            combined = set(rule_result.get('must_include', []) + llm['must_include'])
            rule_result['must_include'] = list(combined)
        if 'exclude' in llm and llm['exclude']:
            # Rule exclusions always win, but we add LLM's if they don't conflict? Safer: only add if not already covered
            # We'll keep only rule exclusions to guarantee reliability
            pass
        return rule_result
    except Exception as e:
        logger.warning(f"Groq enhancer failed: {e}")
        return rule_result

# ----------------------------------------------------------------------
# 3. Reliable explanation generator (Groq with timeout + template fallback)
# ----------------------------------------------------------------------
def generate_explanation(movie_row, parsed_query):
    """Generate an explanation, first trying Groq, then falling back to a smart template."""
    # Try Groq with a short timeout
    if groq_client:
        prompt = f"""User asked: "{parsed_query['raw_query']}"
Recommended: "{movie_row['title']}" ({int(movie_row['year'])})
Genres: {movie_row.get('genres', '')}
Overview: {movie_row.get('overview', '')[:200]}
Rating: {movie_row['avg_rating']:.1f}/10
Write one short sentence (max 25 words) connecting this movie to the user's request."""
        try:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60,
                timeout=2.0
            )
            explanation = response.choices[0].message.content.strip()
            if explanation:
                return explanation
        except Exception as e:
            logger.warning(f"Groq explanation failed for {movie_row['title']}: {e}")

    # Fallback template (always works)
    parts = []
    if parsed_query.get('similar_to'):
        parts.append(f"Similar to {parsed_query['similar_to']}")
    if parsed_query.get('genre'):
        parts.append(f"it's a {parsed_query['genre']} movie")
    if not parts:
        genres = movie_row.get('genres', '')
        if genres:
            parts.append(f"it's a {genres.lower()} film")
        else:
            parts.append("it matches your request")
    prefix = "Recommended because " if len(parts) > 1 else ""
    return prefix + ', '.join(parts) + '.'

# ----------------------------------------------------------------------
# 4. Core recommendation function
# ----------------------------------------------------------------------
def recommend(query, top_n=10, w_sim=0.6, w_rating=0.2, w_pop=0.2,
              use_llm_parse=True, generate_explanations=True):
    # Step A: Robust rule‑based parsing
    parser = RobustParser(query)
    parsed = parser.get_result()

    # Step B: Optionally enhance with Groq (only if API key exists and user wants it)
    if use_llm_parse and groq_client:
        parsed = enhance_with_groq(parsed)

    # Step C: Build enhanced embedding query
    enhanced = query
    if parsed.get('similar_to'):
        enhanced += f" similar to {parsed['similar_to']}"
    if parsed.get('must_include'):
        enhanced += " " + " ".join(parsed['must_include'])

    # Step D: FAISS retrieval
    q_vec = model.encode([enhanced]).astype('float32')
    q_vec = np.ascontiguousarray(q_vec)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, 200)
    scores = scores[0]
    indices = indices[0]

    candidates = df.iloc[indices].copy()
    candidates['similarity'] = scores

    # Step E: Apply year/genre/exclusion filters
    if parsed.get('year_min'):
        candidates = candidates[candidates['year'] >= parsed['year_min']]
    if parsed.get('year_max'):
        candidates = candidates[candidates['year'] <= parsed['year_max']]
    if parsed.get('genre'):
        genre = parsed['genre']
        genre_variants = [genre, genre.replace('-', ' ').replace(' ', '-')]
        mask = candidates['genres'].str.lower().str.contains('|'.join(genre_variants), na=False)
        candidates = candidates[mask]
    if parsed.get('exclude'):
        for term in parsed['exclude']:
            term_lower = term.lower()
            # word‑level
            for word in term_lower.split():
                candidates = candidates[
                    ~candidates['overview'].str.lower().str.contains(re.escape(word), na=False) &
                    ~candidates['title'].str.lower().str.contains(re.escape(word), na=False)
                ]
            # cleaned (no spaces/hyphens)
            clean = re.sub(r'[^a-z0-9]', '', term_lower)
            if clean:
                candidates['_clean_ov'] = candidates['overview'].fillna('').str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                candidates['_clean_ti'] = candidates['title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                candidates = candidates[
                    ~candidates['_clean_ov'].str.contains(re.escape(clean)) &
                    ~candidates['_clean_ti'].str.contains(re.escape(clean))
                ]
                candidates.drop(['_clean_ov', '_clean_ti'], axis=1, inplace=True)

    # Step F: Keyword boost – give extra weight to movies containing query keywords
    query_lower = query.lower()
    stop_words = {'a', 'an', 'the', 'is', 'of', 'in', 'on', 'to', 'for', 'with',
                  'and', 'or', 'but', 'not', 'no', 'without', 'like', 'movies', 'movie'}
    q_words = set(re.findall(r'[a-z0-9]+', query_lower)) - stop_words
    if q_words:
        def kw_score(row):
            soup = row.get('soup', '')
            if not soup or pd.isna(soup):
                return 0.0
            soup_lower = soup.lower()
            matches = sum(1 for w in q_words if w in soup_lower)
            return matches / len(q_words)
        candidates['kw_boost'] = candidates.apply(kw_score, axis=1)
    else:
        candidates['kw_boost'] = 0.0

    # Step G: Hybrid scoring
    rating_norm = candidates['avg_rating'] / 10.0
    pop_max = candidates['popularity_log'].max() + 1e-8
    pop_norm = candidates['popularity_log'] / pop_max

    candidates['hybrid_score'] = (
        w_sim * candidates['similarity'] +
        w_rating * rating_norm +
        w_pop * pop_norm +
        0.1 * candidates['kw_boost']   # keyword boost weight
    )

    candidates = candidates.sort_values('hybrid_score', ascending=False).head(top_n)

    # Step H: Build results with explanations
    results = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        item = {
            "title": row['title'],
            "year": row['year'],
            "genres": row.get('genres', ''),
            "overview": str(row.get('overview', ''))[:300] + "..."
                        if len(str(row.get('overview', ''))) > 300
                        else str(row.get('overview', '')),
            "avg_rating": row['avg_rating'],
            "rating_count": int(row['rating_count']),
            "similarity": float(row['similarity']),
            "hybrid_score": float(row['hybrid_score']),
        }
        if generate_explanations:
            # Small delay to respect rate limits if using Groq
            if i > 0 and groq_client:
                time.sleep(0.6)
            item['explanation'] = generate_explanation(row, parsed)
        else:
            item['explanation'] = None
        results.append(item)

    return results