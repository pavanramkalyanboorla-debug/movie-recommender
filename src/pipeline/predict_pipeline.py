# src/pipeline/predict_pipeline.py
"""
Hybrid retrieval & constraint‑aware recommendation pipeline.

Architecture:
    User Query
       │
       ▼
    ┌─────────────────────────────┐
    │  Structured NLU (Groq JSON)  │  ← intent / entities / exclusions / rewrite
    └─────────────────────────────┘
       │
       ▼
    ┌─────────────────────────────┐
    │  JSON Schema Validation      │  ← repair malformed / hallucinated fields
    └─────────────────────────────┘
       │
       ▼
    ┌─────────────────────────────┐
    │  Heuristic Validation       │  ← entity counts, confidence checks
    └─────────────────────────────┘
       │
       ├── director intent → _pull_director_films()
       ├── actor intent    → _pull_actor_films()
       ├── franchise       → _pull_franchise_films()
       └── semantic        → FAISS hybrid retrieval
       │
       ▼
    ┌─────────────────────────────┐
    │  Constraint Filters         │  ← year / genre / exclusion / vote guards
    └─────────────────────────────┘
       │
       ▼
    ┌─────────────────────────────┐
    │  Hybrid Scoring + Reranking │  ← similarity + rating + popularity + boosts
    └─────────────────────────────┘
       │
       ▼
    Final results (with streaming badges)
"""

import os, re, json, time, logging
import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from src.constants import ARTIFACTS_DIR, PROCESSED_PARQUET, FAISS_INDEX, EMBEDDING_MODEL

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ────────────────────────────────────────────────
# 1. Robust Rule‑based Parser (fallback when Groq is unavailable)
# ────────────────────────────────────────────────
class RobustParser:
    def __init__(self, query: str):
        self.original = query
        self.lower = query.lower()
        self.tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", self.lower)
        self.result = {
            'genre': None, 'year_min': None, 'year_max': None,
            'exclude': [], 'must_include': [], 'similar_to': None,
            'raw_query': query
        }
        self._parse()

    GENRE_NORMAL = {
        'documentaries': 'documentary', 'docs': 'documentary', 'docu': 'documentary',
        'comedy': 'comedy', 'comedies': 'comedy',
        'thriller': 'thriller', 'thrillers': 'thriller',
        'animation': 'animation', 'anime': 'animation',
        'sci-fi': 'sci-fi', 'scifi': 'sci-fi', 'sci fi': 'sci-fi',
        'musical': 'musical', 'musicals': 'musical',
        'short': 'short', 'shorts': 'short',
        'adventure': 'adventure', 'crime': 'crime',
        'drama': 'drama', 'family': 'family', 'fantasy': 'fantasy',
        'history': 'history', 'horror': 'horror', 'mystery': 'mystery',
        'romance': 'romance', 'war': 'war', 'western': 'western',
        'rom-com': 'romance', 'romcom': 'romance', 'rom com': 'romance',
        'heist': 'crime', 'heists': 'crime',
        'superhero': 'action', 'superheroes': 'action',
    }
    KNOWN_GENRES = set(GENRE_NORMAL.values())

    def _parse(self):
        # Genre extraction
        genres_found = []
        for token in self.tokens:
            norm = self.GENRE_NORMAL.get(token)
            if norm and norm in self.KNOWN_GENRES:
                genres_found.append(norm)
        if 'romance' in genres_found and 'comedy' not in genres_found:
            genres_found.append('comedy')
        if genres_found:
            self.result['genre'] = genres_found[0]

        # Exclusion words
        triggers = ['without', 'not', 'no']
        for trig in triggers:
            for i, tok in enumerate(self.tokens):
                if tok == trig:
                    phrase_parts = []
                    j = i + 1
                    while j < len(self.tokens) and self.tokens[j] not in (
                        'and', 'or', 'but', 'without', 'not', 'no'
                    ):
                        phrase_parts.append(self.tokens[j])
                        j += 1
                    if phrase_parts:
                        phrase = ' '.join(phrase_parts)
                        self.result['exclude'].append(phrase)

        # "like / similar to" detection
        for pattern in [
            r'like\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)',
            r'similar\s+to\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)'
        ]:
            m = re.search(pattern, self.lower)
            if m:
                candidate = m.group(1).strip()
                if candidate and not any(trig in candidate for trig in triggers):
                    self.result['similar_to'] = candidate
                    break

        # "must include / with"
        for pattern in [
            r'must\s+include\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)',
            r'with\s+([\w\s\-\']+?)(?:\s*(?:,|\.|!|\?|;|and|or|but)\s|$)'
        ]:
            m = re.search(pattern, self.lower)
            if m:
                phrase = m.group(1).strip()
                if phrase and not any(trig in phrase for trig in triggers):
                    self.result['must_include'] = [phrase]
                break

        # Year range
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

        # ── Detect "[Person] movies" optionally followed by extra words ──
        person_candidate = None
        m = re.search(r'^([\w\s\-\']{2,40}?)\s+movies(?:\s+(.*))?$', self.lower)
        if m:
            person_candidate = m.group(1).strip()
            # Exclude well‑known brand/franchise words
            if person_candidate.lower() not in (
                'marvel', 'dc', 'disney', 'pixar', 'lego', 'monster',
                'science', 'fiction', 'romance', 'comedy', 'thriller', 'horror',
                'action', 'adventure', 'documentary', 'western', 'animation',
                'musical', 'history', 'war'
            ):
                self.result['potential_person'] = person_candidate

        # Also catch plain proper‑name queries (e.g. "Tom Cruise")
        if not person_candidate:
            bare_name = re.match(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)$', self.original)
            if bare_name:
                person = bare_name.group(1).strip()
                if person.lower() not in (
                    'marvel', 'dc', 'disney', 'pixar', 'lego', 'monster'
                ):
                    self.result['potential_person'] = person

        self.result['exclude'] = list(set(self.result['exclude']))

    def get_result(self):
        return self.result


# ────────────────────────────────────────────────
# 2. Groq helper
# ────────────────────────────────────────────────
def _call_groq(model, messages, max_tokens=200, temperature=0.1, timeout=3.0, client=None):
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return None


# ────────────────────────────────────────────────
# 3. Explanation generator
# ────────────────────────────────────────────────
def _generate_explanation(movie_row, parsed_query, groq_client):
    if groq_client:
        prompt = f"""User asked: "{parsed_query['raw_query']}"
Recommended: "{movie_row['title']}" ({int(movie_row['year'])})
Genres: {movie_row.get('genres', '')}
Overview: {movie_row.get('overview', '')[:200]}
Rating: {movie_row['avg_rating']:.1f}/10
Write one short sentence (max 25 words) connecting this movie to the user's request."""
        explanation = _call_groq("llama-3.1-8b-instant", [{"role":"user","content":prompt}],
                                 max_tokens=60, temperature=0.7, timeout=2.0, client=groq_client)
        if explanation:
            return explanation

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


# ────────────────────────────────────────────────
# 4. PredictPipeline
# ────────────────────────────────────────────────
class PredictPipeline:
    def __init__(self):
        self.df = None
        self.index = None
        self.model = None
        self.groq_client = None
        self.streaming_lookup = {}
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            logger.info("Loading artifacts...")
            self.df = pd.read_parquet(PROCESSED_PARQUET)
            self.index = faiss.read_index(FAISS_INDEX)
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            groq_key = os.environ.get("GROQ_API_KEY")
            self.groq_client = Groq(api_key=groq_key) if groq_key else None
            streaming_path = os.path.join(ARTIFACTS_DIR, "streaming_lookup.json")
            if os.path.exists(streaming_path):
                with open(streaming_path, "r") as f:
                    self.streaming_lookup = json.load(f)
                logger.info(f"✅ Streaming lookup loaded for {len(self.streaming_lookup)} movies")
            else:
                logger.info("ℹ️  No streaming lookup found")
            logger.info(f"✅ Artifacts loaded ({len(self.df)} movies)")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    # ───── Structured NLU (LLM) ────────────────────
    def _llm_parse_query(self, query: str):
        if not self.groq_client:
            return None

        system_prompt = """You are a movie retrieval router. Output a JSON object with:
- intent: "director" / "actor" / "franchise" / "semantic"
- entities: [{"type": "director"/"actor"/"franchise", "name": "..."}]
- exclusions: ["genre or keyword"]
- year_range: {"min": int|null, "max": int|null}
- rewritten_query: short cleaned query for semantic search

Rules:
- "Martin Scorsese movies" → intent=director, entity type=director
- "marvel movies" → intent=franchise, entity type=franchise
- return ONLY valid JSON."""
        try:
            raw = _call_groq("llama-3.1-8b-instant",
                             [{"role":"system","content":system_prompt},
                              {"role":"user","content":query}],
                             max_tokens=250, temperature=0.1, timeout=3.0,
                             client=self.groq_client)
            if not raw:
                return None
            parsed = json.loads(raw)
            return self._validate_and_fix_llm_json(parsed)
        except Exception as e:
            logger.warning(f"LLM parse failed: {e}")
            return None

    def _validate_and_fix_llm_json(self, obj: dict) -> dict:
        safe = {
            "intent": obj.get("intent", "semantic"),
            "entities": obj.get("entities", []) if isinstance(obj.get("entities"), list) else [],
            "exclusions": obj.get("exclusions", []) if isinstance(obj.get("exclusions"), list) else [],
            "year_range": obj.get("year_range", {}) if isinstance(obj.get("year_range"), dict) else {},
            "rewritten_query": obj.get("rewritten_query", ""),
        }
        if safe["intent"] not in ("director", "actor", "franchise", "semantic"):
            safe["intent"] = "semantic"
        validated_entities = []
        for ent in safe["entities"]:
            if not isinstance(ent, dict):
                continue
            ent_type = ent.get("type")
            ent_name = ent.get("name")
            if ent_type not in ("director", "actor", "franchise"):
                continue
            if not isinstance(ent_name, str) or not ent_name.strip():
                continue
            validated_entities.append({
                "type": ent_type,
                "name": ent_name.strip()[:80]
            })
        safe["entities"] = validated_entities
        safe["exclusions"] = [str(e).strip()[:50] for e in safe["exclusions"] if isinstance(e, str) and e.strip()][:5]
        yr = safe["year_range"]
        yr.setdefault("min", None)
        yr.setdefault("max", None)
        if isinstance(yr["min"], float): yr["min"] = int(yr["min"])
        if isinstance(yr["max"], float): yr["max"] = int(yr["max"])
        safe["rewritten_query"] = re.sub(r"\s+", " ", str(safe["rewritten_query"]).strip())[:200]
        return safe

    def _validate_entities(self, entities):
        director_name = None
        actor_name = None
        for ent in entities:
            name = ent["name"]
            if ent["type"] == "director" and self._count_director_movies(name) >= 3:
                director_name = name
                break
            elif ent["type"] == "actor" and self._count_actor_movies(name) >= 3:
                actor_name = name
                break
        return director_name, actor_name

    def _count_director_movies(self, name: str) -> int:
        return int((self.df['director'].str.lower() == name.lower()).sum())

    def _count_actor_movies(self, name: str) -> int:
        return int(self.df['cast'].str.lower().str.contains(r'\b' + re.escape(name.lower()) + r'\b', na=False).sum())

    # ───── Filmography pulls ───────────────────────
    def _pull_director_films(self, director_name: str):
        director_lower = director_name.lower()
        mask = self.df['director'].str.lower() == director_lower
        if mask.sum() < 3:
            return None
        df = self.df[mask].copy()
        df = df[df['vote_count'] > 50]
        if df.empty:
            return None
        return df.sort_values('popularity_log', ascending=False).head(500)

    def _pull_actor_films(self, actor_name: str):
        actor_lower = actor_name.lower()
        mask = self.df['cast'].str.lower().str.contains(r'\b' + re.escape(actor_lower) + r'\b', na=False)
        if mask.sum() < 3:
            return None
        df = self.df[mask].copy()
        df = df[df['vote_count'] > 5]   # lowered from 20 to catch more films
        if df.empty:
            return None
        return df.sort_values('popularity_log', ascending=False).head(500)

    def _pull_franchise_films(self, franchise_name: str):
        """Return movies whose title, overview, or genres mention the franchise keyword."""
        name_lower = franchise_name.lower()
        mask = (
            self.df['title'].str.lower().str.contains(name_lower, na=False) |
            self.df['overview'].str.lower().str.contains(name_lower, na=False) |
            self.df['genres'].str.lower().str.contains(name_lower, na=False)
        )
        if mask.sum() < 3:
            return None
        df = self.df[mask].copy()
        return df.sort_values('popularity_log', ascending=False).head(500)

    # ───── Main recommend ──────────────────────────
    def recommend(self, query, top_n=10, w_sim=0.6, w_rating=0.2, w_pop=0.2,
                  use_llm_parse=True, generate_explanations=True,
                  exclude_ids: list = None):
        try:
            # 1. Structured NLU
            parsed_llm = None
            if use_llm_parse and self.groq_client:
                parsed_llm = self._llm_parse_query(query)

            robust = RobustParser(query)
            robust_result = robust.get_result()

            if parsed_llm:
                parsed_llm.setdefault("exclusions", [])
                for excl in robust_result.get("exclude", []):
                    if excl.lower() not in (e.lower() for e in parsed_llm["exclusions"]):
                        parsed_llm["exclusions"].append(excl)
                parsed = parsed_llm
            else:
                parsed = {
                    "intent": "semantic",
                    "entities": [],
                    "exclusions": robust_result.get("exclude", []),
                    "year_range": {"min": robust_result.get("year_min"), "max": robust_result.get("year_max")},
                    "rewritten_query": query,
                }

            # ── Robust person resolution (no LLM) ──
            potential_person = robust_result.get('potential_person')
            if potential_person and not parsed.get("entities"):
                director_cnt = self._count_director_movies(potential_person)
                actor_cnt = self._count_actor_movies(potential_person)
                if director_cnt >= 3:
                    parsed["entities"] = [{"type": "director", "name": potential_person}]
                    parsed["intent"] = "director"
                    logger.info(f"Robust parser resolved director: {potential_person}")
                elif actor_cnt >= 3:
                    parsed["entities"] = [{"type": "actor", "name": potential_person}]
                    parsed["intent"] = "actor"
                    logger.info(f"Robust parser resolved actor: {potential_person}")

            # 2. Retrieval routing
            candidates = None
            used_pull = False

            if parsed["intent"] == "director" and parsed.get("entities"):
                director_name, _ = self._validate_entities(parsed["entities"])
                if director_name:
                    director_df = self._pull_director_films(director_name)
                    if director_df is not None and not director_df.empty:
                        candidates = director_df.copy()
                        candidates['similarity'] = 1.0
                        used_pull = True
            elif parsed["intent"] == "actor" and parsed.get("entities"):
                _, actor_name = self._validate_entities(parsed["entities"])
                if actor_name:
                    actor_df = self._pull_actor_films(actor_name)
                    if actor_df is not None and not actor_df.empty:
                        candidates = actor_df.copy()
                        candidates['similarity'] = 1.0
                        used_pull = True
            elif parsed["intent"] == "franchise" and parsed.get("entities"):
                # New franchise handling
                franchise_name = parsed["entities"][0]["name"]
                franchise_df = self._pull_franchise_films(franchise_name)
                if franchise_df is not None and not franchise_df.empty:
                    candidates = franchise_df.copy()
                    candidates['similarity'] = 1.0
                    used_pull = True

            # Semantic / fallback search
            if candidates is None:
                enhanced = parsed.get("rewritten_query", query)
                if robust_result.get('similar_to'):
                    enhanced += f" similar to {robust_result['similar_to']}"
                if robust_result.get('must_include'):
                    enhanced += " " + " ".join(robust_result['must_include'])
                q_vec = self.model.encode([enhanced]).astype('float32')
                q_vec = np.ascontiguousarray(q_vec)
                faiss.normalize_L2(q_vec)
                scores, indices = self.index.search(q_vec, 200)
                candidates = self.df.iloc[indices[0]].copy()
                candidates['similarity'] = scores[0].astype(float)

            if candidates is None or candidates.empty:
                return []

            # 3. Filters
            yr = parsed.get("year_range", {})
            if yr.get("min"):
                candidates = candidates[candidates['year'] >= yr["min"]]
            if yr.get("max"):
                candidates = candidates[candidates['year'] <= yr["max"]]

            if not used_pull and robust_result.get('genre'):
                genre = robust_result['genre'].lower()
                variants = ['sci-fi', 'science fiction', 'scifi'] if genre == 'sci-fi' else [genre]
                candidates = candidates[candidates['genres'].str.lower().str.contains('|'.join(variants), na=False)]

            exclusions = parsed.get("exclusions", [])
            if exclusions:
                genre_keywords = {'documentary','documentaries','comedy','comedies','action','adventure',
                                  'animation','anime','sci-fi','scifi','science fiction','thriller',
                                  'horror','romance','musical','short','western','war','history','family',
                                  'fantasy','mystery','crime','drama'}
                for term in exclusions:
                    term_lower = term.lower().strip()
                    if term_lower in genre_keywords:
                        genre_map = {'documentaries':'documentary','comedies':'comedy','sci-fi':'sci-fi',
                                     'scifi':'sci-fi','science fiction':'sci-fi','anime':'animation',
                                     'thrillers':'thriller','musicals':'musical','shorts':'short'}
                        search_genre = genre_map.get(term_lower, term_lower)
                        candidates = candidates[~candidates['genres'].str.lower().str.contains(re.escape(search_genre), na=False)]
                    # Title and overview exclusion (handles movie titles like "Oppenheimer")
                    candidates = candidates[
                        ~candidates['title'].str.lower().str.contains(r'\b'+re.escape(term_lower)+r'\b', na=False) &
                        ~candidates['overview'].str.lower().str.contains(r'\b'+re.escape(term_lower)+r'\b', na=False)
                    ]

            # 4. Boosts & scoring
            if not used_pull:
                q_words = set(re.findall(r'[a-z0-9]+', query.lower())) - {'a','an','the','is','of','in','on','to','for','with','and','or','but','not','no','without','like','movies','movie'}
                if q_words:
                    def kw_score(row):
                        soup = str(row.get('soup','')).lower()
                        return sum(1 for w in q_words if w in soup) / len(q_words)
                    candidates['kw_boost'] = candidates.apply(kw_score, axis=1).astype(float)
                else:
                    candidates['kw_boost'] = 0.0

                # ── Optional: extra boost for franchise keywords ──
                franchise_keywords = {'marvel', 'dc', 'pixar', 'disney', 'lego', 'potter'}
                q_words_franchise = q_words & franchise_keywords
                if q_words_franchise:
                    # Additional boost for each franchise word found in soup
                    candidates['kw_boost'] += candidates.apply(
                        lambda row: sum(1 for w in q_words_franchise if w in str(row.get('soup','')).lower()) / max(1, len(q_words)),
                        axis=1
                    ).astype(float) * 0.5

            similar_to_title = robust_result.get('similar_to')
            if similar_to_title and not used_pull:
                target_movie = self.df[self.df['title'].str.lower() == similar_to_title.lower()]
                if target_movie.empty:
                    target_movie = self.df[self.df['title'].str.lower().str.contains(re.escape(similar_to_title.lower()), na=False)]
                if not target_movie.empty:
                    target_genres = target_movie.iloc[0]['genres'].lower().split()
                    if target_genres:
                        genre_pattern = '|'.join(re.escape(g) for g in target_genres if g)
                        if genre_pattern:
                            candidates['genre_similarity_boost'] = candidates['genres'].str.lower().str.contains(genre_pattern, na=False).astype(float) * 0.50
                        else:
                            candidates['genre_similarity_boost'] = 0.0
                    else:
                        candidates['genre_similarity_boost'] = 0.0
                else:
                    candidates['genre_similarity_boost'] = 0.0
            else:
                candidates['genre_similarity_boost'] = 0.0

            candidates['actor_boost'] = 0.0
            candidates['director_boost'] = 0.0
            if not used_pull and parsed.get("entities"):
                for ent in parsed["entities"]:
                    if ent["type"] == "actor":
                        actor_lower = ent["name"].lower()
                        candidates['actor_boost'] += candidates['cast'].fillna('').str.lower().apply(lambda x: 0.50 if actor_lower in x else 0.0).astype(float)
                    elif ent["type"] == "director":
                        director_lower = ent["name"].lower()
                        candidates['director_boost'] += candidates['director'].fillna('').str.lower().apply(lambda x: 0.50 if director_lower in x else 0.0).astype(float)

            candidates['year_boost'] = 0.0
            if yr.get("min") or yr.get("max"):
                target = yr.get("min") or yr.get("max")
                if target:
                    candidates['year_boost'] = candidates['year'].apply(lambda x: 0.30 if abs(int(x)-target)<=2 else 0.0).astype(float)

            CURRENT_YEAR = 2026
            candidates['recency_boost'] = (
                (candidates['year'] >= CURRENT_YEAR - 2).astype(float) * 0.15 +
                (candidates['year'] >= CURRENT_YEAR - 5).astype(float) * 0.05
            )

            is_doc = candidates['genres'].str.lower().str.contains('documentary', na=False)
            if not robust_result.get('genre') or robust_result['genre'] != 'documentary':
                candidates['doc_penalty'] = is_doc.astype(float) * -0.30
                candidates['movie_priority'] = (~is_doc).astype(float) * 0.10
            else:
                candidates['doc_penalty'] = 0.0
                candidates['movie_priority'] = 0.0

            if exclude_ids:
                candidates = candidates[~candidates['movie_id'].isin(exclude_ids)]

            for col in ['similarity','kw_boost','director_boost','actor_boost','year_boost','doc_penalty','movie_priority','avg_rating','popularity_log','genre_similarity_boost','recency_boost']:
                if col not in candidates.columns:
                    candidates[col] = 0.0
                candidates[col] = candidates[col].astype(float)

            rating_norm = candidates['avg_rating'] / 10.0
            pop_norm = candidates['popularity_log'] / (candidates['popularity_log'].max() + 1e-8)

            candidates['hybrid_score'] = (
                w_sim * candidates['similarity'] +
                w_rating * rating_norm +
                w_pop * pop_norm +
                0.1 * candidates['kw_boost'] +
                candidates['director_boost'] +
                candidates['actor_boost'] +
                candidates['year_boost'] +
                candidates['doc_penalty'] +
                candidates['movie_priority'] +
                candidates['genre_similarity_boost'] +
                candidates['recency_boost']
            )
            candidates = candidates.sort_values('hybrid_score', ascending=False).head(top_n)

            results = []
            for _, row in candidates.iterrows():
                item = {
                    "movie_id": int(row['movie_id']),
                    "title": row['title'],
                    "year": int(row['year']),
                    "genres": row.get('genres', ''),
                    "overview": str(row.get('overview', ''))[:300] + ("..." if len(str(row.get('overview', '')))>300 else ""),
                    "avg_rating": float(row['avg_rating']),
                    "rating_count": int(row['popularity_log']),
                    "similarity": float(row['similarity']),
                    "hybrid_score": float(row['hybrid_score']),
                }
                if generate_explanations:
                    item['explanation'] = _generate_explanation(row, robust_result, self.groq_client)
                else:
                    item['explanation'] = None
                results.append(item)

            if self.streaming_lookup:
                for item in results:
                    key = f"{item['title'].lower()}||{item['year']}"
                    if key in self.streaming_lookup:
                        item["watch_providers"] = self.streaming_lookup[key]

            return results

        except Exception as e:
            logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
            return []