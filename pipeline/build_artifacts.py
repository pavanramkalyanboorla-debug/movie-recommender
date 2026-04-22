import os
import re
import pickle
import logging
import ast
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "./data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def clean_title(title):
    """Remove year suffix like ' (1989)' from title."""
    if pd.isna(title):
        return ""
    return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()


def clean_genres(genre_str):
    """Remove noise tags like 'IMAX' and '(no genres listed)'."""
    if pd.isna(genre_str) or genre_str == "(no genres listed)":
        return ""
    noise = {'IMAX', '(no genres listed)'}
    genres = [g.strip() for g in str(genre_str).split('|') if g.strip() not in noise]
    return " ".join(genres)


def extract_json_names(json_str, top_n=None):
    """
    Convert TMDB JSON strings (e.g., keywords, cast, crew) to space‑separated names.
    If top_n is provided, only take first N entries.
    """
    if not json_str or pd.isna(json_str) or json_str == "[]":
        return ""
    try:
        data = ast.literal_eval(json_str)
        names = [i['name'] for i in data]
        if top_n:
            names = names[:top_n]
        return " ".join(names)
    except (ValueError, SyntaxError, KeyError):
        return str(json_str).replace('|', ' ')


def extract_director(crew_json):
    """Extract director name from crew JSON."""
    if not crew_json or pd.isna(crew_json) or crew_json == "[]":
        return ""
    try:
        data = ast.literal_eval(crew_json)
        for person in data:
            if person.get('job') == 'Director':
                return person.get('name', '')
    except:
        pass
    return ""


def main():
    logger.info("Starting data pipeline...")

    # ------------------------------------------------------------
    # Notebook 1: Data Preparation
    # ------------------------------------------------------------
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    tmdb = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))

    # Clean links and merge
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    df = movies.merge(tmdb, left_on='tmdbId', right_on='id', how='left')

    # Select columns we need (add cast and crew if present)
    base_cols = ['movieId', 'title_x', 'genres_x', 'keywords', 'overview',
                 'popularity', 'vote_average', 'vote_count', 'release_date']
    extra_cols = []
    if 'cast' in df.columns:
        extra_cols.append('cast')
    if 'crew' in df.columns:
        extra_cols.append('crew')
    df = df[base_cols + extra_cols]
    df = df.rename(columns={'title_x': 'title', 'genres_x': 'genres'})

    # Clean title (remove year suffix)
    df['title'] = df['title'].apply(clean_title)

    # Extract year from release_date
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['year'] = df['year'].fillna(2000).astype(int)

    # Fill missing text fields
    df['overview'] = df['overview'].fillna("")
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)

    # Clean genres
    df['cleaned_genres'] = df['genres'].apply(clean_genres)

    # Fallback for empty overviews
    df['overview'] = df.apply(
        lambda x: f"{x['title']} {x['cleaned_genres']}" if not x['overview'] or x['overview'] == ""
        else x['overview'],
        axis=1
    )

    # Deduplicate by title+year, keeping most popular first
    df = df.sort_values(by='vote_count', ascending=False)
    df = df.drop_duplicates(subset=['title', 'year'], keep='first')

    # Extract cast and director if available
    if 'cast' in df.columns:
        df['cast_names'] = df['cast'].apply(lambda x: extract_json_names(x, top_n=5))
    else:
        df['cast_names'] = ""
    if 'crew' in df.columns:
        df['director'] = df['crew'].apply(extract_director)
    else:
        df['director'] = ""

    # Build the "soup" for embeddings
    df['soup'] = df.apply(
        lambda row: " ".join([
            str(row['title']),
            str(row['cleaned_genres']),
            extract_json_names(row['keywords']),  # all keywords
            str(row['overview']),
            str(row['cast_names']),
            str(row['director'])
        ]).lower(),
        axis=1
    )

    # Compute movie‑level stats from ratings (for ranking)
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    df = df.merge(movie_stats, on='movieId', how='left')
    df['avg_rating'] = df['avg_rating'].fillna(0)
    df['rating_count'] = df['rating_count'].fillna(0)
    df['popularity_log'] = np.log1p(df['rating_count'])
    df['norm_popularity'] = (df['popularity_log'] - df['popularity_log'].min()) / \
                            (df['popularity_log'].max() - df['popularity_log'].min())

    # Keep final columns (include genres, overview, and any extra for UI)
    final_cols = ['movieId', 'title', 'year', 'genres', 'overview', 'soup',
                  'norm_popularity', 'vote_average', 'vote_count',
                  'avg_rating', 'rating_count', 'popularity_log']
    # Optionally keep cast/director if you want to display them later
    if 'cast_names' in df.columns:
        final_cols.append('cast_names')
    if 'director' in df.columns:
        final_cols.append('director')

    df_final = df[final_cols].copy()

    # Save processed dataframe
    parquet_path = os.path.join(ARTIFACTS_DIR, "movies_processed_final.parquet")
    df_final.to_parquet(parquet_path, index=False)
    logger.info(f"Saved processed dataframe to {parquet_path}")

    # ------------------------------------------------------------
    # Notebook 2: Retrieval Artifacts
    # ------------------------------------------------------------
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Generating embeddings for all movies...")
    embeddings = model.encode(df_final['soup'].tolist(), show_progress_bar=True, convert_to_numpy=True)

    # FAISS Index (cosine similarity via inner product)
    embeddings_f32 = np.ascontiguousarray(embeddings.astype('float32'))
    dimension = embeddings_f32.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_f32)
    index.add(embeddings_f32)

    faiss_path = os.path.join(ARTIFACTS_DIR, "movies_faiss.index")
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index saved to {faiss_path}")

    # TF-IDF vectorizer (optional fallback)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    _ = tfidf.fit_transform(df_final['soup'])
    tfidf_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    logger.info(f"TF-IDF vectorizer saved to {tfidf_path}")

    # Model name
    with open(os.path.join(ARTIFACTS_DIR, "model_name.txt"), "w") as f:
        f.write("all-MiniLM-L6-v2")

    logger.info("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()