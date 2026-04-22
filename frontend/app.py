"""
frontend/app.py
---------------
Streamlit UI for MovieMind with improved layout and details.
"""

import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")

# Custom CSS for better card styling
st.markdown("""
<style>
    .movie-card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff4b4b;
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .movie-year {
        color: #aaaaaa;
        font-size: 1.2rem;
        margin-left: 0.5rem;
    }
    .explanation {
        color: #90ee90;
        font-style: italic;
        margin: 0.5rem 0;
    }
    .overview {
        color: #cccccc;
        margin: 0.8rem 0;
    }
    .genre-tag {
        background-color: #333;
        padding: 4px 10px;
        border-radius: 16px;
        margin-right: 6px;
        font-size: 0.8rem;
        display: inline-block;
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 MovieMind: Conversational Movie Recommender")
st.caption("Describe what you want to watch – I'll find it and tell you why.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.checkbox("Use AI to understand my query", value=True)
    gen_explain = st.checkbox("Generate explanations", value=True)
    st.divider()
    st.header("🎚️ Ranking Weights")
    w_sim = st.slider("Similarity", 0.0, 1.0, 0.6, 0.1, help="How important is semantic match?")
    w_rating = st.slider("Rating", 0.0, 1.0, 0.2, 0.1, help="How important is average user rating?")
    w_pop = 1.0 - w_sim - w_rating
    st.caption(f"Popularity weight (auto): {w_pop:.2f}")
    top_n = st.slider("Number of results", 5, 20, 10)

# Main input
query = st.text_input(
    "What kind of movie are you in the mood for?",
    placeholder="e.g., a mind-bending sci-fi like Inception but not too violent"
)

if st.button("🔍 Recommend", type="primary") and query:
    with st.spinner("Finding the perfect movies for you..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "query": query,
                    "top_n": top_n,
                    "w_sim": w_sim,
                    "w_rating": w_rating,
                    "w_pop": w_pop,
                    "use_llm_parse": use_llm,
                    "generate_explanations": gen_explain,
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                if not data:
                    st.warning("No movies found. Try a different query.")
                else:
                    for movie in data:
                        with st.container():
                            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                            
                            # Title and year
                            st.markdown(
                                f'<span class="movie-title">{movie["title"]}</span>'
                                f'<span class="movie-year">({int(movie["year"])})</span>',
                                unsafe_allow_html=True
                            )
                            
                            # Explanation
                            if movie.get('explanation'):
                                st.markdown(f'<div class="explanation">💡 {movie["explanation"]}</div>', unsafe_allow_html=True)
                            
                            # Genres as chips
                            if movie.get('genres'):
                                genres = movie['genres'].split()
                                genre_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in genres[:5]])
                                st.markdown(f'<div style="margin:8px 0;">{genre_html}</div>', unsafe_allow_html=True)
                            
                            # Overview
                            if movie.get('overview'):
                                st.markdown(f'<div class="overview">📖 {movie["overview"]}</div>', unsafe_allow_html=True)
                            
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("⭐ Avg Rating", f"{movie['avg_rating']:.1f}")
                            with col2:
                                st.metric("👥 Votes", f"{movie['rating_count']:,}")
                            with col3:
                                st.metric("🔍 Similarity", f"{movie['similarity']:.2f}")
                            with col4:
                                st.metric("🏆 Hybrid Score", f"{movie['hybrid_score']:.2f}")
                            
                            # Score bar
                            st.progress(
                                min(max(movie['hybrid_score'], 0.0), 1.0),
                                text=f"Match Score: {movie['hybrid_score']:.2f}"
                            )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"API error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend. Is the API running?")