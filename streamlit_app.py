"""
streamlit_app.py - MovieMind user interface.
Uses the recommender module to fetch and display results.
"""
import streamlit as st
from recommender import recommend

st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")
st.title("🎬 MovieMind: Conversational Movie Recommender")
st.caption("Describe what you want to watch – I'll find it and tell you why.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.checkbox("Use AI to understand my query", value=True)
    gen_explain = st.checkbox("Generate explanations", value=True)
    w_sim = st.slider("Similarity", 0.0, 1.0, 0.6)
    w_rating = st.slider("Rating", 0.0, 1.0, 0.2)
    w_pop = 1.0 - w_sim - w_rating
    st.caption(f"Popularity weight: {w_pop:.2f}")
    top_n = st.slider("Results", 5, 20, 10)

# Main input
query = st.text_input("What kind of movie are you in the mood for?",
                      placeholder="e.g., a mind‑bending sci‑fi like Inception")

if st.button("Recommend", type="primary") and query:
    with st.spinner("Finding movies..."):
        results = recommend(query, top_n, w_sim, w_rating, w_pop, use_llm, gen_explain)
    if not results:
        st.warning("No movies found.")
    else:
        for movie in results:
            with st.container():
                st.markdown(f"### {movie['title']} ({int(movie['year'])})")
                if movie['explanation']:
                    st.markdown(f"*💡 {movie['explanation']}*")
                if movie['genres']:
                    st.markdown(" ".join([f"`{g}`" for g in movie['genres'].split()]))
                if movie['overview']:
                    st.caption(movie['overview'])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rating", f"{movie['avg_rating']:.1f}")
                c2.metric("Votes", f"{movie['rating_count']:,}")
                c3.metric("Similarity", f"{movie['similarity']:.2f}")
                c4.metric("Score", f"{movie['hybrid_score']:.2f}")
                st.progress(movie['hybrid_score'], text=f"Match: {movie['hybrid_score']:.2f}")
                st.divider()