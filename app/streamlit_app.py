"""
streamlit_app.py — MovieMind Premium UI (Netflix‑inspired)
Dark theme · Red accent · Glassmorphism cards · Show‑more‑different button
"""
import streamlit as st
import requests

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MovieMind — Conversational Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# NETFLIX‑INSPIRED DARK CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #0b0b0b;
    }
    .stApp {
        background: linear-gradient(180deg, #0b0b0b 0%, #1a1a1a 100%);
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #e50914, #b20710);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 0 10px rgba(229,9,20,0.5));
    }
    .sub-header {
        text-align: center;
        color: #b3b3b3;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(20,20,20,0.9);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(229,9,20,0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e50914;
        font-weight: 600;
    }

    [data-testid="stTextInput"] input {
        border-radius: 6px;
        border: 1px solid rgba(229,9,20,0.3);
        background: rgba(255,255,255,0.06);
        color: #fff;
        padding: 0.8rem 1rem;
        font-size: 1.05rem;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #e50914;
        box-shadow: 0 0 0 3px rgba(229,9,20,0.2);
    }

    .stButton > button {
        width: 100%;
        border-radius: 6px;
        background: linear-gradient(135deg, #e50914, #b20710);
        color: #fff;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        padding: 0.7rem 1.5rem;
        transition: all 0.25s;
        box-shadow: 0 2px 12px rgba(229,9,20,0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #f40612, #e50914);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(229,9,20,0.6);
    }

    .movie-card {
        background: rgba(30,30,30,0.7);
        backdrop-filter: blur(12px);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid rgba(229,9,20,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .movie-card:hover {
        border-color: rgba(229,9,20,0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(229,9,20,0.15);
    }
    .card-title { font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem; }
    .card-year { font-size: 0.85rem; color: #808080; margin-left: 0.5rem; }
    .card-explanation { font-size: 0.95rem; color: #e50914; font-style: italic; margin-bottom: 0.6rem; }
    .card-overview { font-size: 0.85rem; color: #b3b3b3; line-height: 1.5; margin-bottom: 0.6rem; }
    .card-stats { display: flex; gap: 1.2rem; flex-wrap: wrap; }
    .stat-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.4px; }
    .stat-value { font-size: 0.95rem; font-weight: 600; color: #e0e0e0; }

    .genre-chip {
        display: inline-block;
        background: rgba(229,9,20,0.15);
        color: #e50914;
        border: 1px solid rgba(229,9,20,0.3);
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.7rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }

    /* 🆕 Streaming provider badges */
    .provider-badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.7rem;
        margin-right: 0.3rem;
        margin-top: 0.4rem;
    }

    .empty-state {
        display: flex; align-items: center; justify-content: center;
        min-height: 350px;
        border: 2px dashed rgba(229,9,20,0.15);
        border-radius: 8px;
        background: rgba(229,9,20,0.02);
        color: #6b7280;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0b0b0b; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKEND API HELPER
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000"

def get_recommendations(query, top_n, w_sim, w_rating, w_pop, use_llm, gen_explain, exclude_ids=None):
    payload = {
        "query": query,
        "top_n": top_n,
        "w_sim": w_sim,
        "w_rating": w_rating,
        "w_pop": w_pop,
        "use_llm_parse": use_llm,
        "generate_explanations": gen_explain,
    }
    if exclude_ids:
        payload["exclude_ids"] = exclude_ids
    try:
        resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["results"]
    except Exception:
        pass
    return []

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "shown_movie_ids" not in st.session_state:
    st.session_state.shown_movie_ids = set()
if "all_results" not in st.session_state:
    st.session_state.all_results = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🎬 MovieMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Conversational movie discovery · Semantic search · AI‑powered explanations</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Tuning")
    use_llm = st.checkbox("Use AI to understand my query", value=True)
    gen_explain = st.checkbox("Generate explanations", value=True)
    w_sim = st.slider("Content Similarity", 0.0, 1.0, 0.60)
    w_rating = st.slider("Rating Weight", 0.0, 1.0, 0.20)
    w_pop = max(0.0, 1.0 - w_sim - w_rating)
    st.caption(f"Popularity weight: **{w_pop:.2f}**")
    top_n = st.slider("Results", 3, 20, 8)

    st.divider()
    st.markdown("### 🧪 Example Queries")
    for ex in [
        "Mind‑bending sci‑fi like Inception",
        "Romantic comedy but not cheesy",
        "Batman movies without the Joker",
        "Space exploration before 2000",
        "Dark psychological thriller like Gone Girl"
    ]:
        st.caption(f"• {ex}")

# ─────────────────────────────────────────────
# MAIN INPUT
# ─────────────────────────────────────────────
query = st.text_input(
    "What kind of movie are you in the mood for?",
    placeholder="e.g. Christopher Nolan movies without documentary",
    key="main_query"
)

if st.button("🔍 Discover Movies", type="primary") and query.strip():
    # Reset for new search
    st.session_state.shown_movie_ids = set()
    st.session_state.all_results = []
    st.session_state.current_query = query.strip()

    with st.spinner("🎥 Searching across thousands of movies..."):
        results = get_recommendations(
            st.session_state.current_query,
            top_n=top_n,
            w_sim=w_sim,
            w_rating=w_rating,
            w_pop=w_pop,
            use_llm=use_llm,
            gen_explain=gen_explain,
        )

    if not results:
        st.warning("No movies found. Try a different description — for example, a movie title or a genre.")
    else:
        st.session_state.shown_movie_ids = {r["movie_id"] for r in results}
        st.session_state.all_results = results
        st.rerun()

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
if st.session_state.all_results and st.session_state.current_query:
    st.markdown(f"### 📽️ {len(st.session_state.all_results)} recommendations for \"{st.session_state.current_query}\"")

    for movie in st.session_state.all_results:
        genres_str = ""
        if movie.get("genres"):
            for g in movie["genres"].split():
                genres_str += f'<span class="genre-chip">{g}</span>'

        explanation_html = ""
        if movie.get("explanation"):
            explanation_html = f'<div class="card-explanation">💡 {movie["explanation"]}</div>'

        # 🆕 Watch provider badges
        watch_html = ""
        if movie.get("watch_providers"):
            for p in movie["watch_providers"][:3]:   # show max 3
                watch_html += f'<span class="provider-badge">📺 {p}</span>'

        st.markdown(f"""
        <div class="movie-card">
            <div class="card-title">
                {movie['title']}
                <span class="card-year">({int(movie['year'])})</span>
            </div>
            {explanation_html}
            <div class="card-overview">{movie.get('overview', '')[:280]}{"..." if len(movie.get("overview", "")) > 280 else ""}</div>
            <div>{genres_str} {watch_html}</div>
            <div class="card-stats" style="margin-top: 0.8rem;">
                <div><div class="stat-label">Rating</div><div class="stat-value">⭐ {movie['avg_rating']:.1f}</div></div>
                <div><div class="stat-label">Votes</div><div class="stat-value">{movie['rating_count']:,}</div></div>
                <div><div class="stat-label">Similarity</div><div class="stat-value">{movie['similarity']:.2f}</div></div>
                <div><div class="stat-label">Score</div><div class="stat-value" style="color:#e50914;">{movie['hybrid_score']:.2f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── "Show more, just different" ──
    if st.button("🔄 Show more, just different", key="show_more"):
        with st.spinner("Finding something different…"):
            new_results = get_recommendations(
                st.session_state.current_query,
                top_n=top_n,
                w_sim=w_sim,
                w_rating=w_rating,
                w_pop=w_pop,
                use_llm=use_llm,
                gen_explain=gen_explain,
                exclude_ids=list(st.session_state.shown_movie_ids),
            )
        if new_results:
            for r in new_results:
                st.session_state.shown_movie_ids.add(r["movie_id"])
            st.session_state.all_results.extend(new_results)
            st.rerun()
        else:
            st.info("No more different recommendations found.")

else:
    st.markdown("""
    <div class="empty-state">
        <div style="text-align:center;">
            <p style="font-size:3rem; margin-bottom:0;">🎬</p>
            <p style="font-size:1.2rem; font-weight:500;">Describe a movie you'd love to watch</p>
            <p style="color:#6b7280;">Use natural language — the smarter your description, the better the match</p>
        </div>
    </div>
    """, unsafe_allow_html=True)