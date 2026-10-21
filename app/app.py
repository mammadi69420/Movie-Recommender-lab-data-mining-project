import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data, RANDOM_STATE
from src.popularity import PopularityRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import ItemItemKNNRecommender, SVDRecommender
from src.association_rules import AssociationRuleRecommender
from src.evaluation import temporal_train_test_split, precision_at_k, recall_at_k, ndcg_at_k
from src.tmdb import fetch_movie_details, get_tmdb_api_key
from src.eda import (
    compute_kpis,
    plot_rating_distribution,
    plot_user_rating_tendencies,
    plot_genre_popularity,
    plot_genre_quality,
    plot_genre_cooccurrence,
    plot_long_tail,
    plot_user_activity,
    plot_temporal_trends,
    get_blockbusters_and_gems,
)

# Page setup
st.set_page_config(
    page_title="Movie Recommender Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with movie poster collage background
st.markdown("""
<style>
    /* Fixed background collage with dark gradient overlay */
    [data-testid="stAppViewContainer"] {
        background: 
            linear-gradient(135deg, rgba(13, 17, 23, 0.90) 0%, rgba(15, 23, 42, 0.93) 100%),
            url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=2000&q=80") center center / cover no-repeat fixed;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.96) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Force all text inside sidebar to be bright and clear */
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
    }

    /* Individual outlined borders and clear highlights for sidebar navigation options */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(30, 41, 59, 0.90) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        transition: all 0.25s ease;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.25) !important;
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8);
    }

    /* Main text clarity, subheaders, and section headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    }

    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FF6B6B, #FFA07A, #FEE140);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .sub-header {
        color: #F8FAFC !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 1.6rem;
        text-shadow: 0 1px 8px rgba(0,0,0,0.9);
    }

    /* Movie card styling & Streamlit container card styling */
    .movie-card,
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        padding: 14px 12px !important;
        background: rgba(30, 41, 59, 0.90) !important;
        backdrop-filter: blur(14px);
        border: 1.5px solid rgba(255, 255, 255, 0.20) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
        margin-bottom: 14px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .movie-card:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(56, 189, 248, 0.35);
        border-color: #38BDF8 !important;
    }
    
    /* Uniform movie poster styling with consistent 2:3 aspect ratio */
    [data-testid="stVerticalBlockBorderWrapper"] img,
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        width: 100% !important;
        aspect-ratio: 2 / 3 !important;
        object-fit: cover !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    }

    .movie-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF !important;
        margin-top: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-shadow: 0 1px 4px rgba(0,0,0,0.9);
    }
    .genre-tag {
        font-size: 0.82rem;
        font-weight: 700;
        color: #38BDF8 !important;
        margin-top: 2px;
        text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }
    
    /* Rank badge with hover tooltip */
    .rank-badge-wrapper {
        position: relative;
        display: inline-block;
        margin-top: 6px;
        margin-bottom: 6px;
        cursor: pointer;
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: #0F172A !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        padding: 5px 14px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(245, 158, 11, 0.5) !important;
        border: 1px solid #FCD34D !important;
        letter-spacing: 0.3px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .rank-badge:hover {
        transform: scale(1.08);
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.7) !important;
    }
    .rank-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 110%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.97);
        color: #F8FAFC;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid rgba(56, 189, 248, 0.4);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
        white-space: nowrap;
        z-index: 999;
        transition: opacity 0.25s ease, visibility 0.25s ease;
        pointer-events: none;
    }
    .rank-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: rgba(15, 23, 42, 0.97) transparent transparent transparent;
    }
    .rank-badge-wrapper:hover .rank-tooltip {
        visibility: visible;
        opacity: 1;
    }

    .movie-card p, .movie-card span, .movie-card [data-testid="stCaptionContainer"] {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }

    /* Selectbox label, input box, and dropdown clarity */
    [data-testid="stSelectbox"] label p,
    [data-testid="stSelectbox"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.9) !important;
    }
    
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(30, 41, 59, 0.95) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.5) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Slider text and labels: make 'Rating' and score numbers bright */
    [data-testid="stSlider"] label p,
    [data-testid="stSlider"] div,
    [data-testid="stSlider"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.9);
    }
    
    /* Popover synopsis button */
    [data-testid="stPopover"] button {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stPopover"] button:hover {
        border-color: #38BDF8 !important;
    }

    /* Team cards and badges */
    .team-badge-leader {
        display: inline-block;
        background: linear-gradient(90deg, #F59E0B, #D97706);
        color: white !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4);
    }
    .team-badge-member {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #F8FAFC !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .team-card {
        background: rgba(30, 41, 59, 0.92) !important;
        backdrop-filter: blur(14px);
        border: 1.5px solid rgba(255, 255, 255, 0.20) !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .team-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF !important;
        letter-spacing: 0.2px;
    }

    /* List items in side-by-side comparison */
    .rec-item {
        color: #FFFFFF !important;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 8px 12px;
        background: rgba(30, 41, 59, 0.85) !important;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Tabs text styling */
    button[data-baseweb="tab"] {
        color: #E2E8F0 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }

    /* Distinct section panels for algorithm comparison */
    .section-panel {
        background: rgba(30, 41, 59, 0.88);
        backdrop-filter: blur(14px);
        border-radius: 14px;
        padding: 20px 18px 16px 18px;
        margin-bottom: 10px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    }
    .section-panel-cb {
        border: 2px solid rgba(20, 184, 166, 0.5);
    }
    .section-panel-cb:hover {
        border-color: #14B8A6;
        box-shadow: 0 6px 28px rgba(20, 184, 166, 0.25);
    }
    .section-panel-cf {
        border: 2px solid rgba(168, 85, 247, 0.5);
    }
    .section-panel-cf:hover {
        border-color: #A855F7;
        box-shadow: 0 6px 28px rgba(168, 85, 247, 0.25);
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        padding-bottom: 12px;
    }
    .section-header-cb {
        border-bottom: 2px solid rgba(20, 184, 166, 0.4);
    }
    .section-header-cf {
        border-bottom: 2px solid rgba(168, 85, 247, 0.4);
    }
    .section-icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .section-title-cb {
        color: #14B8A6 !important;
    }
    .section-title-cf {
        color: #A855F7 !important;
    }
    .section-desc {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 2px;
    }
    .section-divider {
        width: 3px;
        background: linear-gradient(180deg, transparent 0%, rgba(148, 163, 184, 0.3) 30%, rgba(148, 163, 184, 0.3) 70%, transparent 100%);
        border-radius: 3px;
        min-height: 100px;
        align-self: stretch;
    }

    /* EDA KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.90) !important;
        backdrop-filter: blur(14px);
        border: 1.5px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #38BDF8 !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.70rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
        line-height: 1.1;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
    }
    .kpi-sub {
        font-size: 0.74rem;
        color: #38BDF8 !important;
        margin-top: 5px;
        font-weight: 600;
    }

    /* EDA Chart Box Container */
    .chart-box {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(12px);
        border: 1.5px solid rgba(255, 255, 255, 0.16);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }

    /* EDA Data Mining Takeaway Alert Box */
    .eda-insight-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-left: 4px solid #38BDF8;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .eda-insight-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: 0.3px;
        margin-bottom: 6px;
    }
    .eda-insight-desc {
        font-size: 0.85rem;
        color: #E2E8F0;
        line-height: 1.6;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_and_prepare_models():
    """Download data and train recommenders with fixed RANDOM_STATE."""
    data = load_data()
    ratings = data["ratings"]
    movies = data["movies"]
    tags = data["tags"]
    links = data["links"]

    # Temporal split 80/20
    train_ratings, test_ratings = temporal_train_test_split(ratings, test_ratio=0.2)

    # 1. Popularity Baseline
    pop_model = PopularityRecommender().fit(train_ratings, movies)

    # 2. Content-Based
    cb_model = ContentBasedRecommender(max_features=4000).fit(movies, tags, train_ratings)

    # 3. Collaborative Filtering (Item-Item kNN)
    knn_model = ItemItemKNNRecommender(n_neighbors=15).fit(train_ratings, movies)

    # 4. Association Rules (FP-Growth)
    ar_model = AssociationRuleRecommender(min_support=0.03, min_confidence=0.2).fit(train_ratings, movies)

    return {
        "ratings": ratings,
        "train_ratings": train_ratings,
        "test_ratings": test_ratings,
        "movies": movies,
        "tags": tags,
        "links": links,
        "pop_model": pop_model,
        "cb_model": cb_model,
        "knn_model": knn_model,
        "ar_model": ar_model,
    }


with st.spinner("Loading MovieLens dataset and fitting recommender models..."):
    state = load_and_prepare_models()

movies_df = state["movies"]
links_df = state["links"]

# Merge links for TMDB ids
movies_with_links = movies_df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")
movie_lookup = movies_with_links.set_index("movieId").to_dict(orient="index")

# Sidebar
st.sidebar.markdown("## 🎬 Movie Recommender Lab")
st.sidebar.caption("DS-311 Data Mining · Fall 2026 · NUST")
app_mode = st.sidebar.radio(
    "Navigation",
    [
        "🌟 Recommender Comparison",
        "📊 Exploratory Data Analysis (EDA)",
        "⭐ Rate Movies (Interactive)",
        "ℹ️ About & Team",
    ]
)

# TMDB Key Configuration in sidebar
with st.sidebar.expander("🔑 TMDB API Key Settings"):
    custom_key = st.text_input("Enter TMDB API Key", value=get_tmdb_api_key(), type="password")
    if custom_key:
        os.environ["TMDB_API_KEY"] = custom_key
        st.success("API Key applied for TMDB posters!")
    if st.button("🔄 Refresh Posters / Clear Cache"):
        st.cache_data.clear()
        st.rerun()


# Tooltip descriptions per algorithm
ALGORITHM_TOOLTIPS = {
    "popularity": "Ranked by global weighted rating (IMDB formula: vote count × avg rating)",
    "content_based": "Ranked by genre, title & tag similarity to your taste (TF-IDF cosine)",
    "collaborative": "Ranked by how much similar users enjoyed this (item-item kNN)",
    "association_rules": "Ranked by 'people who liked your movies also liked this' (FP-Growth)",
}


def render_movie_grid(movies_list, cols_count=5, algorithm=""):
    """Render a responsive grid of movie cards with posters and rank badges."""
    if movies_list.empty:
        st.info("No recommendations found for this criterion.")
        return

    df = movies_list.copy()
    tooltip_text = ALGORITHM_TOOLTIPS.get(algorithm, "Ranked by model relevance score")

    # Render row-by-row in chunks of cols_count to ensure clean layout
    for row_start in range(0, len(df), cols_count):
        row_slice = df.iloc[row_start : row_start + cols_count]
        cols = st.columns(cols_count)
        for idx_in_row, (_, row) in enumerate(row_slice.iterrows()):
            rank = row_start + idx_in_row + 1
            col = cols[idx_in_row]
            mid = row["movieId"]
            info = movie_lookup.get(mid, {})
            tmdb_id = row.get("tmdbId", info.get("tmdbId"))
            title = info.get("title", row.get("title", f"Movie #{mid}"))
            genres = info.get("genres", row.get("genres", []))
            genres_display = ", ".join(genres[:2]) if isinstance(genres, list) else str(genres)

            details = fetch_movie_details(tmdb_id, title=title)

            with col:
                with st.container(border=True):
                    st.image(details["poster_url"], use_container_width=True)
                    st.markdown(f'<div class="movie-title" title="{title}">{title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="genre-tag">{genres_display} · {details["release_year"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div class="rank-badge-wrapper">
                        <div class="rank-badge">🏆 #{rank} Match</div>
                        <div class="rank-tooltip">{tooltip_text}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    with st.popover("📖 Synopsis"):
                        st.write(f"**{title}**")
                        st.write(details["overview"])


# --- PAGE 1: Side-by-Side Comparison ---
if app_mode == "🌟 Recommender Comparison":
    st.markdown('<div class="main-header">Multi-Algorithm Recommender Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare top-10 recommendations side by side across 4 distinct data mining methods.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="team-card" style="margin-bottom: 20px;">
        <span style="font-size: 1.05rem; color: #FFFFFF; font-weight: 600;">
            👤 Target User Profile & Selection
        </span>
    </div>
    """, unsafe_allow_html=True)

    test_users = sorted(state["train_ratings"]["userId"].unique())[:100]
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_user = st.selectbox("Select Test User ID", test_users, index=0)

    user_train_history = state["train_ratings"][state["train_ratings"]["userId"] == selected_user]
    top_liked = user_train_history.sort_values(by="rating", ascending=False).head(5)
    liked_titles = movies_df[movies_df["movieId"].isin(top_liked["movieId"])]["title"].tolist()

    with col2:
        st.markdown(f"""
        <div class="movie-card" style="margin-bottom: 0;">
            <div style="font-size: 1rem; font-weight: 700; color: #38BDF8;">User #{selected_user} Profile</div>
            <div style="font-size: 0.9rem; color: #E2E8F0; margin-top: 4px;">Rated {len(user_train_history)} movies in training set.</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Top liked: {", ".join(liked_titles[:4]) if liked_titles else "None"}</div>
        </div>
        """, unsafe_allow_html=True)

    tab_all, tab1, tab2, tab3, tab4 = st.tabs([
        "All Models Side-by-Side", "1. Popularity Baseline", "2. Content-Based", "3. Item-Item kNN", "4. Association Rules"
    ])

    rec_pop = state["pop_model"].recommend(user_id=selected_user, n=5)
    rec_cb = state["cb_model"].recommend(user_id=selected_user, n=5)
    rec_knn = state["knn_model"].recommend(user_id=selected_user, n=5)
    rec_ar = state["ar_model"].recommend(user_id=selected_user, n=5)

    with tab_all:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown("""
            <div class="team-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Popularity Baseline</span>
            </div>
            """, unsafe_allow_html=True)
            for _, r in rec_pop.iterrows():
                st.markdown(f'<div class="rec-item">🎬 <strong>{r["title"]}</strong></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="team-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Content-Based (TF-IDF)</span>
            </div>
            """, unsafe_allow_html=True)
            for _, r in rec_cb.iterrows():
                st.markdown(f'<div class="rec-item">🎬 <strong>{r["title"]}</strong></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown("""
            <div class="team-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Collaborative (Item kNN)</span>
            </div>
            """, unsafe_allow_html=True)
            for _, r in rec_knn.iterrows():
                st.markdown(f'<div class="rec-item">🎬 <strong>{r["title"]}</strong></div>', unsafe_allow_html=True)
        with col_d:
            st.markdown("""
            <div class="team-card" style="padding: 12px 14px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Association Rules</span>
            </div>
            """, unsafe_allow_html=True)
            if not rec_ar.empty:
                for _, r in rec_ar.iterrows():
                    st.markdown(f'<div class="rec-item">🎬 <strong>{r["title"]}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rec-item" style="color: #94A3B8 !important;">No matching association rules for this user.</div>', unsafe_allow_html=True)

    with tab1:
        st.subheader("Popularity Baseline (Top 10)")
        render_movie_grid(state["pop_model"].recommend(user_id=selected_user, n=10), algorithm="popularity")

    with tab2:
        st.subheader("Content-Based Recommender (Top 10)")
        render_movie_grid(state["cb_model"].recommend(user_id=selected_user, n=10), algorithm="content_based")

    with tab3:
        st.subheader("Item-Item kNN Collaborative Filtering (Top 10)")
        render_movie_grid(state["knn_model"].recommend(user_id=selected_user, n=10), algorithm="collaborative")

    with tab4:
        st.subheader("Association Rules with FP-Growth (Top 10)")
        render_movie_grid(state["ar_model"].recommend(user_id=selected_user, n=10), algorithm="association_rules")


# --- PAGE 2: Exploratory Data Analysis (EDA) ---
elif app_mode == "📊 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistical distributions, genre preferences, rating trends, and sparsity patterns across the MovieLens dataset.</div>', unsafe_allow_html=True)

    # 1. Executive Summary KPIs
    kpi_data = compute_kpis(state["ratings"], state["movies"], state["tags"])

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Ratings</div>
            <div class="kpi-value">{kpi_data["n_ratings"]:,}</div>
            <div class="kpi-sub">0.5 to 5.0 Stars</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Unique Users</div>
            <div class="kpi-value">{kpi_data["n_users"]:,}</div>
            <div class="kpi-sub">Min 20 Ratings/User</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Catalog Movies</div>
            <div class="kpi-value">{kpi_data["n_movies"]:,}</div>
            <div class="kpi-sub">{kpi_data["n_rated_movies"]:,} with ≥1 rating</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Matrix Sparsity</div>
            <div class="kpi-value">{kpi_data["sparsity"]:.2f}%</div>
            <div class="kpi-sub">Unrated Pairs</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Rating</div>
            <div class="kpi-value">{kpi_data["mean_rating"]:.2f} ★</div>
            <div class="kpi-sub">Median: {kpi_data["median_rating"]:.1f} ★ (±{kpi_data["std_rating"]:.2f})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">User Tags</div>
            <div class="kpi-value">{kpi_data["n_tags"]:,}</div>
            <div class="kpi-sub">Free-text Metadata</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. EDA Tabs
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "⭐ Rating Distributions",
        "🎭 Genre Insights & Co-occurrence",
        "📉 Long-Tail & Cold-Start (RQ2/RQ3)",
        "📅 Temporal Trends (1996-2018)",
        "🏆 Blockbusters vs Hidden Gems",
    ])

    # Tab 1: Rating Distributions
    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_rating_distribution(state["ratings"]), use_container_width=True)
        with c2:
            st.plotly_chart(plot_user_rating_tendencies(state["ratings"]), use_container_width=True)

        st.markdown("""
        <div class="eda-insight-box">
            <div class="eda-insight-title">💡 Recommender Systems Key Takeaway: Rating Skew & User Normalization</div>
            <div class="eda-insight-desc">
                • <strong>Positive Bias:</strong> Ratings are heavily skewed towards 4.0 ★ and 3.0 ★. Negative ratings (0.5–2.0 ★) constitute less than 14% of interactions because users self-select movies they anticipate enjoying.<br>
                • <strong>Half-Star Scarcity:</strong> Half-star ratings represent only ~15% of all records because they were introduced in MovieLens later (2003).<br>
                • <strong>User Leniency Skew:</strong> Individual user average ratings range from 1.5 ★ (strict critics) to 4.8 ★ (generous raters). This justifies why <strong>Collaborative Filtering (kNN & SVD)</strong> must mean-center user ratings or use Pearson correlation to avoid misinterpreting strict users' high scores.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 2: Genre Insights & Co-occurrence
    with eda_tab2:
        col_g1, col_g2 = st.columns([1.1, 1])
        with col_g1:
            st.plotly_chart(plot_genre_popularity(state["movies"], state["ratings"]), use_container_width=True)
        with col_g2:
            st.plotly_chart(plot_genre_quality(state["movies"], state["ratings"], min_ratings=50), use_container_width=True)

        st.plotly_chart(plot_genre_cooccurrence(state["movies"], top_n=14), use_container_width=True)

        st.markdown("""
        <div class="eda-insight-box">
            <div class="eda-insight-title">💡 Recommender Systems Key Takeaway: Semantic Clustering for Content-Based Filtering</div>
            <div class="eda-insight-desc">
                • <strong>Catalog vs Engagement:</strong> Drama and Comedy dominate both movie catalog volume (>4,000 titles) and ratings (>40,000 ratings).<br>
                • <strong>Niche Quality:</strong> Film-Noir, War, and Documentary achieve the highest average ratings (>3.8 ★), whereas Horror ranks lowest (~3.25 ★).<br>
                • <strong>Multi-Genre Co-occurrence:</strong> Strong correlation clusters exist between <em>Action · Adventure · Sci-Fi</em> and <em>Animation · Children</em>. This cross-genre overlap forms the bedrock of <strong>Content-Based TF-IDF vector similarity</strong> when comparing movie feature vectors.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 3: Long-Tail & Cold-Start
    with eda_tab3:
        st.plotly_chart(plot_long_tail(state["ratings"]), use_container_width=True)

        col_lt1, col_lt2 = st.columns([1.2, 1])
        with col_lt1:
            st.plotly_chart(plot_user_activity(state["ratings"]), use_container_width=True)
        with col_lt2:
            st.markdown("""
            <div class="chart-box" style="margin-top: 15px;">
                <div style="font-size: 1.05rem; font-weight: 800; color: #38BDF8; margin-bottom: 12px;">
                    🎯 Research Questions Connection (RQ2 & RQ3)
                </div>
                <div style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.6;">
                    <p><strong>RQ3 (Popularity Bias):</strong> The top 15% of movies attract 80% of all ratings! Meanwhile, <strong>3,446 movies (35.4% of the catalog) have only 1 rating</strong>. The Popularity Baseline will strictly recommend head blockbusters, ignoring 85% of catalog diversity.</p>
                    <p style="margin-top: 10px;"><strong>RQ2 (Cold-Start Users):</strong> Over 32% of users have fewer than 50 ratings. Collaborative Filtering struggles on sparse user vectors, whereas Content-Based filtering can personalize after as few as 1–3 ratings.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="eda-insight-box">
            <div class="eda-insight-title">💡 Recommender Systems Key Takeaway: The Extreme Sparsity Challenge</div>
            <div class="eda-insight-desc">
                With 98.3% matrix sparsity, 98 out of every 100 possible user-movie interactions are unobserved. Matrix Factorization (TruncatedSVD) and Item-Item kNN use dimensionality reduction and item similarity graphs to bridge this sparse gap.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 4: Temporal Dynamics
    with eda_tab4:
        st.plotly_chart(plot_temporal_trends(state["ratings"]), use_container_width=True)

        st.markdown("""
        <div class="eda-insight-box">
            <div class="eda-insight-title">💡 Recommender Systems Key Takeaway: Temporal Splitting vs Random Splitting</div>
            <div class="eda-insight-desc">
                • <strong>22-Year Evolution (1996–2018):</strong> Rating volume peaked in 2000 and 2015-2016. Average rating drifted slightly downward from ~3.7 ★ in early years to ~3.4 ★ in the mid-2000s as the user base broadened.<br>
                • <strong>Why Temporal Split Matters:</strong> In production recommenders, you can never train on future interactions to predict past behavior. This is why our project strictly employs a <strong>Temporal Train/Test Split (80% past ratings for training, 20% future ratings for testing)</strong> to prevent lookahead data leakage.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tab 5: Blockbusters vs Hidden Gems
    with eda_tab5:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">Movie Catalog Highlights</span>
                <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #94A3B8;">Comparing the most ubiquitous blockbusters against high-rated hidden gems.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        gem_mode = st.radio(
            "Select Collection",
            ["🔥 Top 15 Most Rated Blockbusters", "💎 Top 15 High-Rated Hidden Gems (35-120 Votes)"],
            horizontal=True,
        )

        blockbusters, gems = get_blockbusters_and_gems(state["ratings"], state["movies"], state["links"], top_k=15)

        if "Blockbusters" in gem_mode:
            st.markdown('<div class="movie-card" style="margin-bottom: 15px; color: #38BDF8; font-weight: 700;">Blockbusters dominate raw volume and form the core of the Popularity Baseline recommender.</div>', unsafe_allow_html=True)
            render_movie_grid(blockbusters, cols_count=5, algorithm="popularity")
        else:
            st.markdown('<div class="movie-card" style="margin-bottom: 15px; color: #10B981; font-weight: 700;">Hidden gems have stellar ratings (≥4.0 ★) but moderate exposure—where personalized CF and Content-Based models shine!</div>', unsafe_allow_html=True)
            render_movie_grid(gems, cols_count=5, algorithm="content_based")


# --- PAGE 3: Interactive Rate Movies Page ---
elif app_mode == "⭐ Rate Movies (Interactive)":
    st.markdown('<div class="main-header">Interactive Movie Rater & Cold-Start Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rate 5 or more movies to test how the recommenders personalize suggestions for a new user.</div>', unsafe_allow_html=True)

    # Preset sample of popular recognizable movies
    sample_movie_ids = [1, 296, 318, 356, 260, 593, 2571, 589, 4993, 7153]
    sample_movies = movies_with_links[movies_with_links["movieId"].isin(sample_movie_ids)].copy()

    st.markdown("""
    <div class="team-card" style="margin-bottom: 20px;">
        <span style="font-size: 1.05rem; color: #FFFFFF; font-weight: 600;">
            ⭐ Step 1: Rate any of these classic movies (1 to 5 stars) to generate your profile
        </span>
    </div>
    """, unsafe_allow_html=True)

    user_ratings = {}
    for row_start in range(0, len(sample_movies), 5):
        row_slice = sample_movies.iloc[row_start : row_start + 5]
        cols = st.columns(5)
        for idx_in_row, (_, movie) in enumerate(row_slice.iterrows()):
            mid = movie["movieId"]
            col = cols[idx_in_row]
            details = fetch_movie_details(movie["tmdbId"], title=movie["title"])
            with col:
                with st.container(border=True):
                    st.image(details["poster_url"], use_container_width=True)
                    st.markdown(f'<div class="movie-title" title="{movie["title"]}">{movie["title"]}</div>', unsafe_allow_html=True)
                    rating = st.slider(f"Rating", 0.0, 5.0, 0.0, step=0.5, key=f"rate_{mid}")
                    if rating > 0:
                        user_ratings[mid] = rating

    st.markdown(f"""
    <div class="team-card" style="margin-top: 25px; margin-bottom: 15px;">
        <span style="font-size: 1.05rem; color: #FFFFFF; font-weight: 600;">
            🎯 Step 2: Recommendations for your rating profile ({len(user_ratings)} rated)
        </span>
    </div>
    """, unsafe_allow_html=True)

    if len(user_ratings) < 1:
        st.info("Please give at least 1 star rating to any movie above to generate personalized recommendations!")
    else:
        col1, col_div, col2 = st.columns([1, 0.04, 1])
        with col1:
            st.markdown('''
            <div class="section-panel section-panel-cb">
                <div class="section-header section-header-cb">
                    <span class="section-icon">🧬</span>
                    <div>
                        <div class="section-title section-title-cb">Content-Based Suggestions</div>
                        <div class="section-desc">Matches by genre, title & tag similarity (TF-IDF cosine)</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            cb_recs = state["cb_model"].recommend(user_rated_dict=user_ratings, n=5)
            render_movie_grid(cb_recs, cols_count=3, algorithm="content_based")

        with col_div:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        with col2:
            st.markdown('''
            <div class="section-panel section-panel-cf">
                <div class="section-header section-header-cf">
                    <span class="section-icon">👥</span>
                    <div>
                        <div class="section-title section-title-cf">Collaborative Filtering (kNN)</div>
                        <div class="section-desc">Based on what similar users enjoyed (item-item kNN)</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            knn_recs = state["knn_model"].recommend(user_id=None, n=5, user_rated_dict=user_ratings)
            render_movie_grid(knn_recs, cols_count=3, algorithm="collaborative")


# --- PAGE 3: About & Team ---
elif app_mode == "ℹ️ About & Team":
    st.markdown('<div class="main-header">Project Overview & Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">DS-311 Data Mining · Fall 2026 · NUST</div>', unsafe_allow_html=True)

    col_team, col_info = st.columns([1.5, 1])

    with col_team:
        st.markdown("### 👥 Team Members")
        st.markdown("""
        <div class="team-card">
            <span class="team-name">Aiman Aslam</span>
            <span class="team-badge-leader">Team Leader</span>
        </div>
        <div class="team-card">
            <span class="team-name">M Wajahat</span>
            <span class="team-badge-member">Team Member</span>
        </div>
        <div class="team-card">
            <span class="team-name">Muhammad Abdul Rehman Nosherwani</span>
            <span class="team-badge-member">Team Member</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("### 🎓 Instruction & Supervision")
        st.markdown("""
        <div class="movie-card">
            <p style="margin: 0; font-size: 1rem; color: #FFFFFF;"><strong>Instructor:</strong> Dr. Zirak Khan</p>
            <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #CBD5E1;"><strong>Lab Engineer:</strong> Daniyal Zaidi</p>
        </div>
        <div class="movie-card">
            <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;"><strong>Dataset:</strong> MovieLens latest-small (GroupLens Research)</p>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;"><strong>Posters & Metadata:</strong> The Movie Database (TMDB) API</p>
        </div>
        """, unsafe_allow_html=True)
