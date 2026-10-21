"""Exploratory Data Analysis (EDA) helpers and interactive Plotly visualizations.

Provides statistical summaries and figures for the MovieLens dataset:
- Rating distributions and user rating tendencies
- Genre popularity and average rating comparisons
- Genre co-occurrence heatmap
- Long-tail item popularity curve and user activity skew
- Temporal trends from 1996 to 2018
- Top blockbusters vs hidden gems
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def apply_dark_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
    x_title: str = "",
    y_title: str = "",
    **extra_layout
) -> go.Figure:
    """Apply consistent glassmorphic dark styling to any Plotly figure."""
    top_margin = 85 if title else 45
    layout_update = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        font=dict(color="#F8FAFC", family="Inter, system-ui, sans-serif"),
        margin=dict(l=45, r=45, t=top_margin, b=45),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_color="#FFFFFF",
            bordercolor="#38BDF8",
        ),
        height=height,
    )
    if title:
        layout_update["title"] = dict(
            text=title,
            font=dict(size=15, color="#FFFFFF", family="Inter, sans-serif"),
            x=0.01,
            y=0.97,
            xanchor="left",
            yanchor="top",
            yref="container",
        )

    fig.update_layout(**layout_update)

    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#CBD5E1", size=11),
        title_font=dict(color="#F8FAFC", size=12),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)

    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.08)",
        zerolinecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#CBD5E1", size=11),
        title_font=dict(color="#F8FAFC", size=12),
    )
    if y_title:
        fig.update_yaxes(title_text=y_title)

    if extra_layout:
        fig.update_layout(**extra_layout)

    return fig


def compute_kpis(ratings: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame) -> dict:
    """Compute top-level summary metrics for the dataset."""
    n_ratings = len(ratings)
    n_users = ratings["userId"].nunique()
    n_movies = len(movies)
    n_rated_movies = ratings["movieId"].nunique()
    n_tags = len(tags)
    
    # User-item matrix sparsity: percentage of unrated pairs
    sparsity = (1.0 - (n_ratings / (n_users * n_movies))) * 100.0
    
    mean_rating = ratings["rating"].mean()
    median_rating = ratings["rating"].median()
    std_rating = ratings["rating"].std()

    # User activity
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    return {
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_movies": n_movies,
        "n_rated_movies": n_rated_movies,
        "n_tags": n_tags,
        "sparsity": sparsity,
        "mean_rating": mean_rating,
        "median_rating": median_rating,
        "std_rating": std_rating,
        "user_counts": user_counts,
        "movie_counts": movie_counts,
    }


def plot_rating_distribution(ratings: pd.DataFrame) -> go.Figure:
    """Bar chart of rating frequencies (0.5 to 5.0 stars) with percentages and mean line."""
    counts = ratings["rating"].value_counts().sort_index().reset_index()
    counts.columns = ["Rating", "Count"]
    total = len(ratings)
    counts["Percentage"] = (counts["Count"] / total) * 100.0

    colors = [
        "#EF4444" if r < 2.0 else
        "#F59E0B" if r < 3.0 else
        "#38BDF8" if r < 4.0 else
        "#10B981"
        for r in counts["Rating"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["Rating"],
        y=counts["Count"],
        text=[f"{p:.1f}%<br>({c:,})" for p, c in zip(counts["Percentage"], counts["Count"])],
        textposition="outside",
        marker=dict(
            color=colors,
            line=dict(color="#FFFFFF", width=1),
        ),
        hovertemplate="<b>Rating: %{x} ★</b><br>Count: %{y:,}<br>Share: %{customdata:.2f}%<extra></extra>",
        customdata=counts["Percentage"],
    ))

    # Mean rating indicator line
    mean_val = ratings["rating"].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"Global Mean ({mean_val:.2f} ★)",
        annotation_position="top left",
        annotation_font=dict(color="#FBBF24", size=11),
    )

    apply_dark_theme(
        fig,
        title="<b>Distribution of Ratings (0.5 to 5.0 Stars)</b>",
        x_title="Rating (Stars)",
        y_title="Total Ratings Count",
        height=420,
    )
    fig.update_xaxes(tickmode="linear", tick0=0.5, dtick=0.5)
    return fig


def plot_user_rating_tendencies(ratings: pd.DataFrame) -> go.Figure:
    """Histogram of each user's average rating (shows strict vs generous users)."""
    user_means = ratings.groupby("userId")["rating"].mean().reset_index()
    user_means.columns = ["userId", "MeanRating"]

    fig = px.histogram(
        user_means,
        x="MeanRating",
        nbins=28,
        color_discrete_sequence=["#A855F7"],
        marginal="box",
    )

    fig.update_traces(
        marker=dict(line=dict(color="#FFFFFF", width=0.8)),
        hovertemplate="<b>User Mean Rating:</b> %{x:.2f} ★<br>Users Count: %{y}<extra></extra>",
    )

    apply_dark_theme(
        fig,
        title="<b>User Rating Leniency (Mean Rating per User)</b>",
        x_title="User Average Rating (★)",
        y_title="Number of Users",
        height=420,
    )
    return fig


def plot_genre_popularity(movies: pd.DataFrame, ratings: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing movie catalog counts vs rating counts per genre."""
    m_exp = movies.explode("genres")
    m_exp = m_exp[m_exp["genres"].notna() & (m_exp["genres"] != "") & (m_exp["genres"] != "(no genres listed)")]

    cat_counts = m_exp["genres"].value_counts().reset_index()
    cat_counts.columns = ["Genre", "CatalogCount"]

    merged = ratings[["movieId"]].merge(m_exp[["movieId", "genres"]], on="movieId", how="inner")
    rat_counts = merged["genres"].value_counts().reset_index()
    rat_counts.columns = ["Genre", "RatingCount"]

    genre_df = cat_counts.merge(rat_counts, on="Genre", how="left").fillna(0)
    genre_df = genre_df.sort_values(by="RatingCount", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=genre_df["Genre"],
        x=genre_df["RatingCount"],
        name="Ratings Received",
        orientation="h",
        marker=dict(color="#38BDF8"),
        hovertemplate="<b>%{y}</b><br>Ratings: %{x:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=genre_df["Genre"],
        x=genre_df["CatalogCount"] * 10,
        name="Catalog Titles (×10)",
        orientation="h",
        marker=dict(color="#F59E0B"),
        hovertemplate="<b>%{y}</b><br>Catalog Movies: %{customdata:,}<extra></extra>",
        customdata=genre_df["CatalogCount"],
    ))

    apply_dark_theme(
        fig,
        title="<b>Genre Distribution: Ratings Volume vs Catalog Titles</b>",
        x_title="Volume",
        y_title="",
        height=540,
        barmode="group",
        margin=dict(l=45, r=45, t=95, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig


def plot_genre_quality(movies: pd.DataFrame, ratings: pd.DataFrame, min_ratings: int = 100) -> go.Figure:
    """Average rating by genre sorted from highest to lowest."""
    m_exp = movies.explode("genres")
    m_exp = m_exp[m_exp["genres"].notna() & (m_exp["genres"] != "") & (m_exp["genres"] != "(no genres listed)")]

    merged = ratings[["movieId", "rating"]].merge(m_exp[["movieId", "genres"]], on="movieId", how="inner")
    stats = merged.groupby("genres")["rating"].agg(["mean", "count", "std"]).reset_index()
    stats = stats[stats["count"] >= min_ratings].sort_values(by="mean", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=stats["genres"],
        x=stats["mean"],
        orientation="h",
        marker=dict(
            color=stats["mean"],
            colorscale="Viridis",
            line=dict(color="#FFFFFF", width=0.5),
        ),
        text=[f"{m:.2f} ★" for m in stats["mean"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Average Rating: %{x:.3f} ★<br>Ratings Count: %{customdata:,}<extra></extra>",
        customdata=stats["count"],
    ))

    apply_dark_theme(
        fig,
        title=f"<b>Average Rating by Genre (Min {min_ratings} Ratings)</b>",
        x_title="Average Rating (★)",
        y_title="",
        height=520,
    )
    fig.update_xaxes(range=[2.5, 4.3])
    return fig


def plot_genre_cooccurrence(movies: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Co-occurrence correlation matrix heatmap for top genres."""
    m_exp = movies.explode("genres")
    m_exp = m_exp[m_exp["genres"].notna() & (m_exp["genres"] != "") & (m_exp["genres"] != "(no genres listed)")]
    
    top_genres = m_exp["genres"].value_counts().head(top_n).index.tolist()

    dummies = pd.get_dummies(m_exp[m_exp["genres"].isin(top_genres)].set_index("movieId")["genres"]).groupby("movieId").max()
    coocc = dummies.T.dot(dummies)

    diag = np.diag(coocc)
    norm_matrix = coocc / np.sqrt(np.outer(diag, diag))

    fig = go.Figure(data=go.Heatmap(
        z=norm_matrix.values,
        x=norm_matrix.columns,
        y=norm_matrix.index,
        colorscale="Blues",
        text=np.round(norm_matrix.values, 2),
        texttemplate="%{text}",
        hoverongaps=False,
        hovertemplate="<b>%{y} & %{x}</b><br>Co-occurrence Correlation: %{z:.3f}<extra></extra>",
    ))

    apply_dark_theme(
        fig,
        title=f"<b>Genre Co-occurrence Matrix (Top {top_n} Genres)</b>",
        height=480,
    )
    return fig


def plot_long_tail(ratings: pd.DataFrame) -> go.Figure:
    """The 80/20 Long-Tail curve: movies ranked by popularity vs rating count."""
    movie_counts = ratings["movieId"].value_counts().values
    n_movies = len(movie_counts)
    ranks = np.arange(1, n_movies + 1)
    
    cumsum = np.cumsum(movie_counts)
    cum_pct = (cumsum / cumsum[-1]) * 100.0

    cutoff_idx = np.where(cum_pct >= 80.0)[0][0]
    cutoff_rank = ranks[cutoff_idx]
    pct_catalog = (cutoff_rank / n_movies) * 100.0

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ranks,
        y=movie_counts,
        mode="lines",
        name="Ratings per Movie",
        line=dict(color="#38BDF8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.15)",
        hovertemplate="<b>Rank #%{x:,}</b><br>Ratings Count: %{y:,}<extra></extra>",
    ))

    fig.add_vline(
        x=cutoff_rank,
        line_dash="dash",
        line_color="#EC4899",
        line_width=2,
        annotation_text=f"80% Ratings Cutoff<br>(Top {pct_catalog:.1f}% movies)",
        annotation_position="top right",
        annotation_font=dict(color="#EC4899", size=11),
    )

    apply_dark_theme(
        fig,
        title="<b>The Long-Tail Distribution (Item Popularity Skew)</b>",
        x_title="Movie Popularity Rank (1 = Most Rated)",
        y_title="Number of Ratings (Log Scale)",
        height=430,
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig


def plot_user_activity(ratings: pd.DataFrame) -> go.Figure:
    """User rating activity distribution highlighting cold-start vs power users."""
    user_counts = ratings["userId"].value_counts().values

    bins = [0, 20, 50, 150, 500, 10000]
    labels = ["Cold Start (<20)", "Casual (20-50)", "Moderate (50-150)", "Active (150-500)", "Power User (500+)"]
    categorized = pd.cut(user_counts, bins=bins, labels=labels).value_counts()[labels].reset_index()
    categorized.columns = ["Segment", "UserCount"]

    colors = ["#EF4444", "#F59E0B", "#10B981", "#38BDF8", "#A855F7"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categorized["Segment"],
        y=categorized["UserCount"],
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
        text=[f"{c:,} users<br>({c/len(user_counts)*100:.1f}%)" for c in categorized["UserCount"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Users: %{y:,}<extra></extra>",
    ))

    apply_dark_theme(
        fig,
        title="<b>User Engagement Segments (Cold Start vs Power Users)</b>",
        x_title="User Activity Segment",
        y_title="Number of Users",
        height=420,
    )
    return fig


def plot_temporal_trends(ratings: pd.DataFrame) -> go.Figure:
    """Annual volume of ratings and average rating trajectory from 1996 to 2018."""
    df = ratings.copy()
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year

    yearly = df.groupby("year").agg(
        RatingsCount=("rating", "count"),
        AverageRating=("rating", "mean"),
    ).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=yearly["year"],
        y=yearly["RatingsCount"],
        name="Ratings Count",
        marker=dict(color="rgba(56, 189, 248, 0.45)", line=dict(color="#38BDF8", width=1)),
        yaxis="y",
        hovertemplate="<b>Year %{x}</b><br>Total Ratings: %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=yearly["year"],
        y=yearly["AverageRating"],
        name="Average Rating",
        mode="lines+markers",
        line=dict(color="#F59E0B", width=3),
        marker=dict(size=6, color="#FBBF24"),
        yaxis="y2",
        hovertemplate="<b>Year %{x}</b><br>Average Rating: %{y:.2f} ★<extra></extra>",
    ))

    apply_dark_theme(
        fig,
        title="<b>Temporal Dynamics: Annual Ratings Volume & Mean Rating (1996–2018)</b>",
        x_title="Year",
        y_title="Ratings Count",
        height=430,
        yaxis2=dict(
            title=dict(text="Average Rating (★)", font=dict(color="#F59E0B")),
            tickfont=dict(color="#F59E0B"),
            overlaying="y",
            side="right",
            range=[3.0, 4.0],
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=45, r=45, t=95, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig


def get_blockbusters_and_gems(ratings: pd.DataFrame, movies: pd.DataFrame, links: pd.DataFrame, top_k: int = 15):
    """Retrieve top blockbusters (most rated) and hidden gems (highest rated with 35-120 votes)."""
    stats = ratings.groupby("movieId").agg(
        vote_count=("rating", "count"),
        vote_average=("rating", "mean"),
    ).reset_index()

    C = ratings["rating"].mean()
    m = 20
    stats["weighted_score"] = (stats["vote_count"] / (stats["vote_count"] + m)) * stats["vote_average"] + (m / (stats["vote_count"] + m)) * C

    merged = stats.merge(movies, on="movieId", how="inner").merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    blockbusters = merged.sort_values(by="vote_count", ascending=False).head(top_k)
    gems = merged[(merged["vote_count"] >= 35) & (merged["vote_count"] <= 120)].sort_values(by="vote_average", ascending=False).head(top_k)

    return blockbusters, gems
