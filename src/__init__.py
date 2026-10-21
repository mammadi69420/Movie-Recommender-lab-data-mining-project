"""MovieLens data loading, models, evaluation, and TMDB integration."""

from .data import load_data, download_movielens_small, load_ml32m_sample, RANDOM_STATE
from .evaluation import (
    temporal_train_test_split,
    evaluate_recommender,
    compute_rmse,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)
from .popularity import PopularityRecommender
from .content_based import ContentBasedRecommender
from .collaborative import ItemItemKNNRecommender, SVDRecommender
from .association_rules import AssociationRuleRecommender
from .tmdb import fetch_movie_details, get_tmdb_api_key
from .eda import (
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

__all__ = [
    "load_data",
    "download_movielens_small",
    "load_ml32m_sample",
    "RANDOM_STATE",
    "temporal_train_test_split",
    "evaluate_recommender",
    "compute_rmse",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "PopularityRecommender",
    "ContentBasedRecommender",
    "ItemItemKNNRecommender",
    "SVDRecommender",
    "AssociationRuleRecommender",
    "fetch_movie_details",
    "get_tmdb_api_key",
    "compute_kpis",
    "plot_rating_distribution",
    "plot_user_rating_tendencies",
    "plot_genre_popularity",
    "plot_genre_quality",
    "plot_genre_cooccurrence",
    "plot_long_tail",
    "plot_user_activity",
    "plot_temporal_trends",
    "get_blockbusters_and_gems",
]

