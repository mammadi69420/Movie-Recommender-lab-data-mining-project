"""Evaluation metrics and temporal train/test split for movie recommendations.

Split Rule (Fair test):
- For each user, the older 80% of ratings are used for training and the newest 20% for testing.
- Random state is fixed where relevant.
- Metrics: Precision@k, Recall@k, NDCG@k, RMSE.
"""

import numpy as np
import pandas as pd
from .data import RANDOM_STATE


def temporal_train_test_split(ratings, test_ratio=0.2):
    """Split each user's rating history temporally.

    For each user, sort by timestamp:
    - older (1 - test_ratio) -> train
    - newest test_ratio -> test

    Returns (train_df, test_df).
    """
    ratings_sorted = ratings.sort_values(by=["userId", "timestamp"]).copy()

    # Calculate per-user cutoff index
    train_list = []
    test_list = []

    for user_id, group in ratings_sorted.groupby("userId"):
        n = len(group)
        if n == 1:
            # If user has only 1 rating, place in train
            train_list.append(group)
        else:
            n_test = max(1, int(np.round(n * test_ratio)))
            train_list.append(group.iloc[:-n_test])
            test_list.append(group.iloc[-n_test:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=ratings.columns)

    return train_df, test_df


def compute_rmse(actual_ratings, predicted_ratings):
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted ratings."""
    actual = np.array(actual_ratings)
    pred = np.array(predicted_ratings)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def precision_at_k(recommended_movie_ids, ground_truth_movie_ids, k=10):
    """Compute Precision@K.

    Precision@K = (# of recommended items in top-K that are relevant) / K
    """
    if k == 0:
        return 0.0
    rec_k = recommended_movie_ids[:k]
    hits = len(set(rec_k).intersection(set(ground_truth_movie_ids)))
    return hits / float(k)


def recall_at_k(recommended_movie_ids, ground_truth_movie_ids, k=10):
    """Compute Recall@K.

    Recall@K = (# of recommended items in top-K that are relevant) / (# total relevant items)
    """
    if len(ground_truth_movie_ids) == 0:
        return 0.0
    rec_k = recommended_movie_ids[:k]
    hits = len(set(rec_k).intersection(set(ground_truth_movie_ids)))
    return hits / float(len(ground_truth_movie_ids))


def ndcg_at_k(recommended_movie_ids, ground_truth_movie_ids, k=10):
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if k == 0 or len(ground_truth_movie_ids) == 0:
        return 0.0

    rec_k = recommended_movie_ids[:k]
    dcg = 0.0
    for i, movie_id in enumerate(rec_k):
        if movie_id in ground_truth_movie_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because 1-based index: i=0 -> log2(2) = 1

    # Ideal DCG (all top positions are relevant up to min(len(ground_truth), k))
    ideal_hits = min(len(ground_truth_movie_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_recommender(recommender, test_df, k=10, rating_threshold=4.0):
    """Evaluate recommender model on test set for Precision@K, Recall@K, NDCG@K.

    Relevant items in test_df are defined as those with rating >= rating_threshold.
    """
    precisions = []
    recalls = []
    ndcgs = []

    # Group test ratings by user
    relevant_by_user = (
        test_df[test_df["rating"] >= rating_threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    for user_id, actual_relevant in relevant_by_user.items():
        if not actual_relevant:
            continue
        try:
            recs = recommender.recommend(user_id, n=k)
            # recs can be a list of movieIds or DataFrame with 'movieId'
            if isinstance(recs, pd.DataFrame):
                rec_ids = recs["movieId"].tolist()
            else:
                rec_ids = list(recs)
        except Exception:
            rec_ids = []

        precisions.append(precision_at_k(rec_ids, actual_relevant, k=k))
        recalls.append(recall_at_k(rec_ids, actual_relevant, k=k))
        ndcgs.append(ndcg_at_k(rec_ids, actual_relevant, k=k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
