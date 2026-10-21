"""Collaborative Filtering Recommender.

Includes:
1. Item-Item kNN Recommender
2. Matrix Factorization via TruncatedSVD wrapped in sklearn Pipeline
Adheres strictly to RANDOM_STATE = 42 and free CPU constraints.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from .data import RANDOM_STATE


class ItemItemKNNRecommender:
    """Item-Item Collaborative Filtering using k-Nearest Neighbors."""

    def __init__(self, n_neighbors=20, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = NearestNeighbors(metric=self.metric, algorithm="brute", n_neighbors=self.n_neighbors)
        self.item_user_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movies_df = None
        self.train_ratings = None

    def fit(self, train_ratings, movies_df=None):
        """Fit item-item kNN on user-item interaction matrix."""
        self.train_ratings = train_ratings.copy()
        self.movies_df = movies_df

        unique_movies = sorted(train_ratings["movieId"].unique())
        unique_users = sorted(train_ratings["userId"].unique())

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        rows = train_ratings["movieId"].map(self.movie_id_to_idx)
        cols = train_ratings["userId"].map(self.user_id_to_idx)
        data = train_ratings["rating"].values

        # Shape: (n_items, n_users)
        self.item_user_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_movies), len(unique_users))
        )

        self.knn.fit(self.item_user_matrix)
        return self

    def get_similar_items(self, movie_id, n=10):
        """Return nearest neighbor movies for a given movie."""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()

        idx = self.movie_id_to_idx[movie_id]
        distances, indices = self.knn.kneighbors(
            self.item_user_matrix[idx],
            n_neighbors=min(n + 1, self.item_user_matrix.shape[0])
        )

        # Exclude self
        neighbor_indices = [i for i in indices.flatten() if i != idx][:n]
        sim_scores = [1.0 - d for i, d in zip(indices.flatten(), distances.flatten()) if i != idx][:n]

        rec_movie_ids = [self.idx_to_movie_id[i] for i in neighbor_indices]
        res = pd.DataFrame({"movieId": rec_movie_ids, "similarity": sim_scores})
        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res

    def recommend(self, user_id, n=10, user_rated_dict=None):
        """Recommend movies for a user by aggregating ratings with item similarities."""
        ratings_map = {}
        if user_rated_dict is not None:
            ratings_map = user_rated_dict
        elif user_id in self.user_id_to_idx:
            u_data = self.train_ratings[self.train_ratings["userId"] == user_id]
            ratings_map = dict(zip(u_data["movieId"], u_data["rating"]))

        if not ratings_map:
            # Fallback
            rec_ids = list(self.idx_to_movie_id.values())[:n]
            res = pd.DataFrame({"movieId": rec_ids, "score": [0.0] * len(rec_ids)})
            if self.movies_df is not None:
                res = res.merge(self.movies_df, on="movieId", how="left")
            return res

        seen_movies = set(ratings_map.keys())
        candidate_scores = {}

        # Consider top-rated items from user history (e.g. rating >= 3.5)
        for mid, r in ratings_map.items():
            if mid not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[mid]
            distances, indices = self.knn.kneighbors(
                self.item_user_matrix[idx],
                n_neighbors=min(self.n_neighbors, self.item_user_matrix.shape[0])
            )
            for neighbor_idx, dist in zip(indices.flatten(), distances.flatten()):
                neighbor_mid = self.idx_to_movie_id[neighbor_idx]
                if neighbor_mid in seen_movies:
                    continue
                sim = max(0.0, 1.0 - dist)
                candidate_scores[neighbor_mid] = candidate_scores.get(neighbor_mid, 0.0) + (sim * r)

        if not candidate_scores:
            unseen = [mid for mid in self.idx_to_movie_id.values() if mid not in seen_movies][:n]
            res = pd.DataFrame({"movieId": unseen, "score": [0.0] * len(unseen)})
        else:
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            res = pd.DataFrame(sorted_candidates, columns=["movieId", "score"])

        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res


class SVDRecommender:
    """Matrix Factorization Collaborative Filtering via TruncatedSVD in an sklearn Pipeline."""

    def __init__(self, n_components=20, random_state=RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("svd", TruncatedSVD(n_components=self.n_components, random_state=self.random_state))
        ])
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.predicted_matrix = None
        self.user_history_ = {}
        self.movies_df = None
        self.user_means = None

    def fit(self, train_ratings, movies_df=None):
        """Fit TruncatedSVD pipeline on normalized user-item matrix."""
        self.movies_df = movies_df
        unique_movies = sorted(train_ratings["movieId"].unique())
        unique_users = sorted(train_ratings["userId"].unique())

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        self.user_history_ = (
            train_ratings.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # Build dense / normalized matrix
        pivot = train_ratings.pivot(index="userId", columns="movieId", values="rating").reindex(
            index=unique_users, columns=unique_movies
        )

        # Mean center per user for unrated entries
        self.user_means = pivot.mean(axis=1)
        matrix_imputed = pivot.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0).values

        # Decompose using Pipeline
        user_factors = self.pipeline.fit_transform(matrix_imputed)
        svd_step = self.pipeline.named_steps["svd"]
        item_factors = svd_step.components_

        # Reconstruct predicted rating matrix
        self.predicted_matrix = np.dot(user_factors, item_factors)
        return self

    def predict_rating(self, user_id, movie_id):
        """Predict continuous rating for a specific (user, movie) pair."""
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            # Fallback to global average or user mean
            return 3.5
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        return float(self.predicted_matrix[u_idx, m_idx])

    def recommend(self, user_id, n=10):
        """Recommend top-N unseen movies with highest predicted rating."""
        if user_id not in self.user_id_to_idx:
            # Fallback to first movies
            rec_ids = list(self.idx_to_movie_id.values())[:n]
            res = pd.DataFrame({"movieId": rec_ids, "score": [0.0] * len(rec_ids)})
            if self.movies_df is not None:
                res = res.merge(self.movies_df, on="movieId", how="left")
            return res

        u_idx = self.user_id_to_idx[user_id]
        user_preds = self.predicted_matrix[u_idx]
        seen_movies = self.user_history_.get(user_id, set())

        # Sort item indices by predicted score descending
        sorted_indices = np.argsort(user_preds)[::-1]
        recs = []
        scores = []
        for idx in sorted_indices:
            mid = self.idx_to_movie_id[idx]
            if mid not in seen_movies:
                recs.append(mid)
                scores.append(float(user_preds[idx]))
            if len(recs) >= n:
                break

        res = pd.DataFrame({"movieId": recs, "score": scores})
        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res
