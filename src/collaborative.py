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
    """Item-Item Collaborative Filtering using k-Nearest Neighbors (cosine or Pearson)."""

    def __init__(self, n_neighbors=20, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        # Pearson correlation on sparse ratings equals cosine similarity on mean-centered ratings
        knn_metric = "cosine" if metric == "pearson" else metric
        self.knn = NearestNeighbors(metric=knn_metric, algorithm="brute", n_neighbors=self.n_neighbors)
        self.item_user_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movies_df = None
        self.train_ratings = None
        self.title_lookup_ = {}

    def fit(self, train_ratings, movies_df=None):
        """Fit item-item kNN on user-item interaction matrix."""
        self.train_ratings = train_ratings.copy()
        self.movies_df = movies_df
        if movies_df is not None:
            self.title_lookup_ = dict(zip(movies_df["movieId"], movies_df["title"]))

        unique_movies = sorted(train_ratings["movieId"].unique())
        unique_users = sorted(train_ratings["userId"].unique())

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        ratings_vals = train_ratings["rating"].values.astype(float)
        if self.metric == "pearson":
            item_means = train_ratings.groupby("movieId")["rating"].transform("mean").values
            ratings_vals = ratings_vals - item_means

        rows = train_ratings["movieId"].map(self.movie_id_to_idx)
        cols = train_ratings["userId"].map(self.user_id_to_idx)

        # Shape: (n_items, n_users)
        self.item_user_matrix = csr_matrix(
            (ratings_vals, (rows, cols)),
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
        candidate_best_src = {}

        # Batch kneighbors query across all rated items for this user (20x faster)
        valid_pairs = [(mid, r, self.movie_id_to_idx[mid]) for mid, r in ratings_map.items() if mid in self.movie_id_to_idx]
        if valid_pairs:
            query_indices = [p[2] for p in valid_pairs]
            all_distances, all_indices = self.knn.kneighbors(
                self.item_user_matrix[query_indices],
                n_neighbors=min(self.n_neighbors + 1, self.item_user_matrix.shape[0])
            )
            for (mid, r, _), distances, indices in zip(valid_pairs, all_distances, all_indices):
                for neighbor_idx, dist in zip(indices, distances):
                    neighbor_mid = self.idx_to_movie_id[neighbor_idx]
                    if neighbor_mid in seen_movies:
                        continue
                    sim = max(0.0, 1.0 - dist)
                    contrib = sim * r
                    candidate_scores[neighbor_mid] = candidate_scores.get(neighbor_mid, 0.0) + contrib
                    # Only attribute "Because you liked" to positively rated movies (>= 3.5)
                    if r >= 3.5:
                        prev_best_mid, prev_best_c = candidate_best_src.get(neighbor_mid, (None, -1.0))
                        if contrib > prev_best_c:
                            candidate_best_src[neighbor_mid] = (mid, contrib)

        if not candidate_scores:
            unseen = [mid for mid in self.idx_to_movie_id.values() if mid not in seen_movies][:n]
            res = pd.DataFrame({
                "movieId": unseen,
                "score": [0.0] * len(unseen),
                "explanation": ["Popular fallback recommendation"] * len(unseen),
            })
        else:
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            res = pd.DataFrame(sorted_candidates, columns=["movieId", "score"])
            explanations = []
            for rec_mid in res["movieId"]:
                src_mid, _ = candidate_best_src.get(rec_mid, (None, 0.0))
                if src_mid is not None:
                    src_title = self.title_lookup_.get(src_mid, f"Movie #{src_mid}")
                    explanations.append(f"Because you liked {src_title}")
                else:
                    explanations.append("Recommended by item-item similarity")
            res["explanation"] = explanations

        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res


class UserUserKNNRecommender:
    """User-User Collaborative Filtering using k-Nearest Neighbors (cosine or Pearson)."""

    def __init__(self, n_neighbors=20, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        knn_metric = "cosine" if metric == "pearson" else metric
        self.knn = NearestNeighbors(metric=knn_metric, algorithm="brute", n_neighbors=self.n_neighbors)
        self.user_item_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movies_df = None
        self.train_ratings = None
        self.title_lookup_ = {}

    def fit(self, train_ratings, movies_df=None):
        """Fit user-user kNN on (n_users, n_items) sparse rating matrix."""
        self.train_ratings = train_ratings.copy()
        self.movies_df = movies_df
        if movies_df is not None:
            self.title_lookup_ = dict(zip(movies_df["movieId"], movies_df["title"]))

        unique_movies = sorted(train_ratings["movieId"].unique())
        unique_users = sorted(train_ratings["userId"].unique())

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        ratings_vals = train_ratings["rating"].values.astype(float)
        if self.metric == "pearson":
            user_means = train_ratings.groupby("userId")["rating"].transform("mean").values
            ratings_vals = ratings_vals - user_means

        rows = train_ratings["userId"].map(self.user_id_to_idx)
        cols = train_ratings["movieId"].map(self.movie_id_to_idx)

        self.user_item_matrix = csr_matrix(
            (ratings_vals, (rows, cols)),
            shape=(len(unique_users), len(unique_movies)),
        )
        self.knn.fit(self.user_item_matrix)
        return self

    def recommend(self, user_id=None, n=10, user_rated_dict=None):
        """Recommend top-N unseen movies from similar users' ratings."""
        if user_rated_dict is not None:
            seen_movies = set(user_rated_dict.keys())
            cols, vals = [], []
            mean_val = float(np.mean(list(user_rated_dict.values()))) if user_rated_dict else 0.0
            for mid, r in user_rated_dict.items():
                if mid in self.movie_id_to_idx:
                    cols.append(self.movie_id_to_idx[mid])
                    centered = r - mean_val if self.metric == "pearson" else r
                    vals.append(centered)
            # Avoid all-zero query vector when all user ratings are identical under Pearson
            if cols and all(v == 0.0 for v in vals):
                vals = [float(user_rated_dict[self.idx_to_movie_id[c]]) for c in cols]
            if not cols:
                rec_ids = list(self.idx_to_movie_id.values())[:n]
                res = pd.DataFrame({"movieId": rec_ids, "score": [0.0] * len(rec_ids), "explanation": ["Popular fallback recommendation"] * len(rec_ids)})
                if self.movies_df is not None:
                    res = res.merge(self.movies_df, on="movieId", how="left")
                return res
            query_vec = csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.movie_id_to_idx)))
            distances, indices = self.knn.kneighbors(
                query_vec, n_neighbors=min(self.n_neighbors, self.user_item_matrix.shape[0])
            )
            neighbor_pairs = list(zip(indices.flatten(), distances.flatten()))
        elif user_id in self.user_id_to_idx:
            u_idx = self.user_id_to_idx[user_id]
            u_data = self.train_ratings[self.train_ratings["userId"] == user_id]
            seen_movies = set(u_data["movieId"].tolist())
            distances, indices = self.knn.kneighbors(
                self.user_item_matrix[u_idx],
                n_neighbors=min(self.n_neighbors + 1, self.user_item_matrix.shape[0]),
            )
            neighbor_pairs = [(i, d) for i, d in zip(indices.flatten(), distances.flatten()) if i != u_idx]
        else:
            rec_ids = list(self.idx_to_movie_id.values())[:n]
            res = pd.DataFrame({"movieId": rec_ids, "score": [0.0] * len(rec_ids), "explanation": ["Popular fallback recommendation"] * len(rec_ids)})
            if self.movies_df is not None:
                res = res.merge(self.movies_df, on="movieId", how="left")
            return res

        # Vectorized neighbor rating aggregation weighted by similarity
        neighbor_uids = [self.idx_to_user_id[i] for i, _ in neighbor_pairs]
        sim_by_uid = {self.idx_to_user_id[i]: max(0.0, 1.0 - d) for i, d in neighbor_pairs}
        nbr_ratings = self.train_ratings[
            self.train_ratings["userId"].isin(neighbor_uids)
            & ~self.train_ratings["movieId"].isin(seen_movies)
        ].copy()

        if not nbr_ratings.empty:
            nbr_ratings["sim"] = nbr_ratings["userId"].map(sim_by_uid).fillna(0.0)
            nbr_ratings = nbr_ratings[nbr_ratings["sim"] > 0]
            nbr_ratings["weighted"] = nbr_ratings["sim"] * nbr_ratings["rating"]
            agg = (
                nbr_ratings.groupby("movieId")["weighted"]
                .sum()
                .sort_values(ascending=False)
                .head(n)
            )
            res = pd.DataFrame({"movieId": agg.index.astype(int), "score": agg.values.astype(float)})
            res["explanation"] = "Recommended by similar viewers"
        else:
            unseen = [mid for mid in self.idx_to_movie_id.values() if mid not in seen_movies][:n]
            res = pd.DataFrame({
                "movieId": unseen,
                "score": [0.0] * len(unseen),
                "explanation": ["Popular fallback recommendation"] * len(unseen),
            })

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

        # Store global mean and user means; center ratings around user mean before sparse TruncatedSVD
        self.global_mean_ = float(train_ratings["rating"].mean())
        user_means_series = train_ratings.groupby("userId")["rating"].mean()
        self.user_means = user_means_series.to_dict()

        centered = train_ratings["rating"].values.astype(float) - train_ratings["userId"].map(self.user_means).values
        rows = train_ratings["userId"].map(self.user_id_to_idx)
        cols = train_ratings["movieId"].map(self.movie_id_to_idx)
        sparse_centered = csr_matrix((centered, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

        # Fit sparse TruncatedSVD pipeline (avoids dense O(U*M) memory explosion on 32M sample)
        self.user_factors_ = self.pipeline.fit_transform(sparse_centered)
        self.item_factors_ = self.pipeline.named_steps["svd"].components_
        return self

    def predict_rating(self, user_id, movie_id):
        """Predict continuous rating for a specific (user, movie) pair, clipped to [0.5, 5.0]."""
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            base = self.user_means.get(user_id, getattr(self, "global_mean_", 3.5))
            return float(np.clip(base, 0.5, 5.0))
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        pred = self.user_means.get(user_id, self.global_mean_) + float(
            np.dot(self.user_factors_[u_idx], self.item_factors_[:, m_idx])
        )
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id, n=10):
        """Recommend top-N unseen movies with highest predicted rating."""
        if user_id not in self.user_id_to_idx:
            rec_ids = list(self.idx_to_movie_id.values())[:n]
            res = pd.DataFrame({"movieId": rec_ids, "score": [0.0] * len(rec_ids)})
            if self.movies_df is not None:
                res = res.merge(self.movies_df, on="movieId", how="left")
            return res

        u_idx = self.user_id_to_idx[user_id]
        user_preds = self.user_means.get(user_id, self.global_mean_) + np.dot(
            self.user_factors_[u_idx], self.item_factors_
        )
        seen_movies = self.user_history_.get(user_id, set())

        # Sort item indices by predicted score descending
        sorted_indices = np.argsort(user_preds)[::-1]
        recs = []
        scores = []
        for idx in sorted_indices:
            mid = self.idx_to_movie_id[idx]
            if mid not in seen_movies:
                recs.append(mid)
                scores.append(float(np.clip(user_preds[idx], 0.5, 5.0)))
            if len(recs) >= n:
                break

        res = pd.DataFrame({"movieId": recs, "score": scores})
        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res
