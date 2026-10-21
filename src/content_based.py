"""Content-Based Recommender.

Computes TF-IDF vectors from movie genres, titles, and user-assigned tags.
Calculates cosine similarity between items and aggregates user profiles
to suggest similar unseen movies.
Follows CLAUDE.md: Uses sklearn Pipeline and fixed RANDOM_STATE.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .data import RANDOM_STATE


class ContentBasedRecommender:
    """Recommender based on TF-IDF features of movie metadata and tags."""

    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=self.max_features))
        ])
        self.movies_df = None
        self.tfidf_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_history_ = {}
        self.train_ratings = None

    def fit(self, movies_df, tags_df=None, train_ratings=None):
        """Build content representation from genres, tags, and titles."""
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Build text description per movie
        # Combine genres into space-separated string
        genres_str = self.movies_df["genres"].apply(
            lambda g: " ".join(g) if isinstance(g, list) else str(g)
        )

        content_corpus = self.movies_df["title"] + " " + genres_str

        # Add tags if available
        if tags_df is not None and not tags_df.empty:
            movie_tags = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda tags: " ".join(str(t) for t in tags.dropna()))
                .reset_index()
            )
            df_merged = self.movies_df.merge(movie_tags, on="movieId", how="left")
            df_merged["tag"] = df_merged["tag"].fillna("")
            content_corpus = content_corpus + " " + df_merged["tag"]

        # Fit TF-IDF via sklearn Pipeline
        self.tfidf_matrix = self.pipeline.fit_transform(content_corpus)

        # Mapping between movieId and row index
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self.idx_to_movie_id = {
            idx: mid for mid, idx in self.movie_id_to_idx.items()
        }

        if train_ratings is not None:
            self.train_ratings = train_ratings.copy()
            self.user_history_ = (
                train_ratings.groupby("userId")["movieId"]
                .apply(set)
                .to_dict()
            )

        return self

    def get_similar_movies(self, movie_id, top_n=10):
        """Find movies most similar to a given movie based on cosine similarity."""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()

        idx = self.movie_id_to_idx[movie_id]
        # Cosine similarity for normalized tf-idf vectors equals dot product
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Sort indices excluding the movie itself
        sim_indices = cosine_sim.argsort()[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:top_n]

        results = self.movies_df.iloc[sim_indices].copy()
        results["similarity"] = cosine_sim[sim_indices]
        return results

    def recommend(self, user_id=None, n=10, user_rated_dict=None):
        """Recommend movies for a user based on user profile vector.

        user_rated_dict: dict of {movieId: rating} for cold-start or ad-hoc scoring.
        """
        user_ratings_map = {}
        if user_rated_dict is not None:
            user_ratings_map = user_rated_dict
        elif self.train_ratings is not None and user_id is not None:
            user_data = self.train_ratings[self.train_ratings["userId"] == user_id]
            user_ratings_map = dict(zip(user_data["movieId"], user_data["rating"]))

        if not user_ratings_map:
            # Cold start fallback: return top movies
            return self.movies_df.head(n).copy()

        # Create user profile vector as weighted sum of rated movies' tf-idf vectors
        rated_indices = []
        weights = []

        mean_rating = np.mean(list(user_ratings_map.values()))

        for mid, rating in user_ratings_map.items():
            if mid in self.movie_id_to_idx:
                rated_indices.append(self.movie_id_to_idx[mid])
                # Center weights around mean to upweight liked and downweight disliked
                weights.append(rating - mean_rating + 1.0)

        if not rated_indices:
            return self.movies_df.head(n).copy()

        weights = np.array(weights).reshape(-1, 1)
        sub_matrix = self.tfidf_matrix[rated_indices].toarray()
        user_profile = np.sum(sub_matrix * weights, axis=0, keepdims=True)

        scores = linear_kernel(user_profile, self.tfidf_matrix).flatten()

        # Exclude movies already seen
        seen_movie_ids = set(user_ratings_map.keys())
        ranked_indices = scores.argsort()[::-1]

        rec_indices = []
        for idx in ranked_indices:
            mid = self.idx_to_movie_id[idx]
            if mid not in seen_movie_ids:
                rec_indices.append(idx)
            if len(rec_indices) >= n:
                break

        res = self.movies_df.iloc[rec_indices].copy()
        res["score"] = scores[rec_indices]
        return res
