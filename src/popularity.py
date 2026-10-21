"""Popularity Baseline Recommender.

Recommends the most popular (frequently rated / highest weighted rating) movies
that the target user has not yet rated.
All other recommender methods must benchmark against and attempt to beat this baseline.
"""

import pandas as pd


class PopularityRecommender:
    """Popularity-based baseline recommender."""

    def __init__(self, min_ratings=5):
        self.min_ratings = min_ratings
        self.popular_movies_ = None
        self.user_history_ = {}

    def fit(self, train_ratings, movies_df=None):
        """Compute popularity ranking based on rating count and mean score."""
        # Record movies already rated by each user in train set
        self.user_history_ = (
            train_ratings.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        stats = (
            train_ratings.groupby("movieId")
            .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"))
            .reset_index()
        )

        # Global average rating
        c = train_ratings["rating"].mean()
        m = self.min_ratings

        # IMDB style weighted rating formula: (v/(v+m) * R) + (m/(v+m) * C)
        stats["weighted_score"] = (
            (stats["rating_count"] / (stats["rating_count"] + m)) * stats["rating_mean"]
            + (m / (stats["rating_count"] + m)) * c
        )

        # Sort descending by popularity
        self.popular_movies_ = stats.sort_values(
            by=["weighted_score", "rating_count"], ascending=False
        )

        if movies_df is not None:
            self.popular_movies_ = self.popular_movies_.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return self

    def recommend(self, user_id=None, n=10, user_rated_movies=None):
        """Recommend top-N unseen popular movies for the user.

        If user_rated_movies is supplied (e.g. for new user), exclude those.
        Otherwise exclude movies from train set history.
        """
        seen_movies = set()
        if user_rated_movies is not None:
            seen_movies.update(user_rated_movies)
        elif user_id in self.user_history_:
            seen_movies.update(self.user_history_[user_id])

        candidates = self.popular_movies_[~self.popular_movies_["movieId"].isin(seen_movies)]
        return candidates.head(n).copy()
