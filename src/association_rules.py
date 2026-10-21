"""Association Rules Recommender using FP-Growth.

Transforms each user's liked movies (rating >= 4.0) into transactions.
Applies FP-Growth to discover frequent itemsets and association rules
in the form "people who liked movie A also liked movie B".
Uses mlxtend.frequent_patterns.fpgrowth and association_rules.
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


class AssociationRuleRecommender:
    """Recommender based on FP-Growth association rules."""

    def __init__(self, min_support=0.03, min_confidence=0.2, min_rating=4.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_rating = min_rating
        self.rules_ = None
        self.movies_df = None
        self.user_history_ = {}

    def fit(self, train_ratings, movies_df=None):
        """Extract transactions of liked movies per user and mine association rules."""
        self.movies_df = movies_df
        self.user_history_ = (
            train_ratings.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # Filter liked movies (rating >= min_rating)
        liked_df = train_ratings[train_ratings["rating"] >= self.min_rating]

        # Group into user transactions
        transactions = (
            liked_df.groupby("userId")["movieId"]
            .apply(list)
            .tolist()
        )

        # Filter out empty or single-item transactions
        transactions = [t for t in transactions if len(t) > 1]
        if not transactions:
            self.rules_ = pd.DataFrame()
            return self

        # Encode transactions to one-hot boolean DataFrame
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Mine frequent itemsets with FP-Growth
        frequent_itemsets = fpgrowth(df_encoded, min_support=self.min_support, use_colnames=True)

        if frequent_itemsets.empty:
            self.rules_ = pd.DataFrame()
            return self

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        if not rules.empty:
            # Keep highest quality rules to maintain high performance
            rules = rules.sort_values(by=["confidence", "lift"], ascending=False).head(5000)

        self.rules_ = rules
        return self

    def recommend(self, user_id=None, n=10, user_liked_movies=None):
        """Recommend movies based on antecedents matched against user's liked movies."""
        liked_set = set()
        if user_liked_movies is not None:
            liked_set.update(user_liked_movies)
        elif user_id in self.user_history_:
            liked_set.update(self.user_history_[user_id])

        if self.rules_ is None or self.rules_.empty or not liked_set:
            return pd.DataFrame(columns=["movieId", "score"])

        candidate_scores = {}

        # Look for rules where antecedents are subset of liked_set
        for _, rule in self.rules_.iterrows():
            antecedents = rule["antecedents"]
            if antecedents.issubset(liked_set):
                consequents = rule["consequents"]
                score = float(rule["confidence"] * rule["lift"])
                for rec_item in consequents:
                    if rec_item not in liked_set:
                        if rec_item not in candidate_scores or score > candidate_scores[rec_item]:
                            candidate_scores[rec_item] = score
                if len(candidate_scores) >= n * 3:
                    break

        if not candidate_scores:
            return pd.DataFrame(columns=["movieId", "score"])

        sorted_recs = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        res = pd.DataFrame(sorted_recs, columns=["movieId", "score"])

        if self.movies_df is not None:
            res = res.merge(self.movies_df, on="movieId", how="left")
        return res
