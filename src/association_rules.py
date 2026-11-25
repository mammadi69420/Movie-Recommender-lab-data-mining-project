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
        self.user_liked_history_ = {}

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
        self.user_liked_history_ = (
            liked_df.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        # Group into user transactions
        transactions = (
            liked_df.groupby("userId")["movieId"]
            .apply(list)
            .tolist()
        )

        # Filter out empty or single-item transactions; cast IDs to str for sparse DataFrame
        transactions = [[str(m) for m in t] for t in transactions if len(t) > 1]
        if not transactions:
            self.rules_ = pd.DataFrame()
            return self

        # Encode transactions to sparse boolean DataFrame for speed & low RAM
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions, sparse=True)
        df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=[str(c) for c in te.columns_])

        # Mine frequent itemsets (max_len=2 keeps pairs fast in Colab CPU)
        frequent_itemsets = fpgrowth(
            df_encoded, min_support=self.min_support, use_colnames=True, max_len=2
        )

        if frequent_itemsets.empty:
            self.rules_ = pd.DataFrame()
            return self

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        if not rules.empty:
            # Convert string movieIds back to int frozensets
            rules["antecedents"] = rules["antecedents"].apply(lambda s: frozenset(int(x) for x in s))
            rules["consequents"] = rules["consequents"].apply(lambda s: frozenset(int(x) for x in s))
            # Keep highest quality rules to maintain high performance
            rules = rules.sort_values(by=["confidence", "lift"], ascending=False).head(5000)

        self.rules_ = rules
        return self

    def recommend(self, user_id=None, n=10, user_liked_movies=None):
        """Recommend movies based on antecedents matched against user's liked movies."""
        liked_set = set()
        seen_set = set()
        if user_liked_movies is not None:
            liked_set.update(user_liked_movies)
            seen_set.update(user_liked_movies)
        elif user_id is not None:
            liked_set.update(self.user_liked_history_.get(user_id, set()))
            seen_set.update(self.user_history_.get(user_id, set()))

        if self.rules_ is None or self.rules_.empty or not liked_set:
            return pd.DataFrame(columns=["movieId", "score", "explanation"])

        candidate_scores = {}
        candidate_reasons = {}

        # Look for rules where antecedents are subset of liked_set
        for _, rule in self.rules_.iterrows():
            antecedents = rule["antecedents"]
            if antecedents.issubset(liked_set):
                consequents = rule["consequents"]
                score = float(rule["confidence"] * rule["lift"])
                for rec_item in consequents:
                    if rec_item not in seen_set:
                        if rec_item not in candidate_scores or score > candidate_scores[rec_item]:
                            candidate_scores[rec_item] = score
                            candidate_reasons[rec_item] = sorted(antecedents)

        if not candidate_scores:
            return pd.DataFrame(columns=["movieId", "score", "explanation"])

        sorted_recs = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        res = pd.DataFrame(sorted_recs, columns=["movieId", "score"])

        title_map = {}
        if self.movies_df is not None:
            title_map = dict(zip(self.movies_df["movieId"], self.movies_df["title"]))
            res = res.merge(self.movies_df, on="movieId", how="left")

        explanations = []
        for mid in res["movieId"]:
            ants = candidate_reasons.get(mid, [])
            if ants:
                ant_title = title_map.get(ants[0], f"Movie #{ants[0]}")
                explanations.append(f"Because you liked {ant_title}")
            else:
                explanations.append("Frequently co-liked by similar viewers")
        res["explanation"] = explanations
        return res
