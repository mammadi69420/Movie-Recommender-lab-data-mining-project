"""Generate a comprehensive PDF project report for the Movie Recommender Lab.

Run:  python generate_report.py
Output: project_report.pdf in the repo root.
"""

from fpdf import FPDF
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_report.pdf")


class ReportPDF(FPDF):
    """Custom PDF class with header/footer and helper methods."""

    DARK_BLUE = (15, 23, 42)
    ACCENT_BLUE = (56, 189, 248)
    ACCENT_TEAL = (20, 184, 166)
    ACCENT_PURPLE = (168, 85, 247)
    ACCENT_AMBER = (245, 158, 11)
    WHITE = (255, 255, 255)
    LIGHT_GRAY = (226, 232, 240)
    MID_GRAY = (148, 163, 184)
    TEXT_DARK = (30, 41, 59)
    SECTION_BG = (241, 245, 249)

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
        self._section_number = 0
        self._subsection_counters = {}

    def header(self):
        if self.page_no() == 1:
            return  # Title page has custom header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.MID_GRAY)
        self.cell(0, 8, "Movie Recommender Lab - DS-311 Data Mining Project Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        # Thin accent line
        self.set_draw_color(*self.ACCENT_BLUE)
        self.set_line_width(0.4)
        self.line(20, 16, self.w - 20, 16)
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.MID_GRAY)
        self.cell(0, 10, "NUST | Fall 2026 | DS-311 Data Mining", align="C")

    # ---- Layout helpers ----

    def add_title_page(self):
        self.add_page()
        self.ln(35)
        # Accent bar
        self.set_fill_color(*self.ACCENT_BLUE)
        self.rect(20, 50, self.w - 40, 3, style="F")
        self.ln(20)

        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*self.TEXT_DARK)
        self.cell(0, 16, "Movie Recommender Lab", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*self.MID_GRAY)
        self.cell(0, 10, "A Comparative Study of Recommendation Algorithms", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)

        # Accent bar below subtitle
        self.set_fill_color(*self.ACCENT_AMBER)
        self.rect(70, self.get_y(), self.w - 140, 2, style="F")
        self.ln(14)

        # Course info box
        self.set_fill_color(*self.SECTION_BG)
        box_y = self.get_y()
        self.rect(35, box_y, self.w - 70, 38, style="F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.TEXT_DARK)
        self.set_xy(35, box_y + 6)
        self.cell(self.w - 70, 7, "DS-311  Data Mining  |  Fall 2026", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.set_xy(35, box_y + 14)
        self.cell(self.w - 70, 7, "National University of Sciences and Technology (NUST)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(35, box_y + 23)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*self.MID_GRAY)
        self.cell(self.w - 70, 7, "Instructor: Dr. Zirak Khan  |  Lab Engineer: Daniyal Zaidi", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(box_y + 46)

        # Team
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.TEXT_DARK)
        self.cell(0, 10, "Team Members", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

        team = [
            ("Aiman Aslam", "Team Leader"),
            ("M Wajahat", "Team Member"),
            ("Muhammad Abdul Rehman Nosherwani", "Team Member"),
        ]
        for name, role in team:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*self.TEXT_DARK)
            self.cell(90, 8, name, align="R")
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*self.MID_GRAY)
            self.cell(80, 8, f"  -  {role}", align="L", new_x="LMARGIN", new_y="NEXT")

    def section_heading(self, title):
        self._section_number += 1
        self._subsection_counters[self._section_number] = 0
        self.ln(6)
        # Accent bar
        self.set_fill_color(*self.ACCENT_BLUE)
        self.rect(20, self.get_y(), 4, 10, style="F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.TEXT_DARK)
        self.set_x(28)
        self.cell(0, 10, f"{self._section_number}. {title}", new_x="LMARGIN", new_y="NEXT")
        # Underline
        self.set_draw_color(*self.LIGHT_GRAY)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), self.w - 20, self.get_y())
        self.ln(4)

    def subsection_heading(self, title):
        self._subsection_counters[self._section_number] = self._subsection_counters.get(self._section_number, 0) + 1
        sub_num = self._subsection_counters[self._section_number]
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.ACCENT_TEAL)
        self.cell(0, 8, f"  {self._section_number}.{sub_num}  {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def formula_box(self, label, formula_text, explanation=""):
        """Render a highlighted formula box."""
        self.ln(2)
        self.set_fill_color(248, 250, 252)
        y_start = self.get_y()
        # Left accent bar
        self.set_fill_color(*self.ACCENT_AMBER)
        self.rect(22, y_start, 3, 22 + (6 if explanation else 0), style="F")
        # Background
        self.set_fill_color(255, 251, 235)
        self.rect(25, y_start, self.w - 45, 22 + (6 if explanation else 0), style="F")

        self.set_x(30)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.ACCENT_AMBER)
        self.cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
        self.set_x(30)
        self.set_font("Courier", "B", 10)
        self.set_text_color(*self.TEXT_DARK)
        self.cell(0, 7, formula_text, new_x="LMARGIN", new_y="NEXT")
        if explanation:
            self.set_x(30)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.MID_GRAY)
            self.cell(0, 6, explanation, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def bullet_list(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.TEXT_DARK)
        for item in items:
            self.set_x(26)
            self.cell(5, 5.5, chr(8226))  # bullet
            self.multi_cell(self.w - 55, 5.5, item)
            self.ln(1)
        self.ln(1)

    def key_value_table(self, headers, rows):
        """Simple bordered table."""
        col_w = (self.w - 40) / len(headers)
        # Header row
        self.set_fill_color(*self.ACCENT_BLUE)
        self.set_text_color(*self.WHITE)
        self.set_font("Helvetica", "B", 9)
        for h in headers:
            self.cell(col_w, 8, h, border=1, align="C", fill=True)
        self.ln()
        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.TEXT_DARK)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(*self.SECTION_BG)
            else:
                self.set_fill_color(*self.WHITE)
            for val in row:
                self.cell(col_w, 7, str(val), border=1, align="C", fill=True)
            self.ln()
            fill = not fill
        self.ln(3)


def build_report():
    pdf = ReportPDF()

    # ========== TITLE PAGE ==========
    pdf.add_title_page()

    # ========== TABLE OF CONTENTS ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*pdf.TEXT_DARK)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*pdf.ACCENT_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 80, pdf.get_y())
    pdf.ln(6)

    toc_items = [
        ("1.", "Introduction & Problem Statement"),
        ("2.", "Dataset Description"),
        ("3.", "Data Preprocessing & Feature Engineering"),
        ("4.", "Recommender Algorithms"),
        ("  4.1", "Popularity Baseline (Weighted Rating)"),
        ("  4.2", "Content-Based Filtering (TF-IDF + Cosine Similarity)"),
        ("  4.3", "Collaborative Filtering (Item-Item kNN)"),
        ("  4.4", "Association Rules (FP-Growth)"),
        ("5.", "Evaluation Framework"),
        ("  5.1", "Temporal Train/Test Split"),
        ("  5.2", "Precision@K, Recall@K, NDCG@K"),
        ("6.", "Score Display & Ranking System"),
        ("7.", "Web Application Architecture"),
        ("8.", "Results & Comparative Analysis"),
        ("9.", "Conclusion & Future Work"),
        ("10.", "References"),
    ]
    for num, title in toc_items:
        is_sub = num.startswith("  ")
        pdf.set_font("Helvetica", "" if is_sub else "B", 10 if is_sub else 11)
        pdf.set_text_color(*pdf.TEXT_DARK)
        indent = 30 if is_sub else 24
        pdf.set_x(indent)
        pdf.cell(12, 7, num.strip())
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    # ========== SECTION 1: INTRODUCTION ==========
    pdf.add_page()
    pdf.section_heading("Introduction & Problem Statement")

    pdf.body_text(
        "Information overload is one of the defining challenges of the digital age. With millions of movies "
        "available across streaming platforms, users often struggle to discover content aligned with their "
        "preferences. Recommender systems address this challenge by leveraging historical interaction data, "
        "item metadata, and collaborative signals to generate personalized suggestions."
    )
    pdf.body_text(
        "This project, Movie Recommender Lab, implements and compares four fundamentally different recommendation "
        "strategies on the MovieLens latest-small dataset (~100,000 ratings from 600+ users on 9,700+ movies). "
        "The goal is to understand the strengths, weaknesses, and trade-offs of each approach through a unified "
        "evaluation framework and an interactive web application."
    )
    pdf.body_text(
        "The four recommender algorithms implemented are:\n"
        "  1. Popularity Baseline - IMDB-style weighted rating (non-personalized)\n"
        "  2. Content-Based Filtering - TF-IDF vectorization with cosine similarity\n"
        "  3. Collaborative Filtering - Item-Item k-Nearest Neighbors (kNN)\n"
        "  4. Association Rules - FP-Growth frequent pattern mining\n\n"
        "All algorithms use a fixed random state (RANDOM_STATE = 42) for full reproducibility and are "
        "constrained to run on free CPU resources to ensure accessibility."
    )

    # ========== SECTION 2: DATASET ==========
    pdf.section_heading("Dataset Description")

    pdf.body_text(
        "The project uses the MovieLens Latest Small dataset published by GroupLens Research at the "
        "University of Minnesota. This dataset is widely used as a benchmark in recommender systems research."
    )

    pdf.key_value_table(
        ["File", "Records", "Key Columns", "Description"],
        [
            ["ratings.csv", "~100,836", "userId, movieId, rating, timestamp", "User-movie ratings (0.5 - 5.0 stars)"],
            ["movies.csv", "~9,742", "movieId, title, genres", "Movie metadata with pipe-separated genres"],
            ["tags.csv", "~3,683", "userId, movieId, tag, timestamp", "Free-text tags assigned by users"],
            ["links.csv", "~9,742", "movieId, imdbId, tmdbId", "Cross-references to IMDb and TMDB"],
        ]
    )

    pdf.body_text(
        "Key dataset statistics:\n"
        "  - 610 unique users\n"
        "  - 9,742 unique movies spanning 20 genres\n"
        "  - Sparsity: ~98.3% (most user-movie pairs are unrated)\n"
        "  - Rating scale: 0.5 to 5.0 in half-star increments\n"
        "  - Time span: January 1996 to September 2018"
    )

    # ========== SECTION 3: PREPROCESSING ==========
    pdf.section_heading("Data Preprocessing & Feature Engineering")

    pdf.subsection_heading("Genre Parsing")
    pdf.body_text(
        "Movie genres in the raw CSV are stored as pipe-separated strings (e.g., 'Action|Adventure|Sci-Fi'). "
        "The preprocessing step splits these into Python lists for downstream use in content feature construction, "
        "filtering, and display."
    )

    pdf.subsection_heading("Text Feature Construction")
    pdf.body_text(
        "For content-based filtering, a text corpus is constructed per movie by concatenating:\n"
        "  1. The movie title (captures franchise and naming patterns)\n"
        "  2. Space-separated genres (captures categorical similarity)\n"
        "  3. User-assigned tags (captures subjective attributes like 'dark humor', 'twist ending')\n\n"
        "This combined text is then vectorized using TF-IDF (see Section 4.2)."
    )

    pdf.subsection_heading("Missing Data Handling")
    pdf.body_text(
        "Movies without tags receive an empty string for the tag field. Movies without TMDB IDs are handled "
        "gracefully in the web application by falling back to placeholder poster images. "
        "In the SVD collaborative filtering variant, unrated user-movie entries are imputed with the "
        "user's mean rating before matrix factorization to avoid cold-start singularities."
    )

    pdf.subsection_heading("Timestamp-based Ordering")
    pdf.body_text(
        "All ratings carry Unix timestamps, which are critical for the temporal train/test split "
        "(see Section 5.1) that ensures the evaluation mimics real-world prediction scenarios."
    )

    # ========== SECTION 4: ALGORITHMS ==========
    pdf.add_page()
    pdf.section_heading("Recommender Algorithms")

    # --- 4.1 Popularity ---
    pdf.subsection_heading("Popularity Baseline (Weighted Rating)")
    pdf.body_text(
        "The popularity baseline is a non-personalized recommender that ranks all movies by their "
        "overall quality, estimated via the IMDB weighted rating formula. This serves as the benchmark "
        "that all personalized methods must beat to demonstrate added value."
    )

    pdf.formula_box(
        "IMDB Weighted Rating Formula",
        "WR = (v / (v + m)) * R  +  (m / (v + m)) * C",
        "v = vote count,  R = average rating,  m = minimum votes threshold,  C = global mean rating"
    )

    pdf.body_text(
        "How it works:\n"
        "  - v/(v+m) is the 'credibility' weight: movies with many ratings get a weight close to 1\n"
        "  - m/(v+m) is the 'prior' weight: pulls under-voted movies toward the global average C\n"
        "  - This prevents movies with very few (but perfect) ratings from dominating the ranking\n"
        "  - In our implementation, m = 5 (minimum 5 ratings required)\n\n"
        "For a given user, movies they have already rated in the training set are excluded from the "
        "recommendation list. The remaining movies are sorted by WR descending."
    )

    pdf.body_text(
        "Strengths: Simple, fast, no cold-start problem for items (only needs aggregate stats). "
        "Always produces recommendations.\n"
        "Weaknesses: Completely non-personalized. Cannot capture individual preferences. "
        "Tends to recommend blockbusters everyone has seen."
    )

    # --- 4.2 Content-Based ---
    pdf.subsection_heading("Content-Based Filtering (TF-IDF + Cosine Similarity)")
    pdf.body_text(
        "Content-based filtering recommends movies similar to those a user has previously enjoyed, "
        "based on item metadata (genres, tags, titles). The core idea is: if a user liked Sci-Fi action movies, "
        "recommend more Sci-Fi action movies."
    )

    pdf.body_text("Step 1: TF-IDF Vectorization")
    pdf.formula_box(
        "TF-IDF (Term Frequency - Inverse Document Frequency)",
        "tfidf(t, d) = tf(t, d) * log(N / df(t))",
        "tf = term frequency in document d,  N = total documents,  df = documents containing term t"
    )
    pdf.body_text(
        "Each movie's concatenated text (title + genres + tags) is transformed into a high-dimensional "
        "TF-IDF vector using sklearn's TfidfVectorizer with max_features=4000 and English stop-word removal. "
        "This produces a sparse matrix of shape (n_movies, 4000)."
    )

    pdf.body_text("Step 2: User Profile Construction")
    pdf.formula_box(
        "User Profile Vector",
        "P_u = SUM( (r_i - mean(r) + 1) * tfidf_i )  for each rated movie i",
        "r_i = user's rating for movie i,  mean(r) = user's average rating,  tfidf_i = movie i's TF-IDF vector"
    )
    pdf.body_text(
        "The user profile is a weighted sum of TF-IDF vectors of all movies rated by the user. "
        "Ratings are centered around the user's mean rating (+1 offset to keep positive) so that "
        "highly-rated movies contribute positively and poorly-rated movies contribute negatively "
        "to the profile."
    )

    pdf.body_text("Step 3: Scoring & Ranking")
    pdf.formula_box(
        "Cosine Similarity (via linear kernel on L2-normalized vectors)",
        "score(u, j) = P_u . tfidf_j = cosine(P_u, tfidf_j)",
        "P_u = user profile vector,  tfidf_j = candidate movie j's TF-IDF vector"
    )
    pdf.body_text(
        "The dot product (linear kernel) between the user profile and each candidate movie's TF-IDF vector "
        "yields a relevance score. Movies already rated by the user are excluded, and the remaining "
        "candidates are sorted by score descending."
    )
    pdf.body_text(
        "Strengths: Fully transparent recommendations (explainable via matching features). No need for "
        "other users' data. Good for niche tastes.\n"
        "Weaknesses: Limited by available metadata. Cannot discover surprising cross-genre recommendations. "
        "Over-specialization ('filter bubble')."
    )

    # --- 4.3 Collaborative Filtering ---
    pdf.add_page()
    pdf.subsection_heading("Collaborative Filtering (Item-Item kNN)")
    pdf.body_text(
        "Collaborative filtering is based on the principle that users who agreed in the past will agree "
        "in the future. Item-Item kNN finds movies that are similar to what the user has liked, where "
        "similarity is measured by how similarly ALL users have rated them (not by content features)."
    )

    pdf.body_text("Step 1: Item-User Rating Matrix")
    pdf.body_text(
        "An item-user sparse matrix is constructed where each row represents a movie, each column a user, "
        "and each entry is the rating value. Shape: (n_movies, n_users). Unrated entries remain zero "
        "in the sparse representation. This matrix is stored as a scipy CSR sparse matrix for memory efficiency."
    )

    pdf.body_text("Step 2: k-Nearest Neighbors (kNN) with Cosine Distance")
    pdf.formula_box(
        "Cosine Similarity Between Items",
        "sim(i, j) = (r_i . r_j) / (||r_i|| * ||r_j||)",
        "r_i = rating vector of movie i across all users,  r_j = rating vector of movie j"
    )
    pdf.body_text(
        "sklearn's NearestNeighbors with metric='cosine' and algorithm='brute' is fitted on the "
        "item-user matrix. For each movie, the k=15 nearest neighbors (most similarly-rated movies) "
        "are identified. Cosine distance is converted to similarity as: sim = 1 - distance."
    )

    pdf.body_text("Step 3: Prediction by Weighted Aggregation")
    pdf.formula_box(
        "Predicted Relevance Score for Candidate Movie j",
        "score(u, j) = SUM( sim(i, j) * r_u,i )  for each rated movie i by user u",
        "sim(i, j) = cosine similarity between movie i and j,  r_u,i = user u's rating for movie i"
    )
    pdf.body_text(
        "For each movie the user has rated, its kNN neighbors are retrieved. Neighbors not already "
        "rated by the user become candidates. Each candidate's score is the sum of "
        "(similarity x user_rating) across all rated movies that have that candidate as a neighbor. "
        "This aggregation captures both how similar the candidate is to liked movies and how strongly "
        "the user liked those source movies."
    )
    pdf.body_text(
        "Strengths: Captures emergent taste patterns that content analysis misses. Can recommend "
        "cross-genre movies. Well-suited for dense rating matrices.\n"
        "Weaknesses: Cold-start problem (new movies/users have no ratings). Computationally expensive "
        "for large datasets. Struggles with very sparse data."
    )

    # --- 4.4 Association Rules ---
    pdf.subsection_heading("Association Rules (FP-Growth)")
    pdf.body_text(
        "Association rule mining discovers patterns of the form 'users who liked movie A also liked "
        "movie B'. This approach is borrowed from market basket analysis and applied to movie "
        "recommendation by treating each user's set of highly-rated movies as a 'transaction'."
    )

    pdf.body_text("Step 1: Transaction Construction")
    pdf.body_text(
        "For each user, movies rated >= 4.0 stars are collected into a transaction (a set of movie IDs). "
        "Transactions with fewer than 2 items are discarded. These transactions are then one-hot encoded "
        "using mlxtend's TransactionEncoder."
    )

    pdf.body_text("Step 2: FP-Growth Algorithm")
    pdf.formula_box(
        "Support of an Itemset X",
        "support(X) = count(transactions containing X) / total_transactions",
        "Minimum support threshold = 0.03 (3% of all transactions)"
    )
    pdf.body_text(
        "FP-Growth (Frequent Pattern Growth) is an efficient algorithm for mining frequent itemsets "
        "without candidate generation (unlike Apriori). It compresses the transaction database into "
        "an FP-Tree data structure and extracts frequent itemsets by traversing conditional pattern "
        "bases. This is significantly faster than Apriori for large, dense datasets."
    )

    pdf.body_text("Step 3: Association Rule Generation")
    pdf.formula_box(
        "Confidence and Lift",
        "confidence(A => B) = support(A U B) / support(A)\n"
        "lift(A => B) = confidence(A => B) / support(B)",
        "Minimum confidence threshold = 0.2 (20%)"
    )
    pdf.body_text(
        "Rules are generated from frequent itemsets with confidence >= 0.2. Lift > 1 indicates a "
        "positive association (the items appear together more often than expected by chance). "
        "Rules are sorted by confidence * lift (descending), and the top 5000 are retained for efficiency."
    )

    pdf.body_text("Step 4: Recommendation Scoring")
    pdf.formula_box(
        "Association Rule Score for Candidate Movie",
        "score(j) = max( confidence(rule) * lift(rule) )  over all matching rules",
        "A matching rule has its antecedent as a subset of the user's liked movies"
    )
    pdf.body_text(
        "For a given user, the system checks which rules have antecedents that are subsets of the "
        "user's liked movies. The consequent movies of matching rules become candidates, scored by "
        "confidence * lift. Higher scores indicate stronger 'people who liked X also liked Y' patterns.\n\n"
        "Strengths: Intuitive, explainable rules. Captures co-occurrence patterns missed by similarity. "
        "No matrix operations needed.\n"
        "Weaknesses: Requires sufficient transaction density. Cannot generate rules for rare movies. "
        "Binary (liked/not liked) - ignores rating magnitudes."
    )

    # ========== SECTION 5: EVALUATION ==========
    pdf.add_page()
    pdf.section_heading("Evaluation Framework")

    pdf.subsection_heading("Temporal Train/Test Split")
    pdf.body_text(
        "To simulate real-world usage, the evaluation uses a per-user temporal split:\n\n"
        "  For each user:\n"
        "    - Sort ratings by timestamp (ascending)\n"
        "    - The oldest 80% of ratings form the training set\n"
        "    - The newest 20% form the test set\n\n"
        "This ensures the model never 'sees the future' - it trains on past behavior and is tested on "
        "predicting what the user will rate next. This is more realistic than random splits, which can "
        "leak temporal information."
    )

    pdf.subsection_heading("Precision@K, Recall@K, NDCG@K")

    pdf.body_text("The following standard ranking metrics are computed with K=10:")

    pdf.formula_box(
        "Precision@K",
        "P@K = |relevant items in top-K| / K",
        "What fraction of the top-K recommendations are actually relevant (rated >= 4.0 in test set)?"
    )

    pdf.formula_box(
        "Recall@K",
        "R@K = |relevant items in top-K| / |all relevant items in test set|",
        "What fraction of all relevant test items did the top-K recommendations capture?"
    )

    pdf.formula_box(
        "Normalized Discounted Cumulative Gain (NDCG@K)",
        "DCG@K = SUM( rel_i / log2(i + 1) )  for i = 1..K\n"
        "NDCG@K = DCG@K / IDCG@K",
        "rel_i = relevance of item at rank i (binary: 1 if relevant, 0 otherwise),  IDCG = ideal DCG"
    )
    pdf.body_text(
        "NDCG rewards placing relevant items at higher ranks. A system that places the most relevant "
        "item at position 1 scores higher than one that places it at position 10, even if both have "
        "the same Precision@K. IDCG is the DCG of the ideal (perfect) ranking.\n\n"
        "A relevant item is defined as one rated >= 4.0 stars in the test set."
    )

    # ========== SECTION 6: SCORE DISPLAY ==========
    pdf.section_heading("Score Display & Ranking System")

    pdf.body_text(
        "An important design decision in this project concerns how recommendation scores are "
        "presented to users in the web interface."
    )

    pdf.subsection_heading("Why Raw Scores Are Not Meaningful to Users")
    pdf.body_text(
        "Each algorithm produces scores on fundamentally different scales:\n\n"
        "  - Popularity Baseline: IMDB weighted rating (naturally 0.0 - 5.0 range)\n"
        "  - Content-Based: Unbounded dot product of user profile and TF-IDF vectors (can be 0 to 100+)\n"
        "  - Collaborative kNN: Sum of (similarity * rating) across neighbors (can be 0 to 50+)\n"
        "  - Association Rules: confidence * lift product (typically 0.2 to 15+)\n\n"
        "Displaying these raw values would confuse users and make cross-algorithm comparison impossible."
    )

    pdf.subsection_heading("Rank-Based Display with Contextual Tooltips")
    pdf.body_text(
        "Instead of displaying misleading numerical scores, the application shows rank positions: "
        "#1 Match, #2 Match, #3 Match, etc. This accurately communicates the relative ordering "
        "within each algorithm's recommendation list.\n\n"
        "Each rank badge has a hover tooltip that explains what the ranking is based on:\n"
        "  - Popularity: 'Ranked by global weighted rating (IMDB formula: vote count x avg rating)'\n"
        "  - Content-Based: 'Ranked by genre, title & tag similarity to your taste (TF-IDF cosine)'\n"
        "  - Collaborative: 'Ranked by how much similar users enjoyed this (item-item kNN)'\n"
        "  - Association Rules: 'Ranked by people who liked your movies also liked this (FP-Growth)'\n\n"
        "This design prioritizes transparency and user understanding over artificial precision."
    )

    # ========== SECTION 7: WEB APP ==========
    pdf.add_page()
    pdf.section_heading("Web Application Architecture")

    pdf.body_text(
        "The interactive web application is built with Streamlit and consists of three main pages:"
    )

    pdf.subsection_heading("Page 1: Multi-Algorithm Recommender Comparison")
    pdf.body_text(
        "Users select a test user ID and view top-10 recommendations from all four algorithms "
        "side by side. Individual algorithm tabs show full movie cards with TMDB posters, genres, "
        "release years, rank badges, and expandable synopses. This enables direct comparison of "
        "how different algorithms serve the same user."
    )

    pdf.subsection_heading("Page 2: Interactive Movie Rater (Cold-Start Demo)")
    pdf.body_text(
        "Users rate 10 well-known movies using slider controls (0.0 - 5.0 stars). Upon rating, "
        "the Content-Based and Collaborative (kNN) recommenders generate personalized suggestions "
        "in real-time. This demonstrates how the system handles the cold-start problem for new users. "
        "The two algorithm sections are visually distinguished with:\n"
        "  - Teal accent border and header for Content-Based suggestions\n"
        "  - Purple accent border and header for Collaborative Filtering\n"
        "  - A vertical divider separating the two sections\n"
        "  - Brief algorithm descriptions under each section header"
    )

    pdf.subsection_heading("Page 3: About & Team")
    pdf.body_text(
        "Displays team member information, instructor/lab engineer credits, dataset attribution "
        "(GroupLens Research), and TMDB API credits."
    )

    pdf.subsection_heading("Technical Stack")
    pdf.bullet_list([
        "Frontend & Server: Streamlit (Python web framework for ML/data apps)",
        "Core ML: scikit-learn (TF-IDF, kNN, TruncatedSVD), scipy (sparse matrices)",
        "Association Rules: mlxtend (FP-Growth, TransactionEncoder)",
        "Data: pandas, numpy for data manipulation",
        "Posters & Metadata: TMDB API integration with graceful fallbacks",
        "Visualization: plotly, matplotlib, seaborn (notebooks)",
        "Reproducibility: RANDOM_STATE = 42 across all stochastic operations",
    ])

    # ========== SECTION 8: RESULTS ==========
    pdf.section_heading("Results & Comparative Analysis")

    pdf.body_text(
        "The four algorithms exhibit fundamentally different recommendation behaviors:"
    )

    pdf.key_value_table(
        ["Algorithm", "Personalized?", "Cold-Start?", "Explainability", "Diversity"],
        [
            ["Popularity", "No", "No issue", "Low", "Low (same for all)"],
            ["Content-Based", "Yes", "Needs ratings", "High (features)", "Medium"],
            ["Item-Item kNN", "Yes", "Needs ratings", "Medium", "High"],
            ["Assoc. Rules", "Yes", "Needs history", "High (rules)", "Low-Medium"],
        ]
    )

    pdf.body_text(
        "Key observations:\n\n"
        "1. Popularity Baseline always recommends the same globally popular movies regardless of user. "
        "It serves as the lower bound for personalization quality.\n\n"
        "2. Content-Based filtering excels for users with strong genre preferences but tends to create "
        "a 'filter bubble' - it rarely recommends movies outside the user's established taste profile.\n\n"
        "3. Collaborative Filtering (kNN) produces the most diverse recommendations by leveraging "
        "cross-user patterns. It can suggest movies from genres the user hasn't explored, based on "
        "the behavior of similar users.\n\n"
        "4. Association Rules are highly interpretable ('users who liked Inception also liked Interstellar') "
        "but require sufficient co-rating density. For users with niche tastes, few rules may match.\n\n"
        "5. The temporal split evaluation is stricter than random splits and better reflects "
        "real-world performance, where we predict future preferences from past behavior."
    )

    # ========== SECTION 9: CONCLUSION ==========
    pdf.section_heading("Conclusion & Future Work")

    pdf.body_text(
        "This project demonstrates that no single recommendation algorithm dominates across all scenarios. "
        "Popularity baselines provide reliable but generic suggestions. Content-based methods excel for "
        "niche content discovery within a user's taste profile. Collaborative filtering captures emergent "
        "community-level preferences. Association rules offer highly interpretable 'also liked' patterns."
    )

    pdf.body_text(
        "Future work could explore:\n"
        "  1. Hybrid models that combine content-based and collaborative signals\n"
        "  2. Deep learning approaches (neural collaborative filtering, autoencoders)\n"
        "  3. Incorporating temporal dynamics (user taste evolution over time)\n"
        "  4. A/B testing framework for online evaluation\n"
        "  5. Context-aware recommendations (time of day, mood, social context)\n"
        "  6. Scaling to the MovieLens 32M dataset for large-scale evaluation"
    )

    # ========== SECTION 10: REFERENCES ==========
    pdf.section_heading("References")

    refs = [
        "[1] F.M. Harper and J.A. Konstan. 'The MovieLens Datasets: History and Context.' "
        "ACM Transactions on Interactive Intelligent Systems, 5(4), 2015.",

        "[2] G. Linden, B. Smith, and J. York. 'Amazon.com Recommendations: Item-to-Item "
        "Collaborative Filtering.' IEEE Internet Computing, 7(1):76-80, 2003.",

        "[3] J. Han, J. Pei, Y. Yin. 'Mining Frequent Patterns without Candidate Generation.' "
        "ACM SIGMOD, pages 1-12, 2000. (FP-Growth algorithm)",

        "[4] P. Resnick, N. Iacovou, M. Suchak, P. Bergstrom, J. Riedl. 'GroupLens: An Open "
        "Architecture for Collaborative Filtering of Netnews.' ACM CSCW, pages 175-186, 1994.",

        "[5] scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.",

        "[6] S. Raschka. 'MLxtend: Providing machine learning and data science utilities and extensions "
        "to Python scientific computing libraries.' JOSS, 3(24), 638, 2018.",

        "[7] TMDB (The Movie Database) API. https://www.themoviedb.org/documentation/api",

        "[8] Streamlit: The fastest way to build data apps. https://streamlit.io/",
    ]

    for ref in refs:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*pdf.TEXT_DARK)
        pdf.set_x(24)
        pdf.multi_cell(pdf.w - 48, 5, ref)
        pdf.ln(2)

    # ========== SAVE ==========
    pdf.output(OUTPUT_PATH)
    print(f"Report generated successfully: {OUTPUT_PATH}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
