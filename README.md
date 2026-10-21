# 🎬 Movie Recommender Lab

**DS-311 Data Mining · Fall 2026 · NUST**

> How well do different ways of recommending movies work, and for which users do they fail?

We build three movie recommenders on the MovieLens dataset, compare them in one fair test, and show them side by side in a Streamlit web app with movie posters from TMDB.

---

## 👥 Team

| Member | Role |
|---|---|
| **Aiman Aslam** | Team Leader |
| **M Wajahat** | Team Member |
| **Muhammad Abdul Rehman Nosherwani** | Team Member |

**Instructor:** Dr. Zirak Khan  ·  **Lab Engineer:** Daniyal Zaidi

---

## ❓ Research questions

1. **RQ1:** Which method gives the best top-10 movie list?
2. **RQ2:** How well does each method work for new users who have rated only a few movies?
3. **RQ3:** How much does each method prefer popular movies over less-known ones?

---

## 🧠 Methods

| Recommender | How it works |
|---|---|
| **Popularity baseline** | Suggests the most-rated movies the user has not seen. Every other method must beat it. |
| **Content-based** | TF-IDF on genres and tags, then cosine similarity between movies. |
| **Collaborative filtering** | Item-item and user-user k-nearest neighbours (cosine / Pearson) and matrix factorisation (TruncatedSVD). |
| **Association rules** | FP-Growth on each user's liked movies (rated 4 or more), turned into "people who liked A also liked B". |

**Fair test:** for each user, the older 80% of ratings are used for training and the newest 20% for testing.
**Scores:** precision@10, recall@10, NDCG@10 and RMSE. For RQ2 we keep only each user's first 5, 10 or 20 ratings; for RQ3 we measure how popular the suggested movies are.

---

## 📊 Data

- **[MovieLens latest-small](https://grouplens.org/datasets/movielens/)** from GroupLens: 100,836 ratings and 9,742 movies, used for development.
- **MovieLens 32M**: a sample of users, used to test at larger scale.
- **[TMDB API](https://www.themoviedb.org/)**: posters, release years, genres and summaries for the app, looked up with the `tmdbId` from MovieLens `links.csv`.

The MovieLens data may not be shared again, so **it is not stored in this repository**. The code downloads it automatically the first time it runs.

---

## 🚀 How to run in Google Colab

Paste this into the first cell of a notebook:

```python
%cd /content
import os
if not os.path.exists("Movie-Recommender-lab-data-mining-project"):
    !git clone https://github.com/wajahat2005/Movie-Recommender-lab-data-mining-project.git
%cd /content/Movie-Recommender-lab-data-mining-project
!git pull -q
!pip install -q -r requirements.txt
```

Then load the data:

```python
from src.data import load_data

data = load_data()          # downloads MovieLens the first time
ratings = data["ratings"]
movies = data["movies"]
print(ratings.shape, movies.shape)
```

---

## 💻 How to run the app locally

```bash
git clone https://github.com/wajahat2005/Movie-Recommender-lab-data-mining-project.git
cd Movie-Recommender-lab-data-mining-project
pip install -r requirements.txt
streamlit run app/app.py
```

### TMDB API key

The app needs a free TMDB API key for posters and movie details.

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/) and request an API key in your account settings.
2. Create the file `.streamlit/secrets.toml` and add:
   ```toml
   TMDB_API_KEY = "your-key-here"
   ```
3. **Never commit this file.** It is listed in `.gitignore`. On Streamlit Cloud, add the key under the app's **Settings → Secrets**.

---

## ☁️ Streamlit Cloud Deployment

1. Connect your repository on [share.streamlit.io](https://share.streamlit.io/).
2. Set **Main file path** to `streamlit.py` (or `streamlit_app.py`).
3. Under **Advanced settings → Secrets**, add your TMDB key:
   ```toml
   TMDB_API_KEY = "your-tmdb-api-key"
   ```
4. Click **Deploy!**

---

## ✅ Progress

- [x] Repository set up (folders, requirements, README, project rules)
- [x] MovieLens loads in Colab with `load_data()`
- [x] Project proposal (M0)
- [ ] Data-quality report and EDA (M1)
- [ ] Fair test, scores and popularity baseline (M1–M2)
- [ ] Content-based, collaborative filtering and association rules (M2–M3)
- [ ] Streamlit app with TMDB posters (M3)
- [ ] Full comparison and app online (M4)
- [ ] Final report, demo and viva

---

## 📝 Contribution log

| Date | Member | What was done |
|---|---|---|
| | M Wajahat | |
| | Aiman Aslam | |
| | Muhammad Abdul Rehman Nosherwani | |

---

## 📚 Credits and citation

**MovieLens:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. https://doi.org/10.1145/2827872

**TMDB:** This product uses the TMDB API but is not endorsed or certified by TMDB.

**AI use:** Parts of this project were written with help from Claude (Anthropic), as allowed by course policy. Every member has reviewed and can explain the code.
