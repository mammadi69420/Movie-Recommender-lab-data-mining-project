# CLAUDE.md

University data-mining project: a MovieLens movie recommender that compares
**content-based**, **collaborative filtering (kNN + SVD)** and
**association-rule** recommenders.

## Repo layout

- `notebooks/` – Colab notebooks (EDA, each recommender, evaluation)
- `src/` – reusable Python helpers imported by notebooks and the app
- `app/` – Streamlit demo app
- `reports/` – final report, figures and slides
- `data/` – local dataset copy (git-ignored, never committed)

## Rules (always follow)

1. **Fixed random seed 42.** Use `RANDOM_STATE = 42` everywhere randomness
   appears: train/test splits, `random_state=` arguments, `np.random.seed(42)`,
   sampling, SVD initialisation.
2. **Use sklearn `Pipeline`s wherever preprocessing is fitted** (e.g.
   TF-IDF, scalers, encoders, `TruncatedSVD`). Fit only on training data so no
   information leaks from the test set.
3. **No data files committed.** Datasets are downloaded at runtime into
   `data/`, which is git-ignored along with `*.zip`. Do not add CSVs, zips or
   pickled models to git.
4. **Must run on free Google Colab CPU.** No GPU, no heavy libraries, keep
   memory modest (prefer MovieLens "small"/100K-sized data, sparse matrices,
   sampling where needed). Only use packages listed in `requirements.txt`.
5. **Simple, commented code.** Students must be able to explain every line in
   their viva. Prefer clear, step-by-step code over clever one-liners, use
   descriptive names, and add short comments explaining *why* each step is done.
