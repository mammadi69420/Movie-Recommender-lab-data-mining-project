"""Download and load the MovieLens datasets.

Usage (from the repo root, e.g. in Colab after `%cd` into the repo):

    from src.data import load_data
    data = load_data()            # downloads ml-latest-small the first time
    ratings = data["ratings"]

The data is saved in the `data/` folder, which is git-ignored.
"""

import os                 # for building file paths and checking if files exist
import urllib.request     # for downloading the zip file from GroupLens
import zipfile            # for unzipping the downloaded file

import pandas as pd       # for loading the CSV files into DataFrames

# Fixed random seed used across the whole project (see CLAUDE.md)
RANDOM_STATE = 42

# Download links from the official GroupLens website
SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def _download_and_unzip(url, dest, folder_name):
    """Download a zip from `url` and unzip it into `dest`, unless already there.

    Returns the path of the unzipped folder, e.g. "data/ml-latest-small".
    """
    folder_path = os.path.join(dest, folder_name)  # where the CSVs will end up

    # If ratings.csv already exists we downloaded it before, so skip the work.
    # (We check a file, not just the folder, in case an earlier unzip was
    # interrupted and left a half-empty folder behind.)
    if os.path.isfile(os.path.join(folder_path, "ratings.csv")):
        print(f"Found existing data in {folder_path}, skipping download.")
        return folder_path

    os.makedirs(dest, exist_ok=True)                    # create data/ if missing
    zip_path = os.path.join(dest, folder_name + ".zip")  # where to save the zip

    import ssl

    print(f"Downloading {url} ...")
    # Handle SSL verification in environments where default cert store is missing/custom
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception:
        # Fallback without ssl verification
        insecure_ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=insecure_ctx) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    print(f"Unzipping to {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)  # the zip contains a folder named `folder_name`

    os.remove(zip_path)  # delete the zip to save disk space
    print("Done.")
    return folder_path


def download_movielens_small(dest="data"):
    """Download and unzip MovieLens 'latest-small' (~100k ratings) into `dest`.

    Does nothing if the data is already there. Returns the folder path.
    """
    return _download_and_unzip(SMALL_URL, dest, "ml-latest-small")


def _clean_tables(ratings, movies, tags, links):
    """Apply the same small cleaning steps to every MovieLens version."""
    # Timestamps are stored as seconds since 1970-01-01; turn them into dates
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")

    # Genres look like "Adventure|Animation|Children"; split them into a list
    movies["genres"] = movies["genres"].str.split("|")

    # Movies without genres are marked "(no genres listed)"; use an empty list
    movies["genres"] = movies["genres"].apply(
        lambda g: [] if g == ["(no genres listed)"] else g
    )

    # Return everything in one dictionary so it is easy to pass around
    return {"ratings": ratings, "movies": movies, "tags": tags, "links": links}


def load_data(dest="data"):
    """Load MovieLens 'latest-small' as a dict of DataFrames.

    Keys: "ratings", "movies", "tags", "links".
    Downloads the data first if needed.
    """
    folder = download_movielens_small(dest)  # make sure the files exist

    # Read each CSV file into a pandas DataFrame
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    tags = pd.read_csv(os.path.join(folder, "tags.csv"))
    links = pd.read_csv(os.path.join(folder, "links.csv"))

    return _clean_tables(ratings, movies, tags, links)


def load_ml32m_sample(frac=0.05, dest="data", seed=RANDOM_STATE):
    """Load a random sample of USERS from MovieLens 32M.

    We keep `frac` of the users (5% by default) and ALL ratings of each kept
    user. Sampling users instead of single ratings keeps every user's full
    history, which collaborative filtering needs.

    Returns the same dict as `load_data()`: ratings, movies, tags, links.
    Note: the first call downloads a ~240 MB zip.
    """
    folder = _download_and_unzip(ML32M_URL, dest, "ml-32m")

    # Read the 32 million ratings with small number types to save memory
    # (int32/float32 use half the RAM of pandas' default int64/float64)
    ratings = pd.read_csv(
        os.path.join(folder, "ratings.csv"),
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
    )

    # Get the list of all distinct users, then randomly pick `frac` of them
    all_users = pd.Series(ratings["userId"].unique())
    sampled_users = all_users.sample(frac=frac, random_state=seed)

    # Keep only the ratings that belong to the sampled users
    ratings = ratings[ratings["userId"].isin(sampled_users)].reset_index(drop=True)

    # Movies and links are small, so we keep all of them
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    links = pd.read_csv(os.path.join(folder, "links.csv"))

    # Keep only the tags written by the sampled users
    tags = pd.read_csv(os.path.join(folder, "tags.csv"))
    tags = tags[tags["userId"].isin(sampled_users)].reset_index(drop=True)

    print(f"Kept {len(sampled_users):,} users and {len(ratings):,} ratings.")
    return _clean_tables(ratings, movies, tags, links)
