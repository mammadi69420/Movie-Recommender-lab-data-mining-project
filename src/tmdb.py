"""TMDB API integration for fetching movie posters, overviews, and ratings.

Requires TMDB_API_KEY in st.secrets or environment variable TMDB_API_KEY.
Falls back to clean placeholder posters if API key is not provided or network fails.
"""

import os
import re
import urllib.parse
import requests

try:
    import streamlit.logger as _sl
    _sl.set_log_level("ERROR")
except Exception:
    pass

import streamlit as st

TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def make_svg_poster(title: str, year: str = "") -> str:
    """Generate a responsive inline SVG movie poster data URI as fallback."""
    short_title = (title[:30] + "...") if len(title) > 33 else title
    safe_title = (
        short_title.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    safe_year = year.replace("&", "&amp;") if year else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" viewBox="0 0 500 750">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="50%" stop-color="#1E293B"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
  </defs>
  <rect width="500" height="750" fill="url(#bg)" rx="16"/>
  <rect x="18" y="18" width="464" height="714" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="8 6" opacity="0.35" rx="12"/>
  <text x="250" y="310" font-size="64" text-anchor="middle">🎬</text>
  <text x="250" y="390" font-family="system-ui, sans-serif" font-weight="bold" font-size="22" fill="#FFFFFF" text-anchor="middle">{safe_title}</text>
  <text x="250" y="430" font-family="system-ui, sans-serif" font-weight="600" font-size="16" fill="#38BDF8" text-anchor="middle">{safe_year}</text>
  <text x="250" y="490" font-family="system-ui, sans-serif" font-size="13" fill="#94A3B8" text-anchor="middle">Movie Recommender Lab</text>
</svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def clean_movie_title(raw_title: str):
    """Extract release year and normalize title for search (e.g. 'Matrix, The (1999)' -> 'The Matrix', '1999')."""
    if not raw_title:
        return "", ""

    # Extract year in parentheses at the end
    year = ""
    match_year = re.search(r"\((\d{4})\)\s*$", raw_title)
    if match_year:
        year = match_year.group(1)
        cleaned = raw_title[: match_year.start()].strip()
    else:
        cleaned = raw_title.strip()

    # Remove alternate foreign titles in parentheses like (Fabuleux destin...)
    cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()

    # Invert trailing articles: 'Shawshank Redemption, The' -> 'The Shawshank Redemption'
    for article in [", The", ", A", ", An", ", Le", ", La", ", Il", ", Der", ", Die", ", Das"]:
        if cleaned.endswith(article):
            cleaned = article[2:] + " " + cleaned[: -len(article)].strip()
            break

    return cleaned.strip(), year


def get_tmdb_api_key():
    """Retrieve TMDB API key from st.secrets, os.environ, or fallback."""
    try:
        if "TMDB_API_KEY" in st.secrets and st.secrets["TMDB_API_KEY"]:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY", "")


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_details(tmdb_id, title=None):
    """Fetch movie metadata (poster_url, overview, release_year, vote_average) from TMDB.

    Uses tmdb_id first; falls back to search by clean title + year if id lookup fails.
    Returns custom SVG poster if poster is unavailable.
    """
    clean_t, extracted_year = clean_movie_title(title or "")
    display_title = clean_t or (title or "Movie")
    fallback_poster = make_svg_poster(display_title, extracted_year)

    api_key = get_tmdb_api_key()
    if not api_key:
        return {
            "poster_url": fallback_poster,
            "overview": "Enter TMDB API Key in sidebar settings to view synopsis and posters.",
            "release_year": extracted_year or "N/A",
            "vote_average": "N/A",
            "tagline": "",
        }

    # 1. Try fetching via numeric tmdb_id
    if tmdb_id is not None and not pd_isna(tmdb_id):
        try:
            tid_int = int(float(tmdb_id))
            url = f"{TMDB_BASE_URL}/movie/{tid_int}?api_key={api_key}&language=en-US"
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    release_date = data.get("release_date", "")
                    release_year = release_date.split("-")[0] if release_date else extracted_year or "N/A"
                    return {
                        "poster_url": f"{POSTER_BASE_URL}{poster_path}",
                        "overview": data.get("overview") or "No synopsis available.",
                        "release_year": release_year,
                        "vote_average": data.get("vote_average", "N/A"),
                        "tagline": data.get("tagline", ""),
                    }
        except Exception:
            pass

    # 2. Fallback: Search by clean title + year
    if clean_t:
        try:
            query_encoded = urllib.parse.quote(clean_t)
            search_url = f"{TMDB_BASE_URL}/search/movie?api_key={api_key}&query={query_encoded}"
            if extracted_year:
                search_url += f"&year={extracted_year}"

            resp = requests.get(search_url, headers=REQUEST_HEADERS, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results and results[0].get("poster_path"):
                    first = results[0]
                    release_date = first.get("release_date", "")
                    release_year = release_date.split("-")[0] if release_date else extracted_year or "N/A"
                    return {
                        "poster_url": f"{POSTER_BASE_URL}{first['poster_path']}",
                        "overview": first.get("overview") or "No synopsis available.",
                        "release_year": release_year,
                        "vote_average": first.get("vote_average", "N/A"),
                        "tagline": "",
                    }
        except Exception:
            pass

    return {
        "poster_url": fallback_poster,
        "overview": "Synopsis not found on TMDB.",
        "release_year": extracted_year or "N/A",
        "vote_average": "N/A",
        "tagline": "",
    }


def pd_isna(val):
    """Helper to check NaN without strict pandas dependency."""
    try:
        import pandas as pd
        return pd.isna(val)
    except Exception:
        return val is None or val != val
