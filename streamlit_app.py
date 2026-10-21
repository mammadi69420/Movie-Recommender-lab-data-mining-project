"""
Streamlit Cloud Entrypoint for Movie Recommender Lab.
DS-311 Data Mining · Fall 2026 · NUST

Streamlit Community Cloud defaults to detecting 'streamlit_app.py' or 'streamlit.py'.
This file seamlessly routes to the primary application in app/app.py.
"""

import os
import sys
import runpy

# Ensure repository root is in python search path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Path to the main Streamlit application
APP_PATH = os.path.join(REPO_ROOT, "app", "app.py")

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(APP_PATH, run_name="__main__")
