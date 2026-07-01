"""
app.py — Voyage Analytics · Entry Point
========================================
Run with:  streamlit run app.py

Uses st.navigation() so every page's sidebar label and icon is set
explicitly here in code — not inferred from filenames.

Page files are auto-detected by keyword (see theme.find_page), so this
works regardless of capitalisation, spaces, or underscores in the
actual filenames inside pages/.
"""

import streamlit as st
from theme import find_page

# ── Must be the very first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-detect each page file, then assign explicit titles + icons ────────
home_path = find_page("home")
dashboard_path = find_page("dashboard")
flight_path = find_page("flight")
gender_path = find_page("gender")
hotel_path = find_page("hotel")

home_page = st.Page(home_path, title="Home", icon="🌍", default=True)
dashboard_page = st.Page(dashboard_path, title="Dashboard", icon="📊")
flight_page = st.Page(flight_path, title="Flight Predictor", icon="✈️")
gender_page = st.Page(gender_path, title="Gender Predictor", icon="👤")
hotel_page = st.Page(hotel_path, title="Hotel Recommender", icon="🏨")

nav = st.navigation(
    {
        "Voyage Analytics": [home_page],
        "Modules": [dashboard_page, flight_page, gender_page, hotel_page],
    }
)

nav.run()
