"""
theme.py — Voyage Analytics shared utilities
=============================================
Import this at the TOP of every page (app.py + pages/*.py) and call
`inject_theme()` first thing, before any other st.* calls that render UI.

Also centralises:
  - USD → INR conversion
  - CSV loaders for hotels.csv / flights.csv / users.csv with safe fallbacks
"""

import os
import re
import streamlit as st
import pandas as pd

# ── Page auto-detection ──────────────────────────────────────────────────────
# Finds files inside pages/ by keyword, ignoring case/spaces/underscores/emoji.
# Used by BOTH app.py (st.Page definitions) and pages/home.py (st.page_link
# targets) so there is exactly one source of truth — renaming a file in
# pages/ (different case, spaces, underscores) never breaks navigation as
# long as the filename still contains the right keyword.

_PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")


def _normalise_filename(name: str) -> str:
    """Lowercase, strip extension, remove everything except letters."""
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^a-zA-Z]", "", name)
    return name.lower()


def find_page(*keywords: str) -> str:
    """
    Find a .py file in pages/ whose normalised name contains all given
    keywords (case/space/underscore-insensitive). Returns a path relative
    to the project root, e.g. "pages/Hotel recommender.py" — safe to pass
    directly to st.Page(...) or st.page_link(...).
    Falls back to a sensible default filename if nothing matches.
    """
    if os.path.isdir(_PAGES_DIR):
        for fname in os.listdir(_PAGES_DIR):
            if not fname.endswith(".py"):
                continue
            norm = _normalise_filename(fname)
            if all(kw in norm for kw in keywords):
                return os.path.join("pages", fname)
    return os.path.join("pages", f"{'_'.join(keywords)}.py")

# ── Currency ─────────────────────────────────────────────────────────────────
USD_TO_INR = 83.0          # update this anytime the rate needs to change
CURRENCY_SYMBOL = "₹"

# ── Backend API (Member 3's spec) ────────────────────────────────────────────
# Base URL + endpoints exactly as documented by the backend lead.
# All three prediction pages import API_BASE_URL + call_* functions from here
# so there is one place to update if the contract changes.
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 10  # seconds


def check_backend_status(show_message: bool = True) -> bool:
    """
    Calls GET /health. Returns True if backend is reachable and healthy.
    Call this before showing prediction forms so users get an early signal
    instead of only finding out when they click Predict.
    """
    import requests
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
        if response.status_code == 200:
            if show_message:
                data = response.json()
                service = data.get("service", "Backend")
                version = data.get("version", "")
                st.success(f"✅ Backend API Online — {service} {version}".strip())
            return True
        else:
            if show_message:
                st.error(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        if show_message:
            st.error(f"❌ Cannot connect to backend: {e}")
        return False


def safe_api_call(endpoint: str, payload: dict, retries: int = 3, show_errors: bool = True) -> dict | None:
    """
    Robust POST call matching Member 3's VoyageAPIClient.safe_api_call exactly:
    distinguishes 400/404/500/503/timeout/connection-error, retries with
    exponential backoff on 503 and on timeout, and surfaces a specific
    st.error/st.warning per failure type rather than a single generic message.

    Returns the parsed JSON body on success, or None after all attempts fail
    (caller should then fall back to its local mock model).
    """
    import requests
    import time

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}/{endpoint}",
                json=payload,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                if show_errors:
                    st.error("❌ Invalid request data. Please check your inputs.")
                return None
            elif response.status_code == 404:
                if show_errors:
                    st.error("❌ Endpoint not found. Please check the API configuration.")
                return None
            elif response.status_code == 500:
                if show_errors:
                    st.error("❌ Server error. The ML model may be unavailable.")
                return None
            elif response.status_code == 503:
                if show_errors:
                    st.warning("⏰ Service temporarily unavailable. Retrying…")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                    continue
                return None
            else:
                if show_errors:
                    st.error(f"❌ Unexpected error: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            if show_errors:
                st.warning(f"⏰ Request timed out (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(1)
                continue
            if show_errors:
                st.error("⏰ All retry attempts failed due to timeout.")
            return None

        except requests.exceptions.ConnectionError:
            if attempt == 0 and show_errors:
                st.error("🔌 Cannot connect to backend API. Is the server running?")
            return None

        except requests.exceptions.RequestException as e:
            if show_errors:
                st.error(f"🚨 Network error: {e}")
            return None

        except ValueError as e:
            if show_errors:
                st.error(f"📊 Invalid response format: {e}")
            return None

    return None


def call_predict_flight(
    flight_type: str, agency: str,
    from_city: str, to_city: str,
    distance: float, time: float,
    day: int, month: int, weekday: int,
    show_errors: bool = True,
) -> dict | None:
    """
    POST /predict_flight — actual contract from Member 3's test file.
    Payload: {flightType, agency, from, to, distance, time, day, month, weekday}
    """
    payload = {
        "flightType": flight_type,
        "agency": agency,
        "from": from_city,
        "to": to_city,
        "distance": distance,
        "time": time,
        "day": day,
        "month": month,
        "weekday": weekday,
    }
    return safe_api_call("predict_flight", payload, show_errors=show_errors)


def call_classify_gender(
    first_name: str, company: str, age: int,
    show_errors: bool = True,
) -> dict | None:
    """
    POST /classify_gender — actual contract from Member 3's test file.
    Payload: {first_name, company, name_length, first_letter, last_letter, age}
    name_length, first_letter, last_letter are derived here so the
    Streamlit form only needs to collect first_name, company, age.
    """
    clean = first_name.strip()
    payload = {
        "first_name": clean,
        "company": company,
        "name_length": len(clean),
        "first_letter": clean[0].upper() if clean else "",
        "last_letter": clean[-1].lower() if clean else "",
        "age": age,
    }
    return safe_api_call("classify_gender", payload, show_errors=show_errors)


def call_recommend_hotels(
    destination: str, budget_max: float,
    amenities: list, style: str,
    min_stars: int, min_review: float,
    show_errors: bool = True,
) -> dict | None:
    """
    POST /recommend_hotels — actual contract from Member 3's test file.
    Payload: {destination, budget_max, amenities, style, min_stars, min_review}
    These keys match your recommender.py's user_data dict exactly.
    """
    payload = {
        "destination": destination,
        "budget_max": budget_max,
        "amenities": amenities,
        "style": style,
        "min_stars": min_stars,
        "min_review": min_review,
    }
    return safe_api_call("recommend_hotels", payload, show_errors=show_errors)


def render_api_debug_panel() -> None:
    """
    Sidebar API Testing Dashboard, matching Member 3's spec exactly:
    a checkbox-gated panel showing live health status as JSON, the
    base URL / endpoint list, and a quick-test button for the flight
    endpoint with the example payload from the spec.
    Call this once from any page (typically home.py) — Streamlit's
    sidebar persists across pages within st.navigation.
    """
    import requests

    st.sidebar.header("🔧 Developer Tools")

    if st.sidebar.checkbox("Show API Debug Panel"):
        st.sidebar.subheader("🔍 API Status")

        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
            if resp.status_code == 200:
                st.sidebar.success("✅ Backend Online")
                st.sidebar.json(resp.json())
            else:
                st.sidebar.error(f"❌ Backend returned {resp.status_code}")
        except requests.exceptions.RequestException:
            st.sidebar.error("❌ Backend Offline")

        st.sidebar.write("**API Endpoints:**")
        st.sidebar.code(f"Base URL: {API_BASE_URL}")
        st.sidebar.code("Endpoints: /health, /predict_flight, /classify_gender, /recommend_hotels")

        if st.sidebar.button("🧪 Test Flight API"):
            test_payload = {
                "origin": "NYC", "destination": "LAX",
                "date": "2024-12-25", "airline": "Delta",
            }
            result = safe_api_call("predict_flight", test_payload, show_errors=True)
            st.sidebar.json(result)


def to_inr(usd_amount):
    """
    Convert a USD amount to INR using the fixed rate above.
    Works for a single number OR a numpy array / pandas Series.
    """
    import numpy as np
    converted = np.asarray(usd_amount, dtype=float) * USD_TO_INR
    rounded = np.round(converted, 0)
    # Return a plain Python float for scalars (keeps old behaviour/call sites happy),
    # otherwise return the array/Series unchanged in shape.
    if np.isscalar(usd_amount) or (hasattr(usd_amount, "ndim") and getattr(usd_amount, "ndim", 0) == 0):
        return float(rounded)
    return rounded


def fmt_inr(usd_amount) -> str:
    """Format a USD amount as an INR currency string, e.g. ₹41,500. Scalars only."""
    inr = to_inr(usd_amount)
    return f"{CURRENCY_SYMBOL}{inr:,.0f}"


# ── Data directory resolution ──────────────────────────────────────────────
# Looks in the project root (one level up from /pages) and in the current
# working directory, so it works whether Streamlit is launched from
# voyage_analytics/ or from pages/.
def _candidate_paths(filename: str) -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        filename,                                    # cwd
        os.path.join(here, filename),                 # project root (same dir as this file)
        os.path.join(here, "..", filename),
        os.path.join(here, "data", filename),          # voyage_analytics/data/
        os.path.join(here, "..", "data", filename),
        os.path.join(here, "pages", "data", filename),
    ]


def _find_csv(filename: str) -> str | None:
    for path in _candidate_paths(filename):
        if os.path.exists(path):
            return path
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame | None:
    """
    Load a CSV by filename, searching common locations.
    Returns None if not found — callers should fall back to synthetic data.
    """
    path = _find_csv(filename)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def data_source_badge(filename: str, df: pd.DataFrame | None) -> None:
    """Small inline badge showing whether real or synthetic data is in use."""
    if df is not None:
        st.markdown(
            f'<span style="background:rgba(52,211,153,0.10); color:#34d399; '
            f'border:1px solid rgba(52,211,153,0.25); border-radius:999px; '
            f'padding:0.15rem 0.65rem; font-size:0.72rem; font-family:monospace;">'
            f'● Live data · {filename} · {len(df):,} rows</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="background:rgba(245,158,11,0.10); color:#f59e0b; '
            f'border:1px solid rgba(245,158,11,0.25); border-radius:999px; '
            f'padding:0.15rem 0.65rem; font-size:0.72rem; font-family:monospace;">'
            f'● Demo data · {filename} not found, using synthetic fallback</span>',
            unsafe_allow_html=True,
        )


# ── Typed dataset loaders (Voyage Analytics schema) ─────────────────────────
# Schema (confirmed from uploaded CSVs):
#   hotels.csv  : travelCode, userCode, name, place, days, price, total, date
#   flights.csv : travelCode, userCode, from, to, flightType, price, time, distance, agency, date
#   users.csv   : code, company, name, gender, age

@st.cache_data(ttl=3600, show_spinner=False)
def load_hotels() -> pd.DataFrame | None:
    df = load_csv("hotels.csv")
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["price_inr"] = (df["price"] * USD_TO_INR).round(0)
    df["total_inr"] = (df["total"] * USD_TO_INR).round(0)
    df["city"] = df["place"].str.extract(r"^(.*?)\s*\(")[0].fillna(df["place"])
    df["state_code"] = df["place"].str.extract(r"\(([A-Z]{2})\)")[0]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_flights() -> pd.DataFrame | None:
    df = load_csv("flights.csv")
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["price_inr"] = (df["price"] * USD_TO_INR).round(0)
    df["from_city"] = df["from"].str.extract(r"^(.*?)\s*\(")[0].fillna(df["from"])
    df["to_city"] = df["to"].str.extract(r"^(.*?)\s*\(")[0].fillna(df["to"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_users() -> pd.DataFrame | None:
    df = load_csv("users.csv")
    if df is None:
        return None
    return df.copy()


@st.cache_data(ttl=3600, show_spinner=False)
def load_flights_with_users() -> pd.DataFrame | None:
    """flights joined to users on userCode/code — adds gender, age, company."""
    flights = load_flights()
    users = load_users()
    if flights is None or users is None:
        return flights
    merged = flights.merge(
        users.rename(columns={"code": "userCode"}),
        on="userCode", how="left", suffixes=("", "_user"),
    )
    return merged


@st.cache_data(ttl=3600, show_spinner=False)
def load_hotels_with_users() -> pd.DataFrame | None:
    """hotels joined to users on userCode/code — adds gender, age, company."""
    hotels = load_hotels()
    users = load_users()
    if hotels is None or users is None:
        return hotels
    merged = hotels.merge(
        users.rename(columns={"code": "userCode"}),
        on="userCode", how="left", suffixes=("", "_user"),
    )
    return merged


def unique_cities() -> list[str]:
    """All distinct city names across flights + hotels, for dropdowns."""
    cities = set()
    flights = load_flights()
    hotels = load_hotels()
    if flights is not None:
        cities.update(flights["from_city"].dropna().unique().tolist())
        cities.update(flights["to_city"].dropna().unique().tolist())
    if hotels is not None:
        cities.update(hotels["city"].dropna().unique().tolist())
    return sorted(cities) if cities else [
        "Florianopolis", "Salvador", "Natal", "Aracaju", "Recife",
        "Sao Paulo", "Campo Grande", "Rio de Janeiro", "Brasilia",
    ]


# ── Shared theme CSS ─────────────────────────────────────────────────────────
def inject_theme() -> None:
    """Call this first on every page so the dark-blue theme is consistent."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-base:      #070d1a;
            --bg-card:      #0e1a2e;
            --bg-card-hover:#111f36;
            --border:       rgba(255,255,255,0.07);
            --border-hover: rgba(14,165,233,0.35);
            --primary:      #0ea5e9;
            --primary-glow: rgba(14,165,233,0.20);
            --amber:        #f59e0b;
            --amber-glow:   rgba(245,158,11,0.15);
            --violet:       #a78bfa;
            --green:        #34d399;
            --red:          #f87171;
            --text-primary: #e2e8f0;
            --text-secondary:#94a3b8;
            --text-muted:   #475569;
            --font-sans:    'Inter', sans-serif;
            --font-mono:    'JetBrains Mono', monospace;
        }

        html, body, [class*="css"] { font-family: var(--font-sans) !important; }

        /* App + every root container Streamlit uses */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stMain"],
        section.main {
            background: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }

        [data-testid="stSidebar"] {
            background: var(--bg-card) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebarNav"] a {
            font-size: 0.875rem; font-weight: 500;
            color: var(--text-secondary) !important;
            border-radius: 8px; transition: all 0.15s ease;
            padding: 0.5rem 0.75rem;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: var(--primary-glow) !important; color: var(--primary) !important;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: var(--primary-glow) !important; color: var(--primary) !important; font-weight: 700;
        }

        [data-testid="metric-container"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            transition: border-color 0.2s ease;
        }
        [data-testid="metric-container"]:hover { border-color: var(--border-hover) !important; }
        [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }
        [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

        .stSelectbox > div > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stDateInput > div > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        .stSelectbox > div > div:focus-within,
        .stNumberInput > div > div:focus-within,
        .stTextInput > div > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-glow) !important;
        }
        .stSlider [data-testid="stSlider"] > div > div > div { background: var(--primary) !important; }

        .stButton > button, .stFormSubmitButton > button {
            background: var(--primary) !important; color: white !important;
            border: none !important; border-radius: 8px !important;
            font-weight: 600 !important; font-family: var(--font-sans) !important;
            font-size: 0.875rem !important; padding: 0.6rem 1.25rem !important;
            transition: all 0.15s ease !important; letter-spacing: 0.01em !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: #38bdf8 !important; box-shadow: 0 0 16px var(--primary-glow) !important;
            transform: translateY(-1px);
        }
        .stButton > button:active, .stFormSubmitButton > button:active { transform: translateY(0); }

        [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden; }
        .dvn-scroller { background: var(--bg-card) !important; }

        .stAlert { border-radius: 10px !important; border-left-width: 3px !important; }
        [data-testid="stInfo"]    { background: rgba(14,165,233,0.08) !important; border-color: var(--primary) !important; }
        [data-testid="stSuccess"] { background: rgba(52,211,153,0.08) !important; border-color: var(--green) !important; }
        [data-testid="stError"]   { background: rgba(248,113,113,0.08) !important; border-color: var(--red) !important; }
        [data-testid="stWarning"] { background: rgba(245,158,11,0.08) !important; border-color: var(--amber) !important; }

        .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0.25rem; }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important; color: var(--text-secondary) !important;
            font-weight: 500 !important; border-radius: 8px 8px 0 0 !important;
            padding: 0.5rem 1rem !important; font-size: 0.875rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important;
            background: var(--primary-glow) !important;
        }

        .streamlit-expanderHeader {
            background: var(--bg-card) !important; border: 1px solid var(--border) !important;
            border-radius: 10px !important; color: var(--text-secondary) !important;
        }
        [data-testid="stExpander"] { background: var(--bg-card) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; }

        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

        .hotel-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 14px; padding: 1.1rem 1.25rem; margin-bottom: 0.85rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .hotel-card:hover { border-color: var(--border-hover); box-shadow: 0 0 20px var(--primary-glow); }
        .match-badge {
            display: inline-block; background: var(--amber-glow); color: var(--amber);
            border: 1px solid rgba(245,158,11,0.30); border-radius: 999px;
            padding: 0.1rem 0.6rem; font-size: 0.75rem; font-weight: 700; font-family: var(--font-mono);
        }
        .star-badge {
            display: inline-block; background: rgba(52,211,153,0.10); color: var(--green);
            border: 1px solid rgba(52,211,153,0.25); border-radius: 999px;
            padding: 0.1rem 0.6rem; font-size: 0.75rem; font-family: var(--font-mono);
        }
        .rank-badge {
            display: inline-block; background: var(--bg-base); color: var(--text-muted);
            border: 1px solid var(--border); border-radius: 999px;
            padding: 0.05rem 0.55rem; font-size: 0.72rem; font-family: var(--font-mono);
        }

        .result-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 14px; padding: 1.5rem; text-align: center; margin-bottom: 1rem;
        }
        .result-value { font-size: 2.8rem; font-weight: 800; color: var(--text-primary); line-height: 1.1; font-family: var(--font-mono); }
        .result-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }

        .page-header { margin-bottom: 0.25rem; }
        .page-title { font-size: 1.75rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
        .page-subtitle { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.15rem; }

        .nav-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 16px 16px 0 0; border-bottom: none;
            padding: 1.5rem; height: 100%; transition: all 0.2s ease;
        }
        .nav-card:hover { border-color: var(--primary); box-shadow: 0 0 28px var(--primary-glow); }
        .nav-card-icon { font-size: 2rem; margin-bottom: 0.75rem; }
        .nav-card-title { font-size: 1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.35rem; }
        .nav-card-desc { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; }
        .nav-card-tag {
            display: inline-block; background: var(--primary-glow); color: var(--primary);
            border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.7rem;
            font-weight: 600; font-family: var(--font-mono); margin-top: 0.75rem;
        }

        /* ─── Page link (clickable nav card CTA) ─────────────────── */
        [data-testid="stPageLink"] {
            margin-top: -0.5rem;
        }
        [data-testid="stPageLink"] a {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 16px 16px !important;
            padding: 0.65rem 1rem !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            text-decoration: none !important;
            transition: all 0.15s ease !important;
            display: flex !important;
            justify-content: center !important;
        }
        [data-testid="stPageLink"] a:hover {
            background: var(--primary-glow) !important;
            color: #38bdf8 !important;
            border-color: var(--border-hover) !important;
        }
        [data-testid="stPageLink"] a p {
            color: inherit !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 99px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
        </style>
        """,
        unsafe_allow_html=True,
    )