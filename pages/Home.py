"""
pages/home.py — Voyage Analytics landing page.
Rendered as the default page by st.navigation() in app.py.
"""

import streamlit as st
from datetime import datetime
from theme import inject_theme, load_hotels, load_flights, load_users, find_page, render_api_debug_panel

inject_theme()

# ── Session state bootstrap ────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "guest_001"
if "hotel_results" not in st.session_state:
    st.session_state["hotel_results"] = None
if "last_flight_prediction" not in st.session_state:
    st.session_state["last_flight_prediction"] = None
if "last_gender_prediction" not in st.session_state:
    st.session_state["last_gender_prediction"] = None
if "preferred_theme" not in st.session_state:
    st.session_state["preferred_theme"] = "dark"

# ── Sidebar user identity ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(
        f"""
        <div style="padding: 0 0.5rem;">
            <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:0.4rem;">Active Session</div>
            <div style="font-size:0.85rem; color:var(--text-secondary); font-family:var(--font-mono);
                        background:var(--bg-base); padding:0.4rem 0.7rem; border-radius:8px;
                        border:1px solid var(--border);">
                👤 {st.session_state['user_id']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    new_uid = st.text_input("Change User ID", value=st.session_state["user_id"], label_visibility="collapsed")
    if new_uid != st.session_state["user_id"]:
        st.session_state["user_id"] = new_uid
        st.rerun()
    st.markdown("---")
    st.markdown(
        """
        <div style="padding:0 0.5rem;">
            <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:0.6rem;">System Status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_s1, col_s2 = st.columns(2)
    col_s1.markdown('<div style="font-size:0.72rem; color:#34d399;">● Models Ready</div>', unsafe_allow_html=True)
    col_s2.markdown('<div style="font-size:0.72rem; color:#f59e0b;">● API Mocked</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="padding: 0 0.5rem; margin-top:0.5rem;">
            <div style="font-size:0.7rem; color:var(--text-muted);">
                {datetime.now().strftime("%d %b %Y · %H:%M")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API Testing Dashboard — exact pattern from Member 3's spec ─────────
    render_api_debug_panel()

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 3rem 0 2rem; text-align: center;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🌍</div>
        <div style="font-size: 2.8rem; font-weight: 800; color: #e2e8f0;
                    letter-spacing: -0.03em; line-height: 1.15;">
            Voyage Analytics
        </div>
        <div style="font-size: 1rem; color: #64748b; margin-top: 0.5rem; font-weight: 400;">
            MLOps Capstone &nbsp;·&nbsp; Recommender &amp; UI Engineer Track &nbsp;·&nbsp; June 2026
        </div>
        <div style="margin-top: 1.25rem; display: inline-block;
                    background: rgba(14,165,233,0.10); color: #0ea5e9;
                    border: 1px solid rgba(14,165,233,0.25); border-radius: 999px;
                    padding: 0.35rem 1rem; font-size: 0.78rem; font-weight: 600;
                    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;">
            ● LIVE · 4 ML Models Active
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Navigation cards (directly clickable — no sidebar needed) ──────────────
st.markdown(
    '<div style="text-align:center; font-size:0.8rem; color:#475569; '
    'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.25rem;">'
    'Select a Module</div>',
    unsafe_allow_html=True,
)

cards = [
    {
        "icon": "📊",
        "title": "Analytics Dashboard",
        "desc": "Interactive Plotly visualisations — flight price trends, top destinations, demographic distributions, airline performance.",
        "tag": "Plotly · Altair",
        "page": find_page("dashboard"),
        "label": "Open Dashboard",
    },
    {
        "icon": "✈️",
        "title": "Flight Price Predictor",
        "desc": "Gradient Boosting regression model estimating ticket prices from route, airline, class, and date inputs.",
        "tag": "GBM · Sklearn",
        "page": find_page("flight"),
        "label": "Open Flight Predictor",
    },
    {
        "icon": "👤",
        "title": "Gender Predictor",
        "desc": "Random Forest classifier predicting traveller demographics from behavioural travel patterns.",
        "tag": "Random Forest",
        "page": find_page("gender"),
        "label": "Open Gender Predictor",
    },
    {
        "icon": "🏨",
        "title": "Hotel Recommender",
        "desc": "Content-based filtering with TF-IDF + cosine similarity. Match your preferences to top hotel properties.",
        "tag": "TF-IDF · Cosine Sim",
        "page": find_page("hotel"),
        "label": "Open Hotel Recommender",
    },
]

cols = st.columns(4, gap="medium")
for col, card in zip(cols, cards):
    with col:
        st.markdown(
            f"""
            <div class="nav-card">
                <div class="nav-card-icon">{card['icon']}</div>
                <div class="nav-card-title">{card['title']}</div>
                <div class="nav-card-desc">{card['desc']}</div>
                <div class="nav-card-tag">{card['tag']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(card["page"], label=f"{card['label']} →", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Quick stats bar — live numbers from your uploaded datasets ─────────────
st.markdown(
    '<div style="font-size:0.75rem; color:#475569; text-transform:uppercase; '
    'letter-spacing:0.08em; margin-bottom:0.75rem;">Platform Overview</div>',
    unsafe_allow_html=True,
)

hotels_df = load_hotels()
flights_df = load_flights()
users_df = load_users()

n_hotels = hotels_df["name"].nunique() if hotels_df is not None else "—"
n_records = (len(hotels_df) if hotels_df is not None else 0) + \
            (len(flights_df) if flights_df is not None else 0) + \
            (len(users_df) if users_df is not None else 0)
n_records_str = f"{n_records:,}" if n_records else "—"
data_status = "Live" if (hotels_df is not None or flights_df is not None) else "Synthetic"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("ML Models",        "4",            "Active")
k2.metric("Hotels Indexed",   str(n_hotels),  data_status)
k3.metric("Training Records", n_records_str,  "Combined")
k4.metric("API Endpoints",    "4",            "Mocked")
k5.metric("Avg Accuracy",     "~95%",         "Validated")
k6.metric("MLflow Runs",      "Tracked",      "Logged")

st.divider()

# ── Tech stack ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:0.7rem; color:#334155; text-transform:uppercase;
                    letter-spacing:0.12em; margin-bottom:0.75rem;">Built With</div>
        <div style="display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:center;">
    """,
    unsafe_allow_html=True,
)

stack = ["Streamlit", "Scikit-learn", "Plotly", "MLflow", "Pandas", "NumPy", "Flask API (backend)"]
tags_html = " ".join(
    f'<span style="background:#0e1a2e; color:#475569; border:1px solid rgba(255,255,255,0.06); '
    f'border-radius:6px; padding:0.2rem 0.65rem; font-size:0.72rem; font-family:monospace;">{t}</span>'
    for t in stack
)
st.markdown(tags_html + "</div></div>", unsafe_allow_html=True)