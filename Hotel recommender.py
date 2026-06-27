"""
pages/hotel_recommender.py — Hotel Recommendations
Calls Member 3's real backend: POST http://localhost:8000/recommend_hotels

Actual contract (from test_api.py):
  Request:  {destination, budget_max, amenities, style, min_stars, min_review}
  Response: {recommendations: [{hotel_name, price_per_night, rating, match_score}], ...}

The new keys match your recommender.py user_data dict exactly — so the local
TF-IDF fallback and the live API speak the same language with no translation needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from recommender import HotelRecommender, RecommenderConfig
from theme import (
    inject_theme, check_backend_status, call_recommend_hotels,
    API_BASE_URL, fmt_inr, to_inr, CURRENCY_SYMBOL,
)

inject_theme()

CHART_BG, PRIMARY, AMBER, GREEN, VIOLET = "#0e1a2e", "#0ea5e9", "#f59e0b", "#34d399", "#a78bfa"
FONT_COLOR, GRID_COLOR = "#94a3b8", "rgba(255,255,255,0.05)"

DESTINATIONS  = ["Any", "Florianopolis", "Salvador", "Natal", "Aracaju",
                 "Recife", "Sao Paulo", "Campo Grande", "Rio de Janeiro", "Brasilia"]
STYLES        = ["Any", "luxury", "boutique", "business", "budget / backpacker"]
AMENITY_OPTS  = ["pool", "spa", "wifi", "fine_dining", "gym", "bar",
                 "beach_access", "business_center", "kids_club", "butler_service"]


@st.cache_resource(show_spinner="Loading hotel recommender…")
def get_recommender() -> HotelRecommender:
    rec = HotelRecommender(RecommenderConfig(text_weight=0.60, num_weight=0.40))
    rec.fit()
    return rec


def local_recommend(destination, budget_max, amenities, style, min_stars, min_review) -> dict:
    rec = get_recommender()
    user_data = {
        "destination": "" if destination == "Any" else destination,
        "budget_max":  budget_max,
        "amenities":   amenities,
        "style":       "" if style == "Any" else style,
        "min_stars":   min_stars,
        "min_review":  min_review,
    }
    results = rec.recommend(user_data, top_n=5)

    # Check if destination exists in dataset
    available = set(rec.hotels_["location"].str.lower())
    dest_found = destination.lower() in available or destination == "Any"

    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            "hotel_name":      row["name"],
            "location":        row.get("location", destination),
            "price_per_night": round(float(row["avg_price_per_night"]), 2),
            "price_per_night_inr": float(row.get("avg_price_per_night_inr",
                                                  to_inr(row["avg_price_per_night"]))),
            "rating":          round(float(row["review_score"]) / 2, 1),
            "match_score":     round(float(row["match_score"]) / 100, 4),
        })

    note = None if dest_found else (
        f"'{destination}' not in your hotel dataset — showing best matches across all cities."
    )
    return {
        "status": "success", "recommendations": recommendations,
        "model_version": "local_tfidf_cosine_v1", "note": note,
    }


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-header">'
    '<div class="page-title">🏨 Hotel Recommender</div>'
    f'<div class="page-subtitle">Backend: POST {API_BASE_URL}/recommend_hotels</div>'
    '</div>', unsafe_allow_html=True,
)

with st.expander("🔧 Developer Tools", expanded=False):
    if st.button("🔍 Check Backend Health"):
        check_backend_status(show_message=True)
    st.code(f"POST {API_BASE_URL}/recommend_hotels\n"
            f"Payload: destination, budget_max, amenities, style, min_stars, min_review")

st.divider()

rec = get_recommender()
form_col, result_col = st.columns([1, 1], gap="large")

with form_col:
    with st.form("hotel_form"):
        st.markdown("### Your Preferences")

        c1, c2 = st.columns(2)
        destination = c1.selectbox("Destination", DESTINATIONS)
        style       = c2.selectbox("Hotel Style", STYLES)

        max_possible_inr = int(to_inr(rec.hotels_["avg_price_per_night"].max())) + 5000
        c3, c4 = st.columns(2)
        budget_max_inr = c3.number_input(
            f"Max Budget/Night ({CURRENCY_SYMBOL})",
            min_value=1000, max_value=50000,
            value=min(25000, max_possible_inr), step=500,
        )
        min_stars = c4.selectbox("Min Star Rating", [2, 3, 4, 5], index=1)

        c5, c6 = st.columns(2)
        min_review = c5.slider("Min Review Score", 5.0, 10.0, 7.5, step=0.1)
        top_n = c6.slider("Results", 1, min(10, len(rec.hotels_)), min(5, len(rec.hotels_)))

        amenities = st.multiselect(
            "Must-have Amenities",
            [a.replace("_", " ").title() for a in AMENITY_OPTS],
            default=["Wifi", "Pool"],
        )

        submitted = st.form_submit_button("🎯 Find Hotels", type="primary", use_container_width=True)

with result_col:
    st.markdown("### Recommendations")

    if submitted:
        budget_max_usd = budget_max_inr / 83.0
        amenities_raw  = [a.lower().replace(" ", "_") for a in amenities]
        style_val      = "" if style == "Any" else style

        with st.spinner("🤖 Finding perfect hotels…"):
            result = call_recommend_hotels(
                "" if destination == "Any" else destination,
                budget_max_usd, amenities_raw, style_val,
                min_stars, min_review,
                show_errors=False,
            )
            used_api = result is not None
            if result is None:
                result = local_recommend(
                    destination, budget_max_usd,
                    amenities_raw, style_val, min_stars, min_review,
                )

        if used_api:
            st.caption("✅ Recommendations served by backend API.")
        else:
            st.caption("⚠️ Backend unreachable — showing local recommender (your real hotels.csv model).")

        if result.get("note"):
            st.warning(result["note"], icon="ℹ️")

        hotels = result.get("recommendations", [])

        if hotels:
            st.success(f"🏨 Found **{len(hotels)}** hotels")
            for i, hotel in enumerate(hotels, 1):
                price_inr = hotel.get("price_per_night_inr", to_inr(hotel["price_per_night"]))
                with st.expander(
                    f"#{i} {hotel['hotel_name']} — {CURRENCY_SYMBOL}{price_inr:,.0f}/night",
                    expanded=(i == 1),
                ):
                    h1, h2, h3 = st.columns(3)
                    h1.metric(f"💰 Price/Night", f"{CURRENCY_SYMBOL}{price_inr:,.0f}")
                    h2.metric("⭐ Rating", f"{hotel['rating']}/5.0")
                    h3.metric("🎯 Match", f"{hotel['match_score']:.0%}")

                    st.progress(min(1.0, max(0.0, hotel["match_score"])))

                    budget_used = (hotel["price_per_night"] / budget_max_usd) * 100
                    if budget_used <= 80:
                        st.success(f"💚 Within budget ({budget_used:.0f}% of max)")
                    else:
                        st.warning(f"💛 Close to budget ({budget_used:.0f}% of max)")

            with st.expander("Raw API payload sent"):
                st.json({
                    "destination": "" if destination == "Any" else destination,
                    "budget_max": round(budget_max_usd, 2),
                    "amenities": amenities_raw,
                    "style": style_val,
                    "min_stars": min_stars,
                    "min_review": min_review,
                })

            st.session_state["hotel_results"] = pd.DataFrame(hotels)
        else:
            st.info("🔍 No hotels matched. Try relaxing budget, stars, or review score.", icon="🏨")
    else:
        st.info("Set preferences and click **Find Hotels**.", icon="🏨")
        preview = rec.hotels_[["name", "location", "avg_price_per_night", "star_rating", "review_score"]].copy()
        preview["Price/Night"] = preview["avg_price_per_night"].apply(lambda p: fmt_inr(p))
        st.dataframe(
            preview[["name", "location", "Price/Night", "star_rating", "review_score"]]
            .rename(columns={"name": "Hotel", "location": "City",
                              "star_rating": "Stars", "review_score": "Review"}),
            use_container_width=True, hide_index=True,
        )

# ── Analytics ────────────────────────────────────────────────────────────────
if isinstance(st.session_state.get("hotel_results"), pd.DataFrame) and \
        not st.session_state["hotel_results"].empty:
    df = st.session_state["hotel_results"]
    st.divider()
    st.markdown("### Result Analytics")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df.sort_values("match_score"),
                     x="match_score", y="hotel_name", orientation="h",
                     color="match_score", color_continuous_scale=["#162032", AMBER],
                     text=df["match_score"].apply(lambda x: f"{x:.0%}"),
                     labels={"match_score": "Match", "hotel_name": "Hotel"},
                     title="Match Scores")
        fig.update_traces(textposition="outside", textfont_color=FONT_COLOR)
        fig.update_layout(template="plotly_dark", plot_bgcolor=CHART_BG,
                          paper_bgcolor=CHART_BG, font=dict(color=FONT_COLOR, size=12),
                          margin=dict(l=10, r=10, t=40, b=10),
                          coloraxis_showscale=False, title_font_size=13)
        fig.update_xaxes(gridcolor=GRID_COLOR, tickformat=".0%")
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(df, x="price_per_night", y="rating",
                          size="match_score", color="hotel_name",
                          title="Price vs Rating",
                          labels={"price_per_night": "Price/Night ($)", "rating": "Rating"},
                          size_max=38)
        fig2.update_layout(template="plotly_dark", plot_bgcolor=CHART_BG,
                           paper_bgcolor=CHART_BG, font=dict(color=FONT_COLOR, size=12),
                           margin=dict(l=10, r=10, t=40, b=10),
                           showlegend=False, title_font_size=13)
        fig2.update_xaxes(gridcolor=GRID_COLOR)
        fig2.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)