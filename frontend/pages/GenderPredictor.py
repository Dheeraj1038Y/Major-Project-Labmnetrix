"""
pages/gender_predictor.py — Gender Classification
Calls Member 3's real backend: POST http://localhost:8000/classify_gender

Actual contract (from test_api.py):
  Request:  {first_name, company, name_length, first_letter, last_letter, age}
  Response: {predicted_gender, confidence, ...}

name_length, first_letter, last_letter are derived from first_name automatically —
the user only fills in first_name, company, and age.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

from theme import (
    inject_theme, check_backend_status, call_classify_gender,
    API_BASE_URL, load_users,
)

warnings.filterwarnings("ignore")
inject_theme()

CHART_BG, PRIMARY, VIOLET, GREEN = "#0e1a2e", "#0ea5e9", "#a78bfa", "#34d399"
FONT_COLOR, GRID_COLOR = "#94a3b8", "rgba(255,255,255,0.05)"

BASE_LAYOUT = dict(
    template="plotly_dark", plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
    font=dict(color=FONT_COLOR, size=12), margin=dict(l=10, r=10, t=40, b=10),
)

# Real company values from users.csv
COMPANIES_DEFAULT = ["4You", "Monsters CYA", "Wonka Company", "Acme Factory", "Umbrella LTDA"]


def get_companies():
    users = load_users()
    if users is not None and "company" in users.columns:
        return sorted(users["company"].dropna().unique().tolist())
    return COMPANIES_DEFAULT


@st.cache_resource(show_spinner="Training gender classifier…")
def build_gender_model():
    users = load_users()
    companies = get_companies()
    le_company = LabelEncoder().fit(companies)

    if users is not None and len(users) > 50:
        df = users.copy()
        df = df[df["company"].isin(companies)].copy()
        df["name_length"]  = df["name"].fillna("").apply(lambda n: len(n.split()[0]) if n else 0)
        df["first_letter"] = df["name"].fillna("").apply(lambda n: n[0].upper() if n else "A")
        df["last_letter"]  = df["name"].fillna("").apply(lambda n: n.split()[0][-1].lower() if n else "a")
        df["company_enc"]  = le_company.transform(df["company"])
        df["first_letter_enc"] = df["first_letter"].apply(lambda c: ord(c) - ord("A"))
        df["last_letter_enc"]  = df["last_letter"].apply(lambda c: ord(c) - ord("a"))

        X = df[["age", "company_enc", "name_length", "first_letter_enc", "last_letter_enc"]].fillna(0).values
        le_gender = LabelEncoder().fit(df["gender"])
        y = le_gender.transform(df["gender"])

        clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        return {"model": clf, "le_company": le_company, "le_gender": le_gender,
                "source": "real_data", "n": len(df)}

    # Synthetic fallback
    np.random.seed(11)
    N = 1000
    le_gender = LabelEncoder().fit(["male", "female", "none"])
    X = np.random.rand(N, 5) * [60, len(companies), 10, 26, 26]
    y = np.random.choice([0, 1, 2], N, p=[0.45, 0.45, 0.10])
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y)
    return {"model": clf, "le_company": le_company, "le_gender": le_gender,
            "source": "synthetic", "n": N}


def mock_classify(first_name: str, company: str, age: int) -> dict:
    b = build_gender_model()
    le_company = b["le_company"]
    le_gender  = b["le_gender"]
    clean = first_name.strip()
    company_enc = le_company.transform([company])[0] if company in le_company.classes_ else 0
    first_enc   = ord(clean[0].upper()) - ord("A") if clean else 0
    last_enc    = ord(clean[-1].lower()) - ord("a") if clean else 0
    X = np.array([[age, company_enc, len(clean), first_enc, last_enc]])
    probs     = b["model"].predict_proba(X)[0]
    classes   = le_gender.inverse_transform(b["model"].classes_)
    predicted = str(classes[int(np.argmax(probs))])
    return {
        "predicted_gender": predicted,
        "confidence": round(float(max(probs)), 4),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probs)},
        "model_version": "local_mock_v1",
    }


def prob_chart(classes, probs) -> go.Figure:
    colors = {"male": PRIMARY, "female": VIOLET, "none": GREEN}
    fig = go.Figure()
    for label, prob in zip(classes, probs):
        fig.add_bar(
            x=[prob * 100], y=[label.title()], orientation="h",
            marker_color=colors.get(label, PRIMARY),
            text=[f"{prob*100:.1f}%"], textposition="outside",
            textfont=dict(color="#e2e8f0", size=13), name=label,
        )
    fig.update_layout(**BASE_LAYOUT, showlegend=False,
                      xaxis=dict(range=[0, 115], gridcolor=GRID_COLOR),
                      yaxis=dict(gridcolor=GRID_COLOR),
                      title="Predicted Probability by Class",
                      title_font_size=13, height=200)
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-header">'
    '<div class="page-title">👤 Gender Predictor</div>'
    f'<div class="page-subtitle">Backend: POST {API_BASE_URL}/classify_gender</div>'
    '</div>', unsafe_allow_html=True,
)

with st.expander("🔧 Developer Tools", expanded=False):
    if st.button("🔍 Check Backend Health"):
        check_backend_status(show_message=True)
    st.code(f"POST {API_BASE_URL}/classify_gender\n"
            f"Payload: first_name, company, name_length, first_letter, last_letter, age\n"
            f"(name_length / first_letter / last_letter derived automatically)")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    with st.form("gender_form"):
        st.markdown("### User Profile")

        c1, c2 = st.columns(2)
        first_name = c1.text_input("First Name", value="Alex")
        user_age   = c2.number_input("Age", min_value=13, max_value=100, value=28)

        company = st.selectbox("Company", get_companies())

        st.markdown(
            '<div style="font-size:0.78rem; color:#475569; margin-top:0.25rem;">'
            '💡 Name Length, First Letter, and Last Letter are derived automatically '
            'and sent to the API.</div>',
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button("🎯 Predict Gender", type="primary", use_container_width=True)

with right:
    st.markdown("### Prediction Output")

    if submitted:
        if not first_name.strip():
            st.warning("⚠️ Please enter a valid first name!")
        else:
            clean = first_name.strip()
            derived = {
                "first_name":   clean,
                "company":      company,
                "name_length":  len(clean),
                "first_letter": clean[0].upper(),
                "last_letter":  clean[-1].lower(),
                "age":          user_age,
            }

            with st.spinner("🤖 Analyzing profile…"):
                result = call_classify_gender(clean, company, user_age, show_errors=False)
                used_api = result is not None
                if result is None:
                    result = mock_classify(clean, company, user_age)

            if used_api:
                st.caption("✅ Prediction served by backend API.")
            else:
                st.caption("⚠️ Backend unreachable — showing local model prediction.")

            gender     = result.get("predicted_gender", "Unknown")
            confidence = result.get("confidence", 0)

            emoji = {"male": "🔵", "female": "🟣", "none": "🟢"}.get(gender.lower(), "⚪")
            st.markdown(f"### {emoji} Predicted: **{gender.title()}**")
            st.metric("Confidence", f"{(confidence or 0)*100:.1f}%")

            if confidence >= 0.8:
                st.info("🎯 High confidence prediction")
            elif confidence >= 0.6:
                st.info("📊 Moderate confidence prediction")
            else:
                st.warning("❓ Low confidence prediction")

            probs_dict = result.get("probabilities", {gender.lower(): confidence})
            st.plotly_chart(
                prob_chart(list(probs_dict.keys()), list(probs_dict.values())),
                use_container_width=True,
            )

            with st.expander("Raw API payload sent"):
                st.json(derived)

            st.session_state["last_gender_prediction"] = result
    else:
        st.info("Fill in the profile and click **Predict Gender**.", icon="👤")
        m1, m2 = st.columns(2)
        m1.metric("Endpoint", "/classify_gender")
        m2.metric("Method", "POST")

# ── Model diagnostics ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### Local Mock Model — Feature Importances")
st.caption("Trained on your real users.csv — name features + age + company.")

b = build_gender_model()
features = ["Age", "Company", "Name Length", "First Letter", "Last Letter"]
imp_df = pd.DataFrame({"Feature": features, "Importance": b["model"].feature_importances_}).sort_values("Importance")
fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
             color="Importance", color_continuous_scale=["#162032", VIOLET],
             title="Feature Importances — Random Forest")
fig.update_layout(**BASE_LAYOUT, coloraxis_showscale=False, title_font_size=13)
fig.update_xaxes(gridcolor=GRID_COLOR)
fig.update_yaxes(gridcolor=GRID_COLOR)
st.plotly_chart(fig, use_container_width=True)
