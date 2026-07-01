"""
Page 1 — Analytics Dashboard
Plotly-powered interactive visualisations built directly from
hotels.csv / flights.csv / users.csv (Voyage Analytics dataset).

NOTE: st.set_page_config is NOT called here — it's set once in app.py.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from theme import (
    inject_theme, fmt_inr, CURRENCY_SYMBOL,
    load_hotels, load_flights, load_users,
    load_flights_with_users, load_hotels_with_users,
    data_source_badge,
)

inject_theme()

# ── Palette ──────────────────────────────────────────────────────────────────
CHART_BG, PAPER_BG = "#070d1a", "#0e1a2e"
GRID_COLOR  = "rgba(255,255,255,0.05)"
FONT_COLOR  = "#94a3b8"
PRIMARY, AMBER, VIOLET, GREEN, RED = "#0ea5e9", "#f59e0b", "#a78bfa", "#34d399", "#f87171"

PLOTLY_LAYOUT = dict(
    template="plotly_dark", plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_COLOR, size=12, family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
)


def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**{**PLOTLY_LAYOUT, **kwargs})
    fig.update_xaxes(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    return fig


# ── Page header ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<div class="page-title">📊 Analytics Dashboard</div>'
    '<div class="page-subtitle">Travel intelligence overview · built from your uploaded datasets</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Load real data ───────────────────────────────────────────────────────────
hotels_df  = load_hotels()
flights_df = load_flights()
users_df   = load_users()

c_badge1, c_badge2, c_badge3 = st.columns(3)
with c_badge1: data_source_badge("hotels.csv", hotels_df)
with c_badge2: data_source_badge("flights.csv", flights_df)
with c_badge3: data_source_badge("users.csv", users_df)

if hotels_df is None and flights_df is None:
    st.warning(
        "No CSV files found yet. Place `hotels.csv`, `flights.csv`, and `users.csv` "
        "inside the `data/` folder next to `app.py` (or the project root) and reload this page.",
        icon="⚠️",
    )
    st.stop()

st.divider()

# ── Sidebar filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    if flights_df is not None:
        year_options = sorted(flights_df["date"].dt.year.dropna().unique().tolist())
    elif hotels_df is not None:
        year_options = sorted(hotels_df["date"].dt.year.dropna().unique().tolist())
    else:
        year_options = []

    selected_years = st.multiselect("Year", year_options, default=year_options)

    if flights_df is not None:
        class_options = sorted(flights_df["flightType"].unique().tolist())
        class_filter = st.multiselect("Flight Class", class_options, default=class_options)
    else:
        class_filter = []

    st.markdown("---")
    show_raw = st.toggle("Show raw data tables", value=False)

st.divider()

# ── KPI row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

if flights_df is not None:
    f_filtered = flights_df[flights_df["date"].dt.year.isin(selected_years)] if selected_years else flights_df
    total_bookings = len(f_filtered) + (len(hotels_df) if hotels_df is not None else 0)
    avg_flight_price = f_filtered["price_inr"].mean()
    top_dest = f_filtered["to_city"].value_counts().idxmax()
    top_dest_count = f_filtered["to_city"].value_counts().max()
else:
    f_filtered = pd.DataFrame()
    total_bookings = len(hotels_df) if hotels_df is not None else 0
    avg_flight_price = 0
    top_dest, top_dest_count = "N/A", 0

if hotels_df is not None:
    avg_hotel_price = hotels_df["price_inr"].mean()
    total_hotel_revenue = hotels_df["total_inr"].sum()
else:
    avg_hotel_price = 0
    total_hotel_revenue = 0

k1.metric("Total Bookings", f"{total_bookings:,}")
k2.metric("Avg Flight Price", fmt_inr(avg_flight_price / 83.0) if avg_flight_price else "—")
k3.metric("Top Destination", top_dest, f"{top_dest_count:,} flights")
k4.metric("Avg Hotel Price/Night", fmt_inr(avg_hotel_price / 83.0) if avg_hotel_price else "—")
k5.metric("Total Hotel Revenue", f"{CURRENCY_SYMBOL}{total_hotel_revenue/1e7:,.2f} Cr" if total_hotel_revenue else "—")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_flights, tab_hotels, tab_destinations, tab_demographics = st.tabs([
    "✈️  Flights", "🏨  Hotels", "🌏  Destinations", "👥  Demographics"
])

# ── TAB 1 · FLIGHTS ──────────────────────────────────────────────────────────
with tab_flights:
    if flights_df is None:
        st.info("Upload flights.csv to see this section.", icon="✈️")
    else:
        fdf = flights_df.copy()
        if selected_years:
            fdf = fdf[fdf["date"].dt.year.isin(selected_years)]
        if class_filter:
            fdf = fdf[fdf["flightType"].isin(class_filter)]

        col_a, col_b = st.columns([3, 2])

        with col_a:
            monthly = fdf.copy()
            monthly["Month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
            price_trend = (
                monthly.groupby(["Month", "flightType"])["price_inr"]
                .mean().reset_index()
            )
            fig = px.line(
                price_trend, x="Month", y="price_inr", color="flightType",
                title=f"Average Flight Price Trend ({CURRENCY_SYMBOL})",
                color_discrete_map={"economic": PRIMARY, "premium": AMBER, "firstClass": VIOLET},
                markers=True,
            )
            apply_layout(fig, title_font_size=14, yaxis_title=f"Avg Price ({CURRENCY_SYMBOL})")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            monthly_bookings = monthly.groupby("Month").size().reset_index(name="Bookings")
            fig2 = go.Figure()
            fig2.add_bar(
                x=monthly_bookings["Month"], y=monthly_bookings["Bookings"],
                marker_color=PRIMARY, opacity=0.9, marker_line_width=0,
            )
            fig2.update_layout(**PLOTLY_LAYOUT, title="Monthly Flight Bookings", title_font_size=14)
            fig2.update_xaxes(gridcolor=GRID_COLOR)
            fig2.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            agency_share = fdf["agency"].value_counts().reset_index()
            agency_share.columns = ["Agency", "Bookings"]
            fig_pie = go.Figure()
            fig_pie.add_pie(
                labels=agency_share["Agency"], values=agency_share["Bookings"],
                hole=0.55, marker_colors=[PRIMARY, AMBER, VIOLET],
                textinfo="label+percent", textfont=dict(size=11),
            )
            fig_pie.update_layout(**PLOTLY_LAYOUT, title="Bookings by Agency", title_font_size=14, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_d:
            class_price = fdf.groupby("flightType")["price_inr"].mean().reset_index()
            fig_bar = px.bar(
                class_price, x="flightType", y="price_inr",
                title=f"Avg Price by Class ({CURRENCY_SYMBOL})",
                color="flightType",
                color_discrete_map={"economic": PRIMARY, "premium": AMBER, "firstClass": VIOLET},
                text="price_inr",
            )
            fig_bar.update_traces(
                texttemplate=f"{CURRENCY_SYMBOL}%{{text:,.0f}}",
                textposition="outside", textfont_color=FONT_COLOR, marker_line_width=0,
            )
            apply_layout(fig_bar, title_font_size=14, showlegend=False, yaxis_title=f"Price ({CURRENCY_SYMBOL})")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Distance vs price
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="distance", y="price_inr", color="flightType",
            title="Distance vs Price", opacity=0.5,
            color_discrete_map={"economic": PRIMARY, "premium": AMBER, "firstClass": VIOLET},
            labels={"distance": "Distance (km)", "price_inr": f"Price ({CURRENCY_SYMBOL})"},
        )
        apply_layout(fig_scatter, title_font_size=14)
        st.plotly_chart(fig_scatter, use_container_width=True)

        if show_raw:
            with st.expander("📋 Raw Flight Data (sample)"):
                st.dataframe(fdf.head(200), use_container_width=True, hide_index=True)

# ── TAB 2 · HOTELS ───────────────────────────────────────────────────────────
with tab_hotels:
    if hotels_df is None:
        st.info("Upload hotels.csv to see this section.", icon="🏨")
    else:
        hdf = hotels_df.copy()
        if selected_years:
            hdf = hdf[hdf["date"].dt.year.isin(selected_years)]

        col_e, col_f = st.columns([3, 2])

        with col_e:
            hmonthly = hdf.copy()
            hmonthly["Month"] = hmonthly["date"].dt.to_period("M").dt.to_timestamp()
            price_trend_h = hmonthly.groupby("Month")["price_inr"].mean().reset_index()
            fig_h1 = px.area(
                price_trend_h, x="Month", y="price_inr",
                title=f"Average Hotel Price Trend ({CURRENCY_SYMBOL}/night)",
                color_discrete_sequence=[PRIMARY],
            )
            fig_h1.update_traces(line_width=2.5)
            apply_layout(fig_h1, title_font_size=14, yaxis_title=f"Avg Price ({CURRENCY_SYMBOL})")
            st.plotly_chart(fig_h1, use_container_width=True)

        with col_f:
            hotel_pop = hdf["name"].value_counts().reset_index()
            hotel_pop.columns = ["Hotel", "Bookings"]
            fig_h2 = px.bar(
                hotel_pop.sort_values("Bookings"), x="Bookings", y="Hotel", orientation="h",
                title="Bookings by Hotel", color="Bookings",
                color_continuous_scale=["#0e1a2e", PRIMARY],
            )
            apply_layout(fig_h2, title_font_size=14, coloraxis_showscale=False)
            st.plotly_chart(fig_h2, use_container_width=True)

        col_g, col_h = st.columns(2)
        with col_g:
            city_price = hdf.groupby("city")["price_inr"].mean().sort_values(ascending=False).reset_index()
            fig_h3 = px.bar(
                city_price, x="price_inr", y="city", orientation="h",
                title=f"Avg Hotel Price by City ({CURRENCY_SYMBOL}/night)",
                color="price_inr", color_continuous_scale=["#0e1a2e", AMBER],
                text="price_inr",
            )
            fig_h3.update_traces(
                texttemplate=f"{CURRENCY_SYMBOL}%{{text:,.0f}}",
                textposition="outside", textfont_color=FONT_COLOR, marker_line_width=0,
            )
            apply_layout(fig_h3, title_font_size=14, coloraxis_showscale=False)
            st.plotly_chart(fig_h3, use_container_width=True)

        with col_h:
            days_dist = hdf["days"].value_counts().sort_index().reset_index()
            days_dist.columns = ["Nights Stayed", "Count"]
            fig_h4 = px.bar(
                days_dist, x="Nights Stayed", y="Count",
                title="Stay Duration Distribution",
                color_discrete_sequence=[VIOLET],
            )
            apply_layout(fig_h4, title_font_size=14)
            st.plotly_chart(fig_h4, use_container_width=True)

        if show_raw:
            with st.expander("📋 Raw Hotel Data (sample)"):
                st.dataframe(hdf.head(200), use_container_width=True, hide_index=True)

# ── TAB 3 · DESTINATIONS ─────────────────────────────────────────────────────
with tab_destinations:
    if flights_df is None:
        st.info("Upload flights.csv to see this section.", icon="🌏")
    else:
        dest_counts = flights_df["to_city"].value_counts().reset_index()
        dest_counts.columns = ["Destination", "Bookings"]
        dest_price = flights_df.groupby("to_city")["price_inr"].mean().reset_index()
        dest_price.columns = ["Destination", "Avg Price"]
        dest_merged = dest_counts.merge(dest_price, on="Destination")

        col_i, col_j = st.columns([2, 1])

        with col_i:
            fig3 = px.bar(
                dest_merged.sort_values("Bookings"),
                x="Bookings", y="Destination", orientation="h",
                title="Top Destinations by Bookings",
                color="Avg Price", color_continuous_scale=["#0e1a2e", PRIMARY, AMBER],
                text="Bookings",
            )
            fig3.update_traces(textposition="outside", textfont_color=FONT_COLOR, marker_line_width=0)
            apply_layout(fig3, title_font_size=14, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_j:
            fig4 = px.scatter(
                dest_merged, x="Avg Price", y="Bookings",
                size="Bookings", color="Destination",
                title="Price vs Demand", hover_name="Destination",
                size_max=38,
            )
            apply_layout(fig4, title_font_size=14, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        fig_tree = px.treemap(
            dest_merged, path=["Destination"], values="Bookings", color="Avg Price",
            color_continuous_scale=["#0e1a2e", PRIMARY],
            title="Destination Treemap — size: bookings · colour: avg price",
        )
        fig_tree.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        st.plotly_chart(fig_tree, use_container_width=True)

        # Route popularity matrix
        st.markdown("#### Top Routes")
        route_counts = (
            flights_df.groupby(["from_city", "to_city"]).size()
            .reset_index(name="Bookings")
            .sort_values("Bookings", ascending=False)
            .head(10)
        )
        route_counts["Route"] = route_counts["from_city"] + " → " + route_counts["to_city"]
        fig_routes = px.bar(
            route_counts, x="Bookings", y="Route", orientation="h",
            color="Bookings", color_continuous_scale=["#0e1a2e", GREEN],
        )
        apply_layout(fig_routes, title_font_size=14, coloraxis_showscale=False)
        st.plotly_chart(fig_routes, use_container_width=True)

# ── TAB 4 · DEMOGRAPHICS ─────────────────────────────────────────────────────
with tab_demographics:
    if users_df is None:
        st.info("Upload users.csv to see this section.", icon="👥")
    else:
        udf = users_df.copy()

        col_k, col_l = st.columns(2)

        with col_k:
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
            udf["age_group"] = pd.cut(udf["age"], bins=age_bins, labels=age_labels, right=False)

            demo_grouped = udf.groupby(["age_group", "gender"], observed=True).size().reset_index(name="Count")
            fig5 = px.bar(
                demo_grouped, x="age_group", y="Count", color="gender",
                title="Traveller Demographics by Age Group",
                color_discrete_map={"male": PRIMARY, "female": VIOLET, "none": GREEN},
                barmode="stack",
            )
            apply_layout(fig5, title_font_size=14, xaxis_title="Age Group")
            st.plotly_chart(fig5, use_container_width=True)

        with col_l:
            gender_counts = udf["gender"].value_counts()
            fig_donut = go.Figure()
            fig_donut.add_pie(
                labels=gender_counts.index, values=gender_counts.values,
                hole=0.6, marker_colors=[PRIMARY, VIOLET, GREEN],
                textinfo="label+percent", textfont=dict(size=12),
            )
            fig_donut.update_layout(
                **PLOTLY_LAYOUT, title="Overall Gender Split", title_font_size=14, showlegend=False,
                annotations=[dict(
                    text=f"{len(udf):,}<br><span style='font-size:10px'>users</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#e2e8f0"),
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        col_m, col_n = st.columns(2)
        with col_m:
            company_counts = udf["company"].value_counts().reset_index()
            company_counts.columns = ["Company", "Users"]
            fig6 = px.bar(
                company_counts, x="Company", y="Users",
                title="Users by Company", color_discrete_sequence=[AMBER],
            )
            apply_layout(fig6, title_font_size=14)
            st.plotly_chart(fig6, use_container_width=True)

        with col_n:
            fig7 = px.histogram(
                udf, x="age", nbins=20, title="Age Distribution",
                color_discrete_sequence=[PRIMARY],
            )
            apply_layout(fig7, title_font_size=14, xaxis_title="Age")
            st.plotly_chart(fig7, use_container_width=True)

        # Spending by gender (joined data)
        flights_users = load_flights_with_users()
        if flights_users is not None and "gender" in flights_users.columns:
            st.markdown("#### Avg Spend by Gender")
            spend_gender = flights_users.groupby("gender")["price_inr"].mean().reset_index()
            fig8 = px.bar(
                spend_gender, x="gender", y="price_inr",
                title=f"Avg Flight Spend by Gender ({CURRENCY_SYMBOL})",
                color="gender",
                color_discrete_map={"male": PRIMARY, "female": VIOLET, "none": GREEN},
                text="price_inr",
            )
            fig8.update_traces(
                texttemplate=f"{CURRENCY_SYMBOL}%{{text:,.0f}}",
                textposition="outside", textfont_color=FONT_COLOR, marker_line_width=0,
            )
            apply_layout(fig8, title_font_size=14, showlegend=False, yaxis_title=f"Avg Price ({CURRENCY_SYMBOL})")
            st.plotly_chart(fig8, use_container_width=True)

        if show_raw:
            with st.expander("📋 Raw User Data"):
                st.dataframe(udf, use_container_width=True, hide_index=True)
