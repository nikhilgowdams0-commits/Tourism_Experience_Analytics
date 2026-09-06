import os
import sys
from html import escape

import numpy as np

# Prevent Python 3.14 recursion bug in numpy/scipy
if hasattr(np, "__all__"):
    np.__all__ = [x for x in np.__all__ if x != "core"]
try:
    import numpy._core as _core
    sys.modules["numpy.core"] = _core
    sys.modules["numpy.core.multiarray"] = getattr(_core, "multiarray", _core)
    sys.modules["numpy.core._multiarray_umath"] = getattr(_core, "_multiarray_umath", _core)
    setattr(np, "core", _core)
except Exception:
    pass

import scipy
import sklearn
import sklearn.pipeline
import sklearn.ensemble
import sklearn.compose
import sklearn.preprocessing
import joblib
import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "cleaned_data"))


# ============================================================
# SAFE HTML HELPER
# Uses st.markdown instead of st.html so the app also works
# with older Streamlit versions.
# ============================================================

def html(content):
    st.markdown(content, unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODE_MAPPING = {
    1: "Business",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Solo",
}

MODE_TO_ID = {v: k for k, v in MODE_MAPPING.items()}

MODE_ICON = {
    "Business": "💼",
    "Couples": "💑",
    "Family": "👨‍👩‍👧",
    "Friends": "👥",
    "Solo": "🧭",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

BUSINESS_USE_CASES = {
    "Business": {
        "title": "Corporate & Executive Travel",
        "marketing": "Use weekday business-travel promotions and convenient transport packages.",
        "amenities": "Prioritize reliable connectivity, quiet work areas and flexible check-in options.",
        "incentive": "Offer corporate loyalty benefits and simple expense documentation.",
    },
    "Couples": {
        "title": "Couples & Romantic Travel",
        "marketing": "Promote scenic attractions, relaxed itineraries and couple-oriented experiences.",
        "amenities": "Highlight scenic rooms, flexible checkout and attraction bundles.",
        "incentive": "Offer experience vouchers or couple-oriented package upgrades.",
    },
    "Family": {
        "title": "Family Holiday Bundles",
        "marketing": "Promote attractions with family-friendly activities and bundled entry options.",
        "amenities": "Highlight family rooms, child-friendly services and convenient transport.",
        "incentive": "Use family bundle pricing and multi-ticket promotions.",
    },
    "Friends": {
        "title": "Group Adventure Packages",
        "marketing": "Promote social, outdoor and activity-oriented destination packages.",
        "amenities": "Highlight group transport, shared accommodation and activity rentals.",
        "incentive": "Use group discounts and multi-person booking offers.",
    },
    "Solo": {
        "title": "Independent Discovery",
        "marketing": "Promote flexible itineraries, cultural attractions and independent exploration.",
        "amenities": "Highlight convenient transit, secure storage and flexible booking.",
        "incentive": "Offer flexible single-person packages and itinerary add-ons.",
    },
}


# ============================================================
# PREMIUM CSS
# ============================================================

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070b14;
    --surface: rgba(18, 27, 46, 0.82);
    --surface2: rgba(25, 36, 60, 0.72);
    --border: rgba(255,255,255,0.09);
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #6366f1;
    --cyan: #06b6d4;
    --green: #10b981;
}

.stApp {
    background:
        radial-gradient(circle at 5% 0%, rgba(99,102,241,.14), transparent 35%),
        radial-gradient(circle at 95% 5%, rgba(6,182,212,.10), transparent 32%),
        #070b14 !important;
    color: var(--text);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.block-container {
    max-width: 1450px !important;
    padding: 25px 42px 80px !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:18px 24px;
    border:1px solid var(--border);
    border-radius:22px;
    background:rgba(15,23,42,.82);
    backdrop-filter:blur(18px);
    margin-bottom:18px;
}

.brand {
    display:flex;
    align-items:center;
    gap:14px;
}

.logo {
    width:48px;
    height:48px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:14px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    font-size:24px;
}

.brand-title {
    font-size:20px;
    font-weight:800;
    color:#fff;
}

.brand-subtitle {
    color:#94a3b8;
    font-size:12px;
    margin-top:3px;
}

.status {
    padding:9px 15px;
    border-radius:20px;
    border:1px solid rgba(16,185,129,.25);
    background:rgba(16,185,129,.08);
    color:#34d399;
    font-size:11px;
    font-weight:700;
}

.nav-button button {
    min-height:52px;
    border-radius:15px !important;
    border:1px solid var(--border) !important;
    background:rgba(15,23,42,.75) !important;
    color:#cbd5e1 !important;
    font-weight:700 !important;
}

.nav-button button:hover {
    border-color:rgba(99,102,241,.55) !important;
    color:#fff !important;
    transform:translateY(-1px);
}

.nav-active button {
    background:linear-gradient(135deg,rgba(99,102,241,.42),rgba(139,92,246,.35)) !important;
    border-color:rgba(139,92,246,.65) !important;
    color:#fff !important;
}

.hero {
    padding:42px;
    border-radius:26px;
    border:1px solid var(--border);
    background:
        radial-gradient(circle at 90% 10%,rgba(99,102,241,.25),transparent 35%),
        radial-gradient(circle at 10% 90%,rgba(6,182,212,.15),transparent 35%),
        rgba(15,23,42,.78);
    margin:18px 0 24px;
}

.kicker {
    color:#a5b4fc;
    font-size:11px;
    font-weight:800;
    letter-spacing:1.4px;
    text-transform:uppercase;
    margin-bottom:9px;
}

.hero h1 {
    color:#fff;
    font-size:42px;
    line-height:1.12;
    margin:0 0 14px;
    font-weight:800;
}

.hero p {
    color:#94a3b8;
    max-width:850px;
    line-height:1.7;
    font-size:14px;
}

.card {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:20px;
    padding:23px;
    height:100%;
}

.card h3 {
    color:#fff;
    margin:0 0 8px;
    font-size:17px;
}

.card p {
    color:#94a3b8;
    line-height:1.65;
    font-size:12.5px;
}

.kpi {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:19px;
    padding:21px;
}

.kpi-label {
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:.8px;
    font-size:10px;
    font-weight:800;
}

.kpi-value {
    color:#fff;
    font-size:30px;
    font-weight:800;
    margin-top:8px;
}

.kpi-note {
    color:#94a3b8;
    font-size:11px;
    margin-top:5px;
}

.result {
    padding:25px;
    border-radius:21px;
    border:1px solid rgba(99,102,241,.35);
    background:
        radial-gradient(circle at 90% 10%,rgba(99,102,241,.22),transparent 45%),
        rgba(15,23,42,.82);
}

.result.rating {
    border-color:rgba(16,185,129,.35);
    background:
        radial-gradient(circle at 90% 10%,rgba(16,185,129,.18),transparent 45%),
        rgba(15,23,42,.82);
}

.result-label {
    color:#64748b;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:1px;
    font-weight:800;
}

.result-value {
    color:#fff;
    font-size:35px;
    font-weight:800;
    margin-top:9px;
}

.rec {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    padding:18px 20px;
    margin:10px 0;
    border-radius:17px;
    background:var(--surface);
    border:1px solid var(--border);
}

.rec-title {
    color:#fff;
    font-size:15px;
    font-weight:700;
}

.rec-meta {
    color:#94a3b8;
    font-size:11px;
    margin-top:5px;
}

.rec-score {
    color:#67e8f9;
    font-weight:800;
    font-size:21px;
}

.section-title {
    color:#fff;
    font-size:25px;
    font-weight:800;
    margin:28px 0 6px;
}

.section-text {
    color:#94a3b8;
    font-size:13px;
    margin-bottom:18px;
}

.small-note {
    color:#64748b;
    font-size:11px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
    color:#94a3b8 !important;
    font-size:11px !important;
    font-weight:700 !important;
}

.stButton button {
    border-radius:13px !important;
    font-weight:700 !important;
}

[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}

@media(max-width:900px) {
    .block-container { padding:16px !important; }
    .hero h1 { font-size:30px; }
}
</style>
""")


# ============================================================
# LOAD DATA / MODELS
# ============================================================

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "tourism_main_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "cleaned_data/tourism_main_clean.csv was not found."
        )
    return pd.read_csv(path)


@st.cache_resource
def load_model_file(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {filename}")
    return joblib.load(path)


try:
    tourism_data = load_data()
except Exception as exc:
    st.error("Could not load the cleaned tourism dataset.")
    st.code(str(exc))
    st.stop()


# Load only what the application actually needs.
# This also makes debugging model files much easier.
try:
    regression_model = load_model_file("regression_model.pkl")
    classification_model = load_model_file("classification_model.pkl")
    content_similarity = load_model_file("content_similarity.pkl")
    collaborative_similarity = load_model_file("collaborative_similarity.pkl")
    attraction_data = load_model_file("attraction_data.pkl")
    recommendation_history = load_model_file("recommendation_history.pkl")
    recommendation_config = load_model_file("recommendation_config.pkl")
except Exception as exc:
    st.error("The application started, but one or more model files could not be loaded.")
    st.warning(
        "Make sure the .pkl files were trained and saved with the same Python "
        "environment that is running this Streamlit app."
    )
    st.code(str(exc))
    st.stop()


# ============================================================
# DATA NORMALIZATION
# ============================================================

tourism_data["VisitYear"] = pd.to_numeric(
    tourism_data["VisitYear"], errors="coerce"
)
tourism_data["VisitMonth"] = pd.to_numeric(
    tourism_data["VisitMonth"], errors="coerce"
)
tourism_data["Rating"] = pd.to_numeric(
    tourism_data["Rating"], errors="coerce"
)

# Convert recommendation structures to DataFrames when needed.
if not isinstance(attraction_data, pd.DataFrame):
    attraction_data = pd.DataFrame(attraction_data)

if not isinstance(recommendation_history, pd.DataFrame):
    recommendation_history = pd.DataFrame(recommendation_history)


# ============================================================
# HELPERS
# ============================================================

def get_season(month):
    month = int(month)
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def attraction_stats(name):
    rows = tourism_data.loc[
        tourism_data["Attraction"] == name, "Rating"
    ].dropna()

    if rows.empty:
        return float(tourism_data["Rating"].mean()), 0

    return float(rows.mean()), int(rows.shape[0])


def recommendation_function(
    user_id,
    top_n=5,
    content_weight=0.2,
    collaborative_weight=0.8,
    category_filter="All Categories",
):
    history = recommendation_history[
        recommendation_history["UserId"] == user_id
    ].copy()

    if history.empty:
        return pd.DataFrame()

    visited = set(history["AttractionId"].tolist())
    user_ratings = dict(
        zip(history["AttractionId"], history["Rating"])
    )

    scores = {}

    # Similarity objects saved during notebook work are DataFrames.
    content_df = pd.DataFrame(content_similarity)
    collab_df = pd.DataFrame(collaborative_similarity)

    for attraction_id, rating in user_ratings.items():

        if attraction_id not in content_df.index:
            continue

        content_scores = content_df.loc[attraction_id]

        if attraction_id in collab_df.index:
            collaborative_scores = collab_df.loc[attraction_id]
        else:
            collaborative_scores = pd.Series(
                0.0, index=content_df.columns
            )

        for candidate_id in content_df.columns:

            if candidate_id in visited:
                continue

            content_score = float(
                content_scores.get(candidate_id, 0.0)
            )

            collaborative_score = float(
                collaborative_scores.get(candidate_id, 0.0)
            )

            score = (
                content_weight * content_score
                + collaborative_weight * collaborative_score
            ) * float(rating)

            scores[candidate_id] = (
                scores.get(candidate_id, 0.0) + score
            )

    if not scores:
        return pd.DataFrame()

    result = pd.DataFrame(
        list(scores.items()),
        columns=["AttractionId", "RecommendationScore"],
    )

    metadata_cols = [
        "AttractionId",
        "Attraction",
        "AttractionType",
        "AttractionCityId",
    ]

    result = result.merge(
        attraction_data[metadata_cols],
        on="AttractionId",
        how="left",
    )

    address_map = (
        tourism_data[
            ["AttractionId", "AttractionAddress"]
        ]
        .drop_duplicates("AttractionId")
    )

    result = result.merge(
        address_map,
        on="AttractionId",
        how="left",
    )

    if category_filter != "All Categories":
        result = result[
            result["AttractionType"] == category_filter
        ]

    return (
        result
        .sort_values(
            "RecommendationScore",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def safe_value(value, fallback="Unknown"):
    if pd.isna(value):
        return fallback
    return str(value)


# ============================================================
# TOP BAR
# ============================================================

total_interactions = len(tourism_data)
unique_users = tourism_data["UserId"].nunique()
unique_attractions = tourism_data["AttractionId"].nunique()
mean_rating = tourism_data["Rating"].mean()

html(f"""
<div class="topbar">
    <div class="brand">
        <div class="logo">🌍</div>
        <div>
            <div class="brand-title">Tourism Experience Analytics</div>
            <div class="brand-subtitle">
                Classification · Rating Prediction · Personalized Recommendation
            </div>
        </div>
    </div>
    <div class="status">● ML SYSTEM READY · {total_interactions:,} RECORDS</div>
</div>
""")


# ============================================================
# NAVIGATION
# Compatible with older Streamlit versions.
# No st.segmented_control().
# ============================================================

NAV = [
    ("🏠", "Overview"),
    ("🎯", "Prediction Studio"),
    ("⭐", "Personalized Recommender"),
    ("📊", "Tourism Analytics"),
    ("📑", "Model Evaluation"),
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

nav_cols = st.columns(len(NAV))

for col, (icon, label) in zip(nav_cols, NAV):
    with col:
        active = st.session_state["active_page"] == label
        button_text = f"{icon}  {label}"
        if st.button(
            button_text,
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["active_page"] = label
            st.rerun()

page = st.session_state["active_page"]


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    html(f"""
    <div class="hero">
        <div class="kicker">AI-POWERED TOURISM INTELLIGENCE</div>
        <h1>
            Turn visitor interactions into
            predictive and personalized travel intelligence.
        </h1>
        <p>
            An end-to-end machine learning application for attraction
            rating prediction, visitor-mode classification and
            personalized attraction recommendation.
        </p>
    </div>
    """)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        html(f"""
        <div class="kpi">
            <div class="kpi-label">Transactions</div>
            <div class="kpi-value">{total_interactions:,}</div>
            <div class="kpi-note">Cleaned tourism interactions</div>
        </div>
        """)

    with k2:
        html(f"""
        <div class="kpi">
            <div class="kpi-label">Visitors</div>
            <div class="kpi-value">{unique_users:,}</div>
            <div class="kpi-note">Unique user profiles</div>
        </div>
        """)

    with k3:
        html(f"""
        <div class="kpi">
            <div class="kpi-label">Attractions</div>
            <div class="kpi-value">{unique_attractions}</div>
            <div class="kpi-note">Attractions in transaction data</div>
        </div>
        """)

    with k4:
        html(f"""
        <div class="kpi">
            <div class="kpi-label">Mean Rating</div>
            <div class="kpi-value">{mean_rating:.2f}/5</div>
            <div class="kpi-note">Historical average</div>
        </div>
        """)

    html("""
    <div class="section-title">Three Machine Learning Objectives</div>
    <div class="section-text">
        The application exposes all three required ML components in one interface.
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        html("""
        <div class="card">
            <div class="kicker">01 · REGRESSION</div>
            <h3>Attraction Rating Prediction</h3>
            <p>
                Gradient Boosting predicts the expected attraction rating
                from travel timing, visitor origin, attraction and visit mode.
            </p>
        </div>
        """)

    with c2:
        html("""
        <div class="card">
            <div class="kicker">02 · CLASSIFICATION</div>
            <h3>Visit Mode Prediction</h3>
            <p>
                Random Forest classifies a visitor into Business, Couples,
                Family, Friends or Solo travel modes.
            </p>
        </div>
        """)

    with c3:
        html("""
        <div class="card">
            <div class="kicker">03 · RECOMMENDATION</div>
            <h3>Personalized Attraction Discovery</h3>
            <p>
                A hybrid recommender combines content similarity and
                item-based collaborative similarity.
            </p>
        </div>
        """)

    html("""
    <div class="section-title">Quick Dataset Insights</div>
    <div class="section-text">
        These values are calculated directly from the cleaned application dataset.
    </div>
    """)

    insight1, insight2 = st.columns(2)

    with insight1:
        rating_share = (
            tourism_data["Rating"].isin([4, 5]).mean() * 100
        )
        html(f"""
        <div class="card">
            <h3>⭐ Rating concentration</h3>
            <p>
                Ratings of 4 or 5 account for
                <strong style="color:#fff">{rating_share:.1f}%</strong>
                of recorded interactions. This concentration is important
                when interpreting rating-prediction performance.
            </p>
        </div>
        """)

    with insight2:
        mode_counts = tourism_data["VisitMode"].value_counts()
        top_mode = mode_counts.index[0]
        top_mode_pct = mode_counts.iloc[0] / total_interactions * 100
        html(f"""
        <div class="card">
            <h3>👥 Visitor-mode distribution</h3>
            <p>
                <strong style="color:#fff">{escape(str(top_mode))}</strong>
                is the largest recorded visit mode at
                <strong style="color:#fff">{top_mode_pct:.1f}%</strong>
                of interactions. The classification task is therefore
                affected by class imbalance.
            </p>
        </div>
        """)


# ============================================================
# PAGE 2: PREDICTION STUDIO
# ============================================================

elif page == "Prediction Studio":

    html("""
    <div class="section-title">Prediction Studio</div>
    <div class="section-text">
        Enter a visitor origin, travel period and target attraction.
        The classification model predicts VisitMode first; the selected
        or predicted mode is then supplied to the rating regression model.
    </div>
    """)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:

        continents = sorted(
            tourism_data["Continent"].dropna().astype(str).unique()
        )

        selected_continent = st.selectbox(
            "🌍 Traveler Continent",
            continents,
            key="pred_continent",
        )

        region_df = tourism_data[
            tourism_data["Continent"].astype(str)
            == selected_continent
        ]

        regions = sorted(
            region_df["Region"].dropna().astype(str).unique()
        )

        selected_region = st.selectbox(
            "📍 Region",
            regions,
            key="pred_region",
        )

        country_df = region_df[
            region_df["Region"].astype(str)
            == selected_region
        ]

        countries = sorted(
            country_df["Country"].dropna().astype(str).unique()
        )

        selected_country = st.selectbox(
            "🏳️ Country",
            countries,
            key="pred_country",
        )

        city_df = country_df[
            country_df["Country"].astype(str)
            == selected_country
        ]

        cities = sorted(
            city_df["CityName"].dropna().astype(str).unique()
        )

        selected_city = st.selectbox(
            "🏙️ Origin City",
            cities,
            key="pred_city",
        )

        years = sorted(
            tourism_data["VisitYear"]
            .dropna()
            .astype(int)
            .unique(),
            reverse=True,
        )

        p1, p2 = st.columns(2)

        with p1:
            visit_year = st.selectbox(
                "📅 Travel Year",
                years,
                key="pred_year",
            )

        with p2:
            visit_month = st.selectbox(
                "🗓️ Travel Month",
                list(MONTH_NAMES.keys()),
                format_func=lambda x: MONTH_NAMES[x],
                key="pred_month",
            )

        visit_quarter = ((visit_month - 1) // 3) + 1
        visit_season = get_season(visit_month)

        st.info(
            f"Derived temporal features: Q{visit_quarter} · {visit_season}"
        )

        attractions = sorted(
            tourism_data["Attraction"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_attraction = st.selectbox(
            "🏝️ Target Attraction",
            attractions,
            key="pred_attraction",
        )

        mode_override = st.selectbox(
            "👥 Visit Mode",
            [
                "Let AI Predict Visit Mode",
                "Business",
                "Couples",
                "Family",
                "Friends",
                "Solo",
            ],
            key="pred_mode",
        )

        run_prediction = st.button(
            "✦ Run Dual-Model Prediction",
            type="primary",
            use_container_width=True,
        )

    with right:

        attr_rows = tourism_data[
            tourism_data["Attraction"] == selected_attraction
        ]

        if attr_rows.empty:
            st.warning("No metadata found for this attraction.")
            st.stop()

        attr_meta = attr_rows.iloc[0]

        attr_type = safe_value(
            attr_meta.get("AttractionType"),
            "Unknown",
        )

        attr_city_id = int(
            pd.to_numeric(
                attr_meta.get("AttractionCityId"),
                errors="coerce",
            )
        )

        hist_rating, hist_visits = attraction_stats(
            selected_attraction
        )

        html(f"""
        <div class="card">
            <div class="kicker">DESTINATION PREVIEW</div>
            <h3>{escape(selected_attraction)}</h3>
            <p>
                Category:
                <strong style="color:#fff">{escape(attr_type)}</strong>
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px;">
                <div style="padding:16px;border-radius:13px;background:rgba(255,255,255,.035);">
                    <div class="kpi-label">Historical Rating</div>
                    <div style="font-size:24px;font-weight:800;color:#fbbf24;margin-top:5px;">
                        ★ {hist_rating:.2f}
                    </div>
                </div>
                <div style="padding:16px;border-radius:13px;background:rgba(255,255,255,.035);">
                    <div class="kpi-label">Recorded Visits</div>
                    <div style="font-size:24px;font-weight:800;color:#fff;margin-top:5px;">
                        {hist_visits:,}
                    </div>
                </div>
            </div>
        </div>
        """)

    if run_prediction:

        classification_input = pd.DataFrame({
            "VisitYear": [visit_year],
            "VisitMonth": [visit_month],
            "VisitQuarter": [visit_quarter],
            "VisitSeason": [visit_season],
            "Continent": [selected_continent],
            "Region": [selected_region],
            "Country": [selected_country],
            "CityName": [selected_city],
            "Attraction": [selected_attraction],
            "AttractionType": [attr_type],
            "AttractionCityId": [attr_city_id],
        })

        try:
            predicted_mode_id = int(
                classification_model.predict(
                    classification_input
                )[0]
            )

            predicted_mode = MODE_MAPPING.get(
                predicted_mode_id,
                "Unknown",
            )

            if mode_override == "Let AI Predict Visit Mode":
                effective_mode = predicted_mode
                effective_mode_id = predicted_mode_id
                source_text = "Random Forest classification"
            else:
                effective_mode = mode_override
                effective_mode_id = MODE_TO_ID[mode_override]
                source_text = "User-selected visit mode"

            regression_input = pd.DataFrame({
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "VisitQuarter": [visit_quarter],
                "VisitSeason": [visit_season],
                "VisitModeId": [effective_mode_id],
                "Continent": [selected_continent],
                "Region": [selected_region],
                "Country": [selected_country],
                "CityName": [selected_city],
                "Attraction": [selected_attraction],
                "AttractionType": [attr_type],
                "AttractionCityId": [attr_city_id],
                "AttractionAvgRating": [hist_rating],
            })

            predicted_rating = float(
                np.clip(
                    regression_model.predict(
                        regression_input
                    )[0],
                    1.0,
                    5.0,
                )
            )

        except Exception as exc:
            st.error("Prediction failed.")
            st.code(str(exc))
            st.info(
                "If the error mentions feature names, confirm that the "
                "models in models/ are the latest models trained from the notebook."
            )
            st.stop()

        r1, r2 = st.columns(2)

        with r1:
            html(f"""
            <div class="result">
                <div class="result-label">Objective 2 · Predicted Visit Mode</div>
                <div class="result-value">
                    {MODE_ICON.get(effective_mode, "🧭")} {escape(effective_mode)}
                </div>
                <div class="small-note" style="margin-top:10px;">
                    Source: {escape(source_text)}
                </div>
            </div>
            """)

        with r2:
            html(f"""
            <div class="result rating">
                <div class="result-label">Objective 1 · Predicted Rating</div>
                <div class="result-value">
                    ★ {predicted_rating:.2f} / 5.00
                </div>
                <div class="small-note" style="margin-top:10px;">
                    Gradient Boosting regression output
                </div>
            </div>
            """)

        # Probability display
        if hasattr(classification_model, "predict_proba"):

            try:
                probabilities = classification_model.predict_proba(
                    classification_input
                )[0]

                classes = classification_model.classes_

                prob_df = pd.DataFrame({
                    "VisitMode": [
                        MODE_MAPPING.get(int(c), str(c))
                        for c in classes
                    ],
                    "Probability": probabilities,
                }).sort_values(
                    "Probability",
                    ascending=False,
                )

                html("""
                <div class="section-title">Visit-Mode Probability</div>
                <div class="section-text">
                    Class probabilities produced by the Random Forest classifier.
                </div>
                """)

                chart = (
                    alt.Chart(prob_df)
                    .mark_bar(cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X(
                            "Probability:Q",
                            title="Probability",
                            axis=alt.Axis(format=".0%"),
                        ),
                        y=alt.Y(
                            "VisitMode:N",
                            sort="-x",
                            title=None,
                        ),
                        tooltip=[
                            "VisitMode",
                            alt.Tooltip(
                                "Probability:Q",
                                format=".2%",
                            ),
                        ],
                    )
                    .properties(height=220)
                )

                st.altair_chart(
                    chart,
                    use_container_width=True,
                )

            except Exception as exc:
                st.caption(
                    f"Probability display unavailable: {exc}"
                )

        strategy = BUSINESS_USE_CASES.get(
            effective_mode,
            BUSINESS_USE_CASES["Solo"],
        )

        html(f"""
        <div class="card" style="margin-top:20px;">
            <div class="kicker">BUSINESS ACTION</div>
            <h3>{escape(strategy["title"])}</h3>
            <p>
                <strong style="color:#fff">Marketing:</strong>
                {escape(strategy["marketing"])}
            </p>
            <p>
                <strong style="color:#fff">Service:</strong>
                {escape(strategy["amenities"])}
            </p>
            <p>
                <strong style="color:#fff">Retention:</strong>
                {escape(strategy["incentive"])}
            </p>
        </div>
        """)


# ============================================================
# PAGE 3: RECOMMENDER
# ============================================================

elif page == "Personalized Recommender":

    html("""
    <div class="section-title">Personalized Attraction Recommender</div>
    <div class="section-text">
        The final recommendation model uses a hybrid of item-based
        collaborative similarity and attraction-content similarity.
        The best evaluated blend was 80% collaborative and 20% content.
    </div>
    """)

    eligible_users = sorted(
        recommendation_history["UserId"]
        .dropna()
        .astype(int)
        .unique()
    )

    if not eligible_users:
        st.warning("No recommendation user profiles are available.")
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        chosen_user = st.selectbox(
            "👤 User Profile ID",
            eligible_users,
            key="rec_user",
        )

    user_history = recommendation_history[
        recommendation_history["UserId"] == chosen_user
    ].copy()

    with c2:
        algorithm = st.selectbox(
            "⚙️ Recommendation Architecture",
            [
                "Hybrid · 80% Collaborative + 20% Content",
                "Collaborative Only",
                "Content-Based Only",
                "Custom Blend",
            ],
            key="rec_algorithm",
        )

    if algorithm == "Collaborative Only":
        w_content, w_collab = 0.0, 1.0
    elif algorithm == "Content-Based Only":
        w_content, w_collab = 1.0, 0.0
    elif algorithm == "Custom Blend":
        w_collab = st.slider(
            "Collaborative Weight",
            0.0,
            1.0,
            0.8,
            0.05,
            key="rec_weight",
        )
        w_content = round(1.0 - w_collab, 2)
    else:
        w_content, w_collab = 0.2, 0.8

    all_categories = [
        "All Categories"
    ] + sorted(
        tourism_data["AttractionType"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    category = st.selectbox(
        "🏛️ Category Filter",
        all_categories,
        key="rec_category",
    )

    top_n = st.slider(
        "Number of Recommendations",
        3,
        10,
        5,
        key="rec_top_n",
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "Unique Visited Attractions",
            user_history["AttractionId"].nunique(),
        )

    with s2:
        st.metric(
            "Average Rating",
            f"{user_history['Rating'].mean():.2f}",
        )

    with s3:
        st.metric(
            "Recommendation Universe",
            len(attraction_data),
        )

    html("""
    <div class="section-title" style="font-size:20px;">
        User Visit History
    </div>
    """)

    history_display = user_history[
        ["Attraction", "AttractionType", "Rating"]
    ].rename(
        columns={
            "Attraction": "Destination",
            "AttractionType": "Category",
            "Rating": "User Rating",
        }
    )

    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "✦ Generate Personalized Recommendations",
        type="primary",
        use_container_width=True,
    ):

        recommendations = recommendation_function(
            chosen_user,
            top_n=top_n,
            content_weight=w_content,
            collaborative_weight=w_collab,
            category_filter=category,
        )

        if recommendations.empty:
            st.warning(
                "No unvisited attractions match the selected criteria."
            )
        else:

            html(f"""
            <div class="section-title" style="font-size:20px;">
                Top Ranked Attractions
            </div>
            <div class="section-text">
                Blend:
                <strong style="color:#fff">
                    {int(w_collab*100)}% Collaborative
                </strong>
                +
                <strong style="color:#fff">
                    {int(w_content*100)}% Content
                </strong>
            </div>
            """)

            for rank, row in recommendations.iterrows():

                name = escape(
                    safe_value(row.get("Attraction"))
                )
                category_name = escape(
                    safe_value(row.get("AttractionType"))
                )
                address = escape(
                    safe_value(
                        row.get("AttractionAddress"),
                        "Location not available",
                    )
                )
                score = float(
                    row["RecommendationScore"]
                )

                html(f"""
                <div class="rec">
                    <div>
                        <div class="rec-title">
                            #{rank + 1} · {name}
                        </div>
                        <div class="rec-meta">
                            {category_name} · {address}
                        </div>
                    </div>
                    <div class="rec-score">
                        {score:.3f}
                    </div>
                </div>
                """)

            st.caption(
                "Recommendation score is an internal ranking score, not a probability."
            )


# ============================================================
# PAGE 4: ANALYTICS
# ============================================================

elif page == "Tourism Analytics":

    html("""
    <div class="section-title">Tourism Analytics & EDA</div>
    <div class="section-text">
        Interactive views of visitor modes, attraction demand,
        geography and rating patterns.
    </div>
    """)

    c1, c2 = st.columns(2)

    with c1:

        rating_df = (
            tourism_data["Rating"]
            .value_counts()
            .sort_index()
            .rename_axis("Rating")
            .reset_index(name="Visits")
        )

        chart = (
            alt.Chart(rating_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(
                    "Rating:O",
                    title="Rating",
                ),
                y=alt.Y(
                    "Visits:Q",
                    title="Recorded Visits",
                ),
                tooltip=["Rating", "Visits"],
            )
            .properties(
                title="Rating Distribution",
                height=300,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with c2:

        mode_df = (
            tourism_data["VisitMode"]
            .value_counts()
            .rename_axis("VisitMode")
            .reset_index(name="Visits")
        )

        chart = (
            alt.Chart(mode_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(
                    "VisitMode:N",
                    title=None,
                ),
                y=alt.Y(
                    "Visits:Q",
                    title="Recorded Visits",
                ),
                tooltip=["VisitMode", "Visits"],
            )
            .properties(
                title="Visit Mode Distribution",
                height=300,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    c3, c4 = st.columns(2)

    with c3:

        top10 = (
            tourism_data["Attraction"]
            .value_counts()
            .head(10)
            .rename_axis("Attraction")
            .reset_index(name="Visits")
        )

        chart = (
            alt.Chart(top10)
            .mark_bar(cornerRadiusTopRight=6)
            .encode(
                y=alt.Y(
                    "Attraction:N",
                    sort="-x",
                    title=None,
                ),
                x=alt.X(
                    "Visits:Q",
                    title="Recorded Visits",
                ),
                tooltip=["Attraction", "Visits"],
            )
            .properties(
                title="Top 10 Most Visited Attractions",
                height=340,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with c4:

        continent_df = (
            tourism_data["Continent"]
            .value_counts()
            .rename_axis("Continent")
            .reset_index(name="Visits")
        )

        chart = (
            alt.Chart(continent_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(
                    "Continent:N",
                    title=None,
                ),
                y=alt.Y(
                    "Visits:Q",
                    title="Recorded Visits",
                ),
                tooltip=["Continent", "Visits"],
            )
            .properties(
                title="Visitor Volume by Continent",
                height=340,
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    year_df = (
        tourism_data
        .groupby("VisitYear")
        .size()
        .reset_index(name="Visits")
    )

    chart = (
        alt.Chart(year_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "VisitYear:O",
                title="Year",
            ),
            y=alt.Y(
                "Visits:Q",
                title="Recorded Visits",
            ),
            tooltip=["VisitYear", "Visits"],
        )
        .properties(
            title="Annual Travel Volume",
            height=300,
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    # Average rating by visit mode
    mode_rating = (
        tourism_data
        .groupby("VisitMode")["Rating"]
        .mean()
        .reset_index()
        .sort_values("Rating", ascending=False)
    )

    chart = (
        alt.Chart(mode_rating)
        .mark_bar(cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "Rating:Q",
                title="Average Rating",
                scale=alt.Scale(domain=[0, 5]),
            ),
            y=alt.Y(
                "VisitMode:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "VisitMode",
                alt.Tooltip(
                    "Rating:Q",
                    format=".2f",
                ),
            ],
        )
        .properties(
            title="Average Rating by Visit Mode",
            height=260,
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    html("""
    <div class="section-title">Model Evaluation & Business Insights</div>
    <div class="section-text">
        Verified evaluation values from the completed model experiments.
    </div>
    """)

    m1, m2, m3 = st.columns(3)

    with m1:
        html("""
        <div class="card">
            <div class="kicker">CLASSIFICATION</div>
            <h3>Random Forest</h3>
            <p><strong style="color:#fff">Accuracy:</strong> 48.60%</p>
            <p><strong style="color:#fff">Weighted Precision:</strong> 46.99%</p>
            <p><strong style="color:#fff">Weighted Recall:</strong> 48.60%</p>
            <p><strong style="color:#fff">Weighted F1:</strong> 46.98%</p>
        </div>
        """)

    with m2:
        html("""
        <div class="card">
            <div class="kicker">REGRESSION</div>
            <h3>Gradient Boosting</h3>
            <p><strong style="color:#fff">MSE:</strong> 0.8301</p>
            <p><strong style="color:#fff">RMSE:</strong> 0.9111</p>
            <p><strong style="color:#fff">R²:</strong> 0.1186</p>
        </div>
        """)

    with m3:
        html("""
        <div class="card">
            <div class="kicker">RECOMMENDATION</div>
            <h3>Hybrid Recommender</h3>
            <p><strong style="color:#fff">Recall@5:</strong> 72.78%</p>
            <p><strong style="color:#fff">Precision@5:</strong> 14.56%</p>
            <p><strong style="color:#fff">MAP@5:</strong> 46.03%</p>
            <p><strong style="color:#fff">Blend:</strong> 80% Collaborative + 20% Content</p>
        </div>
        """)

    html("""
    <div class="section-title" style="font-size:20px;">
        Evidence-Based Findings
    </div>
    """)

    findings = [
        (
            "⭐ Rating distribution",
            "Ratings are strongly concentrated at 4 and 5, with 79.2% of recorded interactions in those two categories."
        ),
        (
            "👥 Visit-mode imbalance",
            "Couples is the largest visit-mode class, while Business is much smaller. This class imbalance helps explain why minority-class prediction is harder."
        ),
        (
            "🏝️ Demand concentration",
            "Nature & Wildlife Areas and Beach attractions account for a large share of recorded visits, showing that popularity is concentrated in a relatively small set of attraction types."
        ),
        (
            "📈 Regression limitation",
            "The Gradient Boosting model improves on the mean baseline, but R² = 0.1186 shows that most rating variation is not explained by the available features."
        ),
        (
            "⭐ Recommendation strength",
            "The hybrid recommender performs substantially better than the content-only evaluation, reaching 72.78% Recall@5 in the evaluated setup."
        ),
        (
            "⚠️ Evaluation caution",
            "The recommendation weights were tuned using the same evaluation split, so the 80/20 result should be presented as a tuned evaluation result rather than an untouched final test estimate."
        ),
    ]

    for title, text in findings:
        html(f"""
        <div class="card" style="margin:10px 0;">
            <h3>{escape(title)}</h3>
            <p>{escape(text)}</p>
        </div>
        """)

    html("""
    <div class="small-note" style="margin-top:28px;text-align:center;">
        Tourism Experience Analytics · Streamlit · Pandas · Scikit-Learn · Altair
    </div>
    """)
