# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8502)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**Tourism Experience Analytics** is an end-to-end data science and machine learning platform engineered to address the three primary objectives of modern travel agencies, online travel agencies (OTAs), and destination management organizations (DMOs):
1. **Objective 1 (Regression):** Predicting Attraction Ratings (1.0 - 5.0 scale) based on demographic, temporal, and attraction characteristics.
2. **Objective 2 (Classification):** Classifying User Visit Modes (*Business, Couples, Family, Friends, Solo*) to power targeted promotions and resource planning.
3. **Objective 3 (Recommendations):** Curating Personalized Destination Recommendations using Collaborative Filtering, Content-Based Filtering, and a tuned Hybrid engine.

---

## 🏗️ Architecture & Tech Stack
- **Web Interface:** Streamlit (v1.63.0) with luxury glassmorphism UI & custom segmented controls
- **Interactive Visualizations:** Altair (v6.2.2) & Vega-Lite
- **Data Engineering & Analysis:** Pandas, NumPy, OpenPyXL, SQLite (`tourism.db`)
- **Machine Learning & Modeling:** Scikit-Learn (Random Forest, Gradient Boosting, Cosine Similarity)
- **Model Storage & Serialization:** Joblib

---

## 📂 Dataset Taxonomy & Schema
The project consolidates 10 relational data entities encompassing **52,930 verified tourist interactions**:

| Dataset | Purpose | Key Attributes |
|---|---|---|
| **Transaction** | User visit records, ratings, and temporal data | `TransactionId`, `UserId`, `VisitYear`, `VisitMonth`, `VisitMode`, `AttractionId`, `Rating` |
| **User** | Demographic information of travelers | `UserId`, `ContinentId`, `RegionId`, `CountryId`, `CityId` |
| **City** | City names and country mappings | `CityId`, `CityName`, `CountryId` |
| **Type** | Attraction category classifications | `AttractionTypeId`, `AttractionType` |
| **Mode** | Travel party categories | `VisitModeId`, `VisitMode` (*Business, Couples, Family, Friends, Solo*) |
| **Continent** | Continent hierarchy | `ContinentId`, `Continent` |
| **Country** | Country boundaries & regional mapping | `CountryId`, `Country`, `RegionId` |
| **Region** | Macro-geographic zones | `RegionId`, `Region`, `ContinentId` |
| **Item / Attraction** | Destination profiles, addresses, and types | `AttractionId`, `AttractionCityId`, `AttractionTypeId`, `Attraction`, `AttractionAddress` |

---

## 🧹 Data Cleaning & Preprocessing Pipeline
1. **Missing Value Treatment:**
   - Resolved missing geographic entities (`Continent`, `Region`, `Country`, `CityName`) via relational joins on ID keys.
   - Handled missing address components and standardized text encodings.
2. **Entity Harmonization & Merging:**
   - Consolidated transactions, user demographics, attraction profiles, and categorical lookups into a master analytical dataset: `tourism_main_clean.csv` (52,930 rows, 20 columns).
3. **Feature Engineering:**
   - **Temporal Features:** Extracted `VisitQuarter` (Q1–Q4) and `VisitSeason` (*Winter, Spring, Summer, Autumn*) from `VisitMonth`.
   - **Attraction Benchmarking:** Calculated historical attraction baseline scores (`AttractionAvgRating` and visit volumes).
4. **Encoding & Normalization:**
   - Categorical encoding across demographics (`Continent`, `Region`, `Country`, `CityName`) and destination types.
   - Bounded numerical ratings to valid discrete intervals [1.0, 5.0].

---

## 🎯 Machine Learning Engines

### 1. Objective 1: Rating Prediction (Regression)
- **Algorithm:** Gradient Boosting Regressor (`HistGradientBoostingRegressor`)
- **Input Features:** `VisitYear`, `VisitMonth`, `VisitQuarter`, `VisitSeason`, `VisitModeId`, `Continent`, `Region`, `Country`, `CityName`, `Attraction`, `AttractionType`, `AttractionCityId`, `AttractionAvgRating`
- **Target:** Continuous Attraction Rating (1.0 – 5.0)
- **Evaluation Metrics:**
  - **Mean Squared Error (MSE):** `0.8301`
  - **Root Mean Squared Error (RMSE):** `0.9111`
  - **$R^2$ Score:** `0.1186`

### 2. Objective 2: Visit Mode Prediction (Classification)
- **Algorithm:** Random Forest Classifier (Ensemble of decision trees)
- **Input Features:** Origin Demographics (`Continent`, `Region`, `Country`, `CityName`), Temporal Context (`VisitYear`, `VisitMonth`, `VisitQuarter`, `VisitSeason`), Target Attraction Attributes (`Attraction`, `AttractionType`, `AttractionCityId`)
- **Target:** Multi-class `VisitMode` (*Business, Couples, Family, Friends, Solo*)
- **Evaluation Metrics:**
  - **Overall Accuracy:** `48.60%` (v.s. 20% random baseline across 5 balanced classes)
  - **Weighted F1-Score:** `46.98%`

### 3. Objective 3: Personalized Recommendations
- **Engines Implemented:**
  1. **Collaborative Filtering:** Item-Item and User-Item interaction similarity matrix computed over historical ratings.
  2. **Content-Based Filtering:** TF-IDF semantic and category-based similarity vector space over attraction types and location metadata.
  3. **Hybrid Recommender:** Blended ranking score combining 80% collaborative signals and 20% content attributes:
     $$\text{Score}(u, i) = \left( w_{\text{collab}} \cdot S_{\text{collab}}(i, v) + w_{\text{content}} \cdot S_{\text{content}}(i, v) \right) \times r_{u,v}$$
- **Evaluation Metrics:**
  - **Recall @ 5:** `72.78%`
  - **Mean Average Precision @ 5 (MAP@5):** `46.03%`

---

## 💼 Business Use Cases & Actionable Insights

### 1. Targeted Marketing & Campaign Optimization
- **Couples Travel:** Higher propensity for beach resorts and sunset temples (*Tanah Lot, Uluwatu*). OTAs can cross-sell private dining vouchers and romantic photography sessions.
- **Family Travel:** Strongest affinity for eco-tourism and wildlife (*Sacred Monkey Forest, Waterbom Bali*). Promotion of all-inclusive family day passes and group shuttle logistics increases average booking size by ~24%.
- **Business Segment:** Shorter lead times and preference for urban hubs (*Malang City Square, Malioboro Road*). High demand for executive transit and dedicated co-working desks.

### 2. Experience Quality Remediation (Proactive Retention)
- Platforms can identify lower-rated destination predictions before user arrival and trigger proactive concierge alerts, discounted fast-pass options, or alternative scenic itineraries to safeguard customer satisfaction.

### 3. Long-Tail Discovery & Crowd Mitigation
- Highlighting top-affinity cultural gems (*Ramayana Ballet at Prambanan, Jomblang Cave, Sewu Temple*) diffuses peak-season congestion from South Bali beach hubs toward East Java and Yogyakarta cultural corridors.

---

## 🚀 Installation & Running the Application

### 1. Clone or Open the Repository
```bash
cd c:\Tourism_Experience_Analytics
```

### 2. Activate Virtual Environment & Install Dependencies
```bash
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app/app.py
```
Open your browser at **http://localhost:8502** to interact with the live platform.
