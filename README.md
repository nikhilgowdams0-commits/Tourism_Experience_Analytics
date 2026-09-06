# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8502)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Project Overview
**Tourism Experience Analytics** is an end-to-end data science and machine learning platform engineered to address the three primary objectives of modern travel agencies, online travel agencies (OTAs), and destination management organizations (DMOs):
1. **Objective 1 (Regression):** Predicting Attraction Ratings (1.0 - 5.0 scale) based on demographic, temporal, and attraction characteristics.
2. **Objective 2 (Classification):** Classifying User Visit Modes (*Business, Couples, Family, Friends, Solo*) to power targeted promotions and resource planning.
3. **Objective 3 (Recommendations):** Curating Personalized Destination Recommendations using Collaborative Filtering, Content-Based Filtering, and a tuned Hybrid engine.

---

## 🏗️ Architecture & Tech Stack
- **Web Interface:** Streamlit with luxury glassmorphism UI & custom segmented controls
- **Interactive Visualizations:** Altair & Vega-Lite
- **Data Engineering & Analysis:** Pandas, NumPy, OpenPyXL, SQLite (`tourism.db`)
- **Machine Learning & Modeling:** Scikit-Learn (Random Forest, Gradient Boosting, Cosine Similarity)
- **Model Storage & Serialization:** Joblib

---

## 🎯 Machine Learning Engines & Benchmarks

| Objective | Task | Model Architecture | Key Evaluation Metrics |
|---|---|---|---|
| **Objective 1** | Rating Prediction | Gradient Boosting Regressor | **MSE:** 0.8301 · **RMSE:** 0.9111 · **$R^2$:** 0.1186 |
| **Objective 2** | Visit Mode Prediction | Random Forest Classifier | **Accuracy:** 48.60% · **Weighted F1:** 46.98% |
| **Objective 3** | Personalized Recommender | Hybrid (80% Collab / 20% Content) | **Recall @ 5:** 72.78% · **MAP @ 5:** 46.03% |

---

## 🚀 Running the Application
```bash
# 1. Activate Virtual Environment
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit UI
streamlit run app/app.py
```
Open **http://localhost:8502** in your browser.
