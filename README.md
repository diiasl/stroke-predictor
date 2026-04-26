# 🧠 Stroke Risk Predictor — MaKaLe MSc ML Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

A Streamlit web application that predicts stroke risk using a tuned **Gradient Boosting** classifier with domain-informed interaction features, trained on the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## 🏫 Team — MaKaLe

| Name | Student ID | Role |
|------|-----------|------|
| Magzumov Amir | 25MD0218 | Report Lead |
| Kaziyeva Dana | 25MD0204 | Feature Engineer |
| Les Dias | 25MD0501 | Modelling & Deployment Lead |

---

## 🗂️ Repository Structure

```
stroke_app/
├── app.py                         # Main Streamlit application
├── makale_week3_best_model.joblib  # Trained pipeline (Gradient Boosting)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stroke-predictor.git
cd stroke-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `makale_week3_best_model.joblib` is committed — it's ~MB so it fits under GitHub's 100 MB limit).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

---

## 🧪 Model Details

| Item | Detail |
|------|--------|
| **Dataset** | Stroke Prediction Dataset (5 110 rows, 4.9 % positive) |
| **Best Model** | Gradient Boosting (tuned via RandomizedSearchCV) |
| **Primary Metrics** | Recall + PR-AUC (imbalanced dataset) |
| **Pipeline** | IterativeImputer → Log-transform glucose → StandardScaler → OneHotEncoder → DomainInteractionAdder → SelectKBest (k=7) → GBM |
| **Interaction Features** | age×hypertension, age×heart_disease, bmi×log_glucose, age×log_glucose |
| **Resampling** | SMOTE / ADASYN / RandomUnderSampler (evaluated in Week 3) |
| **Interpretability** | SHAP TreeExplainer waterfall per prediction |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.  
It must **not** be used as a substitute for clinical diagnosis or professional medical advice.

---

## 📚 Reference

Dev, S., Wang, H., Nwosu, C. S., et al. (2022). A predictive analytics approach for stroke prediction using machine learning and neural networks. *Healthcare Analytics*, 2, 100032.
