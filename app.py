import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-low    { background:#d4edda; color:#155724; padding:16px 20px;
                   border-radius:10px; font-size:1.3rem; font-weight:700; text-align:center; }
    .risk-medium { background:#fff3cd; color:#856404; padding:16px 20px;
                   border-radius:10px; font-size:1.3rem; font-weight:700; text-align:center; }
    .risk-high   { background:#f8d7da; color:#721c24; padding:16px 20px;
                   border-radius:10px; font-size:1.3rem; font-weight:700; text-align:center; }
    .metric-card { background:#f8f9fa; border-radius:8px; padding:12px 16px;
                   border-left:4px solid #0d6efd; margin-bottom:8px; }
    .section-header { font-size:1.05rem; font-weight:600; color:#343a40;
                      border-bottom:2px solid #dee2e6; padding-bottom:4px; margin-top:20px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "makale_week3_best_model.joblib"

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Feature constants (mirror Week 3 notebook) ────────────────────────────────
NUMERICAL_FEATURES   = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES      = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ── SHAP helpers ──────────────────────────────────────────────────────────────
def get_feature_names(fitted_pipe):
    fitted_pre = fitted_pipe.named_steps["preprocessor"]
    ohe        = fitted_pre.named_transformers_["categorical"].named_steps["ohe"]
    ohe_names  = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    base_names = [
        "age_scaled", "log_glucose_scaled", "bmi_scaled",
        "hypertension", "heart_disease",
    ] + ohe_names

    interaction_names = [
        "age_x_hypertension", "age_x_heart_disease",
        "bmi_x_log_glucose",  "age_x_log_glucose",
    ]

    selector = fitted_pipe.named_steps["selector"]
    selected = np.array(base_names + interaction_names)[selector.get_support()].tolist()
    return selected


def transform_for_classifier(fitted_pipe, X):
    Xt = fitted_pipe.named_steps["preprocessor"].transform(X)
    Xt = fitted_pipe.named_steps["interaction_adder"].transform(Xt)
    Xt = fitted_pipe.named_steps["selector"].transform(Xt)
    return Xt


@st.cache_resource(show_spinner="Building SHAP explainer…")
def build_explainer(_model):
    clf = _model.named_steps["classifier"]
    return shap.TreeExplainer(clf)

# ── Sidebar — Patient Input ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=72)
    st.title("Patient Data")
    st.markdown("Fill in all fields, then press **Predict**.")

    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female", "Other"])
    age            = st.slider("Age (years)", 1, 100, 55)
    ever_married   = st.selectbox("Ever Married", ["Yes", "No"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    work_type      = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )

    st.markdown('<p class="section-header">Clinical Measurements</p>', unsafe_allow_html=True)
    avg_glucose = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 110.0, step=0.5)
    bmi         = st.slider("BMI", 10.0, 60.0, 28.0, step=0.1)

    st.markdown('<p class="section-header">Medical History</p>', unsafe_allow_html=True)
    hypertension  = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    smoking       = st.selectbox(
        "Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )

    st.markdown("---")
    predict_btn = st.button("🧠 Predict Stroke Risk", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 Stroke Risk Predictor")
st.caption(
    "MaKaLe MSc ML Project · Stroke Prediction Using Domain-Informed Interaction Features · "
    "Best model: **Gradient Boosting** (tuned, Week 3)"
)

if not predict_btn:
    # ── Welcome state ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** — Fill in patient data in the sidebar on the left.")
    with col2:
        st.info("**Step 2** — Click **Predict Stroke Risk**.")
    with col3:
        st.info("**Step 3** — See risk score + SHAP explanation.")

    with st.expander("ℹ️ About this app"):
        st.markdown("""
This application uses a **Gradient Boosting** classifier trained on the
[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
(5 110 patients, 4.9 % positive class).

**Pipeline highlights:**
- IterativeImputer (MICE) for BMI missing values
- Log-transform of average glucose level
- StandardScaler for numerical features
- One-hot encoding for categorical variables
- Domain-informed interaction features: `age × hypertension`, `age × heart_disease`, `bmi × log_glucose`, `age × log_glucose`
- SelectKBest (k = 7) feature selection
- RandomizedSearchCV tuning optimised for **Recall** and **PR-AUC**

**Primary metrics used for model selection:** Recall and PR-AUC (imbalanced dataset).

> ⚠️ This tool is for educational purposes only and must not replace clinical judgement.
        """)

else:
    # ── Build input dataframe ──────────────────────────────────────────────────
    input_data = pd.DataFrame([{
        "age":             age,
        "avg_glucose_level": avg_glucose,
        "bmi":             bmi,
        "hypertension":    int(hypertension),
        "heart_disease":   int(heart_disease),
        "gender":          gender,
        "ever_married":    ever_married,
        "work_type":       work_type,
        "Residence_type":  residence_type,
        "smoking_status":  smoking,
    }])

    # ── Predict ────────────────────────────────────────────────────────────────
    proba      = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    # ── Risk level ────────────────────────────────────────────────────────────
    if proba < 0.30:
        risk_label = "🟢 LOW RISK"
        risk_class = "risk-low"
        risk_desc  = "The model estimates a low probability of stroke."
    elif proba < 0.55:
        risk_label = "🟡 MEDIUM RISK"
        risk_class = "risk-medium"
        risk_desc  = "Some risk factors are present. Clinical follow-up is recommended."
    else:
        risk_label = "🔴 HIGH RISK"
        risk_class = "risk-high"
        risk_desc  = "Multiple strong risk factors detected. Prompt clinical evaluation is advised."

    # ── Layout: result + input summary ────────────────────────────────────────
    col_res, col_inp = st.columns([1, 1], gap="large")

    with col_res:
        st.subheader("Prediction Result")
        st.markdown(f'<div class="{risk_class}">{risk_label}<br><span style="font-size:0.9rem;font-weight:400">{risk_desc}</span></div>', unsafe_allow_html=True)
        st.markdown("")

        # Probability gauge
        st.metric("Stroke Probability", f"{proba:.1%}")
        st.progress(float(proba))

        # Confidence bar breakdown
        fig_prob, ax_prob = plt.subplots(figsize=(5, 1.2))
        ax_prob.barh(["No Stroke", "Stroke"], [1 - proba, proba],
                     color=["#198754", "#dc3545"])
        ax_prob.set_xlim(0, 1)
        ax_prob.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        for i, v in enumerate([1 - proba, proba]):
            ax_prob.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
        ax_prob.set_xlabel("Probability")
        ax_prob.set_title("Class Probabilities")
        plt.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)
        plt.close()

    with col_inp:
        st.subheader("Patient Summary")
        summary = {
            "Age": age,
            "Gender": gender,
            "BMI": f"{bmi:.1f}",
            "Avg Glucose (mg/dL)": f"{avg_glucose:.1f}",
            "Hypertension": "Yes" if hypertension else "No",
            "Heart Disease": "Yes" if heart_disease else "No",
            "Ever Married": ever_married,
            "Work Type": work_type,
            "Residence": residence_type,
            "Smoking Status": smoking,
        }
        for k, v in summary.items():
            st.markdown(
                f'<div class="metric-card"><strong>{k}</strong>: {v}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Explanation — What drove this prediction?")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows the contribution of each feature "
        "to the predicted risk. Red bars push risk **up**; blue bars push risk **down**."
    )

    try:
        explainer    = build_explainer(model)
        feature_names = get_feature_names(model)
        X_transformed = transform_for_classifier(model, input_data)
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        shap_values_raw = explainer.shap_values(X_transformed_df)
        # For GBM / XGB, shap_values may be a list [class0, class1] or a 2D array
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[1][0]
        else:
            sv = shap_values_raw[0]

        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            ev = expected_val[1] if len(np.atleast_1d(expected_val)) > 1 else float(np.atleast_1d(expected_val)[0])
        else:
            ev = float(expected_val)

        expl = shap.Explanation(
            values=sv,
            base_values=ev,
            data=X_transformed_df.iloc[0].values,
            feature_names=feature_names,
        )

        fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close()

        # ── Feature importance table ───────────────────────────────────────────
        with st.expander("📊 View feature contribution table"):
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": X_transformed_df.iloc[0].values.round(4),
                "SHAP Contribution": sv.round(4),
            }).sort_values("SHAP Contribution", key=abs, ascending=False)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")

    st.divider()

    # ── Clinical context ──────────────────────────────────────────────────────
    with st.expander("📖 Clinical context for this prediction"):
        st.markdown("""
**Key stroke risk factors (from the SHAP model):**

| Factor | Direction | Clinical note |
|--------|-----------|---------------|
| **Age** | ↑ risk as age increases | Strongest predictor in this model |
| **Avg Glucose Level** | ↑ risk with hyperglycaemia | Linked to diabetes-related vascular damage |
| **Hypertension** | ↑ risk if present | Classic modifiable risk factor |
| **Heart Disease** | ↑ risk if present | Associated with cardioembolic stroke |
| **BMI** | Moderate effect | Obesity increases vascular strain |
| **Smoking** | ↑ risk (active/former) | Accelerates atherosclerosis |

> ⚠️ **Disclaimer:** This tool is for *educational and research purposes only*.  
> It must **not** be used as a substitute for clinical diagnosis or medical advice.
> Always consult a qualified healthcare professional.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MaKaLe · MSc Machine Learning Group Project · Stroke Prediction · Week 3 Deployment")
