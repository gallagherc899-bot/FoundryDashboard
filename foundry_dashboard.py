# foundry_dashboard_full.py
# Run with: streamlit run foundry_dashboard_full.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import plotly.figure_factory as ff

# -----------------------------
# Default Settings
# -----------------------------
DEFAULTS = {
    "scrap_threshold": 6.5,
    "prior_shift": True,
    "prior_shift_guard": 20,
    "use_quick_hook": False,
    "manual_s": 1.0,
    "manual_gamma": 0.5,
    "rolling_validation": True,
    "include_rate_features": True
}

def initialize_session_state():
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Dashboard Settings")

if st.sidebar.button("Reset to Recommended Defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.experimental_rerun()

thr_label = st.sidebar.slider("Scrap % Threshold", 1.0, 15.0, key="scrap_threshold", step=0.1)
prior_shift_guard = st.sidebar.slider("Prior Shift Guard", 0, 50, key="prior_shift_guard")
st.session_state["include_rate_features"] = st.sidebar.checkbox("Include *_rate Features", value=st.session_state["include_rate_features"])
st.session_state["prior_shift"] = st.sidebar.checkbox("Enable Prior Shift", value=st.session_state["prior_shift"])
st.session_state["use_quick_hook"] = st.sidebar.checkbox("Use Manual Quick‑Hook", value=st.session_state["use_quick_hook"])

if st.session_state["use_quick_hook"]:
    s_manual = st.sidebar.slider("Manual s", 0.1, 2.0, key="manual_s", step=0.1)
    gamma_manual = st.sidebar.slider("Manual γ", 0.1, 1.0, key="manual_gamma", step=0.05)

st.session_state["rolling_validation"] = st.sidebar.checkbox("Run 6‑2‑1 Rolling Validation", value=st.session_state["rolling_validation"])
settings = {key: st.session_state[key] for key in DEFAULTS}

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Model Comparison",
    layout="wide"
)

# -----------------------------
# Upload & Process Dataset
# -----------------------------
st.title("🧪 Foundry Scrap Risk Model Comparison Dashboard")

csv_file = st.file_uploader("Upload your foundry dataset CSV", type="csv")

if csv_file is not None:
    df = pd.read_csv(csv_file)
    df.columns = (df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.replace("#", "num", regex=False))
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])
    df = df[df["order_quantity"] > 0].copy()

    df["scrap_flag"] = (df["scrap%"] > thr_label).astype(int)

    # Compute MTTFscrap
    mtbf_df = df.groupby("part_id").agg(
        total_runs=("scrap%", "count"),
        failures=("scrap_flag", "sum")
    )
    mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
    mtbf_df["mttf_scrap"] = mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"])
    df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

    df["part_id_encoded"] = LabelEncoder().fit_transform(df["part_id"])
    features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
    X = df[features]
    y = df["scrap_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Pre‑SMOTE Model
    rf_pre = RandomForestClassifier(random_state=42)
    rf_pre.fit(X_train, y_train)

    # Post‑SMOTE Model
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    rf_post = RandomForestClassifier(random_state=42)
    rf_post.fit(X_resampled, y_resampled)

    # Enhanced Model (for demonstration use rf_post, replace with full pipeline if desired)
    rf_enhanced = rf_post

    models = {
        "Pre‑SMOTE": rf_pre,
        "Post‑SMOTE": rf_post,
        "Enhanced MTTFscrap": rf_enhanced
    }

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        mtbf = float(mtbf_df["mttf_scrap"].median())
        reliability = np.exp(-1.0/mtbf) if mtbf > 0 else np.nan
        try:
            mid = len(y_proba)//2
            stat, pval = wilcoxon(y_proba[:mid], y_proba[mid:])
        except:
            pval = np.nan
        expected_loss = float(np.mean(y_proba * df["order_quantity"].mean() * 30.0))
        results.append([name, round(brier,3), round(acc,3), f"${expected_loss:.2f}", round(mtbf,2), f"{reliability*100:.2f}%", f"{pval:.3f}" if not np.isnan(pval) else "N/A"])

    comp_df = pd.DataFrame(results, columns=["Model","Brier Score","Accuracy","Expected Loss","MTTFscrap","Reliability","Wilcoxon p‑value"])
    st.subheader("📊 Model Comparison Table")
    st.dataframe(comp_df, use_container_width=True)
    st.caption("Better models → lower Brier & p‑value, higher Accuracy, MTTFscrap & Reliability.")

    # Optionally, you could add SHAP visualization or deeper model‑specific panels below.
    st.markdown("---")
    st.subheader("📌 Detailed Model Diagnostics")
    model_choice = st.selectbox("Select Model for deeper view", list(models.keys()))
    model = models[model_choice]

    st.write(f"### {model_choice} Model Diagnostics")
    st.write(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
    st.write(f"Brier Score: {brier_score_loss(y_test, model.predict_proba(X_test)[:,1]):.3f}")

    if st.button(f"Explain {model_choice} Predictions via SHAP"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap_values_single = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(shap_values_single, X_test, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.error(f"SHAP visualization failed: {e}")
