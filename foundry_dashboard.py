import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix
import warnings

# --- Setup ---
st.set_page_config(page_title="Enhanced Foundry Analytics", layout="wide")
warnings.filterwarnings("ignore")

# ------------------------------------------------
# 1. CAMPBELL PROCESS MAPPING (The Enhanced Logic)
# ------------------------------------------------
# Based on Campbell (2003) - Mapping defects to multiple foundry processes
PROCESS_GROUPS = {
    "Melt & Pour": ["dross_rate", "short_pour_rate", "runout_rate", "gas_porosity_rate", "shrink_porosity_rate"],
    "Molding & Sand": ["sand_rate", "crush_rate", "tear_up_rate", "shift_rate", "missrun_rate"],
    "Core Room": ["core_rate", "missrun_rate", "shift_rate"], # Note: Overlaps per Campbell logic
    "Finishing": ["gouged_rate", "over_grind_rate", "cut_into_rate", "bent_rate"]
}

# ------------------------------------------------
# 2. DATA LOADER (Fixed for KeyErrors)
# ------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("anonymized_parts.csv")
    except FileNotFoundError:
        st.error("File 'anonymized_parts.csv' not found.")
        return None, []

    # Clean headers: remove special chars, spaces to underscores, lowercase
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Robust Date Detection
    date_options = ["week_ending", "weekending", "date", "week"]
    date_col = next((c for c in df.columns if c in date_options), None)
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.rename(columns={date_col: "week_ending"}).dropna(subset=["week_ending"])
    else:
        st.error("No date column detected. Ensure 'Week Ending' exists.")
        st.stop()

    # Identify defect columns (any column with 'rate' in name)
    defect_cols = [c for c in df.columns if "rate" in c and c != "scrap_percent"]
    
    return df, defect_cols

# ------------------------------------------------
# 3. CORE LOGIC
# ------------------------------------------------
def run_enhanced_analytics(df, defect_cols, threshold):
    # 1. Labeling
    df["label"] = (df["scrap_percent"] > threshold).astype(int)
    
    # 2. Time-based Split (Rolling 6-2-1)
    df = df.sort_values("week_ending")
    weeks = sorted(df["week_ending"].unique())
    train_weeks = weeks[:-3]
    test_weeks = weeks[-3:]
    
    train_df = df[df["week_ending"].isin(train_weeks)]
    test_df = df[df["week_ending"].isin(test_weeks)]
    
    # Features: Use defect rates to predict overall scrap risk
    X_train, y_train = train_df[defect_cols], train_df["label"]
    X_test, y_test = test_df[defect_cols], test_df["label"]
    
    # 3. Model
    model = CalibratedClassifierCV(RandomForestClassifier(n_estimators=150, random_state=42))
    model.fit(X_train, y_train)
    
    # 4. Process Influence Calculation (Campbell Theory)
    # We calculate which process contributes most to the predicted scrap
    probs = model.predict_proba(X_test)[:, 1]
    test_df["risk_score"] = probs
    
    process_impact = {}
    for process, defects in PROCESS_GROUPS.items():
        # Find which of these defects actually exist in our dataset
        valid_defects = [d for d in defects if d in defect_cols]
        if valid_defects:
            # Impact = Correlation of process defects with high-risk predictions
            impact = test_df[valid_defects].mean(axis=1).corr(test_df["risk_score"])
            process_impact[process] = max(0, impact) # Ensure non-negative

    return model, test_df, process_impact, X_test, y_test

# ------------------------------------------------
# 4. USER INTERFACE
# ------------------------------------------------
st.title("🏭 Campbell Process-Aware Foundry Dashboard")

df, defect_cols = load_data()

if df is not None:
    # Sidebar Controls
    st.sidebar.header("Model Settings")
    scrap_threshold = st.sidebar.slider("Scrap Risk Threshold (%)", 1.0, 15.0, 6.5)
    
    if st.sidebar.button("Run Analysis"):
        model, results, impact, X_test, y_test = run_enhanced_analytics(df, defect_cols, scrap_threshold)
        
        tab1, tab2 = st.tabs(["🚀 Enhanced Prediction", "📊 Model Validation"])
        
        with tab1:
            st.header("Process Risk Analysis")
            st.info("This view identifies the 'Weakest Link' in the foundry process according to Campbell research.")
            
            # Impact Metrics
            cols = st.columns(len(impact))
            for i, (process, score) in enumerate(impact.items()):
                cols[i].metric(process, f"{score:.2%}", delta="High Risk" if score > 0.5 else "Stable")

            # Pareto of Predicted Defects
            st.subheader("Predicted Defect Pareto")
            pred_defect_sums = results[defect_cols].multiply(results["risk_score"], axis=0).sum().sort_values(ascending=False)
            st.bar_chart(pred_defect_sums.head(10))

            # Recommendations
            st.subheader("💡 Managerial Recommendations")
            top_process = max(impact, key=impact.get)
            st.warning(f"**Primary Constraint identified:** {top_process}. Focus quality audits on these defects to reduce variance.")

        with tab2:
            st.header("Model Performance")
            col_a, col_b = st.columns(2)
            
            preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            brier = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
            
            col_a.metric("Prediction Accuracy", f"{acc:.1%}")
            col_b.metric("Brier Score (Calibration)", f"{brier:.4f}")
            
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, preds), 
                                      index=["Actual OK", "Actual Scrap"], 
                                      columns=["Pred OK", "Pred Scrap"]))
else:
    st.warning("Please ensure 'anonymized_parts.csv' is in the root directory.")
