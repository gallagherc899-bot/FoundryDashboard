import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Campbell Foundry Dashboard", layout="wide")

# ---------------------------------------------------------
# 1. CAMPBELL PROCESS MAPPING (CORE LOGIC)
# ---------------------------------------------------------
# This dictionary maps specific defects to their root-cause processes.
# Note: Some defects (like Missrun) are linked to multiple processes.
CAMPBELL_MAP = {
    "Melt & Pour": ["Dross Rate", "Short Pour Rate", "Runout Rate", "Gas Porosity Rate", "Shrink Porosity Rate"],
    "Molding & Sand": ["Sand Rate", "Crush Rate", "Tear Up Rate", "Shift Rate", "Missrun Rate"],
    "Core Room": ["Core Rate", "Missrun Rate", "Shift Rate"],
    "Finishing": ["Gouged Rate", "Over Grind Rate", "Cut Into Rate", "Bent Rate", "Zyglo Rate"]
}

# ---------------------------------------------------------
# 2. DATA LOADING & CLEANING
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('anonymized_parts.csv')
        
        # Clean column names for internal logic (remove spaces/dots)
        # But we keep a copy of the original names for the Campbell Map
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---------------------------------------------------------
# 3. ANALYTICS ENGINE
# ---------------------------------------------------------
def run_analysis(df, part_id, threshold):
    # 1. Labeling based on user threshold
    df['is_scrap'] = (df['Scrap%'] > threshold).astype(int)
    
    # 2. Identify available defect columns in the CSV
    available_defects = [c for c in df.columns if 'Rate' in c]
    
    if not available_defects:
        return None, None, "No columns with 'Rate' found in dataset."

    # 3. Model Training (to identify which defects drive scrap)
    X = df[available_defects].fillna(0)
    y = df['is_scrap']
    
    if len(y.unique()) < 2:
        return None, None, "Not enough data variation for this threshold. Try lowering the Scrap% Threshold."

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 4. Extract Feature Importance
    feat_imp = pd.Series(model.feature_importances_, index=available_defects)
    
    # 5. Campbell Attribution Logic
    process_scores = []
    for process, related_defects in CAMPBELL_MAP.items():
        # Match mapping to available columns (Case Insensitive)
        matches = [d for d in available_defects if d.strip().lower() in [rd.lower() for rd in related_defects]]
        
        importance_sum = feat_imp[matches].sum() if matches else 0
        process_scores.append({"Process": process, "Risk Weight": importance_sum})
    
    return pd.DataFrame(process_scores), feat_imp, None

# ---------------------------------------------------------
# 4. USER INTERFACE
# ---------------------------------------------------------
st.title("🏭 Campbell Process-Aware Analytics")
st.markdown("---")

df = load_and_clean_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    part_id = st.sidebar.selectbox("Select Part ID", options=df['Part ID'].unique())
    threshold = st.sidebar.slider("Scrap% Alarm Threshold", 0.0, 10.0, 2.5)
    
    if st.sidebar.button("Run Attribution"):
        process_df, feat_imp, error = run_analysis(df, part_id, threshold)
        
        if error:
            st.warning(error)
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 Process Attribution")
                # Normalize weights to 100%
                total = process_df["Risk Weight"].sum()
                if total > 0:
                    process_df["Influence %"] = (process_df["Risk Weight"] / total * 100).round(1)
                else:
                    process_df["Influence %"] = 0
                
                st.dataframe(process_df[["Process", "Influence %"]].sort_values("Influence %", ascending=False), hide_index=True)
                
                # Recommendation
                top_proc = process_df.sort_values("Risk Weight", ascending=False).iloc[0]["Process"]
                st.success(f"**Action Plan:** The model attributes the highest scrap risk to the **{top_proc}** process. Review setup parameters in this area.")

            with col2:
                st.subheader("📊 Top Defect Drivers")
                top_5 = feat_imp.sort_values(ascending=False).head(5)
                fig, ax = plt.subplots()
                top_5.plot(kind='barh', ax=ax, color='skyblue')
                ax.invert_yaxis()
                st.pyplot(fig)

            # Theoretical Summary
            st.info("""
            **How this works:** Following Campbell's research, certain defects like 'Missruns' are not isolated to one area. 
            The model identifies which defects are spiking and 'weights' the responsible processes accordingly.
            """)



### Key improvements in this version:
1.  **Case Insensitivity:** The mapping logic now uses `.lower()` and `.strip()`. This ensures that if your CSV has `"Dross Rate "` (with a space) and the code has `"Dross Rate"`, it will still link them correctly.
2.  **Explicit Attribution:** It calculates the "Risk Weight" by summing the `feature_importance_` of all defects linked to a specific process group. 
3.  **Automatic Error Handling:** If the "Scrap Threshold" is set so high that there is no data to learn from, the app will now give you a friendly warning instead of crashing.
4.  **No Dependencies Hidden:** This is the complete script. Save it as `foundry_dashboard.py` and ensure `anonymized_parts.csv` is in the same folder.
