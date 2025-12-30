import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('anonymized_parts.csv')
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("🔧 Manager Input Controls")
part_id = st.sidebar.text_input("Enter Part ID (matches 'Part ID' column exactly)")
order_qty = st.sidebar.number_input("Order Quantity", min_value=1, value=100)
weight = st.sidebar.number_input("Piece Weight (lbs)", min_value=0.01, value=10.0)
cost = st.sidebar.number_input("Cost per Part ($)", min_value=0.01, value=50.0)
threshold = st.sidebar.slider("Scrap% Threshold", 0.0, 5.0, 2.5, step=0.05)

# ---------------------------------------------------------
# Helper: Adaptive Similarity Expansion
# ---------------------------------------------------------
def get_similar_parts(df, part_id, scrap_col='Scrap%', weight_col='Piece Weight (lbs)'):
    target = df[df['Part ID'].astype(str) == str(part_id)]
    if len(target) == 0:
        return pd.DataFrame()

    target_scrap = target[scrap_col].mean()
    target_weight = target[weight_col].mean()

    expansion_scrap = 1.0
    expansion_weight = 10.0
    for _ in range(6):
        scrap_low, scrap_high = target_scrap - expansion_scrap, target_scrap + expansion_scrap
        w_low, w_high = target_weight * (1 - expansion_weight / 100), target_weight * (1 + expansion_weight / 100)
        similar = df[
            (df[scrap_col].between(scrap_low, scrap_high)) &
            (df[weight_col].between(w_low, w_high))
        ]
        if len(similar) >= 20:
            return similar
        expansion_scrap += 0.5
        expansion_weight += 5
    return similar

# ---------------------------------------------------------
# Train and Evaluate Model
# ---------------------------------------------------------
def train_and_evaluate(df_part, threshold):
    X = df_part.select_dtypes(include=[np.number]).drop(columns=['Scrap%'], errors='ignore')
    y = (df_part['Scrap%'] > threshold).astype(int)

    # Skip training if one class only
    if len(np.unique(y)) < 2:
        return None, None

    model = RandomForestClassifier(random_state=42, n_estimators=200)
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }

    try:
        y_prob = model.predict_proba(X)[:, 1]
        metrics['brier'] = brier_score_loss(y, y_prob)
    except Exception:
        metrics['brier'] = np.nan

    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return metrics, feature_imp

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
if st.sidebar.button("🔮 Predict"):
    if part_id == "":
        st.warning("Please enter a valid Part ID.")
    else:
        df_part = df[df['Part ID'].astype(str) == str(part_id)]

        if len(df_part) == 0:
            st.warning("⚠️ No matching Part ID found.")
        else:
            y_check = (df_part['Scrap%'] > threshold).astype(int)

            # Dual-class safeguard
            if len(df_part) >= 10 and len(np.unique(y_check)) == 2:
                st.info(f"✅ Direct training on Part {part_id} (dual-class dataset).")
                active_df = df_part
            else:
                similar_df = get_similar_parts(df, part_id)
                if len(similar_df) >= 10 and len(np.unique((similar_df['Scrap%'] > threshold).astype(int))) == 2:
                    st.info(f"🔍 Using expanded dataset (± scrap/weight range) with {len(similar_df)} samples.")
                    active_df = similar_df
                else:
                    st.warning("⚠️ All scrap values on one side of threshold — switching to SPC-based insights.")
                    active_df = df_part

            metrics, feature_imp = train_and_evaluate(active_df, threshold)

            # ---------------------------------------------------------
            # Charts
            # ---------------------------------------------------------
            defect_cols = [c for c in df.columns if 'rate' in c.lower()]
            hist_pareto = df_part[defect_cols].mean().sort_values(ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Historical Pareto (Observed)")
                plt.figure(figsize=(6, 4))
                plt.bar(hist_pareto.index, hist_pareto.values, color='steelblue')
                plt.xticks(rotation=90)
                st.pyplot(plt)

            with col2:
                st.subheader("Predicted Pareto (Enhanced ML-PHM)")
                if feature_imp is not None:
                    pred_pareto = feature_imp.set_index('Feature').head(10)
                    plt.figure(figsize=(6, 4))
                    plt.bar(pred_pareto.index, pred_pareto['Importance'], color='seagreen')
                    plt.xticks(rotation=90)
                    st.pyplot(plt)
                else:
                    st.warning("⚠️ Not enough variation for prediction. SPC-based patterns shown instead.")

            # ---------------------------------------------------------
            # Defect–Process Influence Table
            # ---------------------------------------------------------
            process_map = {
                'Sand System and Preparation': ['sand_rate', 'dirty_pattern_rate', 'crush_rate', 'runout_rate', 'gas_porosity_rate'],
                'Core Making': ['core_rate', 'gas_porosity_rate', 'shrink_porosity_rate', 'crush_rate'],
                'Pattern Making and Maintenance': ['shift_rate', 'bent_rate', 'dirty_pattern_rate'],
                'Mold Making and Assembly': ['shift_rate', 'runout_rate', 'missrun_rate', 'short_pour_rate', 'gas_porosity_rate'],
                'Melting and Alloy Treatment': ['dross_rate', 'gas_porosity_rate', 'shrink_rate', 'shrink_porosity_rate', 'gouged_rate'],
                'Pouring and Mold Filling': ['missrun_rate', 'short_pour_rate', 'dross_rate', 'tear_up_rate'],
                'Solidification and Cooling': ['shrink_rate', 'shrink_porosity_rate', 'gas_porosity_rate', 'missrun_rate'],
                'Shakeout and Cleaning': ['tear_up_rate', 'over_grind_rate', 'sand_rate'],
                'Inspection and Finishing': ['failed_zyglo_rate', 'zyglo_rate', 'outside_process_scrap_rate']
            }

            influence_data = []
            if feature_imp is not None:
                for process, defects in process_map.items():
                    for defect in defects:
                        imp = feature_imp.loc[feature_imp['Feature'] == defect, 'Importance']
                        if not imp.empty:
                            influence_data.append({
                                'Defect': defect,
                                'Process': process,
                                'Importance': imp.values[0]
                            })

            if influence_data:
                df_inf = pd.DataFrame(influence_data).sort_values(by='Importance', ascending=False)
                df_inf['Influence_%'] = 100 * df_inf['Importance'] / df_inf['Importance'].sum()
                st.subheader("📊 Defect–Process Influence (Campbell PHM)")
                st.dataframe(df_inf, hide_index=True)
            else:
                st.info("No measurable process influence detected for this threshold.")

            if metrics:
                st.subheader("📈 Model Performance")
                st.table(pd.DataFrame(metrics.items(), columns=['Metric', 'Score']).set_index('Metric'))
                st.success("✅ Prediction Complete!")
