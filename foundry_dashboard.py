import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap
import plotly.figure_factory as ff

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# Initial threshold for MTTFscrap calculation
initial_threshold = 5.0
df["scrap_flag"] = df["scrap%"] > initial_threshold
mtbf_df = df.groupby("part_id").agg(
    total_runs=("scrap%", "count"),
    failures=("scrap_flag", "sum")
)
mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
mtbf_df["mttf_scrap"] = mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"])
df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

# Encode part_id
le = LabelEncoder()
df["part_id_encoded"] = le.fit_transform(df["part_id"])

# Define features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)

# Split into train, calibration, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

# Train model with class weighting
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# Calibrate using sigmoid (Platt scaling)
calibrated_model = CalibratedClassifierCV(estimator=rf_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

# Optimized scrap prediction weights
optimal_weights = [0.1283, 2.2272]  # [order_quantity, piece_weight_lbs]
benchmark_df = df.head(20).copy()
benchmark_df['actual_scrap_pounds'] = benchmark_df['order_quantity'] * benchmark_df['piece_weight_lbs'] * (benchmark_df['scrap%'] / 100)
benchmark_df['optimized_predicted_scrap'] = (
    benchmark_df['order_quantity'] * optimal_weights[0] + benchmark_df['piece_weight_lbs'] * optimal_weights[1]
)
benchmark_df['lower_bound'] = benchmark_df['actual_scrap_pounds'] * 0.90
benchmark_df['upper_bound'] = benchmark_df['actual_scrap_pounds'] * 1.10
benchmark_df['within_tolerance'] = (
    (benchmark_df['optimized_predicted_scrap'] >= benchmark_df['lower_bound']) &
    (benchmark_df['optimized_predicted_scrap'] <= benchmark_df['upper_bound'])
)

# Streamlit UI
st.title("🧪 Foundry Scrap Risk Dashboard")
part_id_input = st.selectbox("Select Part ID", ["New"] + sorted(df["part_id"].astype(str).unique().tolist()))
quantity = st.number_input("Order Quantity", min_value=1, value=10)
weight = st.number_input("Piece Weight (lbs)", min_value=0.01, value=1.0)
cost_per_part = st.number_input("Cost per Part ($)", min_value=0.01, value=5.0)

if st.button("Predict Scrap Risk"):
    part_known = part_id_input != "New"
    part_index = le.transform([int(part_id_input)])[0] if part_known else -1
    mttf_value = mtbf_df.loc[int(part_id_input), "mttf_scrap"] if part_known else df["mttf_scrap"].mean()
    input_data = pd.DataFrame([[quantity, weight, part_index, mttf_value]], columns=features)
    
    pred_proba = calibrated_model.predict_proba(input_data)[0][1]
    st.metric("Predicted Scrap Risk", f"{pred_proba * 100:.2f}%")

    expected_scrap = pred_proba * quantity
    expected_loss = expected_scrap * cost_per_part
    st.write(f"Expected Scrap Count: {expected_scrap:.1f} parts")
    st.write(f"Expected Financial Loss: ${expected_loss:.2f}")

    # Confusion Matrix
    y_pred = calibrated_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    z = [[tp, fn], [fp, tn]]
    fig = ff.create_annotated_heatmap(z, x=['Pred Scrap', 'Pred OK'], y=['Actual Scrap', 'Actual OK'], colorscale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    # SHAP
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_data)
        shap_val = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        shap_df = pd.DataFrame({
            'Feature': features,
            'SHAP Value': shap_val,
            'Impact': np.abs(shap_val)
        }).sort_values(by="Impact", ascending=False)
        st.write("### Feature Contributions to Prediction")
        st.dataframe(shap_df)
    except Exception as e:
        st.warning(f"SHAP analysis failed: {e}")
