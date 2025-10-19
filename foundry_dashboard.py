import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import plotly.figure_factory as ff

# Load and preprocess data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"], inplace=True)

# Feature Engineering
threshold = 5.0
df["scrap_flag"] = df["scrap%"] > threshold
mtbf_df = df.groupby("part_id").agg(total_runs=("scrap%", "count"), failures=("scrap_flag", "sum"))
mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"], inplace=True)
df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

# Encode categorical features
label_encoder = LabelEncoder()
df["part_id_encoded"] = label_encoder.fit_transform(df["part_id"])

# Prepare features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > threshold).astype(int)

# Split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

# Resample and train model
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Calibrate model
calibrated_model = CalibratedClassifierCV(base_estimator=rf_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

# Streamlit App
st.title("Foundry Scrap Risk & Reliability Dashboard")
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Number of Parts", min_value=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1)
cost = st.number_input("Cost per Part ($)", min_value=0.01)

if st.button("Predict Scrap Risk"):
    known = selected_part != "New"
    part_id = int(float(selected_part)) if known else -1
    mttf = mtbf_df.loc[part_id, "mttf_scrap"] if known and part_id in mtbf_df.index else 1.0
    encoded_id = label_encoder.transform([part_id])[0] if known and part_id in label_encoder.classes_ else -1
    input_data = pd.DataFrame([[quantity, weight, encoded_id, mttf]], columns=features)

    proba = calibrated_model.predict_proba(input_data)[0][1]
    st.metric("Calibrated Scrap Risk", f"{round(proba * 100, 2)}%")

    if known:
        runs = len(df[df["part_id"] == part_id])
        fails = (df[df["part_id"] == part_id]["scrap%"] > threshold).sum()
        mtbf_scrap = runs / fails if fails > 0 else float("inf")
        reliability = np.exp(-1 / mtbf_scrap) if mtbf_scrap != float("inf") else 1.0
        st.metric("MTTFscrap", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)}")
        st.metric("Reliability for Next Run", f"{round(reliability * 100, 2)}%")

    expected_scrap = round(proba * quantity)
    expected_loss = round(expected_scrap * cost, 2)
    st.write(f"Expected scrap: **{expected_scrap} parts** → Loss: **${expected_loss}**")

    # Confusion Matrix
    y_pred = calibrated_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fig = ff.create_annotated_heatmap(
        z=[[tp, fn], [fp, tn]],
        x=['Predicted Scrap', 'Predicted Non-Scrap'],
        y=['Actual Scrap', 'Actual Non-Scrap'],
        colorscale='Blues'
    )
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig)

    # SHAP
    st.subheader("Feature Impact (SHAP)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_data)
    if isinstance(shap_values, list):
        shap_val = shap_values[1][0]
    else:
        shap_val = shap_values[0]
    shap_df = pd.DataFrame({"Feature": features, "SHAP Value": shap_val})
    shap_df["Impact"] = shap_df["SHAP Value"].abs()
    st.dataframe(shap_df.sort_values("Impact", ascending=False))
