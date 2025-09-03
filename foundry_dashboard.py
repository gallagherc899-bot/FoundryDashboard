import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop rows with missing critical values
required_cols = ["part_id", "scrap%", "order_quantity", "piece_weight_(lbs)"]
df = df.dropna(subset=required_cols)

# Train Random Forest model
features = ["order_quantity", "piece_weight_(lbs)", "part_id"]
X = df[features]
y = df["scrap%"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Dashboard UI
st.title("🧪 Foundry Scrap Risk Predictor")
st.markdown("Estimate scrap risk, defect likelihood, and cost impact for any part.")

# Part ID selection with "New" option
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)

# Input fields
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)

# Prediction logic
if st.button("Predict Scrap Risk"):
    part_known = selected_part != "New"
    part_id_input = int(float(selected_part)) if part_known else None

    if part_known:
        input_data = pd.DataFrame([[quantity, weight, part_id_input]], columns=features)
        predicted_scrap = model.predict(input_data)[0]

        # Get top 6 defect types historically associated with this part
        defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
        part_df = df[df["part_id"] == part_id_input]
        defect_means = part_df[defect_cols].mean().sort_values(ascending=False).head(6)

        st.success(f"✅ Known Part ID: {part_id_input}")
        st.metric("Predicted Scrap %", f"{round(predicted_scrap, 2)}%")
        st.write("Top 6 Likely Defects:")
        for defect, rate in defect_means.items():
            st.write(f"- {defect}: {round(rate * 100, 2)}% chance")

    else:
        # Find similar parts by weight and quantity
        similar = df[(df["piece_weight_(lbs)"].between(weight * 0.9, weight * 1.1)) &
                     (df["order_quantity"].between(quantity * 0.9, quantity * 1.1))]

        if not similar.empty:
            avg_scrap = similar["scrap%"].mean()
            defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
            defect_means = similar[defect_cols].mean().sort_values(ascending=False).head(6)
            st.warning("🆕 New Part ID (Similar parts found)")
            st.metric("Estimated Scrap %", f"{round(avg_scrap, 2)}%")
            st.write("Top 6 Likely Defects:")
            for defect, rate in defect_means.items():
                st.write(f"- {defect}: {round(rate * 100, 2)}% chance")
        else:
            avg_scrap = df["scrap%"].mean()
            defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
            defect_means = df[defect_cols].mean().sort_values(ascending=False).head(6)
            st.warning("🆕 New Part ID (No similar parts found)")
            st.metric("Estimated Scrap %", f"{round(avg_scrap, 2)}%")
            st.write("Top 6 Likely Defects:")
            for defect, rate in defect_means.items():
                st.write(f"- {defect}: {round(rate * 100, 2)}% chance")

    # Scrap weight and cost impact
    scrap_rate = predicted_scrap if part_known else avg_scrap
    scrap_weight = quantity * weight * (scrap_rate / 100)
    st.write(f"Estimated Scrap Weight: **{round(scrap_weight, 2)} lbs**")

    # Hypothetical cost (material + melting + labor)
    cost_estimate = scrap_weight * (0.75 + 0.15) + quantity * (scrap_rate / 100) * 2.00
    st.write(f"Hypothetical Cost Impact: **${round(cost_estimate, 2)}**")
