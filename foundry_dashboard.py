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
st.markdown("Estimate scrap risk, defect likelihood, cost impact, and reliability (MTBF) for any part.")

# Part ID selection with "New" option
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)

# Input fields
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)
threshold = st.slider("Scrap % Threshold for Failure (MTBF)", min_value=1.0, max_value=10.0, value=5.0)

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

        # Calculate MTBFscrap
        N = len(part_df)
        failures = (part_df["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")

        st.success(f"✅ Known Part ID: {part_id_input}")
        st.metric("Predicted Scrap %", f"{round(predicted_scrap, 2)}%")
        st.metric("MTBF (Scrap-Based)", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
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

            # Calculate MTBFscrap for similar parts
            N = len(similar)
            failures = (similar["scrap%"] > threshold).sum()
            mtbf_scrap = N / failures if failures > 0 else float("inf")

            st.warning("🆕 New Part ID (Similar parts found)")
            st.metric("Estimated Scrap %", f"{round(avg_scrap, 2)}%")
            st.metric("MTBF (Scrap-Based)", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
            st.write("Top 6 Likely Defects:")
            for defect, rate in defect_means.items():
                st.write(f"- {defect}: {round(rate * 100, 2)}% chance")
        else:
            avg_scrap = df["scrap%"].mean()
            defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
            defect_means = df[defect_cols].mean().sort_values(ascending=False).head(6)

            # Calculate MTBFscrap for entire dataset
            N = len(df)
            failures = (df["scrap%"] > threshold).sum()
            mtbf_scrap = N / failures if failures > 0 else float("inf")

            st.warning("🆕 New Part ID (No similar parts found)")
            st.metric("Estimated Scrap %", f"{round(avg_scrap, 2)}%")
            st.metric("MTBF (Scrap-Based)", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
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
st.markdown("---")
st.subheader("📈 MTBF Trend Visualization")

# Multi-select for up to 5 parts
selected_parts = st.multiselect(
    "Select up to 5 Part IDs to compare MTBFscrap trends:",
    options=sorted(df["part_id"].unique()),
    default=[],
    max_selections=5
)

if selected_parts:
    # Prepare data
    df["week_ending"] = pd.to_datetime(df["week_ending"], format="%m/%d/%Y", errors="coerce")

    grouped = df[df["part_id"].isin(selected_parts)].groupby(["part_id", "week_ending"])

    # Calculate MTBFscrap per week
    mtbf_data = []
    for (pid, week), group in grouped:
        N = len(group)
        failures = (group["scrap%"] > threshold).sum()
        mtbf = N / failures if failures > 0 else float("inf")
        mtbf_data.append({"Part ID": pid, "Week": week, "MTBFscrap": mtbf})

    mtbf_df = pd.DataFrame(mtbf_data)

    # Plot
    import altair as alt
    chart = alt.Chart(mtbf_df).mark_line(point=True).encode(
        x="Week:T",
        y=alt.Y("MTBFscrap:Q", title="MTBF (Runs per Failure)"),
        color="Part ID:N",
        tooltip=["Part ID", "Week", "MTBFscrap"]
    ).properties(
        title="MTBFscrap Trend by Week",
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)
