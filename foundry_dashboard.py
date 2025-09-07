import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import altair as alt
import matplotlib.pyplot as plt

# -----------------------------
# Load and clean data
# -----------------------------
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["week_ending"] = pd.to_datetime(df["week_ending"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_(lbs)", "week_ending"])

# -----------------------------
# Train model
# -----------------------------
features = ["order_quantity", "piece_weight_(lbs)", "part_id"]
X = df[features]
y = df["scrap%"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# UI Header
# -----------------------------
st.title("🧪 Foundry Scrap Risk & Reliability Dashboard")
st.markdown("Predict scrap risk, visualize MTBF trends, and compare defect counts across parts.")

# -----------------------------
# Scrap Prediction Panel
# -----------------------------
st.subheader("🔍 Scrap Risk Prediction")
part_ids = sorted(df["part_id"].unique())
part_id_options = ["New"] + [str(int(pid)) for pid in part_ids]
selected_part = st.selectbox("Select Part ID", part_id_options)
quantity = st.number_input("Number of Parts", min_value=1, step=1)
weight = st.number_input("Weight per Part (lbs)", min_value=0.1, step=0.1)
threshold = st.slider("Scrap % Threshold for Failure (MTBF)", min_value=1.0, max_value=10.0, value=5.0)
st.write(f"Current failure threshold: **{threshold}%**")

# -----------------------------
# MTBFscrap Equation
# -----------------------------
st.subheader("📐 MTBFscrap Equation")
st.latex(r"\text{MTBF}_{\text{scrap}} = \frac{N}{\sum I(S_i > T)}")
st.write("- N = total number of runs")
st.write("- Sᵢ = scrap % for run i")
st.write("- T = scrap threshold (slider value)")
st.write("- I(Sᵢ > T) = 1 if scrap exceeds threshold, else 0")

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict Scrap Risk"):
    part_known = selected_part != "New"
    part_id_input = int(float(selected_part)) if part_known else None

    if part_known:
        input_data = pd.DataFrame([[quantity, weight, part_id_input]], columns=features)
        predicted_scrap = model.predict(input_data)[0]

        part_df = df[df["part_id"] == part_id_input]
        defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
        defect_means = part_df[defect_cols].mean().sort_values(ascending=False).head(6)

        # MTBFscrap calculation
        N = len(part_df)
        failures = (part_df["scrap%"] > threshold).sum()
        mtbf_scrap = N / failures if failures > 0 else float("inf")

        # Confidence band calculation
        scrap_std = part_df["scrap%"].std()
        confidence_band = round(scrap_std, 2)
        lower_bound = round(predicted_scrap - confidence_band, 2)
        upper_bound = round(predicted_scrap + confidence_band, 2)

        # Display results
        st.success(f"✅ Known Part ID: {part_id_input}")
        st.write(f"**Predicted Scrap %:** {round(predicted_scrap, 2)}% ± {confidence_band}%")
        st.metric("MTBFscrap", f"{'∞' if mtbf_scrap == float('inf') else round(mtbf_scrap, 2)} runs per failure")
        st.write(f"Failures above threshold: **{failures}** out of **{N}** runs")

        # Confidence interpretation
        if confidence_band <= 1.5:
            st.success("🔒 High Confidence: Historical scrap variation is low for this part.")
        if upper_bound < threshold:
            st.info(f"📉 90% confident this part will stay below the {threshold}% failure threshold.")

        # Defect breakdown
        st.write("Top 6 Likely Defects:")
        for defect, rate in defect_means.items():
            st.write(f"- {defect}: {round(rate * 100, 2)}% chance")

        # Cost estimate
        scrap_weight = quantity * weight * (predicted_scrap / 100)
        cost_estimate = scrap_weight * (0.75 + 0.15) + quantity * (predicted_scrap / 100) * 2.00
        st.write(f"Estimated Scrap Weight: **{round(scrap_weight, 2)} lbs**")
        st.write(f"Hypothetical Cost Impact: **${round(cost_estimate, 2)}**")

# -----------------------------
# MTBF Trend Visualization
# -----------------------------
st.markdown("---")
st.subheader("📈 MTBFscrap Trend by Week")

date_range = st.date_input("Select Time Window", value=[df["week_ending"].min(), df["week_ending"].max()])
selected_parts = st.multiselect("Select up to 5 Part IDs", options=sorted(df["part_id"].unique()), max_selections=5)

if selected_parts and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = df[(df["week_ending"] >= start_date) & (df["week_ending"] <= end_date) & (df["part_id"].isin(selected_parts))]
    grouped = filtered.groupby(["part_id", "week_ending"])
    mtbf_data = []
    for (pid, week), group in grouped:
        N = len(group)
        failures = (group["scrap%"] > threshold).sum()
        mtbf = N / failures if failures > 0 else float("inf")
        mtbf_data.append({"Part ID": pid, "Week": week, "MTBFscrap": mtbf})
    mtbf_df = pd.DataFrame(mtbf_data)

    chart = alt.Chart(mtbf_df).mark_line(point=True).encode(
        x="Week:T",
        y=alt.Y("MTBFscrap:Q", title="MTBF (Runs per Failure)"),
        color="Part ID:N",
        tooltip=["Part ID", "Week", "MTBFscrap"]
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Defect Count Bar Chart
# -----------------------------
st.subheader("📊 Total Defect Counts by Part")
if selected_parts and len(date_range) == 2:
    defect_cols = [col for col in df.columns if col.endswith("rate") and "scrap" not in col]
    filtered = df[(df["week_ending"] >= start_date) & (df["week_ending"] <= end_date) & (df["part_id"].isin(selected_parts))]
    defect_counts = filtered.groupby("part_id")[defect_cols].sum().reset_index()
    melted = defect_counts.melt(id_vars="part_id", var_name="Defect Type", value_name="Count")
    bar_chart = alt.Chart(melted).mark_bar().encode(
        x="part_id:N",
        y="Count:Q",
        color="Defect Type:N",
        tooltip=["part_id", "Defect Type", "Count"]
    ).properties(width=800, height=400)
    st.altair_chart(bar_chart, use_container_width=True)
