import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import plotly.figure_factory as ff

# ✅ Debug marker to confirm script freshness
print("✅ Running the latest version of foundry_dashboard.py")

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
df["part_id_encoded"] = LabelEncoder().fit_transform(df["part_id"])

# Define features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train models
rf_pre = RandomForestClassifier(random_state=42)
rf_pre.fit(X_train, y_train)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_post = RandomForestClassifier(random_state=42)
rf_post.fit(X_resampled, y_resampled)

# ⛔ No use of CalibratedClassifierCV at all
# ✅ Safe from any 'cv="prefit"' issues

# ... (the rest of your dashboard code remains unchanged)
