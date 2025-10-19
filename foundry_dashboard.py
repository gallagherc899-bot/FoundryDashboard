import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Load and clean data
df = pd.read_csv("anonymized_parts.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "", regex=False).str.replace(")", "", regex=False)
df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
df = df.dropna(subset=["part_id", "scrap%", "order_quantity", "piece_weight_lbs", "week_ending"])

# Calculate MTTFscrap
initial_threshold = 5.0
df["scrap_flag"] = df["scrap%"] > initial_threshold
mtbf_df = df.groupby("part_id").agg(total_runs=("scrap%", "count"), failures=("scrap_flag", "sum"))
mtbf_df["mttf_scrap"] = mtbf_df["total_runs"] / mtbf_df["failures"].replace(0, np.nan)
mtbf_df["mttf_scrap"] = mtbf_df["mttf_scrap"].fillna(mtbf_df["total_runs"])
df = df.merge(mtbf_df[["mttf_scrap"]], on="part_id", how="left")

# Encode part_id
label_encoder = LabelEncoder()
df["part_id_encoded"] = label_encoder.fit_transform(df["part_id"])

# Define features and target
features = ["order_quantity", "piece_weight_lbs", "part_id_encoded", "mttf_scrap"]
X = df[features]
y = (df["scrap%"] > initial_threshold).astype(int)

# Train model with SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Calibrate with Platt scaling and Isotonic regression
platt_model = CalibratedClassifierCV(base_estimator=rf_model, method='sigmoid', cv='prefit')
platt_model.fit(X_test, y_test)

isotonic_model = CalibratedClassifierCV(base_estimator=rf_model, method='isotonic', cv='prefit')
isotonic_model.fit(X_test, y_test)

# Predict probabilities
df["original_scrap_probability"] = rf_model.predict_proba(X)[:, 1]
df["platt_scrap_probability"] = platt_model.predict_proba(X)[:, 1]
df["isotonic_scrap_probability"] = isotonic_model.predict_proba(X)[:, 1]

# Estimate scrap and cost
cost_per_pound = 2.50
for method in ["original", "platt", "isotonic"]:
    prob_col = f"{method}_scrap_probability"
    df[f"{method}_expected_scrap_pieces"] = df["order_quantity"] * df[prob_col]
    df[f"{method}_expected_scrap_pounds"] = df[f"{method}_expected_scrap_pieces"] * df["piece_weight_lbs"]
    df[f"{method}_expected_scrap_cost"] = df[f"{method}_expected_scrap_pounds"] * cost_per_pound

# Actual scrap
df["actual_scrap_pounds"] = df["pieces_scrapped"] * df["piece_weight_lbs"]

# Financial savings
for method in ["original", "platt", "isotonic"]:
    df[f"{method}_poundage_difference"] = df["actual_scrap_pounds"] - df[f"{method}_expected_scrap_pounds"]
    df[f"{method}_financial_savings"] = df[f"{method}_poundage_difference"] * cost_per_pound

# Output summary
summary_cols = [
    "part_id", "order_quantity", "piece_weight_lbs", "actual_scrap_pounds",
    "original_scrap_probability", "platt_scrap_probability", "isotonic_scrap_probability",
    "original_expected_scrap_pounds", "platt_expected_scrap_pounds", "isotonic_expected_scrap_pounds",
    "original_expected_scrap_cost", "platt_expected_scrap_cost", "isotonic_expected_scrap_cost",
    "original_financial_savings", "platt_financial_savings", "isotonic_financial_savings"
]

summary = df[summary_cols].round(2)
print(summary.head(20))

# Total savings comparison
total_original = df["original_financial_savings"].sum()
total_platt = df["platt_financial_savings"].sum()
total_isotonic = df["isotonic_financial_savings"].sum()

print("\nTotal Financial Impact Comparison:")
print(f"Original Model: ${total_original:,.2f}")
print(f"Platt-Calibrated Model: ${total_platt:,.2f}")
print(f"Isotonic-Calibrated Model: ${total_isotonic:,.2f}")
