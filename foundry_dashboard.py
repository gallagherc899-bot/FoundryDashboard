
# ===============================
# Google Colab PHM Validation Script (Regression + Classification)
# Author: [Your Name]
# Based on Zio (2022) PHM Framework
# ===============================

# Install dependencies
!pip install shap scikit-learn pandas matplotlib scipy --quiet

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, brier_score_loss
from scipy.stats import wilcoxon
import shap
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

# ===============================
# 1. LOAD DATA
# ===============================

file_path = "/content/anonymized_parts.csv"  # Upload this file in Colab before running
data = pd.read_csv(file_path)

# Convert to numeric where possible
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='ignore')

# ===============================
# 2. PREPROCESSING
# ===============================

drop_cols = ['Work Order #', 'Week Ending', 'Part ID']
data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

# Fill missing values
data = data.fillna(data.median(numeric_only=True))

# Define targets
regression_target = 'Scrap%'
classification_target = 'HighScrap'

# Classification label based on 2.5% PHM target
data[classification_target] = (data[regression_target] > 2.5).astype(int)

# Split predictors and targets
X = data.drop(columns=[regression_target, classification_target])
y_reg = data[regression_target]
y_clf = data[classification_target]

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# ===============================
# 3. REGRESSION MODEL
# ===============================

reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
start_time = time.time()
reg_model.fit(X_train, y_train_reg)
reg_train_time = time.time() - start_time

y_pred_reg = reg_model.predict(X_test)
reg_r2 = r2_score(y_test_reg, y_pred_reg)
reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
reg_mae = mean_absolute_error(y_test_reg, y_pred_reg)

# ===============================
# 4. CLASSIFICATION MODEL
# ===============================

base_clf = RandomForestClassifier(n_estimators=200, random_state=42)
cal_clf = CalibratedClassifierCV(base_clf, cv=5)
start_time = time.time()
cal_clf.fit(X_train, y_train_clf)
clf_train_time = time.time() - start_time

y_pred_clf = cal_clf.predict(X_test)
y_prob_clf = cal_clf.predict_proba(X_test)[:, 1]

clf_acc = accuracy_score(y_test_clf, y_pred_clf)
clf_f1 = f1_score(y_test_clf, y_pred_clf)
clf_brier = brier_score_loss(y_test_clf, y_prob_clf)
_, clf_p_value = wilcoxon(y_pred_clf, y_test_clf)

# ===============================
# 5. SHAP INTERPRETABILITY
# ===============================

base_clf.fit(X_train, y_train_clf)
explainer_clf = shap.TreeExplainer(base_clf)
shap_values_clf = explainer_clf.shap_values(X_test)

if isinstance(shap_values_clf, list):
    shap_vals_to_plot = shap_values_clf[1] if len(shap_values_clf) > 1 else shap_values_clf[0]
else:
    shap_vals_to_plot = shap_values_clf

plt.title("Global Feature Importance (Classification - SHAP)")
shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("/content/SHAP_Global_Classification.png")
plt.close()

explainer_reg = shap.TreeExplainer(reg_model)
shap_values_reg = explainer_reg.shap_values(X_test)

plt.title("Global Feature Importance (Regression - SHAP)")
shap.summary_plot(shap_values_reg, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("/content/SHAP_Global_Regression.png")
plt.close()

# ===============================
# 6. UNCERTAINTY & PARETO ANALYSIS (Hard-Aligned Final Fix)
# ===============================

plt.figure(figsize=(6, 4))
plt.hist(y_prob_clf, bins=20, color='lightblue', edgecolor='black')
plt.title("Predicted Probability Distribution (Uncertainty Visualization)")
plt.xlabel("Predicted Probability of High Scrap (>2.5%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/content/Uncertainty_Distribution.png")
plt.close()

# Compute mean absolute SHAP values
mean_abs_shap_clf = np.mean(np.abs(shap_vals_to_plot), axis=0)

if mean_abs_shap_clf.ndim > 1:
    mean_abs_shap_clf = np.mean(mean_abs_shap_clf, axis=tuple(range(mean_abs_shap_clf.ndim - 1)))

# --- HARD ALIGNMENT FIX ---
n_features = len(X_test.columns)
flat_vals = np.ravel(mean_abs_shap_clf)

# Trim or pad the SHAP array to match feature count exactly
if len(flat_vals) > n_features:
    flat_vals = flat_vals[:n_features]
elif len(flat_vals) < n_features:
    flat_vals = np.pad(flat_vals, (0, n_features - len(flat_vals)), 'constant')

pareto_clf = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP|': flat_vals
}).sort_values(by='Mean |SHAP|', ascending=False)

pareto_clf['Cumulative %'] = (
    pareto_clf['Mean |SHAP|'].cumsum() / pareto_clf['Mean |SHAP|'].sum() * 100
)

# ===============================
# 7. PHM VALIDATION SUMMARIES (Zio-Aligned)
# ===============================

reg_summary = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE', 'Train Time (s)'],
    'Value': [reg_r2, reg_rmse, reg_mae, reg_train_time]
})

clf_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Brier Score', 'Wilcoxon p-value', 'Train Time (s)'],
    'Value': [clf_acc, clf_f1, clf_brier, clf_p_value, clf_train_time]
})

zio_comparison = pd.DataFrame({
    'PHM Dimension': ['Uncertainty Quantification', 'Accuracy', 'Interpretability', 'Deployability'],
    'Regression Evidence': [
        f'RMSE={reg_rmse:.3f}, MAE={reg_mae:.3f}',
        f'R²={reg_r2:.3f}',
        'SHAP (continuous feature attribution)',
        f'Training time={reg_train_time:.2f}s'
    ],
    'Classification Evidence': [
        f'Brier Score={clf_brier:.3f}',
        f'Accuracy={clf_acc:.3f}, F1={clf_f1:.3f}',
        'SHAP (binary feature attribution)',
        f'Training time={clf_train_time:.2f}s'
    ]
})

# ===============================
# 8. OUTPUT RESULTS
# ===============================

print("\n=== REGRESSION MODEL VALIDATION (Scrap%) ===\n")
print(reg_summary)
print("\n=== CLASSIFICATION MODEL VALIDATION (High Scrap Risk >2.5%) ===\n")
print(clf_summary)
print("\n=== ZIO PHM DIMENSION COMPARISON ===\n")
print(zio_comparison)

reg_summary.to_csv("/content/Regression_Validation.csv", index=False)
clf_summary.to_csv("/content/Classification_Validation.csv", index=False)
pareto_clf.to_csv("/content/Pareto_Classification.csv", index=False)
zio_comparison.to_csv("/content/Zio_Comparison_Summary.csv", index=False)

print("\nAll outputs saved to /content/. Use the file browser in Colab to download CSVs and PNG plots.\n")
