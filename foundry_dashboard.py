import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv("RATESOlAL60.20.20.csv")
df["Week Ending"] = pd.to_datetime(df["Week Ending"].astype(str).str.strip(), format="mixed", errors="coerce")
df = df.dropna(subset=["Week Ending"]).sort_values("Week Ending").reset_index(drop=True)

# Print date range after cleaning
print(f"📅 Date range after cleaning: {df['Week Ending'].min().date()} to {df['Week Ending'].max().date()}")


# Add seasonal and temporal features
df["Month"] = df["Week Ending"].dt.month
df["Week"] = df["Week Ending"].dt.isocalendar().week
df["Season"] = df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
})
df["Sand Rate Lag1"] = df["Sand Rate"].shift(1)
df["Shift Rate Lag1"] = df["Shift Rate"].shift(1)
df["Sand Rate MA3"] = df["Sand Rate"].rolling(window=3).mean()
df["Shift Rate MA3"] = df["Shift Rate"].rolling(window=3).mean()

# Drop rows with NaNs from lagged features
# df = df.dropna().reset_index(drop=True) # Removed this line

# Rolling window setup
start_date = pd.to_datetime("2023-01-01")
end_date = pd.to_datetime("2025-08-31")
window_size = pd.DateOffset(months=6)
val_size = pd.DateOffset(months=2)
test_size = pd.DateOffset(months=2)

results = []
current_start = start_date

while current_start + window_size + val_size + test_size <= end_date:
    print(f"\n🔄 Processing window starting {current_start.date()}...")
    train_end = current_start + window_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    train = df[(df["Week Ending"] >= current_start) & (df["Week Ending"] < train_end)].copy()
    val = df[(df["Week Ending"] >= train_end) & (df["Week Ending"] < val_end)].copy()
    test = df[(df["Week Ending"] >= val_end) & (df["Week Ending"] < test_end)].copy()

    print(f"  ➤ Train rows (before dropna): {len(train)}, Val rows (before dropna): {len(val)}, Test rows (before dropna): {len(test)}")

    if len(train) < 10 or len(val) < 5 or len(test) < 5:
        print("  ⚠️ Skipping window due to insufficient data.")
        current_start += pd.DateOffset(weeks=4)
        continue

    combined = pd.concat([train, val, test])
    encoded = pd.get_dummies(combined[["Customer Name", "Season"]], drop_first=True)

    rate_cols = [col for col in df.columns if col.endswith("Rate")]
    context_cols = ["Piece Weight (lbs)", "Order Quantity", "Month", "Week"]
    lagged_cols = ["Sand Rate Lag1", "Shift Rate Lag1", "Sand Rate MA3", "Shift Rate MA3"]
    all_features = rate_cols + context_cols + lagged_cols + encoded.columns.tolist()

    train_X = pd.concat([train[rate_cols + context_cols + lagged_cols], encoded.loc[train.index]], axis=1).dropna()
    val_X = pd.concat([val[rate_cols + context_cols + lagged_cols], encoded.loc[val.index]], axis=1).dropna()
    test_X = pd.concat([test[rate_cols + context_cols + lagged_cols], encoded.loc[test.index]], axis=1).dropna()

    train_y = train.loc[train_X.index, "Scrap%"]
    val_y = val.loc[val_X.index, "Scrap%"]
    test_y = test.loc[test_X.index, "Scrap%"]

    print(f"  ➤ Train rows (after dropna): {len(train_X)}, Val rows (after dropna): {len(val_X)}, Test rows (after dropna): {len(test_X)}")


    if len(train_X) < 10 or len(val_X) < 5 or len(test_X) < 5:
        print("  ⚠️ Skipping window due to insufficient data after dropping NaNs.")
        current_start += pd.DateOffset(weeks=4)
        continue


    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_X, train_y)

    def evaluate(X, y):
        pred = model.predict(X)
        r2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        return r2, rmse

    r2_train, rmse_train = evaluate(train_X, train_y)
    r2_val, rmse_val = evaluate(val_X, val_y)
    r2_test, rmse_test = evaluate(test_X, test_y)

    baseline_pred = np.full_like(test_y, train_y.mean())
    baseline_r2 = r2_score(test_y, baseline_pred)
    model_gain = r2_test - baseline_r2

    importances = pd.Series(model.feature_importances_, index=train_X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    results.append({
        "Window Start": current_start.date(),
        "Train R²": round(r2_train, 4),
        "Val R²": round(r2_val, 4),
        "Test R²": round(r2_test, 4),
        "Baseline R²": round(baseline_r2, 4),
        "Model Gain": round(model_gain, 4),
        "Train RMSE": round(rmse_train, 4),
        "Val RMSE": round(rmse_val, 4),
        "Test RMSE": round(rmse_test, 4),
        "Top Features": top_features.to_dict()
    })

    current_start += pd.DateOffset(weeks=4)

# Final output
if not results:
    print("❌ No valid windows processed. Try inspecting row counts or adjusting window size.")
else:
    print(f"\n✅ {len(results)} windows processed successfully.")
    for r in results:
        print(f"\n📊 Window starting {r['Window Start']}:")
        print(f"  Train R²: {r['Train R²']}, RMSE: {r['Train RMSE']}")
        print(f"  Val   R²: {r['Val R²']}, RMSE: {r['Val RMSE']}")
        print(f"  Test  R²: {r['Test R²']}, RMSE: {r['Test RMSE']}")
        print(f"  Baseline R²: {r['Baseline R²']}, Model Gain: {r['Model Gain']}")
        print("  Top Features:")
        for k, v in r["Top Features"].items():
            print(f"    {k}: {round(v, 4)}")

# 📊 Monthly row count diagnostic
df["YearMonth"] = df["Week Ending"].dt.to_period("M")
monthly_counts = df["YearMonth"].value_counts().sort_index()

print("\n📅 Row counts per month:")
print(monthly_counts)

# Optional: visualize row density
import matplotlib.pyplot as plt
monthly_counts.plot(kind="bar", title="Rows per Month", figsize=(12, 4))
plt.ylabel("Row Count")
plt.xlabel("Month")
plt.tight_layout()
plt.show()
