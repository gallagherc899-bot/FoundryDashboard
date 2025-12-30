# Foundry Scrap Risk Dashboard — Complete Updated Script
# Drop-in replacement for your current foundry_dashboard.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Foundry Scrap Risk Dashboard", layout="wide")

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
DEFAULT_THRESHOLD = 6.5
MIN_SAMPLES_LEAF = 2


# -------------------------------
# Utilities
# -------------------------------
def _normalize_headers(cols: pd.Index) -> pd.Index:
    return (
        pd.Index(cols)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )


def _canonical_rename(df: pd.DataFrame) -> pd.DataFrame:
    # Canonical renames for typical foundry spreadsheets (includes the image headings you shared)
    rename_map = {
        # Part/Work order ID
        "work_order": "part_id",
        "work_order_number": "part_id",
        "work_order_#": "part_id",
        "work_order_num": "part_id",
        "work_order_": "part_id",
        "work_order_#_": "part_id",
        "work_ord": "part_id",
        "part_id": "part_id",

        # Quantities and weights
        "order_quantity": "order_quantity",
        "order_qty": "order_quantity",
        "pieces": "order_quantity",
        "pieces_scrapped": "pieces_scrapped",
        "total_scrap_weight": "total_scrap_weight_lbs",
        "total_scrap_weight_lbs": "total_scrap_weight_lbs",
        "piece_weight_lbs": "piece_weight_lbs",
        "piece_weight": "piece_weight_lbs",

        # Dates / time
        "week_ending": "week_ending",
        "week_end": "week_ending",
        "week_end_date": "week_ending",

        # Scrap %
        "scrap%": "scrap_percent",
        "scrap_percent": "scrap_percent",
        "scrap_percentage": "scrap_percent",
        "scrap": "scrap_percent",

        # Other common columns
        "sellable": "sellable",
        "saleable": "sellable",
        "heats": "heats",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def _normalize_defect_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize defect headings variations to *_rate (using best-effort mapping)
    # From your dataset: dross_rate, bent_rate, outside_process_scrap_rate, failed_zyglo_rate, gouged_rate, shift_rate,
    # missrun_rate, core_rate, cut_into_rate, dirty_pattern_rate, crush_rate, zyglo_rate, shrink_rate,
    # short_pour_rate, runout_rate, shrink_porosity_rate, gas_porosity_rate, over_grind_rate, sand_rate, tear_up_rate

    # First pass: coerce any column ending in _rate to numeric
    for c in [c for c in df.columns if c.endswith("_rate")]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Second pass: map likely variants from screenshot to canonical names
    variant_map = {
        "gross_rate": "dross_rate",  # screenshot likely mis-OCR of "Dross Rate"
        "dross_rat": "dross_rate",
        "bent_rate": "bent_rate",
        "outside_proc": "outside_process_scrap_rate",
        "outside_process_scrap": "outside_process_scrap_rate",
        "failed_zyglo_rate": "failed_zyglo_rate",
        "failed_zigzag_rate": "failed_zyglo_rate",  # OCR variant
        "gouged_rate": "gouged_rate",
        "gouge_rate": "gouged_rate",
        "shift_rate": "shift_rate",
        "mismatch_rate": "missrun_rate",  # approximate mapping if mismatch ~ missrun
        "missrun_rate": "missrun_rate",
        "core_rate": "core_rate",
        "out_iron_rate": "cut_into_rate",  # approximate mapping (if cut into = out iron)
        "cut_into_rate": "cut_into_rate",
        "dirty_pattern_rate": "dirty_pattern_rate",
        "crush_rate": "crush_rate",
        "zigzag_rate": "zyglo_rate",  # OCR variant
        "zyglo_rate": "zyglo_rate",
        "shrink_rate": "shrink_rate",
        "short_pour_rate": "short_pour_rate",
        "runout_rate": "runout_rate",
        "shrink_porosity_rate": "shrink_porosity_rate",
        "gas_porosity_rate": "gas_porosity_rate",
        "over_grind_rate": "over_grind_rate",
        "sand_rate": "sand_rate",
        "tear_up_rate": "tear_up_rate",
    }

    # Apply variant mapping
    for src, tgt in variant_map.items():
        if src in df.columns and src != tgt:
            if tgt not in df.columns:
                df[tgt] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
            df.drop(columns=[src], inplace=True)

    # Final: any remaining *_rate to numeric
    for c in [c for c in df.columns if c.endswith("_rate")]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# -------------------------------
# Data loading and cleaning
# -------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Flatten multi-index column headers if present (Excel multi-row headers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != "None"]).strip() for col in df.columns.values]

    # Normalize headers
    df.columns = _normalize_headers(df.columns)

    # Canonical renaming
    df = _canonical_rename(df)

    # Handle duplicate 'part_id' columns: keep the first occurrence
    if (df.columns == "part_id").sum() > 1:
        st.warning("⚠ Detected multiple 'part_id' columns — keeping the first one.")
        # Drop duplicate columns beyond the first
        first_idx = np.where(df.columns == "part_id")[0][0]
        keep = np.ones(len(df.columns), dtype=bool)
        dup_indices = np.where(df.columns == "part_id")[0][1:]
        keep[dup_indices] = False
        df = df.loc[:, keep]

    # Fallbacks for missing key columns
    for c in ["part_id", "order_quantity", "piece_weight_lbs", "week_ending"]:
        if c not in df.columns:
            df[c] = np.nan

    # Scrap column: unify to 'scrap_percent'
    scrap_candidates = ["scrap_percent", "scrap%", "scrap", "pieces_scrapped"]
    scrap_col = next((c for c in scrap_candidates if c in df.columns), None)
    if scrap_col is None:
        df["scrap_percent"] = 0.0
        scrap_col = "scrap_percent"
    else:
        df.rename(columns={scrap_col: "scrap_percent"}, inplace=True)

    # Convert data types (coerce errors)
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df["scrap_percent"], errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce")
    df["piece_weight_lbs"] = pd.to_numeric(df["piece_weight_lbs"], errors="coerce")

    # Drop rows with invalid week_ending
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    # Normalize defect rate columns
    df = _normalize_defect_columns(df)
    defect_cols = [c for c in df.columns if c.endswith("_rate")]

    # Flatten part_id if nested, cast to string
    if "part_id" in df.columns:
        if isinstance(df["part_id"], pd.DataFrame):
            st.warning("⚠ 'part_id' detected as multi-column. Flattening first column.")
            df["part_id"] = df["part_id"].iloc[:, 0]
        elif len(df) > 0 and isinstance(df["part_id"].iloc[0], (list, tuple)):
            st.warning("⚠ 'part_id' contains nested values. Flattening with first element.")
            df["part_id"] = df["part_id"].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    else:
        df["part_id"] = np.nan

    df["part_id"] = df["part_id"].astype(str)
    df["part_id"].replace({"nan": "unknown", "": "unknown"}, inplace=True)

    # De-duplicate any remaining columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Informational summary
    st.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns. Detected {len(defect_cols)} defect-type rate columns.")
    st.write("🧩 Final cleaned column names:", list(df.columns))

    return df


# -------------------------------
# Helper functions
# -------------------------------
def time_split(df, train_ratio=0.75, calib_ratio=0.1):
    # Temporal split: sort by week_ending ascending
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:calib_end], df_sorted.iloc[calib_end:]


def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    # Part-level mean scrap_percent as proxy; floor to 1.0 if <= threshold
    df_train = df_train.copy()
    df_train["part_id"] = df_train["part_id"].astype(str)
    df_train["scrap_percent"] = pd.to_numeric(df_train["scrap_percent"], errors="coerce").fillna(0.0)

    grp = (
        df_train.groupby("part_id", dropna=False)["scrap_percent"]
        .mean(numeric_only=True)
        .reset_index()
    )
    grp.rename(columns={"scrap_percent": "mttf_scrap"}, inplace=True)
    grp["mttf_scrap"] = np.where(grp["mttf_scrap"] <= thr_label, 1.0, grp["mttf_scrap"])
    return grp


def attach_train_features(df_sub: pd.DataFrame,
                          mtbf_train: pd.DataFrame,
                          part_freq_train: pd.Series,
                          default_mtbf: float,
                          default_freq: float) -> pd.DataFrame:
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s


def make_xy(df: pd.DataFrame, thr_label: float, use_rate_cols: bool):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]

    # Ensure all features exist
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    # Defensive scrap label handling
    scrap_candidates = ["scrap_percent", "scrap%", "scrap", "pieces_scrapped"]
    scrap_col = next((c for c in scrap_candidates if c in df.columns), None)
    if scrap_col is None:
        df["scrap_percent"] = 0.0
        scrap_col = "scrap_percent"
    df[scrap_col] = pd.to_numeric(df[scrap_col], errors="coerce").fillna(0.0)

    # Label
    y = (df[scrap_col] > thr_label).astype(int)

    X = df[feats].copy()
    return X, y, feats


def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)

    # If calibration set is degenerate, skip calibration gracefully
    pos = int(y_calib.sum())
    if pos == 0 or pos == len(y_calib):
        return rf, rf, "uncalibrated"

    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3).fit(X_calib, y_calib)
    return rf, cal, "calibrated (sigmoid, cv=3)"


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("📂 Data source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("⚙️ Model controls")
thr_label = st.sidebar.slider("Scrap % threshold (Label & MTTF)", 1.0, 15.0, DEFAULT_THRESHOLD, 0.5)
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", True)
n_est = st.sidebar.slider("Number of trees (n_estimators)", 50, 300, DEFAULT_ESTIMATORS, 10)
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation", True)

if not os.path.exists(csv_path):
    st.error("❌ CSV not found.")
    st.stop()


# -------------------------------
# Data preparation
# -------------------------------
df = load_and_clean(csv_path)
df_train, df_calib, df_test = time_split(df)

# Compute part-level signals from train
mtbf_train = compute_mtbf_on_train(df_train, thr_label)
part_freq_train = df_train["part_id"].value_counts(normalize=True)

# Defaults for unseen parts
default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

# Attach features
df_train = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_calib = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
df_test = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

# Train and calibrate
X_train, y_train, feats = make_xy(df_train, thr_label, use_rate_cols)
X_calib, y_calib, _ = make_xy(df_calib, thr_label, use_rate_cols)
rf, cal_model, method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)


# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])


# -------------------------------
# Predict tab
# -------------------------------
with tab1:
    st.subheader("🔮 Predict scrap risk and reliability")

    c0, c1, c2, c3 = st.columns(4)
    part_id_input = c0.text_input("Part ID", value="Unknown")
    order_qty = c1.number_input("Order Quantity", min_value=1, value=100)
    piece_weight = c2.number_input("Piece Weight (lbs)", min_value=0.1, value=5.0)
    cost_per_part = c3.number_input("Cost per Part ($)", min_value=0.1, value=10.0)

    input_df = pd.DataFrame(
        [[part_id_input, order_qty, piece_weight, default_mtbf, default_freq]],
        columns=["part_id", "order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"],
    )

    if st.button("Predict"):
        try:
            X_input = input_df.drop(columns=["part_id"])
            # Ensure feature parity with training
            for col in feats:
                if col not in X_input.columns:
                    X_input[col] = 0.0
            X_input = X_input[feats]

            prob = float(cal_model.predict_proba(X_input)[0, 1])
            adj_prob = np.clip(prob, 0.0, 1.0)
            exp_scrap = order_qty * adj_prob
            exp_loss = exp_scrap * cost_per_part
            reliability = 1.0 - adj_prob

            st.markdown(f"### 🧩 Prediction results for part **{part_id_input}**")
            st.metric("Predicted Scrap Risk (raw)", f"{prob*100:.2f}%")
            st.metric("Adjusted Scrap Risk", f"{adj_prob*100:.2f}%")
            st.metric("Expected Scrap Count", f"{exp_scrap:.1f}")
            st.metric("Expected Loss ($)", f"{exp_loss:,.2f}")
            st.metric("MTTF Scrap (default)", f"{default_mtbf:.1f}")
            st.metric("Reliability", f"{reliability*100:.2f}%")

            # Pareto — Historical
            defect_cols = [c for c in df.columns if c.endswith("_rate")]
            if len(defect_cols) == 0:
                st.warning("⚠ No defect-type rate columns found.")
            else:
                st.markdown("#### 📊 Historical Pareto (Top 10 defect types by actual defects)")
                hist = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Historical Defects": [(df_train["order_quantity"] * df_train[c]).sum() for c in defect_cols]
                    })
                    .sort_values("Historical Defects", ascending=False)
                    .head(10)
                )
                total_hist = hist["Historical Defects"].sum()
                hist["Share (%)"] = np.where(total_hist > 0, hist["Historical Defects"] / total_hist * 100, 0.0)
                st.dataframe(hist)
                st.bar_chart(hist.set_index("Defect Type")["Historical Defects"])

                # Pareto — Predicted
                st.markdown("#### 🔮 Predicted Pareto (Top 10 defect types by expected defects)")
                df_test_local = df_test.copy()
                df_test_local["pred_prob"] = cal_model.predict_proba(make_xy(df_test_local, thr_label, use_rate_cols)[0])[:, 1]
                pred = (
                    pd.DataFrame({
                        "Defect Type": [c.replace("_rate", "").replace("_", " ").title() for c in defect_cols],
                        "Expected Defects": [(df_test_local["order_quantity"] * df_test_local[c] * df_test_local["pred_prob"]).sum()
                                             for c in defect_cols]
                    })
                    .sort_values("Expected Defects", ascending=False)
                    .head(10)
                )
                total_pred = pred["Expected Defects"].sum()
                pred["Share (%)"] = np.where(total_pred > 0, pred["Expected Defects"] / total_pred * 100, 0.0)
                st.dataframe(pred)
                st.bar_chart(pred.set_index("Defect Type")["Expected Defects"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -------------------------------
# Validation tab
# -------------------------------
with tab2:
    st.subheader("📏 Rolling 6–2–1 validation")
    try:
        X_test, y_test, _ = make_xy(df_test, thr_label, use_rate_cols)
        preds = cal_model.predict_proba(X_test)[:, 1]
        st.metric("Brier Score", f"{brier_score_loss(y_test, preds):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, (preds > 0.5)):.3f}")
        st.write(pd.DataFrame(
            confusion_matrix(y_test, (preds > 0.5)),
            index=["Actual OK", "Actual Scrap"],
            columns=["Pred OK", "Pred Scrap"]
        ))
        st.text(classification_report(y_test, (preds > 0.5)))
    except Exception as e:
        st.warning(f"Validation failed: {e}")

st.success(f"✅ Model trained successfully with {method}.")
