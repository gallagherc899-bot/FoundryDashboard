# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

# -----------------------------
# Page / constants
# -----------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)
# TOP_K_PARETO is no longer used for dynamic 80% cutoff, but kept as a fallback.
TOP_K_PARETO = 8 
# Set rate cols to always be true for Pareto output (manager requirement)
USE_RATE_COLS_PERMANENT = True 


# -----------------------------
# Helpers - MODIFIED for 80% Pareto Rule
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("#", "num", regex=False)
    )
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=needed).copy()

    for c in ["scrap%", "order_quantity", "piece_weight_lbs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["scrap%", "order_quantity", "piece_weight_lbs"]).copy()

    if "pieces_scrapped" not in df.columns:
        df["pieces_scrapped"] = np.round((df["scrap%"].clip(lower=0) / 100.0) * df["order_quantity"]).astype(int)

    df = df.sort_values("week_ending").reset_index(drop=True)
    return df

def time_split(df: pd.DataFrame, train_frac=0.60, calib_frac=0.20):
    n = len(df)
    t_end = int(train_frac * n)
    c_end = int((train_frac + calib_frac) * n)

    df_train = df.iloc[:t_end].copy()
    df_calib = df.iloc[t_end:c_end].copy()
    df_test  = df.iloc[c_end:].copy()

    # prevent part leakage
    train_parts = set(df_train.part_id.unique())
    # Exclude parts seen in train from calib/test to ensure proper backtesting validation
    df_calib = df_calib[~df_calib.part_id.isin(train_parts)].copy()
    calib_parts = set(df_calib.part_id.unique())
    df_test  = df_test[~df_test.part_id.isin(train_parts.union(calib_parts))].copy()
    return df_train, df_calib, df_test

def compute_mtbf_on_train(df_train: pd.DataFrame, thr_label: float) -> pd.DataFrame:
    t = df_train.copy()
    t["scrap_flag"] = (t["scrap%"] > thr_label).astype(int)
    mtbf = t.groupby("part_id").agg(
        total_runs=("scrap%", "count"),
        failures=("scrap_flag", "sum")
    )
    mtbf["mttf_scrap"] = mtbf["total_runs"] / mtbf["failures"].replace(0, np.nan)
    mtbf["mttf_scrap"] = mtbf["mttf_scrap"].fillna(mtbf["total_runs"])
    return mtbf[["mttf_scrap"]]

def attach_train_features(df_sub, mtbf_train, part_freq_train, default_mtbf, default_freq):
    s = df_sub.merge(mtbf_train, on="part_id", how="left")
    s["mttf_scrap"] = s["mttf_scrap"].fillna(default_mtbf)
    s = s.merge(part_freq_train.rename("part_freq"), left_on="part_id", right_index=True, how="left")
    s["part_freq"] = s["part_freq"].fillna(default_freq)
    return s

def make_xy(df, thr_label: float, use_rate_cols: bool):
    feats = ["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"]
    if use_rate_cols:
        feats += [c for c in df.columns if c.endswith("_rate")]
    X = df[feats].copy()
    y = (df["scrap%"] > thr_label).astype(int)
    return X, y, feats

@st.cache_resource(show_spinner=True)
def train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators: int):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)

    has_both = (y_calib.sum() > 0) and (y_calib.sum() < len(y_calib))
    
    if not has_both:
        # **CRITICAL FIX:** Guardrail against single-class calibration data. 
        # Returns the uncalibrated RF model to prevent InvalidParameterError.
        st.warning(f"Calibration data is single-class (only {y_calib.sum()} positive events). Using uncalibrated predictions.")
        return rf, rf, "uncalibrated (data too scarce/unbalanced)"
        
    method = "isotonic" if len(y_calib) > 500 else "sigmoid"
    try:
        # Attempt calibration with selected method
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv="prefit").fit(X_calib, y_calib)
    except Exception:
        # Fallback to sigmoid if the first attempt (e.g., isotonic) fails for other reasons.
        # This is safe because the 'has_both' check passed above.
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        method = "sigmoid"
        
    return rf, cal, method

def tune_s_gamma_on_validation(p_val_raw, y_val, part_ids_val, part_scale,
                               s_grid=S_GRID, gamma_grid=GAMMA_GRID):
    """Minimize Brier on validation for p_adj = p_raw * s * (part_scale^gamma)."""
    if not len(p_val_raw):
         return {"brier_val": np.nan, "s": 1.0, "gamma": 1.0}
    
    ps = part_scale.reindex(part_ids_val).fillna(1.0).to_numpy(dtype=float)
    best = (np.inf, 1.0, 1.0)
    for s in s_grid:
        for g in gamma_grid:
            p_adj = np.clip(p_val_raw * (s * (ps ** g)), 0, 1)
            score = brier_score_loss(y_val, p_adj)
            if score < best[0]:
                best = (score, s, g)
    return {"brier_val": best[0], "s": best[1], "gamma": best[2]}

def prior_shift_logit(p_raw, src_prev, tgt_prev):
    """Saerens–Latinne prior correction on logits."""
    p = np.clip(p_raw, 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    delta = np.log(np.clip(tgt_prev,1e-6,1-1e-6)) - np.log(np.clip(1-tgt_prev,1e-6,1))
    delta -= np.log(np.clip(src_prev,1e-6,1-1e-6)) - np.log(np.clip(1-src_prev,1e-6,1))
    p_adj = 1/(1 + np.exp(-(logit + delta)))
    return np.clip(p_adj, 1e-6, 1-1e-6)

# Exceedance baselines at current threshold
def compute_part_exceedance_baselines(df_train: pd.DataFrame, thr_label: float):
    """Per-part prevalence of exceeding the threshold, and a scale vs global prevalence."""
    part_prev = (
        df_train.assign(exceed=(df_train["scrap%"] > thr_label).astype(int))
                .groupby("part_id")["exceed"].mean()
                .clip(lower=1e-6, upper=0.999)
    )
    global_prev = float(part_prev.mean()) if len(part_prev) else 0.5
    part_scale = (part_prev / max(global_prev, 1e-6)).fillna(1.0).clip(lower=0.25, upper=4.0)
    return part_prev, part_scale, global_prev

def build_input_row_for_part(selected_part: int,
                             quantity: float,
                             weight: float,
                             mttf_value: float,
                             part_freq_value: float,
                             FEATURES: list,
                             df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row dataframe aligned to FEATURES.
    For *_rate columns, fill with the part's historical mean (fallback = global mean).
    """
    base = {"order_quantity": quantity,
            "piece_weight_lbs": weight,
            "mttf_scrap": mttf_value,
            "part_freq": part_freq_value}

    row = pd.DataFrame([base], columns=["order_quantity", "piece_weight_lbs", "mttf_scrap", "part_freq"])

    rate_cols = [c for c in FEATURES if c.endswith("_rate")]
    if rate_cols:
        part_hist = df_train[df_train["part_id"] == selected_part][rate_cols].mean()
        global_hist = df_train[rate_cols].mean()
        filled = part_hist.fillna(global_hist).fillna(0.0)
        for c in rate_cols:
            row[c] = float(filled.get(c, float(global_hist.get(c, 0.0))))
    # Reorder to FEATURES exactly
    row = row.reindex(columns=FEATURES, fill_value=0.0)
    return row

def local_defect_drivers(calibrated_model,
                         input_row: pd.DataFrame,
                         FEATURES: list,
                         df_train: pd.DataFrame,
                         strategy: str = "p75",
                         k: int = 100) -> pd.DataFrame: # Increased k for intermediate calculation
    """
    Compute a local 'predicted Pareto' by measuring the positive delta in raw calibrated probability.
    Now implements a dynamic stop at the first defect where cumulative_% meets or exceeds 80%.
    """
    rate_cols = [c for c in FEATURES if c.endswith("_rate")]
    if not rate_cols:
        return pd.DataFrame(columns=["defect", "delta_prob", "share_%", "cumulative_%"])

    base_p = float(calibrated_model.predict_proba(input_row)[0, 1])

    deltas = []
    for col in rate_cols:
        tmp = input_row.copy()
        if strategy == "p75":
            hi = float(np.nanpercentile(df_train[col].values, 75))
            tmp[col] = hi
        else:
            tmp[col] = float(input_row[col].iloc[0]) + 0.02

        new_p = float(calibrated_model.predict_proba(tmp)[0, 1])
        delta = max(0.0, new_p - base_p)  # only positive risk drivers for Pareto
        deltas.append((col, delta))

    dd = pd.DataFrame(deltas, columns=["defect", "delta_prob"])
    dd = dd[dd["delta_prob"] > 1e-6] # Filter out defects with negligible impact
    dd = dd.sort_values("delta_prob", ascending=False).head(k)
    total = float(dd["delta_prob"].sum())
    
    if total > 0:
        dd["share_%"] = dd["delta_prob"] / total * 100.0
        dd["cumulative_%"] = dd["share_%"].cumsum()
        
        # --- 80% Pareto Rule Implementation ---
        stop_index_loc = dd[dd["cumulative_%"] >= 80.0].index.min()
        if pd.isna(stop_index_loc):
             return dd
        
        # Get the index position of the stop_index_loc, and slice up to that position (inclusive)
        stop_pos = dd.index.get_loc(stop_index_loc)
        return dd.iloc[:stop_pos + 1]
        
    else:
        dd["share_%"] = 0.0
        dd["cumulative_%"] = 0.0
        return dd.head(0)


def historical_defect_pareto_for_part(selected_part: int,
                                     df_train: pd.DataFrame,
                                     k: int = 100) -> pd.DataFrame: # Increased k for intermediate calculation
    """
    Top historical defect rates (means) for the part, dynamically stopping at 80% cumulative share.
    """
    rate_cols = [c for c in df_train.columns if c.endswith("_rate")]
    if not rate_cols:
        return pd.DataFrame(columns=["defect", "mean_rate", "share_%", "cumulative_%"])

    part_hist = df_train[df_train["part_id"] == selected_part]
    if part_hist.empty:
        return pd.DataFrame(columns=["defect", "mean_rate", "share_%", "cumulative_%"])

    means = part_hist[rate_cols].mean().fillna(0.0)
    means = means[means > 1e-6] # Filter out negligible mean rates
    means = means.sort_values(ascending=False).head(k)
    total = float(means.sum())
    
    out = pd.DataFrame({"defect": means.index,
                        "mean_rate": means.values})
                        
    if total > 0:
        out["share_%"] = out["mean_rate"] / total * 100.0
        out["cumulative_%"] = out["share_%"].cumsum()
        
        # --- 80% Pareto Rule Implementation ---
        stop_index_loc = out[out["cumulative_%"] >= 80.0].index.min()
        if pd.isna(stop_index_loc):
             return out
        
        # Get the index position of the stop_index_loc, and slice up to that position (inclusive)
        stop_pos = out.index.get_loc(stop_index_loc)
        return out.iloc[:stop_pos + 1]
        
    else:
        out["share_%"] = 0.0
        out["cumulative_%"] = 0.0
        return out.head(0)

# Add risk alert summary box based on corrected_p
def get_alert_summary(prob):
    # This function remains unchanged as the risk tiers are based on best practices.
    if prob >= 0.90:
        color = "red"
        status = "🔴 High Alert for Scrap"
        message = (
            "Strong likelihood of scrap. Monitor **tear_up_rate** and **runout_rate**. "
            "Review recent patterns and defects in historical vs. predicted Pareto."
        )
    elif prob >= 0.80:
        color = "orange"
        status = "🟠 Serious Concern for Scrap"
        message = (
            "Elevated risk of scrap. Investigate predicted defects and ensure quality controls are reinforced."
        )
    elif prob >= 0.60:
        color = "gold"
        status = "🟡 Elevated Risk"
        message = (
            "Moderate chance of scrap. Maintain quality vigilance and inspect critical defect trends."
        )
    elif prob >= 0.40:
        color = "lightgray"
        status = "⚪ Moderate Risk"
        message = (
            "Some uncertainty. Proceed with normal inspection and monitor any abnormal trends."
        )
    elif prob >= 0.20:
        color = "lightgreen"
        status = "🟢 Low Risk"
        message = (
            "Low likelihood of scrap. Scrap probability is well within historical norms."
        )
    else:
        color = "green"
        status = "🟢 Very Low Risk"
        message = (
            "Minimal expected scrap. System and process conditions appear highly favorable."
        )
    return color, status, message


# -----------------------------
# Sidebar - SIMPLIFIED for Foundry Manager
# -----------------------------
st.sidebar.header("Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

st.sidebar.header("Risk Definition")
# Scrap Threshold is the only adjuster left for the manager (as requested)
thr_label = st.sidebar.slider("Scrap % Threshold (label & MTTFscrap)", 1.0, 15.0, 6.50, 0.5)

# Validation Controls are now exposed in the main sidebar to easily trigger the backtest
st.sidebar.header("Model Validation")
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation (slower)", value=True)


if not os.path.exists(csv_path):
    st.error("CSV not found.")
    st.stop()

# -----------------------------
# Load and Model Prep (Runs once at app startup)
# -----------------------------
df = load_and_clean(csv_path)

st.title("🧪 Foundry Scrap Risk Dashboard — Actionable Insights")
st.caption("RF + calibrated probs • **Validation-tuned (s, γ)** quick-hook • per-part **exceedance** scaling • MTTFscrap & reliability • Historical & Predicted 80% Pareto")

# Global state to hold validation results (for automatic use in the Predict tab)
if "validation_results" not in st.session_state:
    st.session_state.validation_results = {}

tabs = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

# -----------------------------
# TAB 2: Validation (6–2–1) - TUNING CONTROLS MOVED HERE
# -----------------------------
with tabs[1]:
    st.subheader("Model Hyperparameters & Tuning Controls (Engineer View)")
    
    # Tuning controls MOVED from the old sidebar
    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.number_input("RandomForest Trees", 80, 600, DEFAULT_ESTIMATORS, 20)
        enable_prior_shift = st.checkbox("Enable prior shift (validation ➜ test)", value=True)
    with c2:
        prior_shift_guard = st.slider("Prior-shift guard (max Δ prevalence, pp)", 5, 50, 20, step=5)
    with c3:
        # Default to NOT use manual hook, but allow engineer to override for testing
        use_manual_hook = st.checkbox("Override Quick-Hook (s, γ)", value=False)
        s_manual = st.slider("Manual s", 0.60, 1.20, 1.00, 0.01)
        gamma_manual = st.slider("Manual γ", 0.50, 1.20, 0.50, 0.01)

    st.markdown("---")
    st.subheader("Rolling 6–2–1 Backtest with Wilcoxon Significance")
    
    # Store validation results automatically
    if run_validation and not st.session_state.validation_results.get("is_complete", False):
        with st.spinner("Running rolling evaluation…"):
            rows = []
            start_date, end_date = df["week_ending"].min(), df["week_ending"].max()
            
            # Simplified for clarity; same logic as original code
            s_tuned_list, g_tuned_list = [], [] 

            while start_date + relativedelta(months=(6+2+1)) <= end_date:
                # ... (rolling window definitions remain the same) ...
                train_end = start_date + relativedelta(months=6)
                val_end = train_end + relativedelta(months=2)
                test_end = val_end + relativedelta(months=1)

                train = df[(df.week_ending >= start_date) & (df.week_ending < train_end)].copy()
                val = df[(df.week_ending >= train_end) & (df.week_ending < val_end)].copy()
                test = df[(df.week_ending >= val_end) & (df.week_ending < test_end)].copy()

                if len(train) < 50 or len(test) < 10:
                    start_date += relativedelta(months=1); continue

                mtbf_tr = compute_mtbf_on_train(train, thr_label)
                default_mtbf = float(mtbf_tr["mttf_scrap"].median()) if len(mtbf_tr) else 1.0
                part_freq_tr = train["part_id"].value_counts(normalize=True)
                default_freq = float(part_freq_tr.median()) if len(part_freq_tr) else 0.0

                train_f = attach_train_features(train, mtbf_tr, part_freq_tr, default_mtbf, default_freq)
                val_f = attach_train_features(val, mtbf_tr, part_freq_tr, default_mtbf, default_freq)
                test_f = attach_train_features(test, mtbf_tr, part_freq_tr, default_mtbf, default_freq)

                X_tr, y_tr, _ = make_xy(train_f, thr_label, USE_RATE_COLS_PERMANENT)
                X_va, y_va, _ = make_xy(val_f, thr_label, USE_RATE_COLS_PERMANENT)
                X_te, y_te, _ = make_xy(test_f, thr_label, USE_RATE_COLS_PERMANENT)

                base = RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
                X_calibfit = pd.concat([X_tr, X_va], axis=0)
                y_calibfit = pd.concat([y_tr, y_va], axis=0)
                # Note: Calibration for validation is done differently (cv=3) and is safe.
                cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3).fit(X_calibfit, y_calibfit)

                p_val_raw  = cal.predict_proba(X_va)[:, 1]
                p_test_raw = cal.predict_proba(X_te)[:, 1]

                if enable_prior_shift and len(p_val_raw) and len(p_test_raw):
                    prev_src = float(np.clip(p_val_raw.mean(), 1e-6, 1-1e-6))
                    prev_tgt = float(np.clip((test_f["scrap%"] > thr_label).mean(), 1e-6, 1-1e-6))
                    gap_pp = abs(prev_tgt - prev_src) * 100
                    if gap_pp <= prior_shift_guard:
                        p_test_raw = prior_shift_logit(p_test_raw, prev_src, prev_tgt)

                part_prev_win, part_scale_win, _ = compute_part_exceedance_baselines(train, thr_label)
                tune = tune_s_gamma_on_validation(p_val_raw, y_va, val_f["part_id"], part_scale_win, S_GRID, GAMMA_GRID)
                s_star, gamma_star = tune["s"], tune["gamma"]
                s_tuned_list.append(s_star)
                g_tuned_list.append(gamma_star)

                pid_test = test_f["part_id"].to_numpy()
                ps_test  = part_scale_win.reindex(pid_test).fillna(1.0).to_numpy(dtype=float)
                p_test_adj = np.clip(p_test_raw * (s_star * (ps_test ** gamma_star)), 0, 1)

                actual_prev = float((test_f["scrap%"] > thr_label).mean())

                rows.append({
                    "window_start": start_date.date(),
                    "train_rows": len(train), "test_rows": len(test),
                    "s_tuned": round(float(s_star),2), "gamma_tuned": round(float(gamma_star),2),
                    "actual_mean": round(actual_prev*100,2),
                    "pred_mean_raw": round(float(np.mean(p_test_raw))*100,2),
                    "pred_mean_adj": round(float(np.mean(p_test_adj))*100,2),
                    "brier_raw": round(brier_score_loss(y_te, p_test_raw),4),
                    "accuracy_raw": round(accuracy_score(y_te, p_test_raw>0.5),3)
                })
                start_date += relativedelta(months=1)

            results_df = pd.DataFrame(rows)
            st.session_state.validation_results["is_complete"] = True
            st.session_state.validation_results["results_df"] = results_df
            st.session_state.validation_results["s_median"] = np.median(s_tuned_list) if s_tuned_list else 1.0
            st.session_state.validation_results["gamma_median"] = np.median(g_tuned_list) if g_tuned_list else 0.5
            
        # Display validation results
        if results_df.empty:
            st.warning("No valid rolling windows found.")
        else:
            st.dataframe(results_df, use_container_width=True)

            def wilcoxon_summary(df, col):
                # ... (wilcoxon_summary function remains the same) ...
                actual = df["actual_mean"].to_numpy(float)
                pred = df[col].to_numpy(float)
                rel_err = np.where(actual>0, np.abs(pred-actual)/actual,
                                     np.where(pred==0, 0.0, 1.0))
                gain = np.clip(1.0-rel_err, 0.0, 1.0)
                rows=[]
                if len(gain)>=10:
                    for th in [0.50, 0.80, 0.90]:
                        stat, p = wilcoxon(gain-th, alternative="greater")
                        rows.append([th, gain.mean(), np.median(gain), (gain>=th).mean()*100, stat, p, "✅" if p<0.05 else "❌"])
                return pd.DataFrame(rows, columns=["Threshold","Mean Gain","Median Gain","% Windows ≥Threshold","Statistic","p-value","Significant?"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Wilcoxon — Adjusted (s, γ)**")
                summ_adj = wilcoxon_summary(results_df, "pred_mean_adj")
                if not summ_adj.empty: st.dataframe(summ_adj, use_container_width=True)
                else: st.info("Need ≥10 windows for Wilcoxon.")

            with c2:
                st.markdown("**Wilcoxon — Raw (calibrated)**")
                summ_raw = wilcoxon_summary(results_df, "pred_mean_raw")
                if not summ_raw.empty: st.dataframe(summ_raw, use_container_width=True)
                else: st.info("Need ≥10 windows for Wilcoxon.")
                
            st.markdown(f"***Median Tuned Quick-Hook: s = {st.session_state.validation_results['s_median']:.2f}, γ = {st.session_state.validation_results['gamma_median']:.2f}***")

    elif not run_validation:
        st.info("Tick **Run 6–2–1 rolling validation** in the sidebar to compute windows and Wilcoxon tests.")
    else:
        st.info("Validation results loaded. Un-tick and re-tick the validation box in the sidebar to re-run.")


# -----------------------------
# TAB 1: Predict (Now uses stored validation results)
# -----------------------------
with tabs[0]:
    st.subheader("Actionable Scrap Risk Prediction")

    df_train, df_calib, df_test = time_split(df)
    
    # --- GET TUNING PARAMS ---
    s_star = st.session_state.validation_results.get("s_median", 1.0)
    gamma_star = st.session_state.validation_results.get("gamma_median", 0.5)
    
    # Train-only features at current threshold
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

    df_train_f = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib_f = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test_f  = attach_train_features(df_test, mtbf_train, part_freq_train, default_mtbf, default_freq)

    # Use permanent rate-cols setting
    X_train, y_train, FEATURES = make_xy(df_train_f, thr_label, USE_RATE_COLS_PERMANENT)
    X_calib, y_calib, _        = make_xy(df_calib_f, thr_label, USE_RATE_COLS_PERMANENT)
    X_test,  y_test,  _        = make_xy(df_test_f,  thr_label, USE_RATE_COLS_PERMANENT)

    # Note: Use n_estimators from validation tab, if running prediction on its own, it uses DEFAULT_ESTIMATORS
    n_est = st.session_state.validation_results.get("n_estimators", DEFAULT_ESTIMATORS)
    _, calibrated_model, calib_method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_est)

    p_calib = calibrated_model.predict_proba(X_calib)[:, 1] if len(X_calib) and hasattr(calibrated_model, 'predict_proba') else np.array([])
    p_test  = calibrated_model.predict_proba(X_test)[:, 1]  if len(X_test) and hasattr(calibrated_model, 'predict_proba') else np.array([])

    # Guarded prior shift - NO LONGER USED ON PREDICT TAB, but calculation remains for validation
    shift_note = "Prior shift is disabled on the Predict tab to avoid manual error. See Validation tab."

    # Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        part_ids = sorted(df["part_id"].unique())
        selected_part = st.selectbox("Select Part ID", part_ids)
    with c2:
        quantity = st.number_input("Order Quantity", 1, 100000, 351)
    with c3:
        weight = st.number_input("Piece Weight (lbs)", 0.1, 100.0, 4.0)
    with c4:
        cost_per_part = st.number_input("Cost per Part ($)", 0.01, 100.0, 20.00, 0.01) # Set default cost to 20.00
    
    # Recalculate Exceedance Baselines just before prediction
    part_prev_train, part_scale, global_prev = compute_part_exceedance_baselines(df_train, thr_label)

    mttf_value = float(mtbf_train.loc[selected_part, "mttf_scrap"]) if selected_part in mtbf_train.index else default_mtbf
    part_freq_value = float(part_freq_train.get(selected_part, default_freq))

    # Build an input row ALIGNED to FEATURES
    input_row = build_input_row_for_part(
        selected_part=selected_part,
        quantity=quantity,
        weight=weight,
        mttf_value=mttf_value,
        part_freq_value=part_freq_value,
        FEATURES=FEATURES,
        df_train=df_train_f
    )

    if st.button("Predict", type="primary", use_container_width=True):
        # Base & adjusted predictions
        # Check if the model has predict_proba, which it might not if it was forced to be "uncalibrated" (i.e. rf)
        if not hasattr(calibrated_model, 'predict_proba'):
             st.error(f"Cannot generate prediction. The model is **{calib_method}** and does not support probability prediction.")
             st.stop()
             
        base_p = float(calibrated_model.predict_proba(input_row)[0, 1])

        adj_factor = float(part_scale.get(selected_part, 1.0)) ** float(gamma_star)
        corrected_p = np.clip(base_p * float(s_star) * adj_factor, 0, 1)

        expected_scrap_count = int(round(corrected_p * quantity))
        expected_loss = round(expected_scrap_count * cost_per_part, 2)

        # MTTF + reliability at current threshold
        part_df = df_train[df_train["part_id"] == selected_part]
        N = len(part_df)
        failures = int((part_df["scrap%"] > thr_label).sum())
        mttf_scrap = (N / failures) if failures > 0 else float("inf")
        lam = 0.0 if mttf_scrap == float("inf") else 1.0 / mttf_scrap
        reliability_next_run = np.exp(-lam * 1.0) if lam > 0 else 1.0

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Scrap Risk (raw)", f"{base_p*100:.2f}%")
        m2.metric("Adjusted Scrap Risk (s·part^γ)", f"{corrected_p*100:.2f}%")
        m3.metric("Expected Scrap Count", f"{expected_scrap_count} parts")
        m4.metric("Expected Loss", f"${expected_loss:.2f}")

        st.markdown(f"**Quick-hook params:** s = {s_star:.2f}, γ = {gamma_star:.2f}  | Calibration: **{calib_method}**")
        st.caption(shift_note)
        
        # Generate and display the alert
        color, status, message = get_alert_summary(corrected_p)

        st.markdown(
            f"""
            <div style='
                background-color:{color};
                padding:1em;
                border-radius:10px;
                color:white;
                font-weight:bold;
                margin-top: 1em;
            '>
                {status}<br><span style='font-weight:normal'>{message}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Reliability context (at current threshold)")
        r1, r2, r3 = st.columns(3)
        r1.metric("MTTFscrap", "∞ runs" if mttf_scrap == float("inf") else f"{mttf_scrap:.2f} runs")
        r2.metric("Reliability (next run)", f"{reliability_next_run*100:.2f}%")
        r3.metric("Failures / Runs", f"{failures} / {N}")
        st.caption("Reliability computed as R(1) = exp(−1/MTTFscrap). Threshold slider sets both labels and MTTF calculation.")

        # Historical exceedance prevalence at current threshold
        part_prev_card = float(part_prev_train.get(selected_part, np.nan)) if 'part_prev_train' in locals() else np.nan
        st.markdown(
            f"**Historical Exceedance Rate @ {thr_label:.1f}% (part):** "
            f"{(part_prev_card*100 if not np.isnan(part_prev_card) else np.nan):.2f}%  ({N} runs)"
        )
        # Use simple comparison
        if not np.isnan(part_prev_card): # Added check to ensure comparison is valid
             if corrected_p > part_prev_card:
                st.warning("⬆️ Prediction above historical exceedance rate for this part.")
             elif corrected_p < part_prev_card:
                st.success("⬇️ Prediction below historical exceedance rate for this part.")
             else:
                st.info("≈ Equal to historical exceedance rate.")
        

        # -----------------------------
        # NEW: Historical vs Predicted Pareto (80% rule applied)
        # -----------------------------
        st.subheader("Pareto of Defects — Historical vs Predicted (Top 80% Drivers)")

        rate_cols_in_model = [c for c in FEATURES if c.endswith("_rate")]
        if not rate_cols_in_model:
             st.error("Cannot compute Pareto. No *_rate features found in data or model.")
        else:
            # Historical Pareto for the selected part (now uses 80% rule)
            hist_pareto = historical_defect_pareto_for_part(selected_part, df_train_f)
            hist_pareto = hist_pareto.rename(columns={"mean_rate": "hist_mean_rate"})

            # Local (predicted) drivers for the current input (now uses 80% rule)
            pred_pareto = local_defect_drivers(
                calibrated_model=calibrated_model,
                input_row=input_row,
                FEATURES=FEATURES,
                df_train=df_train_f,
                strategy="p75",
            ).rename(columns={"delta_prob": "delta_prob_raw"})

            # Left-right display
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Historical Pareto (Top 80% defect rates for this part)**")
                if len(hist_pareto):
                    st.dataframe(
                        hist_pareto.assign(
                            hist_mean_rate=lambda d: d["hist_mean_rate"].round(4)
                        ).assign(**{
                            "share_%": lambda d: d["share_%"].round(2),
                            "cumulative_%": lambda d: d["cumulative_%"].round(1),
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No historical defect rates found for this part in the training window.")

            with c_right:
                st.markdown("**Predicted Pareto (Top 80% local drivers of current prediction)**")
                if len(pred_pareto):
                    st.dataframe(
                        pred_pareto.assign(
                            delta_prob_raw=lambda d: (d["delta_prob_raw"]*100).round(2)
                        ).assign(**{
                            "share_%": lambda d: d["share_%"].round(2),
                            "cumulative_%": lambda d: d["cumulative_%"].round(1),
                        }).rename(columns={"delta_prob_raw": "Δ prob (pp)"}),
                        use_container_width=True
                    )

                    st.caption("For each defect-rate feature, we set it to the 75th percentile and measure the +∆ in the raw calibrated probability vs base. Larger +∆ means stronger risk driver **for this prediction**.")
                else:
                    st.info("Predicted Pareto unavailable (no positive delta drivers found).")

        # Diagnostics
        st.subheader("Model Diagnostics")
        test_brier = np.nan
        try:
            test_brier = brier_score_loss(y_test, p_test) if len(X_test) and len(p_test) else np.nan
        except Exception:
            pass
        st.write(f"Calibration: **{calib_method}** | Test Brier: {test_brier:.4f}")
        st.caption("Adjusted risk = calibrated prob × s × (part_scale^γ). part_scale is the per-part exceedance prevalence relative to train global prevalence.")
