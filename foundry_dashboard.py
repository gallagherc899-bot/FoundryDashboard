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
    page_title="Foundry Scrap Risk Dashboard — Validated Quick-Hook (+MTTF & Exceedance)",
    layout="wide"
)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 2

S_GRID = np.linspace(0.6, 1.2, 13)       # {0.60,...,1.20}
GAMMA_GRID = np.linspace(0.5, 1.2, 15)   # {0.50,...,1.20}
TOP_K_PARETO = 8

# -----------------------------
# Helpers
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
    method = "isotonic" if has_both and len(y_calib) > 500 else "sigmoid"
    try:
        cal = CalibratedClassifierCV(estimator=rf, method=method, cv="prefit").fit(X_calib, y_calib)
    except Exception:
        cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        method = "sigmoid"
    return rf, cal, method

def tune_s_gamma_on_validation(p_val_raw, y_val, part_ids_val, part_scale,
                               s_grid=S_GRID, gamma_grid=GAMMA_GRID):
    """Minimize Brier on validation for p_adj = p_raw * s * (part_scale^gamma)."""
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
                         k: int = TOP_K_PARETO) -> pd.DataFrame:
    """
    Compute a local 'predicted Pareto' by measuring the positive delta in raw calibrated probability
    when bumping each *_rate feature to a higher-risk value (default: 75th percentile).

    Returns a dataframe with columns: ['defect', 'delta_prob', 'share_%', 'cumulative_%'].
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
            # simple +2 percentage points (if features are in [0,1] scale already, this can be tuned)
            tmp[col] = float(input_row[col].iloc[0]) + 0.02

        new_p = float(calibrated_model.predict_proba(tmp)[0, 1])
        delta = max(0.0, new_p - base_p)  # only positive risk drivers for Pareto
        deltas.append((col, delta))

    dd = pd.DataFrame(deltas, columns=["defect", "delta_prob"])
    dd = dd.sort_values("delta_prob", ascending=False).head(k)
    total = float(dd["delta_prob"].sum())
    if total > 0:
        dd["share_%"] = dd["delta_prob"] / total * 100.0
        dd["cumulative_%"] = dd["share_%"].cumsum()
    else:
        dd["share_%"] = 0.0
        dd["cumulative_%"] = 0.0
    return dd

def historical_defect_pareto_for_part(selected_part: int,
                                      df_train: pd.DataFrame,
                                      k: int = TOP_K_PARETO) -> pd.DataFrame:
    """Top-K historical defect rates (means) for the part."""
    rate_cols = [c for c in df_train.columns if c.endswith("_rate")]
    if not rate_cols:
        return pd.DataFrame(columns=["defect", "mean_rate", "share_%", "cumulative_%"])

    part_hist = df_train[df_train["part_id"] == selected_part]
    if part_hist.empty:
        return pd.DataFrame(columns=["defect", "mean_rate", "share_%", "cumulative_%"])

    means = part_hist[rate_cols].mean().fillna(0.0)
    means = means.sort_values(ascending=False).head(k)
    total = float(means.sum())
    out = pd.DataFrame({"defect": means.index,
                        "mean_rate": means.values})
    if total > 0:
        out["share_%"] = out["mean_rate"] / total * 100.0
        out["cumulative_%"] = out["share_%"].cumsum()
    else:
        out["share_%"] = 0.0
        out["cumulative_%"] = 0.0
    return out

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Data & Model")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")
n_estimators = st.sidebar.slider("RandomForest Trees", 80, 600, DEFAULT_ESTIMATORS, 20)

st.sidebar.header("Label & MTTF")
thr_label = st.sidebar.slider("Scrap % Threshold (label & MTTF)", 1.0, 15.0, 5.0, 0.5)

st.sidebar.header("Features & Drift")
use_rate_cols = st.sidebar.checkbox("Include *_rate process features", value=False)
enable_prior_shift = st.sidebar.checkbox("Enable prior shift (validation ➜ test)", value=True)
prior_shift_guard = st.sidebar.slider("Prior-shift guard (max Δ prevalence, pp)", 5, 50, 20, step=5)

st.sidebar.header("Quick-Hook Override")
use_manual_hook = st.sidebar.checkbox("Use manual quick-hook", value=False)
s_manual = st.sidebar.slider("Manual s", 0.60, 1.20, 1.00, 0.01)
gamma_manual = st.sidebar.slider("Manual γ", 0.50, 1.20, 0.50, 0.01)

st.sidebar.header("Validation Controls")
run_validation = st.sidebar.checkbox("Run 6–2–1 rolling validation (slower)", value=False)

if not os.path.exists(csv_path):
    st.error("CSV not found.")
    st.stop()

# -----------------------------
# Load
# -----------------------------
df = load_and_clean(csv_path)

st.title("🧪 Foundry Scrap Risk Dashboard — Validated Quick-Hook (+MTTF & Exceedance)")
st.caption("RF + calibrated probs • tuned (s, γ) quick-hook • optional guarded prior-shift • per-part **exceedance** scaling • rolling 6–2–1 validation with Wilcoxon • MTTFscrap & reliability • Historical & Predicted Pareto")

tabs = st.tabs(["🔮 Predict", "📏 Validation (6–2–1)"])

# -----------------------------
# TAB 1: Predict
# -----------------------------
with tabs[0]:
    st.subheader("Prediction (validation-tuned quick-hook; threshold drives labels & MTTF)")

    df_train, df_calib, df_test = time_split(df)

    # Train-only features at current threshold
    mtbf_train = compute_mtbf_on_train(df_train, thr_label)
    default_mtbf = float(mtbf_train["mttf_scrap"].median()) if len(mtbf_train) else 1.0
    part_freq_train = df_train["part_id"].value_counts(normalize=True)
    default_freq = float(part_freq_train.median()) if len(part_freq_train) else 0.0

    df_train_f = attach_train_features(df_train, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_calib_f = attach_train_features(df_calib, mtbf_train, part_freq_train, default_mtbf, default_freq)
    df_test_f  = attach_train_features(df_test,  mtbf_train, part_freq_train, default_mtbf, default_freq)

    X_train, y_train, FEATURES = make_xy(df_train_f, thr_label, use_rate_cols)
    X_calib, y_calib, _        = make_xy(df_calib_f, thr_label, use_rate_cols)
    X_test,  y_test,  _        = make_xy(df_test_f,  thr_label, use_rate_cols)

    _, calibrated_model, calib_method = train_and_calibrate(X_train, y_train, X_calib, y_calib, n_estimators)

    p_calib = calibrated_model.predict_proba(X_calib)[:, 1] if len(X_calib) else np.array([])
    p_test  = calibrated_model.predict_proba(X_test)[:, 1]  if len(X_test) else np.array([])

    # Guarded prior shift
    shift_note = "Prior shift disabled."
    if enable_prior_shift and len(p_calib) and len(p_test):
        prev_src = float(np.clip(p_calib.mean(), 1e-6, 1-1e-6))
        prev_tgt = float(np.clip((df_test_f["scrap%"] > thr_label).mean(), 1e-6, 1-1e-6))
        gap_pp = abs(prev_tgt - prev_src) * 100
        if gap_pp <= prior_shift_guard:
            p_test = prior_shift_logit(p_test, prev_src, prev_tgt)
            shift_note = f"Prior shift applied (Δ={gap_pp:.1f} pp)."
        else:
            shift_note = f"Prior shift skipped (Δ={gap_pp:.1f} pp too large)."

    # Exceedance baselines & quick-hook tuning
    part_prev_train, part_scale, global_prev = compute_part_exceedance_baselines(df_train, thr_label)
    tune = tune_s_gamma_on_validation(p_calib, y_calib, df_calib_f["part_id"], part_scale) if len(p_calib) else {"s":1.0,"gamma":1.0}
    s_star, gamma_star = float(tune["s"]), float(tune["gamma"])
    if use_manual_hook:
        s_star, gamma_star = float(s_manual), float(gamma_manual)

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
        cost_per_part = st.number_input("Cost per Part ($)", 0.01, 100.0, 0.01)

    mttf_value = float(mtbf_train.loc[selected_part, "mttf_scrap"]) if selected_part in mtbf_train.index else default_mtbf
    part_freq_value = float(part_freq_train.get(selected_part, default_freq))

    # Build an input row ALIGNED to FEATURES, including *_rate columns if in use
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
        if not np.isnan(part_prev_card):
            if corrected_p > part_prev_card:
                st.warning("⬆️ Prediction above historical exceedance rate for this part.")
            elif corrected_p < part_prev_card:
                st.success("⬇️ Prediction below historical exceedance rate for this part.")
            else:
                st.info("≈ Equal to historical exceedance rate.")

        # -----------------------------
        # NEW: Historical vs Predicted Pareto
        # -----------------------------
        st.subheader("Pareto of Defects — Historical vs Predicted (this run)")

        rate_cols_in_model = [c for c in FEATURES if c.endswith("_rate")]
        if not use_rate_cols or not rate_cols_in_model:
            st.info("To view Predicted Pareto, enable **Include *_rate process features** in the sidebar and re-run.")
        else:
            # Historical Pareto for the selected part
            hist_pareto = historical_defect_pareto_for_part(selected_part, df_train_f, k=TOP_K_PARETO)
            hist_pareto = hist_pareto.rename(columns={"mean_rate": "hist_mean_rate"})

            # Local (predicted) drivers for the current input
            pred_pareto = local_defect_drivers(
                calibrated_model=calibrated_model,
                input_row=input_row,
                FEATURES=FEATURES,
                df_train=df_train_f,
                strategy="p75",
                k=TOP_K_PARETO
            ).rename(columns={"delta_prob": "delta_prob_raw"})

            # Left-right display
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Historical Pareto (Top defect rates for this part)**")
                if len(hist_pareto):
                    st.dataframe(hist_pareto.assign(
                        hist_mean_rate=lambda d: d["hist_mean_rate"].round(4),
                        share_%=lambda d: d["share_%"].round(2),
                        cumulative_%=lambda d: d["cumulative_%"].round(1)
                    ), use_container_width=True)
                else:
                    st.info("No historical defect rates found for this part in the training window.")

            with c_right:
                st.markdown("**Predicted Pareto (Local drivers of current prediction)**")
                if len(pred_pareto):
                    st.dataframe(pred_pareto.assign(
                        delta_prob_raw=lambda d: (d["delta_prob_raw"]*100).round(2),  # show in pp
                        share_%=lambda d: d["share_%"].round(2),
                        cumulative_%=lambda d: d["cumulative_%"].round(1)
                    ).rename(columns={"delta_prob_raw": "Δ prob (pp)"}), use_container_width=True)
                    st.caption("For each defect-rate feature, we set it to the 75th percentile and measure the +∆ in the raw calibrated probability vs base. Larger +∆ means stronger risk driver **for this prediction**.")
                else:
                    st.info("Predicted Pareto unavailable (no *_rate features in model or all deltas <= 0).")

    # Diagnostics
    st.subheader("Model Diagnostics")
    test_brier = np.nan
    try:
        test_brier = brier_score_loss(y_test, p_test) if len(X_test) and len(p_test) else np.nan
    except Exception:
        pass
    st.write(f"Calibration: **{calib_method}**, Test Brier: {test_brier:.4f}")
    st.caption("Adjusted risk = calibrated prob × s × (part_scale^γ). part_scale is the per-part exceedance prevalence relative to train global prevalence.")

# -----------------------------
# TAB 2: Validation (6–2–1)
# -----------------------------
with tabs[1]:
    st.subheader("Rolling 6–2–1 Backtest with Wilcoxon Significance")
    if run_validation:
        with st.spinner("Running rolling evaluation…"):
            rows = []
            start_date, end_date = df["week_ending"].min(), df["week_ending"].max()

            while start_date + relativedelta(months=(6+2+1)) <= end_date:
                train_end = start_date + relativedelta(months=6)
                val_end   = train_end + relativedelta(months=2)
                test_end  = val_end   + relativedelta(months=1)

                train = df[(df.week_ending >= start_date) & (df.week_ending < train_end)].copy()
                val   = df[(df.week_ending >= train_end) & (df.week_ending < val_end)].copy()
                test  = df[(df.week_ending >= val_end) & (df.week_ending < test_end)].copy()

                if len(train) < 50 or len(test) < 10:
                    start_date += relativedelta(months=1); continue

                mtbf_tr = compute_mtbf_on_train(train, thr_label)
                default_mtbf = float(mtbf_tr["mttf_scrap"].median()) if len(mtbf_tr) else 1.0
                part_freq_tr = train["part_id"].value_counts(normalize=True)
                default_freq = float(part_freq_tr.median()) if len(part_freq_tr) else 0.0

                train_f = attach_train_features(train, mtbf_tr, part_freq_tr, default_mtbf, default_freq)
                val_f   = attach_train_features(val,   mtbf_tr, part_freq_tr, default_mtbf, default_freq)
                test_f  = attach_train_features(test,  mtbf_tr, part_freq_tr, default_mtbf, default_freq)

                X_tr, y_tr, _ = make_xy(train_f, thr_label, use_rate_cols)
                X_va, y_va, _ = make_xy(val_f,   thr_label, use_rate_cols)
                X_te, y_te, _ = make_xy(test_f,  thr_label, use_rate_cols)

                base = RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
                X_calibfit = pd.concat([X_tr, X_va], axis=0)
                y_calibfit = pd.concat([y_tr, y_va], axis=0)
                cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3).fit(X_calibfit, y_calibfit)

                p_val_raw  = cal.predict_proba(X_va)[:, 1]
                p_test_raw = cal.predict_proba(X_te)[:, 1]

                # Guarded prior shift
                if enable_prior_shift and len(p_val_raw) and len(p_test_raw):
                    prev_src = float(np.clip(p_val_raw.mean(), 1e-6, 1-1e-6))
                    prev_tgt = float(np.clip((test_f["scrap%"] > thr_label).mean(), 1e-6, 1-1e-6))
                    gap_pp = abs(prev_tgt - prev_src) * 100
                    if gap_pp <= prior_shift_guard:
                        p_test_raw = prior_shift_logit(p_test_raw, prev_src, prev_tgt)

                # Exceedance-based scaling + tuning
                part_prev_win, part_scale_win, _ = compute_part_exceedance_baselines(train, thr_label)
                tune = tune_s_gamma_on_validation(p_val_raw, y_va, val_f["part_id"], part_scale_win)
                s_star, gamma_star = tune["s"], tune["gamma"]

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
            if results_df.empty:
                st.warning("No valid rolling windows found.")
            else:
                st.dataframe(results_df, use_container_width=True)

                # Wilcoxon summaries
                def wilcoxon_summary(df, col):
                    actual = df["actual_mean"].to_numpy(float)
                    pred   = df[col].to_numpy(float)
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
                    if not summ_adj.empty:
                        st.dataframe(summ_adj, use_container_width=True)
                    else:
                        st.info("Need ≥10 windows for Wilcoxon.")

                with c2:
                    st.markdown("**Wilcoxon — Raw (calibrated)**")
                    summ_raw = wilcoxon_summary(results_df, "pred_mean_raw")
                    if not summ_raw.empty:
                        st.dataframe(summ_raw, use_container_width=True)
                    else:
                        st.info("Need ≥10 windows for Wilcoxon.")

                # Save CSVs
                try:
                    results_df.to_csv("rolling_window_results.csv", index=False)
                    if not summ_adj.empty: summ_adj.to_csv("rolling_window_threshold_summary_adj.csv", index=False)
                    if not summ_raw.empty: summ_raw.to_csv("rolling_window_threshold_summary_raw.csv", index=False)
                    st.caption("Saved CSVs to working directory.")
                except Exception:
                    pass
    else:
        st.info("Tick **Run 6–2–1 rolling validation** in the sidebar to compute windows and Wilcoxon tests.")
