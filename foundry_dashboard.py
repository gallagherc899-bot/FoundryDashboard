import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 🏭 Aluminum Foundry Scrap Analytics Dashboard — v8.3
# Enhanced ML-PHM model with Campbell process integration
# ✅ Fix: week_ending column now detected before any reference
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")

    # --- Normalize all column names ---
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # --- Identify and rename the date column before referencing it ---
    possible_date_cols = ["week_ending", "weekending", "week_end", "weekendingdate", "week", "date"]
    found_date = [c for c in df.columns if any(p == c for p in possible_date_cols)]

    if found_date:
        df.rename(columns={found_date[0]: "week_ending"}, inplace=True)
    else:
        st.error(f"❌ No valid date column found. Columns detected: {list(df.columns)}")
        st.stop()

    # --- Rename other important fields ---
    rename_map = {
        "part_id": "part_id",
        "partid": "part_id",
        "scrap%": "scrap_percent",
        "scrap": "scrap_percent",
        "order_quantity": "order_quantity",
    }
    df.rename(columns=rename_map, inplace=True)

    # --- Convert and clean key columns ---
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df.get("scrap_percent", 0), errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df.get("order_quantity", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    # --- Detect all defect rate columns ---
    defect_cols = [c for c in df.columns if c.endswith("_rate")]

    return df, defect_cols

# Load data
df, defect_cols = load_data()

# =========================================================
# Campbell Process Mapping
# =========================================================
campbell_mapping = {
    "Sand System and Preparation": ["sand_rate", "dirty_pattern_rate", "crush_rate", "runout_rate", "gas_porosity_rate"],
    "Core Making": ["core_rate", "gas_porosity_rate", "shrink_porosity_rate", "crush_rate"],
    "Pattern Making and Maintenance": ["shift_rate", "bent_rate", "dirty_pattern_rate"],
    "Mold Making and Assembly": ["shift_rate", "runout_rate", "missrun_rate", "short_pour_rate", "gas_porosity_rate"],
    "Melting and Alloy Treatment": ["dross_rate", "gas_porosity_rate", "shrink_rate", "shrink_porosity_rate", "gouged_rate"],
    "Pouring and Mold Filling": ["missrun_rate", "short_pour_rate", "dross_rate", "tear_up_rate"],
    "Solidification and Cooling": ["shrink_rate", "shrink_porosity_rate", "gas_porosity_rate", "missrun_rate"],
    "Shakeout and Cleaning": ["tear_up_rate", "over_grind_rate", "sand_rate"],
    "Inspection and Finishing": ["failed_zyglo_rate", "zyglo_rate", "outside_process_scrap_rate"],
}

# =========================================================
# Helper functions
# =========================================================
def rolling_splits(df, weeks_train=6, weeks_val=2, weeks_test=1):
    weeks = sorted(df["week_ending"].unique())
    total = len(weeks) - (weeks_train + weeks_val + weeks_test) + 1
    for i in range(max(0, total)):
        yield (
            df[df["week_ending"].isin(weeks[i:i+weeks_train])].copy(),
            df[df["week_ending"].isin(weeks[i+weeks_train:i+weeks_train+weeks_val])].copy(),
            df[df["week_ending"].isin(weeks[i+weeks_train+weeks_val:i+weeks_train+weeks_val+weeks_test])].copy(),
        )

def train_and_evaluate(df_part, threshold):
    features = defect_cols
    y_global = (df_part["scrap_percent"] > threshold).astype(int)
    pos, neg = y_global.sum(), len(y_global) - y_global.sum()
    pct_above = 100 * pos / len(y_global)
    avg_scrap = df_part["scrap_percent"].mean()

    if pos == 0 or neg == 0:
        st.info(f"Average scrap% = {avg_scrap:.2f}%. All runs below {threshold:.2f}%. 100% yield achieved.")
        return pd.DataFrame(), None

    st.info(f"{pct_above:.1f}% of runs exceed {threshold:.2f}%, average scrap% = {avg_scrap:.2f}%.")

    results, rf = [], None
    for train, val, test in rolling_splits(df_part):
        for d in [train, val, test]:
            d["Label"] = (d["scrap_percent"] > threshold).astype(int)

        X_train, y_train = train[features], train["Label"]
        X_val, y_val = val[features], val["Label"]
        X_test, y_test = test[features], test["Label"]

        rf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        try:
            cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
            cal.fit(X_val, y_val)
            model = cal
        except ValueError:
            model = rf

        probs = model.predict_proba(X_test)
        probs = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(X_test))
        preds = (probs > 0.5).astype(int)

        results.append({
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "brier": brier_score_loss(y_test, probs),
        })

    if not results or rf is None:
        st.warning("⚠️ Insufficient historical data for rolling validation — using summary statistics only.")
        return pd.DataFrame(), None

    return pd.DataFrame(results), rf

# =========================================================
# UI Sidebar
# =========================================================
with st.sidebar:
    st.header("🔧 Manager Input Controls")
    part_id = st.text_input("Enter Part ID (matches 'Part ID' column exactly)")
    order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    weight = st.number_input("Piece Weight (lbs)", min_value=0.0, value=10.0)
    cost = st.number_input("Cost per Part ($)", min_value=0.0, value=50.0)
    threshold = st.slider("Scrap% Threshold", 0.0, 5.0, 2.5, 0.5)
    predict = st.button("🔮 Predict")

    if part_id:
        df_check = df[df["part_id"].astype(str).str.lower() == str(part_id).lower()]
        weeks = df_check["week_ending"].nunique()
        if weeks >= 9:
            st.success(f"🧩 Data coverage: {weeks} weeks (sufficient for ML training)")
        else:
            st.warning(f"⚠️ Data coverage: {weeks} weeks only — ML insights may be limited")

tab1, tab2 = st.tabs(["📈 Dashboard", "📊 Validation (6–2–1)"])

# =========================================================
# Prediction + ML-PHM Campbell Logic
# =========================================================
if predict:
    with st.spinner("⏳ Training enhanced ML-PHM model..."):
        df_part = df[df["part_id"].astype(str).str.lower() == str(part_id).lower()]
        if df_part.empty:
            st.warning(f"No data found for Part ID '{part_id}'. Using full dataset.")
            df_part = df.copy()

        results, model = train_and_evaluate(df_part, threshold)

        if model is not None and not results.empty:
            scrap_pred = model.predict_proba(df_part[defect_cols])
            scrap_pred = scrap_pred[:, 1] if scrap_pred.shape[1] > 1 else np.zeros(len(df_part))
            scrap_pred = scrap_pred.mean() * 100
            expected_scrap = order_qty * (scrap_pred / 100)
            loss = expected_scrap * cost
            mtts = (len(df_part) / (expected_scrap + 1)) * 7

            pareto_hist = df_part[defect_cols].mean().sort_values(ascending=False)
            pareto_pred = pd.Series(model.feature_importances_, index=defect_cols).sort_values(ascending=False)

            process_influence = {}
            for process, defects in campbell_mapping.items():
                valid = [d for d in defects if d in pareto_pred.index]
                if valid:
                    process_influence[process] = pareto_pred[valid].sum()

            process_df = pd.DataFrame.from_dict(process_influence, orient="index", columns=["Importance"])
            process_df["Influence_%"] = (process_df["Importance"] / process_df["Importance"].sum()) * 100
            process_df = process_df.sort_values("Influence_%", ascending=False)

            spc_top = pareto_hist.head(3).index.tolist()
            ml_top_process = process_df.head(3).index.tolist()
            overlap = {proc for proc, defs in campbell_mapping.items() if any(d in spc_top for d in defs)}

            alignment = len(overlap.intersection(ml_top_process)) / max(1, len(ml_top_process))
            if alignment >= 0.5:
                rec_text = (
                    f"🟢 ML-PHM confirms SPC findings — {', '.join(ml_top_process[:2])} are consistent drivers.\n"
                    f"Focus on reinforcing best practices and monitoring these processes."
                )
            else:
                rec_text = (
                    f"⚠️ ML-PHM detects counter-intuitive process signals.\n"
                    f"SPC suggests {', '.join(list(overlap)[:2]) if overlap else 'different areas'}, "
                    f"but ML indicates {', '.join(ml_top_process[:2])} as emerging contributors.\n"
                    f"Investigate and monitor these for root-cause insights."
                )

            st.session_state.update({
                "results": results,
                "pareto_hist": pareto_hist,
                "pareto_pred": pareto_pred,
                "process_df": process_df,
                "rec_text": rec_text,
                "scrap_pred": scrap_pred,
                "loss": loss,
                "mtts": mtts,
                "df_part": df_part,
            })
            st.success("✅ Prediction Complete!")

# =========================================================
# Dashboard Display
# =========================================================
with tab1:
    if "results" in st.session_state:
        df_part = st.session_state.df_part
        hist_scrap = df_part["scrap_percent"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Scrap Risk", f"{st.session_state.scrap_pred:.2f}%")
        col2.metric("Expected Scrap Cost", f"${st.session_state.loss:,.2f}")
        col3.metric("MTTS (days)", f"{st.session_state.mtts:.1f}")
        col4.metric("Historical Mean Scrap%", f"{hist_scrap:.2f}%")

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5, 3))
            st.session_state.pareto_hist.head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Historical Pareto (SPC Observed)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        with colB:
            fig, ax = plt.subplots(figsize=(5, 3))
            st.session_state.pareto_pred.head(15).plot(kind="bar", ax=ax, color="seagreen")
            ax.set_title("Predicted Pareto (Enhanced ML-PHM)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        st.subheader("📊 Enhanced Defect–Process Influence (ML-PHM Analysis)")
        st.dataframe(st.session_state.process_df.style.format({"Influence_%": "{:.2f}"}))
        st.info(st.session_state.rec_text)

with tab2:
    if "results" in st.session_state:
        val_df = st.session_state.results
        st.dataframe(val_df.describe().T.style.format("{:.3f}"))

        fig, ax = plt.subplots(figsize=(6, 3))
        val_df[["accuracy", "recall", "precision", "f1"]].plot(ax=ax)
        ax.set_title("Rolling Validation Performance (Enhanced ML-PHM)")
        ax.set_xlabel("Rolling Window #")
        ax.set_ylabel("Score")
        st.pyplot(fig)

st.caption("© 2025 Foundry Analytics | Enhanced Campbell Logic (ML-PHM Framework v8.3)")
