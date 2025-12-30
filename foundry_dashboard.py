import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@st.cache_data
def load_data():
    df = pd.read_csv("anonymized_parts.csv")

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    date_cols = ["week_ending", "weekending", "week_end", "weekendingdate", "week", "date"]
    found_date = [c for c in df.columns if c in date_cols]
    if found_date:
        df.rename(columns={found_date[0]: "week_ending"}, inplace=True)

    rename_map = {
        "scrap%": "scrap_percent",
        "scrap": "scrap_percent",
        "order_quantity": "order_quantity",
    }
    df.rename(columns=rename_map, inplace=True)

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df.get("scrap_percent", 0), errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df.get("order_quantity", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)

    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    return df, defect_cols


df, defect_cols = load_data()

# Campbell Process Mapping (remains unchanged)
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


def train_rf_model(X, y):
    """Trains a Random Forest + Calibration safely."""
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    try:
        cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
        cal.fit(X, y)
        return cal
    except ValueError:
        return rf


def compute_metrics(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "brier": brier_score_loss(y, probs),
    }


def model_feature_importances(model, defect_cols):
    if isinstance(model, CalibratedClassifierCV):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model
    return pd.Series(base_model.feature_importances_, index=defect_cols)


def process_importance_from_features(pareto, mapping):
    """Aggregate defect feature importances by process group."""
    influence = {}
    for process, defects in mapping.items():
        valid = [d for d in defects if d in pareto.index]
        if valid:
            influence[process] = pareto[valid].sum()
    process_df = pd.DataFrame.from_dict(influence, orient="index", columns=["Importance"])
    process_df["Influence_%"] = (process_df["Importance"] / process_df["Importance"].sum()) * 100
    return process_df.sort_values("Influence_%", ascending=False)


with st.sidebar:
    st.header("🔧 Manager Input Controls")
    part_id = st.text_input("Enter Part ID (matches 'Part ID' column exactly)")
    order_qty = st.number_input("Order Quantity", min_value=1, value=100)
    weight = st.number_input("Piece Weight (lbs)", min_value=0.0, value=10.0)
    cost = st.number_input("Cost per Part ($)", min_value=0.0, value=50.0)
    threshold = st.slider("Scrap% Threshold", 0.0, 5.0, 2.5, 0.5)
    predict = st.button("🔮 Predict")

tab1, tab2 = st.tabs(["📈 Dashboard", "📊 Model Performance"])

if predict:
    with st.spinner("⏳ Training Global + Local ML-PHM models..."):
        df["Label"] = (df["scrap_percent"] > threshold).astype(int)
        X_all, y_all = df[defect_cols], df["Label"]

        # Global model
        global_model = train_rf_model(X_all, y_all)
        global_metrics = compute_metrics(global_model, X_all, y_all)
        global_feat = model_feature_importances(global_model, defect_cols)
        global_proc = process_importance_from_features(global_feat, campbell_mapping)

        # Local model
        df_part = df[df["part_id"].astype(str).str.lower() == str(part_id).lower()]
        if df_part.empty:
            st.warning(f"No data found for Part ID '{part_id}'. Using overall dataset for local approximation.")
            df_part = df.copy()

        X_part, y_part = df_part[defect_cols], df_part["Label"]
        local_model = train_rf_model(X_part, y_part)
        local_metrics = compute_metrics(local_model, X_part, y_part)
        local_feat = model_feature_importances(local_model, defect_cols)
        local_proc = process_importance_from_features(local_feat, campbell_mapping)

        # Compute overlap confidence between models
        merged = pd.concat([global_feat, local_feat], axis=1, keys=["Global", "Local"]).fillna(0)
        overlap = 1 - (np.abs(merged["Global"] - merged["Local"]).sum() / 2)
        confidence = max(0, min(overlap * 100, 100))

        # Determine systemic vs localized recommendation
        top_global = global_proc.head(3).index.tolist()
        top_local = local_proc.head(3).index.tolist()
        common = list(set(top_global) & set(top_local))

        if len(common) >= 2:
            rec_text = (
                f"🟢 ML-PHM confirms SPC and systemic findings — "
                f"{', '.join(common)} remain consistent scrap drivers across global and local datasets. "
                f"(Confidence: {confidence:.1f}%)\n\n"
                f"Focus on reinforcing best practices and monitoring these processes for stability."
            )
        else:
            rec_text = (
                f"🟠 ML-PHM detects localized variation for Part {part_id}. "
                f"Global model emphasizes {', '.join(top_global[:2])}, but local data suggests "
                f"{', '.join(top_local[:2])} as emerging contributors. "
                f"(Confidence: {confidence:.1f}%)\n\n"
                f"Investigate localized mold setup, tooling, or process control for these stages."
            )

        # Store session data
        st.session_state.update({
            "global_proc": global_proc,
            "local_proc": local_proc,
            "global_metrics": global_metrics,
            "local_metrics": local_metrics,
            "rec_text": rec_text,
            "confidence": confidence,
        })
        st.success("✅ Dual-Model Prediction Complete!")


with tab1:
    if "global_proc" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌍 Global Process Influence")
            fig, ax = plt.subplots(figsize=(5, 3))
            st.session_state.global_proc["Influence_%"].plot(kind="bar", ax=ax, color="steelblue")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        with col2:
            st.subheader("🧩 Local (Part-Specific) Process Influence")
            fig, ax = plt.subplots(figsize=(5, 3))
            st.session_state.local_proc["Influence_%"].plot(kind="bar", ax=ax, color="orange")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        st.info(st.session_state.rec_text)
        st.caption(f"Model agreement confidence: {st.session_state.confidence:.1f}%")

with tab2:
    if "global_metrics" in st.session_state:
        st.subheader("📊 Global vs Local Model Performance")
        metrics_df = pd.DataFrame({
            "Global": st.session_state.global_metrics,
            "Local": st.session_state.local_metrics
        })
        st.dataframe(metrics_df.style.format("{:.3f}"))
        st.caption("v8.5 — Dual-Context ML-PHM Framework with Campbell Process Logic")
