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

    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["scrap_percent"] = pd.to_numeric(df.get("scrap_percent", df.get("scrap", 0)), errors="coerce").fillna(0.0)
    df["order_quantity"] = pd.to_numeric(df.get("order_quantity", 0), errors="coerce").fillna(0.0)
    df["piece_weight_lbs"] = pd.to_numeric(df.get("piece_weight_lbs", df.get("piece_weight", 0)), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["week_ending"]).reset_index(drop=True)
    defect_cols = [c for c in df.columns if c.endswith("_rate")]
    return df, defect_cols


df, defect_cols = load_data()

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
    if len(np.unique(y)) < 2:
        probs = np.full_like(y, np.mean(y), dtype=float)
        return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "brier": 0.0}
    else:
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
    influence = {}
    for process, defects in mapping.items():
        valid = [d for d in defects if d in pareto.index]
        if valid:
            influence[process] = pareto[valid].sum()
    process_df = pd.DataFrame.from_dict(influence, orient="index", columns=["Importance"])
    process_df["Influence_%"] = (process_df["Importance"] / process_df["Importance"].sum()) * 100
    return process_df.sort_values("Influence_%", ascending=False)


def find_similar_parts(df, target_part, scrap_mean, weight_mean, defect_cols, scrap_tol, weight_tol):
    df_avg = df.groupby("part_id").agg({"scrap_percent": "mean", "piece_weight_lbs": "mean"}).reset_index()
    df_avg["scrap_diff"] = abs(df_avg["scrap_percent"] - scrap_mean)
    df_avg["weight_diff"] = abs(df_avg["piece_weight_lbs"] - weight_mean)
    similar = df_avg[
        (df_avg["scrap_diff"] <= scrap_tol)
        & (df_avg["weight_diff"] <= weight_tol * weight_mean / 100)
        & (df_avg["part_id"] != target_part)
    ]

    if similar.empty:
        return pd.DataFrame(), []
    part_ids = similar["part_id"].tolist()
    shared_defects = []
    for pid in part_ids:
        defects_target = [d for d in defect_cols if df[df["part_id"] == target_part][d].mean() > 0]
        defects_other = [d for d in defect_cols if df[df["part_id"] == pid][d].mean() > 0]
        overlap = list(set(defects_target) & set(defects_other))
        if overlap:
            shared_defects.extend(overlap)
    return df[df["part_id"].isin(part_ids)], list(set(shared_defects))


def expand_similarity_context(df, target_part, defect_cols):
    target_df = df[df["part_id"] == target_part]
    scrap_mean = target_df["scrap_percent"].mean()
    weight_mean = target_df["piece_weight_lbs"].mean()

    scrap_tol, weight_tol = 1.0, 10.0
    total_records, matched_parts, shared_defects = 0, [], []
    while scrap_tol <= 3.5 and total_records < 20:
        similar_df, shared_defects = find_similar_parts(df, target_part, scrap_mean, weight_mean, defect_cols, scrap_tol, weight_tol)
        total_records = len(similar_df)
        if total_records >= 20:
            matched_parts = similar_df["part_id"].unique().tolist()
            break
        scrap_tol += 0.5
        weight_tol += 5.0
    if total_records == 0:
        return target_df, "⚠️ No similar parts found.", "Low", 0, []
    confidence = "High" if scrap_tol <= 1.5 else "Medium" if scrap_tol <= 3.0 else "Low"
    merged = pd.concat([target_df, similar_df], axis=0).reset_index(drop=True)
    return merged, confidence, scrap_tol, len(merged), matched_parts, shared_defects


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
    df["Label"] = (df["scrap_percent"] > threshold).astype(int)
    df_part = df[df["part_id"].astype(str).str.lower() == str(part_id).lower()]

    if df_part.empty:
        st.warning(f"No data found for Part ID '{part_id}'. Using SPC averages.")
        df_part = df.copy()

    X_part, y_part = df_part[defect_cols], df_part["Label"]
    local_context_msg = ""
    if len(np.unique(y_part)) < 2 or len(df_part) < 10:
        merged_df, confidence, scrap_tol, total_records, matched_parts, shared_defects = expand_similarity_context(df, df_part["part_id"].iloc[0], defect_cols)
        if total_records > 10:
            local_context_msg = (
                f"🔍 Part {part_id} lacked variation — merged with Parts {', '.join(map(str, matched_parts))} "
                f"(shared: {', '.join(shared_defects[:3])}) ±{scrap_tol:.1f}% scrap window. "
                f"Confidence: {confidence} | Data used: {total_records} samples from {len(matched_parts)+1} parts."
            )
            df_part = merged_df
        else:
            st.warning("⚠️ Insufficient data even after similarity search. Using SPC mean only.")
    else:
        local_context_msg = "✅ Sufficient data available for direct ML-PHM training."

    X_local, y_local = df_part[defect_cols], df_part["Label"]
    model = train_rf_model(X_local, y_local)
    metrics = compute_metrics(model, X_local, y_local)
    pareto_pred = model_feature_importances(model, defect_cols).sort_values(ascending=False)
    process_df = process_importance_from_features(pareto_pred, campbell_mapping)

    st.session_state.update({
        "metrics": metrics,
        "pareto_pred": pareto_pred,
        "process_df": process_df,
        "context_msg": local_context_msg,
    })
    st.success("✅ Prediction Complete!")

with tab1:
    if "pareto_pred" in st.session_state:
        fig, ax = plt.subplots(figsize=(6, 3))
        st.session_state.pareto_pred.head(15).plot(kind="bar", ax=ax, color="seagreen")
        ax.set_title("Predicted Pareto (Enhanced ML-PHM)")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)

        st.subheader("📊 Defect–Process Influence (Campbell PHM)")
        st.dataframe(st.session_state.process_df.style.format({"Influence_%": "{:.2f}"}))
        st.caption(st.session_state.context_msg)

with tab2:
    if "metrics" in st.session_state:
        st.write(pd.DataFrame([st.session_state.metrics]).T.rename(columns={0: "Score"}))
        st.caption("Global ML-PHM model metrics (v8.6)")
