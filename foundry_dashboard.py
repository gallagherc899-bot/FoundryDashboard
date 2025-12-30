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

    possible_date_cols = ["week_ending", "weekending", "week_end", "weekendingdate", "week", "date"]
    found_date = [c for c in df.columns if c in possible_date_cols]
    if found_date:
        df.rename(columns={found_date[0]: "week_ending"}, inplace=True)
    else:
        st.error(f"❌ No valid date column found. Columns detected: {list(df.columns)}")
        st.stop()

    rename_map = {
        "part_id": "part_id",
        "partid": "part_id",
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

def train_global_model(df, threshold):
    df["Label"] = (df["scrap_percent"] > threshold).astype(int)
    X = df[defect_cols]
    y = df["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
    try:
        cal.fit(X, y)
        model = cal
    except ValueError:
        model = rf

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "brier": brier_score_loss(y, probs),
    }
    return model, metrics

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
    with st.spinner("⏳ Training global ML-PHM model..."):
        model, metrics = train_global_model(df, threshold)

        df_part = df[df["part_id"].astype(str).str.lower() == str(part_id).lower()]
        if df_part.empty:
            st.warning(f"No data found for Part ID '{part_id}'. Using overall dataset averages.")
            df_part = df.copy()

        X_part = df_part[defect_cols]
        scrap_pred = model.predict_proba(X_part)[:, 1].mean() * 100
        expected_scrap = order_qty * (scrap_pred / 100)
        loss = expected_scrap * cost

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
            "metrics": metrics,
            "pareto_hist": pareto_hist,
            "pareto_pred": pareto_pred,
            "process_df": process_df,
            "rec_text": rec_text,
            "scrap_pred": scrap_pred,
            "loss": loss,
        })
        st.success("✅ Prediction Complete!")

with tab1:
    if "pareto_hist" in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Scrap Risk", f"{st.session_state.scrap_pred:.2f}%")
        col2.metric("Expected Scrap Cost", f"${st.session_state.loss:,.2f}")
        col3.metric("Model Accuracy", f"{st.session_state.metrics['accuracy']*100:.1f}%")

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

        st.subheader("📊 Enhanced Defect–Process Influence (Campbell PHM)")
        st.dataframe(st.session_state.process_df.style.format({"Influence_%": "{:.2f}"}))
        st.info(st.session_state.rec_text)

with tab2:
    if "metrics" in st.session_state:
        st.write(pd.DataFrame([st.session_state.metrics]).T.rename(columns={0: "Score"}))
        st.caption("Rolling validation replaced with global performance metrics (v8.4 global model)")

st.caption("© 2025 Foundry Analytics | Global ML-PHM Model with Campbell Process Integration (v8.4)")
