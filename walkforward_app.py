"""
walkforward_app.py
==================
Standalone walk-forward demonstration for Parts 3, 14, 74 (forward + reverse).

This is a thin entry point that REUSES the functions in foundry_dashboard.py
(load_data, train_three_stage_model, train_global_model, the feature-engineering
chain, etc.) so its numbers are identical to tab10 of the full dashboard. The only
thing this app adds is speed:

  1. The base three-stage model trains ONCE per session  (st.cache_resource).
  2. Per-step retrained models are cached BY EVENT DATE     (st.cache_resource).
     Forward and reverse traversals of the same part use the same pools (pool =
     every foundry run up to that date), so once forward has trained them reverse
     is free. Dates shared across parts train only once for the whole session.
  3. Each part is computed on demand, so first load only pays for the part you
     select — not all three at once.

Deployment: keep foundry_dashboard.py (renamed if needed) in the same folder /
repo as this file, and point Streamlit Cloud's main module at walkforward_app.py.
foundry_dashboard.py is imported as a library here; its main() does not run
(it is guarded by `if __name__ == "__main__"`).

Once the behavior here is confirmed, the cached _train_at_step-by-date pattern can
be folded back into tab10 of the full dashboard to fix its load time.
"""

# IMPORTANT: import the dashboard module FIRST. Its module body calls
# st.set_page_config(...) and injects the page CSS, which must be the first
# Streamlit commands to run. This app must NOT call st.set_page_config itself.
import foundry_dashboard as fd

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------------------------------
# CONFIG (mirrors tab10)
# --------------------------------------------------------------------------
_DIV_THR = 5.0
_PROC = {3: "shift / Core Making", 14: "misrun / Pouring", 74: "gouged / Pattern/Tooling"}
_SIG_COLOR = {"S1": "#1f77b4", "S2": "#c0392b", "S3": "#2ca02c", "S4": "#7b3f99"}

_DEFECT_TO_PROC = {
    "dross_rate": "Melting", "gas_porosity_rate": "Melting",
    "missrun_rate": "Pouring", "misrun_rate": "Pouring",
    "short_pour_rate": "Pouring", "runout_rate": "Pouring",
    "shrink_rate": "Gating Design", "tear_up_rate": "Gating Design",
    "shrink_porosity_rate": "Gating Design",
    "sand_rate": "Sand System", "dirty_pattern_rate": "Sand System",
    "core_rate": "Core Making", "crush_rate": "Core Making",
    "shift_rate": "Core Making",
    "bent_rate": "Shakeout", "gouged_rate": "Pattern/Tooling",
    "over_grind_rate": "Finishing", "cut_into_rate": "Finishing",
    "zyglo_rate": "Inspection", "failed_zyglo_rate": "Inspection",
    "outside_process_scrap_rate": "Inspection",
}


def _signal(rf_p, mpts_p, last_proc, chronic_proc):
    """±5pp divergence rule with chronic-vs-active S4 branch (identical to tab7/tab10)."""
    div = round(rf_p - mpts_p, 1)
    if abs(div) <= _DIV_THR:
        return "S1", div
    if div > _DIV_THR:
        return "S2", div
    if last_proc not in ("—", chronic_proc) and last_proc:
        return "S4", div
    return "S3", div


# --------------------------------------------------------------------------
# CACHED DATA / MODELS
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(path):
    """fd.load_data is itself cached; this wrapper just gives a stable key."""
    df, defect_cols = fd.load_data(path)
    return df, defect_cols


@st.cache_resource(show_spinner="Training base three-stage model (one time)…")
def get_base_model(path):
    df, defect_cols = _load(path)
    return fd.train_three_stage_model(df, defect_cols)


@st.cache_resource(show_spinner=False)
def train_at_date(path, date_iso, train_thr):
    """Retrain the single-stage global model on the cumulative foundry pool up to
    `date_iso`. Cached by date, so identical pools (forward/reverse, shared dates
    across parts) train only once. Returns the model dict or None on failure
    (pool too small / class imbalance) — caller falls back to the base model."""
    df, defect_cols = _load(path)
    ds = df.sort_values("week_ending").reset_index(drop=True)
    pool = ds[ds["week_ending"] <= pd.Timestamp(date_iso)].copy()
    try:
        gm = fd.train_global_model(pool, train_thr, defect_cols)
        if "cal_model" not in gm or "features" not in gm:
            return None
        return gm
    except Exception:
        return None


def part_event_dates(path, part_id):
    """Ordered unique event dates for a part (used to pre-warm the date cache)."""
    df, _ = _load(path)
    pdat = df[df["part_id"] == str(part_id)].sort_values("week_ending")
    seen, out = set(), []
    for d in pdat["week_ending"].tolist():
        key = pd.Timestamp(d).isoformat()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# --------------------------------------------------------------------------
# WALK-FORWARD  (verbatim port of tab10 _build_walkforward, using the date cache)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_walkforward(path, part_id, direction):
    df, defect_cols = _load(path)
    global_model = get_base_model(path)

    part_id = str(part_id)
    pdat = df[df["part_id"] == part_id].sort_values("week_ending").reset_index(drop=True)
    if len(pdat) == 0:
        return None, None

    thr = pdat["scrap_percent"].mean()

    if direction == "reverse":
        pdat = pdat.iloc[::-1].reset_index(drop=True)

    chronic_proc = "—"
    rates = {c: pdat[c].mean() for c in defect_cols if c in pdat.columns}
    if rates and max(rates.values()) > 0:
        chronic_proc = _DEFECT_TO_PROC.get(max(rates, key=rates.get), "—")

    df_sorted = df.sort_values("week_ending").reset_index(drop=True)
    train_thr = global_model["global_threshold"]

    rows = []
    for i in range(len(pdat)):
        event_date = pdat.iloc[i]["week_ending"]
        date_iso = pd.Timestamp(event_date).isoformat()

        pool = df_sorted[df_sorted["week_ending"] <= event_date].copy()
        pool_size = len(pool)

        gm_step = train_at_date(path, date_iso, train_thr)
        if gm_step is None:
            gm_step = global_model
            retrain_status = "fallback (H2 fixed)"
        else:
            retrain_status = "retrained"

        # ---- MPTS (history-to-date for this part) ----
        hist = pdat.iloc[:i + 1]
        fc = int((hist["scrap_percent"] > thr).sum())
        total_parts = hist["order_quantity"].sum()
        avg_oq = total_parts / len(hist)
        mpts_parts = total_parts / fc if fc > 0 else total_parts
        R = np.exp(-avg_oq / mpts_parts) if mpts_parts > 0 else 0
        mpts_p = round(100 * (1 - R), 1)

        # ---- RF: score this event's own row through the retrained model ----
        rf_p = None
        try:
            enh = pool.copy()
            enh = fd.add_multi_defect_features(enh, defect_cols)
            enh = fd.add_temporal_features(enh)
            enh = fd.add_mtts_sequential_features(enh, train_thr)
            event_rows = enh[
                (enh["part_id"] == part_id) &
                (enh["week_ending"] == event_date)
            ]
            if len(event_rows) > 0:
                last_enh = event_rows.iloc[[-1]].copy()
                last_enh = fd.attach_train_features(
                    last_enh,
                    gm_step["scrap_rate_train"], gm_step["part_freq_train"],
                    gm_step["default_scrap_rate"], gm_step["default_freq"],
                )
                mtts_tr = fd.compute_mtts_on_train(gm_step["df_train"], train_thr)
                last_enh = fd.attach_mtts_aggregate_features(last_enh, mtts_tr)
                feat = last_enh.reindex(columns=gm_step["features"], fill_value=0).fillna(0)
                rf_p = round(float(gm_step["cal_model"].predict_proba(feat)[:, 1][0]) * 100, 1)
        except Exception:
            rf_p = None

        # ---- active process on the most recent run ----
        last_proc = "—"
        rr = {c: float(pdat.iloc[i][c]) for c in defect_cols
              if c in pdat.columns and float(pdat.iloc[i][c]) > 0}
        if rr:
            last_proc = _DEFECT_TO_PROC.get(max(rr, key=rr.get), "—")

        if rf_p is not None:
            sig, delta = _signal(rf_p, mpts_p, last_proc, chronic_proc)
        else:
            sig, delta = "", None

        rows.append(dict(
            step=i,
            date=str(pd.Timestamp(event_date).date()) if pd.notna(event_date) else "",
            work_order=pdat.iloc[i].get("work_order", ""),
            order_qty=int(pdat.iloc[i]["order_quantity"]),
            scrap_pct=round(float(pdat.iloc[i]["scrap_percent"]), 3),
            mpts_prob=mpts_p,
            rf_prob=rf_p if rf_p is not None else "",
            delta=delta if delta is not None else "",
            signal=sig,
            pool_size=pool_size,
            retrain_status=retrain_status,
        ))

    return pd.DataFrame(rows), thr


# --------------------------------------------------------------------------
# FIGURE  (verbatim port of tab10 _figure)
# --------------------------------------------------------------------------
def _figure(wf, part_id, thr, proc_label, direction_label):
    steps = wf["step"].tolist()
    cum = wf["scrap_pct"].expanding().mean().round(3).tolist()
    fig = go.Figure()
    mp = wf["mpts_prob"].tolist()
    fig.add_trace(go.Scatter(
        x=steps + steps[::-1],
        y=[m + _DIV_THR for m in mp] + [max(m - _DIV_THR, 0) for m in mp][::-1],
        fill="toself", fillcolor="rgba(200,200,200,0.4)",
        line=dict(color="rgba(0,0,0,0)"), name="±5 pp band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=steps, y=mp, mode="lines+markers",
        name="MPTS P(scrap)", line=dict(color="#1f77b4", width=3), marker=dict(size=8)))
    rf_y = [v if v != "" else None for v in wf["rf_prob"].tolist()]
    fig.add_trace(go.Scatter(x=steps, y=rf_y, mode="lines+markers",
        name="RF P(scrap)", line=dict(color="#000000", width=3), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=steps, y=wf["scrap_pct"].tolist(), mode="lines+markers",
        name="Actual scrap %", yaxis="y2",
        line=dict(color="#c0392b", width=2.5), marker=dict(size=10, symbol="star")))
    fig.add_trace(go.Scatter(x=steps, y=cum, mode="lines+markers",
        name="Cumulative avg scrap %", yaxis="y2",
        line=dict(color="#2ca02c", width=2, dash="dash"), marker=dict(size=6, symbol="diamond")))
    fig.update_layout(
        title=f"Part {part_id} — Walk-Forward {direction_label}  ({proc_label}; threshold {thr:.2f}%)",
        xaxis=dict(title="Walk-Forward Step (each = next production run in traversal order)",
                   tickmode="array", tickvals=steps),
        yaxis=dict(title="P(scrap) %", range=[-5, 105], tickvals=[0, 20, 40, 60, 80, 100]),
        yaxis2=dict(title="Scrap %", overlaying="y", side="right",
                    range=[0, max(max(wf["scrap_pct"]), max(cum)) * 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        height=520, font=dict(size=14, color="black"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# --------------------------------------------------------------------------
# RENDER ONE PART (forward + reverse) with a pre-warm progress bar
# --------------------------------------------------------------------------
def render_part(path, pid):
    st.markdown(f"### Part {pid} — {_PROC[pid]}")

    base = get_base_model(path)
    train_thr = base["global_threshold"]

    # Pre-warm the per-date model cache with a visible progress bar. After this,
    # both forward and reverse build_walkforward calls hit the warm cache.
    dates = part_event_dates(path, pid)
    if not dates:
        st.warning(f"Part {pid}: no records found.")
        return
    prog = st.progress(0.0, text=f"Part {pid}: retraining {len(dates)} per-event models…")
    for k, d in enumerate(dates):
        train_at_date(path, pd.Timestamp(d).isoformat(), train_thr)
        prog.progress((k + 1) / len(dates),
                      text=f"Part {pid}: retraining per-event models… {k + 1}/{len(dates)}")
    prog.empty()

    for _dir, _dlabel in [("forward", "FORWARD (months 1→32)"),
                          ("reverse", "REVERSE (months 32→1)")]:
        wf, thr = build_walkforward(path, pid, _dir)
        if wf is None:
            st.warning(f"Part {pid}: no records found.")
            continue
        fig = _figure(wf, pid, thr, _PROC[pid], _dlabel)
        st.plotly_chart(fig, use_container_width=True)

        counts = wf[wf["signal"] != ""]["signal"].value_counts().to_dict()
        summary = " | ".join(f"{k}={counts[k]}" for k in ("S1", "S2", "S3", "S4") if k in counts)
        st.caption(f"Part {pid} {_dlabel} signals: {summary if summary else 'RF scoring unavailable'}")

        c1, c2 = st.columns(2)
        with c1:
            try:
                png = fig.to_image(format="png", scale=2, width=1500, height=520)
                st.download_button(f"⬇️ PNG — Part {pid} {_dir}", png,
                                   f"Figure_4-WF_Part{pid}_{_dir}.png", "image/png",
                                   key=f"wf_png_{pid}_{_dir}")
            except Exception:
                st.info("PNG export needs `kaleido` (pip install kaleido).")
        with c2:
            st.download_button(f"⬇️ CSV — Part {pid} {_dir}",
                               wf.to_csv(index=False).encode(),
                               f"walkforward_part{pid}_{_dir}.csv", "text/csv",
                               key=f"wf_csv_{pid}_{_dir}")
    st.markdown("---")


# --------------------------------------------------------------------------
# APP BODY
# --------------------------------------------------------------------------
st.title("Walk-Forward Demonstration — Parts 3, 14, 74")
st.caption(
    "Standalone walk-forward harness (reuses the full dashboard's training and "
    "feature code). MPTS P(scrap) and RF P(scrap) are computed from the live "
    "trained model; the S1–S4 signal applies the ±5 pp divergence rule. Each "
    "chart and its per-step data are downloadable."
)
st.caption(
    "Each part is shown in two traversal orders. **Forward (months 1→32)** is the "
    "chronological operational view. **Reverse (months 32→1)** is an order-robustness "
    "sensitivity view — the same foundry runs traversed in the opposite order. The RF "
    "score for each run is identical in both (it reads that run's own feature vector); "
    "only the MPTS accumulation order changes."
)

path = st.text_input("Data file path", value=fd.DEFAULT_CSV_PATH)
up = st.file_uploader("…or upload the dataset CSV", type=["csv"])
if up is not None:
    path = "_uploaded_walkforward.csv"
    with open(path, "wb") as f:
        f.write(up.getbuffer())

df_check, _ = _load(path)
if df_check is None:
    st.error(f"Could not load data from: {path}")
    st.stop()
st.success(f"Loaded {len(df_check):,} records | {df_check['part_id'].nunique()} parts")

# Warm the base model up front (cached for the session).
get_base_model(path)

choice = st.radio("Which part to compute?",
                  ["Part 3", "Part 14", "Part 74", "All three"],
                  horizontal=True, index=0)
st.caption(
    "First compute of a part retrains one model per event date (slow once, then "
    "cached). Reverse reuses forward's models, so it is near-instant. Picking a "
    "single part keeps first load fast; 'All three' computes them in sequence."
)

run = st.button("Run walk-forward", type="primary")
if run:
    pids = {"Part 3": [3], "Part 14": [14], "Part 74": [74],
            "All three": [3, 14, 74]}[choice]
    for pid in pids:
        render_part(path, pid)
else:
    st.info("Choose a part and click **Run walk-forward**.")
