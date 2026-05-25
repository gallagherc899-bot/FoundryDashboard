# ============================================================================
#  WALK-FORWARD DEMONSTRATION — PARTS 3, 14, 74  (H1-validated instruments)
# ============================================================================
#  PASTE THIS BLOCK INSIDE A `with tabN:` SECTION OF THE DASHBOARD, *after*
#  `global_model = train_three_stage_model(df, defect_cols)` has run, so that
#  `global_model`, `df`, and `defect_cols` are all in scope.
#
#  It reuses the dashboard's OWN scoring path (the same code tab7 uses):
#     - MPTS P(scrap):  R(n)=e^(-avg_oq/MPTS), P=100*(1-R)   [Eq 3.1-3.3]
#     - RF  P(scrap):   global_model["cal_model"].predict_proba on the
#                       enhanced+attached feature row  [H2-validated model]
#     - Signal S1-S4:   ±5pp divergence rule + chronic-vs-active process
#                       (identical to tab7 / compute_dual_model_validation_table)
#
#  Walk-forward semantics: for each step i of a part's chronological run
#  sequence, MPTS is recomputed on history-to-date (runs 0..i); the RF score
#  is the model's probability for run i; the signal compares them.
#
#  For each of Parts 3, 14, 74 it renders an inline Plotly figure and offers
#  TWO download buttons: the PNG figure and the per-step CSV — so the artifact
#  is stored and reproducible for future revisions.
# ============================================================================

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.header("Walk-Forward Demonstration — Parts 3, 14, 74")
st.caption(
    "Observed-sequence walk-forward for the three H1-validated monitoring "
    "instruments. MPTS P(scrap) and RF P(scrap) are computed from the live "
    "trained model; the S1–S4 signal applies the ±5 pp divergence rule "
    "(Eq. 3.x). Each chart and its per-step data are downloadable."
)

# ---- shared scoring helpers (mirror tab7) ---------------------------------
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
_DIV_THR = 5.0

_enh_all   = global_model["df_enhanced"]
_feat_cols = global_model["features"]
_cal_model = global_model["cal_model"]
_mtts_tr   = compute_mtts_on_train(global_model["df_train"],
                                   global_model["global_threshold"])

def _rf_prob_for_row(enh_row):
    """Score ONE enhanced run-row through the H2-validated calibrated model."""
    row = attach_train_features(
        enh_row.copy(),
        global_model["scrap_rate_train"], global_model["part_freq_train"],
        global_model["default_scrap_rate"], global_model["default_freq"],
    )
    row = attach_mtts_aggregate_features(row, _mtts_tr)
    feat = row.reindex(columns=_feat_cols, fill_value=0).fillna(0)
    return round(float(_cal_model.predict_proba(feat)[:, 1][0]) * 100, 1)

def _signal(rf_p, mpts_p, last_proc, chronic_proc):
    """±5pp divergence rule with chronic-vs-active S4 branch (tab7 logic)."""
    div = round(rf_p - mpts_p, 1)
    if abs(div) <= _DIV_THR:
        return "S1", div
    if div > _DIV_THR:
        return "S2", div
    if last_proc not in ("—", chronic_proc) and last_proc:
        return "S4", div
    return "S3", div

def _build_walkforward(part_id):
    """Return a per-step DataFrame: step, date, work_order, order_qty,
    scrap_pct, mpts_prob, rf_prob, delta, signal."""
    pdat = df[df["part_id"] == part_id].sort_values("week_ending").reset_index(drop=True)
    if len(pdat) == 0:
        return None, None
    thr = pdat["scrap_percent"].mean()                     # per-part threshold
    penh = _enh_all[_enh_all["part_id"] == part_id]
    if "week_ending" in penh.columns:
        penh = penh.sort_values("week_ending")
    penh = penh.reset_index(drop=True)

    # chronic (MPTS) process = dominant defect over full history
    chronic_proc = "—"
    rates = {c: pdat[c].mean() for c in defect_cols if c in pdat.columns}
    if rates and max(rates.values()) > 0:
        chronic_proc = _DEFECT_TO_PROC.get(max(rates, key=rates.get), "—")

    rows = []
    for i in range(len(pdat)):
        hist = pdat.iloc[:i + 1]
        fc = int((hist["scrap_percent"] > thr).sum())
        total_parts = hist["order_quantity"].sum()
        avg_oq = total_parts / len(hist)
        mpts_parts = total_parts / fc if fc > 0 else total_parts
        R = np.exp(-avg_oq / mpts_parts) if mpts_parts > 0 else 0
        mpts_p = round(100 * (1 - R), 1)

        # RF score for THIS run (enhanced row i), if available
        rf_p = None
        if i < len(penh):
            try:
                rf_p = _rf_prob_for_row(penh.iloc[[i]].copy())
            except Exception:
                rf_p = None

        # active process from this run's defect rates
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
            date=str(pdat.iloc[i]["week_ending"].date())
                 if pd.notna(pdat.iloc[i]["week_ending"]) else "",
            work_order=pdat.iloc[i].get("work_order", ""),
            order_qty=int(pdat.iloc[i]["order_quantity"]),
            scrap_pct=round(float(pdat.iloc[i]["scrap_percent"]), 3),
            mpts_prob=mpts_p,
            rf_prob=rf_p if rf_p is not None else "",
            delta=delta if delta is not None else "",
            signal=sig,
        ))
    return pd.DataFrame(rows), thr

_SIG_COLOR = {"S1": "#1f77b4", "S2": "#c0392b", "S3": "#2ca02c", "S4": "#7b3f99"}

def _figure(wf, part_id, thr, proc_label):
    steps = wf["step"].tolist()
    cum = wf["scrap_pct"].expanding().mean().round(3).tolist()
    fig = go.Figure()
    # ±5pp band around MPTS
    mp = wf["mpts_prob"].tolist()
    fig.add_trace(go.Scatter(
        x=steps + steps[::-1],
        y=[m + _DIV_THR for m in mp] + [max(m - _DIV_THR, 0) for m in mp][::-1],
        fill="toself", fillcolor="rgba(200,200,200,0.4)",
        line=dict(color="rgba(0,0,0,0)"), name="±5 pp band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=steps, y=mp, mode="lines+markers",
        name="MPTS P(scrap)", line=dict(color="#1f77b4", width=3),
        marker=dict(size=8)))
    rf_y = [v if v != "" else None for v in wf["rf_prob"].tolist()]
    fig.add_trace(go.Scatter(x=steps, y=rf_y, mode="lines+markers",
        name="RF P(scrap)", line=dict(color="#000000", width=3),
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=steps, y=wf["scrap_pct"].tolist(), mode="lines+markers",
        name="Actual scrap %", yaxis="y2",
        line=dict(color="#c0392b", width=2.5), marker=dict(size=10, symbol="star")))
    fig.add_trace(go.Scatter(x=steps, y=cum, mode="lines+markers",
        name="Cumulative avg scrap %", yaxis="y2",
        line=dict(color="#2ca02c", width=2, dash="dash"), marker=dict(size=6, symbol="diamond")))
    fig.update_layout(
        title=f"Part {part_id} — Observed-Sequence Walk-Forward  ({proc_label}; threshold {thr:.2f}%)",
        xaxis=dict(title="Walk-Forward Step (each = next production run)",
                   tickmode="array", tickvals=steps),
        yaxis=dict(title="P(scrap) %", range=[-5, 105],
                   tickvals=[0, 20, 40, 60, 80, 100]),
        yaxis2=dict(title="Scrap %", overlaying="y", side="right",
                    range=[0, max(max(wf["scrap_pct"]), max(cum)) * 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        height=560, font=dict(size=14, color="black"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

_PROC = {3: "shift / Core Making", 14: "misrun / Pouring", 74: "gouged / Pattern/Tooling"}

for _pid in [3, 14, 74]:
    wf, thr = _build_walkforward(_pid)
    if wf is None:
        st.warning(f"Part {_pid}: no records found.")
        continue
    fig = _figure(wf, _pid, thr, _PROC[_pid])
    st.plotly_chart(fig, use_container_width=True)

    # signal-count summary
    counts = wf[wf["signal"] != ""]["signal"].value_counts().to_dict()
    summary = " | ".join(f"{k}={counts[k]}" for k in ("S1", "S2", "S3", "S4") if k in counts)
    st.caption(f"Part {_pid} signals: {summary if summary else 'RF scoring unavailable'}")

    c1, c2 = st.columns(2)
    with c1:
        try:
            png = fig.to_image(format="png", scale=2, width=1500, height=560)
            st.download_button(f"⬇️ Export Part {_pid} Walk-Forward (PNG)", png,
                               f"Figure_4-WF_Part{_pid}_walkforward.png", "image/png",
                               key=f"wf_png_{_pid}")
        except Exception:
            st.info("PNG export needs `kaleido` (pip install kaleido).")
    with c2:
        st.download_button(f"⬇️ Export Part {_pid} Walk-Forward (CSV)",
                           wf.to_csv(index=False).encode(),
                           f"walkforward_part{_pid}.csv", "text/csv",
                           key=f"wf_csv_{_pid}")
