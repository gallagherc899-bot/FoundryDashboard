"""
Drop-in replacement for foundry_dashboard_LOUIT_plus_walkforward__2_.py

REPLACE the existing _build_walkforward function (around line 7463) with the
TWO functions below (a helper plus the rewritten walk-forward).

Behavior change:
  - OLD: scored every event through the H2-validated FIXED model
  - NEW: at each event for the selected part, retrains the global RF on the
         cumulative foundry-wide pool up to and including that event, then
         scores the event through the freshly retrained model.

Why this matters:
  Per-event retraining shows realistic operational behavior — as the foundry
  produces more runs over time, the classifier incorporates that new data.
  The H2-validation calibration (98.6% recall) was on the original 1,257-row
  fit; per-event retrained models inherit the same training procedure but
  are not individually re-validated. State this in §5 if reported as a result.

Required imports (already present in the dashboard, no new imports needed).
"""

# ────────────────────────────────────────────────────────────────────────
# HELPER: retrain global model on a given pool snapshot
# ────────────────────────────────────────────────────────────────────────
def _train_at_step(pool_df, threshold, defect_cols):
    """Train a global model on the pool_df snapshot and return the dict
    needed to score one row through the retrained calibrated classifier.

    Returns the same dict shape as train_global_model() so the rest of the
    walk-forward can score events identically to the dashboard's H2 path.

    On training failure (pool too small or class-imbalance issue), returns
    None and the caller should fall back to MPTS-only output for that step.
    """
    try:
        gm = train_global_model(pool_df.copy(), threshold, defect_cols)
        # Verify the retrained model is usable
        if "cal_model" not in gm or "features" not in gm:
            return None
        return gm
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────
# REWRITTEN: walk-forward with per-event retraining
# ────────────────────────────────────────────────────────────────────────
def _build_walkforward(part_id, direction="forward"):
    """Return a per-step DataFrame: step, date, work_order, order_qty,
    scrap_pct, mpts_prob, rf_prob, delta, signal, pool_size.

    direction='forward'  -> months 1..32 (chronological, operational view)
    direction='reverse'  -> months 32..1 (order-robustness sensitivity view)

    PER-EVENT RETRAINING:
      At each step i for the selected part, the global RF is retrained on
      the cumulative foundry-wide pool of all runs up to and including
      that event's date. The retrained calibrated model then scores the
      event's feature vector.

    If the pool at step i is too small to support a 60-20-20 split with
    both classes present, the function falls back to the H2-validated
    fixed model from global_model['cal_model'] (preserves dashboard
    behavior when retraining is infeasible).
    """
    part_id = str(part_id)
    pdat = df[df["part_id"] == part_id].sort_values("week_ending").reset_index(drop=True)
    if len(pdat) == 0:
        return None, None

    # Per-part threshold (order-invariant; same as H1 §3.3.2)
    thr = pdat["scrap_percent"].mean()

    if direction == "reverse":
        pdat = pdat.iloc[::-1].reset_index(drop=True)

    # Chronic (MPTS) process from full-history dominant defect (order-invariant)
    chronic_proc = "—"
    rates = {c: pdat[c].mean() for c in defect_cols if c in pdat.columns}
    if rates and max(rates.values()) > 0:
        chronic_proc = _DEFECT_TO_PROC.get(max(rates, key=rates.get), "—")

    # Global (foundry-wide) chronological frame for pool slicing
    df_sorted = df.sort_values("week_ending").reset_index(drop=True)

    # Cache training threshold from the dashboard's existing global_model
    train_thr = global_model["global_threshold"]

    rows = []
    for i in range(len(pdat)):
        event_date = pdat.iloc[i]["week_ending"]

        # --- Pool: all foundry runs with date <= this event's date ---
        # (In reverse direction we still use real dates for pool slicing,
        #  matching the operational interpretation: the foundry sees runs
        #  in real chronological order regardless of plotting direction.)
        pool = df_sorted[df_sorted["week_ending"] <= event_date].copy()
        pool_size = len(pool)

        # --- Retrain global model on this pool snapshot ---
        gm_step = _train_at_step(pool, train_thr, defect_cols)
        if gm_step is None:
            # Fallback: use H2-validated fixed model (preserves dashboard behavior)
            gm_step = global_model
            retrain_status = "fallback (H2 fixed)"
        else:
            retrain_status = "retrained"

        # --- MPTS: per-part history accumulated up through step i ---
        hist = pdat.iloc[:i + 1]
        fc = int((hist["scrap_percent"] > thr).sum())
        total_parts = hist["order_quantity"].sum()
        avg_oq = total_parts / len(hist)
        mpts_parts = total_parts / fc if fc > 0 else total_parts
        R = np.exp(-avg_oq / mpts_parts) if mpts_parts > 0 else 0
        mpts_p = round(100 * (1 - R), 1)

        # --- RF score for this event through the (re)trained model ---
        rf_p = None
        try:
            # Re-enhance the pool so this event's enhanced row has all features
            enh = pool.copy()
            enh = add_multi_defect_features(enh, defect_cols)
            enh = add_temporal_features(enh)
            enh = add_mtts_sequential_features(enh, train_thr)
            # Find this event's row in the enhanced pool
            event_rows = enh[
                (enh["part_id"] == part_id) &
                (enh["week_ending"] == event_date)
            ]
            if len(event_rows) > 0:
                last_enh = event_rows.iloc[[-1]].copy()
                last_enh = attach_train_features(
                    last_enh,
                    gm_step["scrap_rate_train"], gm_step["part_freq_train"],
                    gm_step["default_scrap_rate"], gm_step["default_freq"],
                )
                mtts_tr = compute_mtts_on_train(gm_step["df_train"], train_thr)
                last_enh = attach_mtts_aggregate_features(last_enh, mtts_tr)
                feat = last_enh.reindex(columns=gm_step["features"], fill_value=0).fillna(0)
                rf_p = round(float(gm_step["cal_model"].predict_proba(feat)[:, 1][0]) * 100, 1)
        except Exception:
            rf_p = None

        # --- Active process from this run's defect rates ---
        last_proc = "—"
        rr = {c: float(pdat.iloc[i][c]) for c in defect_cols
              if c in pdat.columns and float(pdat.iloc[i][c]) > 0}
        if rr:
            last_proc = _DEFECT_TO_PROC.get(max(rr, key=rr.get), "—")

        # --- Signal: pure Δ-rule (no observed-exceedance override) ---
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
            pool_size=pool_size,               # NEW: training-pool size at this step
            retrain_status=retrain_status,     # NEW: 'retrained' or 'fallback (H2 fixed)'
        ))

    return pd.DataFrame(rows), thr
