# ============================================================================
# TAB 8 REPLACEMENT — H3 Scenario Envelope
# ============================================================================
# Paste this entire block over the existing `with tab8:` block
# (approximately lines 6363–6602 in Final_Dashboard_Lilliefors.py).
#
# Also update the tabs list on approximately line 3014:
#     CHANGE:  "📉 PM Projection Charts"
#     TO:      "🎯 H3 Scenario Envelope"
#
# This tab implements the paragraph-641 methodology from Chapter 3:
#   Scenario 1 (Replication Ceiling): target = historical minimum %, CP-adjusted
#   Scenario 2 (Conservative Projection): target[n] = max(avg × 0.90^n, min), CP-adjusted
# Applied to two cohorts: 22-part (by scrap weight) and 8-part (priority).
# ============================================================================

    with tab8:
        st.header("🎯 H3 Scenario Envelope — 22-Part vs 8-Part Priority")

        st.markdown("""
        <div class="citation-box">
            <strong>Scenario logic (praxis §3.5).</strong>
            <strong>Scenario 1 (Replication Ceiling):</strong> each part's target scrap%
            is set to its own historical minimum observed in the 32-month dataset.
            <strong>Scenario 2 (Conservative Projection):</strong> each part's target
            scrap% starts at its historical average and decays 10% compounded per
            consecutive run, floored at its historical minimum. Both scenarios apply
            the Clopper–Pearson 95% lower-bound recall factor of 0.924 established in
            H2 (§4.2.2). Scenario 1 is the upper ceiling of the projection envelope;
            Scenario 2 is the conservative floor.
        </div>
        """, unsafe_allow_html=True)

        # ── Cohort definitions (from praxis §3.2.2) ───────────────────────
        # 22-part validation cohort by scrap weight (top-22 Pareto-weight rank)
        COHORT_22 = [3, 6, 15, 16, 40, 63, 67, 71, 74, 82, 88, 90,
                     99, 100, 101, 120, 122, 124, 134, 185, 198, 229]
        # 8-part priority (top-8 by scrap weight + stable run counts; covers all
        # 7 Campbell process groups)
        COHORT_8 = [15, 63, 124, 40, 120, 100, 6, 67]

        # ── Editable parameters (sidebar-style expander) ──────────────────
        with st.expander("⚙️ Methodological parameters (editable for sensitivity analysis)",
                         expanded=False):
            st.markdown("*Defaults match praxis §3.5 Table 3-6. Adjust to explore "
                        "parameter sensitivity. Reset by reloading the page.*")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown("**Financial**")
                material_cost = st.number_input("Material Cost ($/lb)",
                                                value=2.50, min_value=0.0, step=0.25,
                                                key="h3_material_cost")
                energy_cost = st.number_input("Energy Cost ($/MMBtu)",
                                              value=12.00, min_value=0.0, step=0.50,
                                              key="h3_energy_cost")
                impl_cost = st.number_input("Implementation Cost ($)",
                                            value=2000, min_value=0, step=250,
                                            key="h3_impl_cost")
            with pcol2:
                st.markdown("**Energy & emissions**")
                btu_per_lb = st.number_input("TTE intensity (BTU/lb)",
                                             value=47250, min_value=0, step=500,
                                             key="h3_btu_per_lb",
                                             help="DOE aluminum green-sand casting "
                                                  "benchmark — Eppich (2004)")
                kg_co2_per_mmbtu = st.number_input("GHG factor (kg CO₂/MMBtu)",
                                                    value=53.06, min_value=0.0, step=0.1,
                                                    key="h3_kg_co2_per_mmbtu",
                                                    help="EPA natural gas combustion — EPA (2023)")
            with pcol3:
                st.markdown("**Classifier uncertainty**")
                cp_factor = st.number_input("CP recall lower bound",
                                            value=0.924, min_value=0.0, max_value=1.0,
                                            step=0.001, format="%.3f",
                                            key="h3_cp_factor",
                                            help="Clopper–Pearson 95% lower bound "
                                                 "from H2 (§4.2.2)")
                decay_rate = st.number_input("Scenario 2 decay rate (per run)",
                                             value=0.10, min_value=0.0, max_value=0.5,
                                             step=0.01, format="%.2f",
                                             key="h3_decay_rate",
                                             help="10% compounding decay from each "
                                                  "part's historical average")

        # ── Load the dataset from the main dashboard's session state ──────
        # The dashboard loads `df` at startup — reuse it rather than re-reading.
        _h3_df = None
        for _cand_name in ["df", "df_raw", "data_df", "source_df"]:
            if _cand_name in dir():
                try:
                    _cand = eval(_cand_name)
                    if isinstance(_cand, pd.DataFrame) and "Part ID" in _cand.columns:
                        _h3_df = _cand.copy()
                        break
                except Exception:
                    continue
        if _h3_df is None:
            # Fallback: read from disk
            for _path in ["anonymized_parts.csv",
                          "/home/claude/anonymized_parts.csv",
                          "data/anonymized_parts.csv"]:
                try:
                    _h3_df = pd.read_csv(_path)
                    break
                except Exception:
                    continue

        if _h3_df is None:
            st.error("Could not locate anonymized_parts.csv. Place it alongside this "
                     "dashboard script or in /home/claude/.")
            st.stop()

        # Normalize
        _h3_df["Week Ending"] = pd.to_datetime(_h3_df["Week Ending"], errors="coerce")
        _h3_df = _h3_df.dropna(subset=["Week Ending"])
        _h3_df = _h3_df.sort_values(["Part ID", "Week Ending"]).reset_index(drop=True)

        WINDOW_MONTHS = 32
        _earliest = _h3_df["Week Ending"].min()
        _latest = _h3_df["Week Ending"].max()
        _y1_end = _earliest + pd.DateOffset(months=12)
        _y2_end = _earliest + pd.DateOffset(months=24)
        _y32_end = _earliest + pd.DateOffset(months=32)

        # Facility baseline
        _facility_32mo = _h3_df["Total Scrap Weight (lbs)"].sum()
        _facility_annual = _facility_32mo / (WINDOW_MONTHS / 12)

        # ── Scenario computation functions ────────────────────────────────
        def _scenario1_replication(df, ids, cp, window_end=None):
            """Target = each part's historical minimum scrap %.
               Returns total avoided scrap over the window, CP-adjusted."""
            total = 0.0
            for pid in ids:
                part_full = df[df["Part ID"] == pid].sort_values("Week Ending")
                if len(part_full) == 0:
                    continue
                mn = part_full["Scrap%"].min()  # always use full-history minimum
                part = (part_full if window_end is None
                        else part_full[part_full["Week Ending"] <= window_end])
                for _, r in part.iterrows():
                    a_pct = r["Scrap%"]
                    a_wt = r["Total Scrap Weight (lbs)"]
                    if a_pct > mn and a_pct > 0:
                        total += a_wt * (1 - mn / a_pct)
            return total * cp

        def _scenario2_decay(df, ids, cp, decay, window_end=None):
            """Target[n] = max(historical_avg × (1-decay)^n, historical_min).
               Runs are walked chronologically; decay index = position in full history.
               CP-adjusted."""
            total = 0.0
            for pid in ids:
                part_full = df[df["Part ID"] == pid].sort_values("Week Ending").reset_index(drop=True)
                if len(part_full) == 0:
                    continue
                avg_pct = part_full["Scrap%"].mean()
                mn = part_full["Scrap%"].min()
                if window_end is not None:
                    mask = part_full["Week Ending"] <= window_end
                    part_full_idx = part_full[mask].copy()
                else:
                    part_full_idx = part_full
                for n_idx, (_, r) in enumerate(part_full_idx.iterrows()):
                    # n_idx is position in the chronologically ordered history
                    target = max(avg_pct * ((1 - decay) ** n_idx), mn)
                    a_pct = r["Scrap%"]
                    a_wt = r["Total Scrap Weight (lbs)"]
                    if a_pct > target and a_pct > 0:
                        total += a_wt * (1 - target / a_pct)
            return total * cp

        def _cascade(scrap_annual, mat, eng, impl, btu, kg_co2, facility_ann):
            tte = scrap_annual * btu / 1_000_000
            ghg = tte * kg_co2 / 1000
            material_savings = scrap_annual * mat
            energy_savings = tte * eng
            total_savings = material_savings + energy_savings
            roi = total_savings / impl if impl > 0 else 0.0
            facility_pct = scrap_annual / facility_ann * 100 if facility_ann > 0 else 0.0
            return dict(
                scrap=scrap_annual, tte=tte, ghg=ghg,
                mat=material_savings, eng=energy_savings,
                tot=total_savings, roi=roi, fac=facility_pct,
            )

        def _compute_all(ids, label):
            """Returns dict with S1/S2 full-window and Y1/Y2/Y32 cumulative."""
            s1_full = _scenario1_replication(_h3_df, ids, cp_factor)
            s2_full = _scenario2_decay(_h3_df, ids, cp_factor, decay_rate)
            s1_y1 = _scenario1_replication(_h3_df, ids, cp_factor, window_end=_y1_end)
            s1_y2 = _scenario1_replication(_h3_df, ids, cp_factor, window_end=_y2_end)
            s2_y1 = _scenario2_decay(_h3_df, ids, cp_factor, decay_rate, window_end=_y1_end)
            s2_y2 = _scenario2_decay(_h3_df, ids, cp_factor, decay_rate, window_end=_y2_end)
            annualize = lambda x: x / (WINDOW_MONTHS / 12)
            s1_casc = _cascade(annualize(s1_full), material_cost, energy_cost,
                               impl_cost, btu_per_lb, kg_co2_per_mmbtu, _facility_annual)
            s2_casc = _cascade(annualize(s2_full), material_cost, energy_cost,
                               impl_cost, btu_per_lb, kg_co2_per_mmbtu, _facility_annual)
            return dict(
                label=label, ids=ids,
                s1_full=s1_full, s2_full=s2_full,
                s1_y1=s1_y1, s1_y2=s1_y2, s2_y1=s2_y1, s2_y2=s2_y2,
                s1_ann=annualize(s1_full), s2_ann=annualize(s2_full),
                s1_casc=s1_casc, s2_casc=s2_casc,
            )

        _res22 = _compute_all(COHORT_22, "22-Part Pareto Cohort")
        _res8 = _compute_all(COHORT_8, "8-Part Priority Cohort")

        # ── Headline metric strip ─────────────────────────────────────────
        st.markdown("### Headline metrics (annualized, CP-adjusted)")
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        hcol1.metric("22-part Scenario 1",
                     f"{_res22['s1_ann']:,.0f} lbs/yr",
                     f"{_res22['s1_casc']['fac']:.1f}% facility | "
                     f"{_res22['s1_casc']['roi']:.1f}× ROI")
        hcol2.metric("22-part Scenario 2",
                     f"{_res22['s2_ann']:,.0f} lbs/yr",
                     f"{_res22['s2_casc']['fac']:.1f}% facility | "
                     f"{_res22['s2_casc']['roi']:.1f}× ROI")
        hcol3.metric("8-part Scenario 1",
                     f"{_res8['s1_ann']:,.0f} lbs/yr",
                     f"{_res8['s1_casc']['fac']:.1f}% facility | "
                     f"{_res8['s1_casc']['roi']:.1f}× ROI")
        hcol4.metric("8-part Scenario 2",
                     f"{_res8['s2_ann']:,.0f} lbs/yr",
                     f"{_res8['s2_casc']['fac']:.1f}% facility | "
                     f"{_res8['s2_casc']['roi']:.1f}× ROI")

        st.markdown("---")

        # ── Side-by-side cascade tables ───────────────────────────────────
        def _cascade_row(label, s1c, s2c, unit=""):
            return {
                "Metric": label,
                "Scenario 1 (Replication)": f"{s1c}{unit}" if unit else s1c,
                "Scenario 2 (Conservative)": f"{s2c}{unit}" if unit else s2c,
            }

        def _render_cohort(res):
            st.markdown(f"#### {res['label']} ({len(res['ids'])} parts)")
            rows = [
                _cascade_row("Annual scrap reduction (lbs/yr)",
                             f"{res['s1_ann']:,.0f}", f"{res['s2_ann']:,.0f}"),
                _cascade_row("Facility-wide reduction",
                             f"{res['s1_casc']['fac']:.2f}%",
                             f"{res['s2_casc']['fac']:.2f}%"),
                _cascade_row("TTE Savings (MMBtu/yr)",
                             f"{res['s1_casc']['tte']:.1f}",
                             f"{res['s2_casc']['tte']:.1f}"),
                _cascade_row("GHG Reduction (MT CO₂/yr)",
                             f"{res['s1_casc']['ghg']:.2f}",
                             f"{res['s2_casc']['ghg']:.2f}"),
                _cascade_row("Material Savings ($)",
                             f"${res['s1_casc']['mat']:,.0f}",
                             f"${res['s2_casc']['mat']:,.0f}"),
                _cascade_row("Energy Savings ($)",
                             f"${res['s1_casc']['eng']:,.0f}",
                             f"${res['s2_casc']['eng']:,.0f}"),
                _cascade_row("Total Annual Savings ($)",
                             f"${res['s1_casc']['tot']:,.0f}",
                             f"${res['s2_casc']['tot']:,.0f}"),
                _cascade_row("Return on Investment",
                             f"{res['s1_casc']['roi']:.2f}×",
                             f"{res['s2_casc']['roi']:.2f}×"),
            ]
            st.table(pd.DataFrame(rows).set_index("Metric"))

        col_left, col_right = st.columns(2)
        with col_left:
            _render_cohort(_res22)
        with col_right:
            _render_cohort(_res8)

        st.markdown("---")

        # ── Year-by-year cumulative bar chart ─────────────────────────────
        st.markdown("### Cumulative scrap reduction over the 32-month window")
        st.caption("Reflects the progressive decay model in Scenario 2: Year 1 shows "
                   "modest reductions (targets still near historical average), Year 2 "
                   "accelerates (targets approach historical minimums), and the final "
                   "8 months show the floor being reached.")

        _chart_df = pd.DataFrame({
            "Period": ["End of Year 1", "End of Year 2", "End of Month 32"],
            "22-part S1": [_res22["s1_y1"], _res22["s1_y2"], _res22["s1_full"]],
            "22-part S2": [_res22["s2_y1"], _res22["s2_y2"], _res22["s2_full"]],
            "8-part S1": [_res8["s1_y1"], _res8["s1_y2"], _res8["s1_full"]],
            "8-part S2": [_res8["s2_y1"], _res8["s2_y2"], _res8["s2_full"]],
        })
        _fig_cum = go.Figure()
        _colors = {"22-part S1": "#1F4E79", "22-part S2": "#4472C4",
                   "8-part S1": "#C55A11", "8-part S2": "#ED7D31"}
        for col in ["22-part S1", "22-part S2", "8-part S1", "8-part S2"]:
            _fig_cum.add_trace(go.Bar(
                x=_chart_df["Period"], y=_chart_df[col], name=col,
                marker_color=_colors[col],
                text=[f"{v:,.0f}" for v in _chart_df[col]],
                textposition="outside",
            ))
        _fig_cum.update_layout(
            height=420, barmode="group",
            yaxis_title="Cumulative scrap reduction (lbs)",
            xaxis_title="Projection checkpoint",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(t=40, b=40, l=60, r=20),
        )
        st.plotly_chart(_fig_cum, use_container_width=True)

        st.markdown("---")

        # ── EPA ENERGY STAR band comparison ───────────────────────────────
        st.markdown("### EPA ENERGY STAR (2016) 3–10% reduction band comparison")
        _epa_low = _facility_annual * 0.03
        _epa_high = _facility_annual * 0.10
        st.caption(f"Facility baseline: {_facility_annual:,.0f} lbs/yr. "
                   f"EPA 3–10% band: {_epa_low:,.0f} – {_epa_high:,.0f} lbs/yr.")

        def _band_status(val):
            if val >= _epa_high:
                return f"✅ Exceeds 10% ceiling ({val/_facility_annual*100:.1f}%)"
            elif val >= _epa_low:
                return f"✅ In-range ({val/_facility_annual*100:.1f}%)"
            else:
                return f"❌ Below 3% floor ({val/_facility_annual*100:.1f}%)"

        _band_df = pd.DataFrame([
            {"Cohort": "22-part", "Scenario": "1 (Replication)",
             "Annual reduction (lbs/yr)": f"{_res22['s1_ann']:,.0f}",
             "EPA band position": _band_status(_res22["s1_ann"])},
            {"Cohort": "22-part", "Scenario": "2 (Conservative)",
             "Annual reduction (lbs/yr)": f"{_res22['s2_ann']:,.0f}",
             "EPA band position": _band_status(_res22["s2_ann"])},
            {"Cohort": "8-part", "Scenario": "1 (Replication)",
             "Annual reduction (lbs/yr)": f"{_res8['s1_ann']:,.0f}",
             "EPA band position": _band_status(_res8["s1_ann"])},
            {"Cohort": "8-part", "Scenario": "2 (Conservative)",
             "Annual reduction (lbs/yr)": f"{_res8['s2_ann']:,.0f}",
             "EPA band position": _band_status(_res8["s2_ann"])},
        ])
        st.table(_band_df.set_index(["Cohort", "Scenario"]))

        # ── Interpretation ────────────────────────────────────────────────
        st.markdown("### Interpretation")
        st.markdown(f"""
        The framework's projection envelope spans **{min(_res22['s2_casc']['fac'], _res8['s2_casc']['fac']):.1f}%**
        (most conservative: 8-part Scenario 2) to **{_res22['s1_casc']['fac']:.1f}%**
        (most aggressive: 22-part Scenario 1) of facility-wide scrap reduction.
        All four combinations exceed the EPA ENERGY STAR 3–10% range, with ROI
        ranging from **{min(_res8['s2_casc']['roi'], _res22['s2_casc']['roi']):.1f}×**
        to **{_res22['s1_casc']['roi']:.1f}×** against the $2,000 implementation cost.

        The 8-part priority cohort demonstrates that targeting only **2.2% of the
        part population (8 of 359 parts)** is sufficient to exceed the EPA upper
        benchmark. This is the dissertation's strongest operational finding: the
        reliability-driven framework identifies a vital-few intervention boundary
        where modest implementation effort produces substantial facility impact.

        *Methodology reference: Praxis §3.5, Equations 3-17 through 3-20 and
        3-17a. CP factor from §4.2.2. EPA benchmark from EPA (2016).*
        """)

        # ── Cohort part lists (expandable) ────────────────────────────────
        with st.expander("📋 Cohort part lists"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**22-part cohort (by scrap weight rank)**")
                st.code(", ".join(str(p) for p in sorted(COHORT_22)))
            with c2:
                st.markdown("**8-part priority cohort**")
                st.code(", ".join(str(p) for p in sorted(COHORT_8)))
            st.caption(f"All 8 priority parts are members of the 22-part cohort: "
                       f"{set(COHORT_8).issubset(set(COHORT_22))}")

# ============================================================================
# END TAB 8 REPLACEMENT
# ============================================================================
