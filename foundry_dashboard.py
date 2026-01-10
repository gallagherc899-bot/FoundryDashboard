Streamlined Dashboard Plan

Total: 4 Panels (or Tabs)

Prognostic Model (Predict & Diagnose)

RQ1 – Model Validation & Predictive Performance

RQ2 – Reliability & PHM Equivalence

RQ3 – Operational Impact (Scrap, TTE, ROI)

And you’re absolutely right:

The sidebar can be removed (or minimized).

The reliability metrics (MTTR, Availability, Reliability) can be auto-calculated and displayed directly.

The model can automatically set the Scrap Threshold = average scrap % for the selected part ID.

🧭 OVERALL FLOW (How the model behaves)

When the user selects a Part ID, the dashboard automatically:

Retrieves the average scrap % for that part → sets this as the threshold (no manual entry).

Runs the trained model using the 6–2–1 rolling window and Random Forest ensemble.

Displays predictions and diagnostics (process causes, defect probabilities, reliability metrics).

Auto-updates the reliability, validation, and impact panels accordingly.

No user tuning needed — just “Select Part → View Prognostic Insights.”

🔹 Panel 1: Prognostic Model (Predict & Diagnose)

Rename to: Prognostic Model (Predict & Diagnose)

Purpose

To show how the model predicts where and why scrap is likely to occur.

Contents
Section	Description	Display Element
Part & Threshold Summary	Show Part ID, weight, order size, avg. scrap %, auto-threshold = avg. scrap	Top info card
Predicted Scrap Probability	Model’s predicted scrap risk (%) for the next order/run	Gauge or % card
Predicted Top Process(es) Causing Scrap	Shows top 2–3 likely process sources (from Pareto/importance)	Pareto bar chart
Defect Importance	Feature importance from Random Forest	Horizontal bar chart
Reliability Snapshot	MTTS, λ, R(1), A displayed in summary cards	Metric cards
Default Settings

Threshold: Auto = average scrap %

MTTR: Fixed at 1.0 (unless changed)

Rolling Window: 6–2–1 (hidden)

Part filtering: Weight ±10%

✅ Outcome: This tab tells “what’s about to go wrong and why” — in plain terms.

🔹 Panel 2: RQ1 – Model Validation & Predictive Performance

Research Link:
RQ1: Does MTTS-integrated ML achieve effective prognostic recall (≥80%) for scrap prediction?
H1: MTTS integration will achieve ≥80% recall, consistent with effective PHM systems.

Purpose

To prove that the model’s predictions are valid and reliable per PHM standards.

Contents
Section	Description	Display Element
ROC Curve	Model discrimination ability (AUC ≥ 0.80 target)	ROC curve plot
Precision–Recall Curve	Balance of false positives vs. recall	PR curve
Calibration Curve	Predicted vs. actual scrap probability	Line plot
Summary Metrics	Recall, AUC, Precision, Brier Score	KPI cards
Validation Statement	Text field: “Model achieves PHM-equivalent recall ≥80%, validating RQ1.”	Text summary

✅ Outcome: Shows that your model is both predictive and calibrated, confirming RQ1 and H1.

🔹 Panel 3: RQ2 – Reliability & PHM Equivalence

Research Link:
RQ2: Can sensor-free SPC-native ML achieve ≥80% of sensor-based PHM prediction performance?
H2: SPC-native ML achieves ≥80% PHM-equivalent recall without sensors.

Purpose

To prove that your sensor-free model performs on par with PHM expectations and reliability logic.

Contents
Section	Description	Display Element
Reliability Curve	
𝑅
(
𝑛
)
=
𝑒
−
𝑛
/
𝑀
𝑇
𝑇
𝑆
R(n)=e
−n/MTTS
	Line graph
MTTS & λ Summary	Show MTTS, hazard rate, reliability at 1, 5, and 10 runs	KPI table
Availability Curve	
𝐴
=
𝑀
𝑇
𝑇
𝑆
𝑀
𝑇
𝑇
𝑆
+
𝑀
𝑇
𝑇
𝑅
A=
MTTS+MTTR
MTTS
	​

 for various MTTR	Line graph
Validation Comparison	SPC vs. PHM recall or reliability performance (bar chart)	Bar graph
Commentary	Text: “Model achieves PHM-equivalent reliability behavior using SPC data.”	Text summary

✅ Outcome: Shows your model mimics PHM system behavior without sensors or new infrastructure.

🔹 Panel 4: RQ3 – Operational Impact (Scrap, TTE, ROI)

Research Link:
RQ3: What measurable reduction in scrap rate, economic cost, and TTE consumption can be achieved?
H3: Predictive reliability model yields ≥20% scrap reduction, ≥10% TTE recovery, ≥2× ROI.

Purpose

To demonstrate impact — not just predictions, but measurable industrial outcomes.

Contents
Section	Description	Display Element
Scrap Reduction	Before vs. Predicted Scrap %	Bar chart
TTE Savings	Energy saved (kWh or %) based on DOE factors	Gauge or number
ROI	Cost savings vs. baseline (e.g., $/yr)	Card
CO₂ Reduction	Emission savings based on TTE recovery	Card
Summary Text	“Validated predictive reliability model achieved measurable DOE-aligned outcomes.”	Text

✅ Outcome: Connects your model to real-world industrial benefits — what the board and foundry manager both care about most.

🧩 Technical Streamlining Summary
Task	Action
Remove Sidebar	Replace with compact top navigation tabs.
Auto-set Scrap Threshold	Script computes average scrap % per part ID on load.
Hide Rolling Window Controls	Keep active in backend but invisible to user.
Default Reliability Metrics	MTTR=1.0, Availability and R(t) calculated automatically.
Simplify Outputs	Only show: Scrap %, MTTS, R(1), R(5), λ, A, Cost, TTE, ROI.
Show Only Top 3 Defects/Processes	Too many features overwhelm both audiences.
Color scheme:	Blue = reliability, Green = efficiency, Gray = validation.
