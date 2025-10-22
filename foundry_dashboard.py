import warnings
# Suppress the NumPy/Pandas warnings that often occur during data manipulation
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import streamlit as st
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import mannwhitneyu

# Use the existing Streamlit page configuration structure
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard — Actionable Insights",
    layout="wide"
)

# --- 1. CONFIGURATION (Moved to Streamlit Sidebar) ---
st.sidebar.title("Simulation Configuration")

# Financial Inputs
st.sidebar.subheader("Financial Metrics (USD)")
MATERIAL_COST_PER_LB = st.sidebar.number_input(
    "Material Cost/Value per lb ($)", value=2.50, step=0.10, format="%.2f"
)
LABOR_OVERHEAD_COST_PER_LB = st.sidebar.number_input(
    "Labor/Overhead Cost per lb ($)", value=0.50, step=0.10, format="%.2f"
)
AVG_NON_MATERIAL_COST_PER_FAILURE = st.sidebar.number_input(
    "Non-Material Cost per Failure ($)", value=150.00, step=10.00
)

# Simulation Target and Effort
st.sidebar.subheader("Target & Effort")
TARGET_SCRAP_PERCENT = st.sidebar.slider(
    "Ultimate Scrap Target (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1
)
REDUCTION_SCENARIOS = [
    0.30, 0.25, 0.20, 0.15, 0.10, 0.05
]
# Hardcoded max cycles for simulation stability
MAX_CYCLES = 100

# --- 2. DATA LOADING AND PREPARATION ---

def clean_col_name(col_raw):
    """
    Cleans column names to a reliable snake_case format.
    Ensures 'Work Order #' becomes 'work_order_id'.
    """
    col = str(col_raw).strip().lower()
    col = col.replace(' ', '_')
    col = col.replace('#', '_id')
    col = col.replace('%', '_percent')
    
    # Remove any other common symbols
    for char in ['(', ')', '.', '/', '\\']:
        col = col.replace(char, '')
        
    # Ensure no double underscores remain
    while '__' in col:
        col = col.replace('__', '_')
        
    return col.strip('_')

@st.cache_data
def load_and_prepare_data():
    """
    Loads, cleans, and prepares data for simulation, ML, and historical context.
    All column names are converted to snake_case for consistency.
    """
    try:
        df_historical = pd.read_csv('anonymized_parts.csv')
        
        # --- Robust Column Name Cleaning (converts to snake_case) ---
        df_historical.columns = [clean_col_name(c) for c in df_historical.columns]

        # --- Standardize Key Column Name for Scrap History ---
        df_historical = df_historical.rename(columns={'scrap_percent': 'scrap_percent_hist'}, errors='ignore')
        
        # Convert historical scrap percentage to decimal
        df_historical['scrap_percent_hist'] = df_historical['scrap_percent_hist'] / 100.0
        
        # --- Data for Simulation (df_avg: Averages/Max rates per Part ID) ---
        # NOTE: All column references are now snake_case
        df_avg = df_historical.groupby('part_id').agg(
            Scrap_Percent_Baseline=('scrap_percent_hist', 'max'),
            Avg_Order_Quantity=('order_quantity', 'mean'),
            Avg_Piece_Weight=('piece_weight_lbs', 'mean'),
            Total_Runs=('work_order_id', 'count') 
        ).reset_index()

        df_avg['Est_Annual_Scrap_Weight_lbs'] = df_avg.apply(
            lambda row: row['Scrap_Percent_Baseline'] * row['Avg_Order_Quantity'] * row['Avg_Piece_Weight'] * min(52, row['Total_Runs']), 
            axis=1
        )
        
        # Calculate total historical scrap pieces for cost avoidance estimation
        df_historical['scrap_pieces'] = df_historical['pieces_scrapped'] 
        total_historical_scrap_pieces = df_historical['scrap_pieces'].sum()
        
        # --- Data for Machine Learning (df_ml: Work Orders with Binary Scrap Causes) ---
        df_ml = df_historical.copy()
        
        # Target variable for ML: 1 if scrap occurred, 0 otherwise
        df_ml['is_scrapped'] = (df_ml['scrap_percent_hist'] > 0).astype(int)
        
        # Features for ML: Scrap Cause columns (Rate columns)
        # Find all columns ending in '_rate'
        rate_cols = [col for col in df_ml.columns if col.endswith('_rate')]
        
        # Convert rate columns into binary features (1 if cause was present, 0 otherwise)
        for col in rate_cols:
            df_ml[col] = (df_ml[col] > 0).astype(int)

        return df_avg, df_ml, total_historical_scrap_pieces, df_historical

    except FileNotFoundError:
        st.error("Error: 'anonymized_parts.csv' not found. Please ensure the file is correctly named and available.")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()
    except Exception as e:
        # Catch and display the specific error
        st.error(f"An error occurred during data processing: {e}")
        # Add a diagnostic check to help identify the problem
        try:
             # This requires the CSV to be loaded successfully before the error
            df_temp = pd.read_csv('anonymized_parts.csv')
            cleaned_cols = [clean_col_name(c) for c in df_temp.columns]
            st.code(f"Available column names after cleaning (Look for 'work_order_id'):\n{cleaned_cols}")
        except Exception:
            pass
            
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()

# Load and prepare data
df_avg, df_ml, total_historical_scrap_pieces, df_historical = load_and_prepare_data()

if df_avg.empty or df_ml.empty:
    st.stop() # Stop the script if data loading failed

# --- 3. SIMULATION & CALCULATION FUNCTIONS (Unchanged logic, uses consistent column names) ---

def calculate_cycles_to_target(initial_scrap_rate, reduction_factor, target_scrap):
    """Calculates the number of runs (cycles) required for a single part to hit the target."""
    target_scrap_decimal = target_scrap / 100.0
    if initial_scrap_rate <= target_scrap_decimal:
        return 0
    
    improvement_factor = 1.0 - reduction_factor
    current_rate = initial_scrap_rate
    cycles = 0
    
    buffer = 1e-9 
    
    while current_rate > target_scrap_decimal + buffer and cycles < MAX_CYCLES: 
        current_rate *= improvement_factor
        current_rate = max(current_rate, target_scrap_decimal)
        cycles += 1
        
    return cycles


def run_universal_improvement_simulation(df_avg, reduction_factor, target_scrap_percent):
    """Runs the full simulation to get total cycles and savings."""
    target_scrap_decimal = target_scrap_percent / 100.0
    improvement_factor = 1.0 - reduction_factor
    
    # Use 'part_id' column from df_avg
    current_part_status = df_avg[['part_id', 'Scrap_Percent_Baseline']].copy()
    current_part_status.rename(columns={'Scrap_Percent_Baseline': 'Latest_Scrap_Percent'}, inplace=True)
    
    total_cycles = 0
    total_cumulative_savings = 0.0
    
    while total_cycles < MAX_CYCLES:
        parts_to_improve = current_part_status[current_part_status['Latest_Scrap_Percent'] > target_scrap_decimal]
        
        if parts_to_improve.empty:
            break
        
        cycle_savings = 0.0
        
        for part_id in parts_to_improve['part_id']:
            current_scrap_rate = current_part_status[current_part_status['part_id'] == part_id]['Latest_Scrap_Percent'].iloc[0]
            part_metrics = df_avg[df_avg['part_id'] == part_id].iloc[0]
            
            new_scrap_rate = current_scrap_rate * improvement_factor
            new_scrap_rate = max(new_scrap_rate, target_scrap_decimal)
            
            scrap_weight_before = current_scrap_rate * part_metrics['Avg_Order_Quantity'] * part_metrics['Avg_Piece_Weight']
            scrap_weight_after = new_scrap_rate * part_metrics['Avg_Order_Quantity'] * part_metrics['Avg_Piece_Weight']
            
            weight_saved = scrap_weight_before - scrap_weight_after
            cycle_savings += weight_saved
            
            current_part_status.loc[current_part_status['part_id'] == part_id, 'Latest_Scrap_Percent'] = new_scrap_rate

        total_cycles += 1
        total_cumulative_savings += cycle_savings
        
    return total_cycles, total_cumulative_savings

# --- 4. MACHINE LEARNING AND VALIDATION FUNCTIONS ---

@st.cache_data
def train_random_forest_model(df_ml):
    """Trains a Calibrated Random Forest classifier to predict if a work order will scrap."""
    # Identify binary scrap cause columns as features (using the snake_case names)
    feature_cols = [col for col in df_ml.columns if col.endswith('_rate')]

    X = df_ml[feature_cols]
    y = df_ml['is_scrapped'] # Use snake_case
    
    # --- FIX for ValueError: Stratification issue ---
    class_counts = y.value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2:
        st.warning("Warning: Scrap target class has fewer than 2 samples. Disabling stratification for train/test split to prevent error.")
        stratify_param = None
    else:
        stratify_param = y
    # --- END FIX ---
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify_param
    )

    # Base Random Forest Model (using balanced weights)
    base_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')

    # Calibrate the probabilities using Isotonic regression for better risk estimation
    model = CalibratedClassifierCV(
        estimator=base_model,
        method='isotonic', 
        cv=5
    )
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, feature_cols


def predict_part_risk(model, part_id, df_ml, feature_cols):
    """
    Generates a prediction vector based on the part's historical average feature profile 
    and predicts the scrap probability.
    """
    # 1. Get the average feature profile for the selected part
    part_data = df_ml[df_ml['part_id'] == part_id] # Use snake_case
    
    if part_data.empty:
        return 0.0, None

    # Calculate the mean of the binary cause features for this specific part.
    mean_features = part_data[feature_cols].mean().to_frame().T
    
    # 2. Predict the probability of scrap (Is_Scrapped=1)
    prob_prediction = model.predict_proba(mean_features)[0][1]
    
    # 3. Format mean features for display
    mean_features_display = mean_features.T
    mean_features_display.columns = ['Mean_Likelihood']
    # Clean up cause names for display (from snake_case back to Title Case)
    mean_features_display['Cause'] = mean_features_display.index.str.replace('_rate', '').str.replace('_', ' ').str.title()
    
    return prob_prediction, mean_features_display.sort_values(by='Mean_Likelihood', ascending=False)


def run_statistical_validation(df_ml, df_avg):
    """
    Performs a Mann-Whitney U Test (non-parametric two-sample test) to compare 
    the distribution of scrap rates between all runs and the top 10 worst parts.
    """
    # 1. Scrap rates for all runs where scrap occurred
    all_scrap_rates = df_ml[df_ml['scrap_percent_hist'] > 0]['scrap_percent_hist'] * 100 # Use snake_case
    
    # 2. Scrap rates for the top 10 worst parts (by max historical rate)
    top_parts = df_avg.sort_values(by='Scrap_Percent_Baseline', ascending=False)['part_id'].head(10).tolist() # Use snake_case
    top_scrap_rates = df_ml[df_ml['part_id'].isin(top_parts) & (df_ml['scrap_percent_hist'] > 0)]['scrap_percent_hist'] * 100 # Use snake_case
    
    if all_scrap_rates.empty or top_scrap_rates.empty:
        return "Insufficient non-zero scrap data to run statistical test.", 1.0, 0, 0

    # Ensure samples are the same size (up to 1000) for a cleaner comparison/faster runtime
    sample_size = min(len(all_scrap_rates), len(top_scrap_rates), 1000)
    
    sample1 = all_scrap_rates.sample(sample_size, random_state=42, replace=True if len(all_scrap_rates) < sample_size else False)
    sample2 = top_scrap_rates.sample(sample_size, random_state=42, replace=True if len(top_scrap_rates) < sample_size else False)

    u_stat, p_value = mannwhitneyu(
        sample1, 
        sample2, 
        alternative='less' # Test if the first sample (all parts) is stochastically smaller than the second (top 10 parts)
    )
    
    return "Mann-Whitney U Test (Non-parametric distribution comparison)", p_value, all_scrap_rates.mean(), top_scrap_rates.mean()


# --- 5. STREAMLIT APP LAYOUT ---

st.title("🏭 Foundry Scrap Risk Dashboard")
# Use snake_case for access
st.markdown(f"**Target Scrap Floor:** **{TARGET_SCRAP_PERCENT:.1f}%** | **Total Parts Analyzed:** {len(df_avg)}")

# Train the model once for all tabs
with st.spinner("Initializing ML Model for Predictions..."):
    model, X_test, y_test, feature_cols = train_random_forest_model(df_ml)

# Define the tabs
tab_sim, tab_pred, tab_val, tab_part_risk = st.tabs(["🚀 Simulation & Tactics", "🔮 Causal Prediction", "📊 Statistical Validation", "🔬 Individual Part Risk Prediction"])

# ==============================================================================
# TAB 1: SIMULATION & TACTICS
# ==============================================================================
with tab_sim:
    st.header("1. Comparative Improvement Forecast")
    st.markdown("Forecasts the total project timeline and cost avoidance based on different levels of continuous improvement effort.")

    report_data = []
    max_savings_lbs = 0.0 

    for factor in REDUCTION_SCENARIOS:
        with st.spinner(f"Running simulation for {factor*100:.0f}% reduction..."):
            cycles, savings_lbs = run_universal_improvement_simulation(df_avg, factor, TARGET_SCRAP_PERCENT)
        
        if factor == REDUCTION_SCENARIOS[0]:
            max_savings_lbs = savings_lbs

        # Financial Calculation
        total_material_value = max_savings_lbs * (MATERIAL_COST_PER_LB + LABOR_OVERHEAD_COST_PER_LB)
        total_failures_avoided_cost = total_historical_scrap_pieces * AVG_NON_MATERIAL_COST_PER_FAILURE * 0.50
        total_cost_avoided = total_material_value + total_failures_avoided_cost

        report_data.append({
            'Reduction (%)': factor * 100,
            'Cycles Required': cycles,
            'Days Required': cycles * 7,
            'Time (Years)': cycles * 7 / 365.25,
            'Total Aluminum Saved (lbs)': max_savings_lbs,
            'Total Cost Avoided (USD)': total_cost_avoided,
        })

    df_report = pd.DataFrame(report_data)

    # Display Metrics for the most aggressive (30%) scenario
    col1, col2, col3 = st.columns(3)
    aggressive_data = df_report[df_report['Reduction (%)'] == 30.0].iloc[0]

    col1.metric("Max Potential Savings", f"${aggressive_data['Total Cost Avoided (USD)']:.0f}", "Total Project Value")
    col2.metric("Fastest Timeline (30% Effort)", f"{aggressive_data['Days Required']:.0f} Days", f"Time to Hit {TARGET_SCRAP_PERCENT:.1f}% Floor")
    col3.metric("Max Weight Saved", f"{aggressive_data['Total Aluminum Saved (lbs)']:.0f} lbs", "Total Project Weight Savings")

    st.dataframe(
        df_report.style.format({
            'Reduction (%)': "{:.0f}%",
            'Days Required': "{:.0f}",
            'Time (Years)': "{:.2f}",
            'Total Aluminum Saved (lbs)': "{:,.0f}",
            'Total Cost Avoided (USD)': "${:,.0f}",
        }),
        use_container_width=True
    )

    st.caption("NOTE: Total savings are consistent across scenarios as they represent the fixed maximum achievable savings to reach the set target.")

    st.header("2. Tactical Manager Actionable Dashboard")
    st.markdown(f"Prioritization tool: Showing the **Top 30 Parts** based on highest historical scrap rate and estimated annual weight loss.")

    # Use snake_case for access
    df_dashboard = df_avg.sort_values(by='Scrap_Percent_Baseline', ascending=False).head(30).copy()

    for factor in [0.30, 0.25, 0.20]:
        col_name_cycles = f'{factor*100:.0f}% Cycles'
        col_name_days = f'{factor*100:.0f}% Days'
        
        df_dashboard[col_name_cycles] = df_dashboard.apply(
            lambda row: calculate_cycles_to_target(row['Scrap_Percent_Baseline'], factor, TARGET_SCRAP_PERCENT),
            axis=1
        )
        df_dashboard[col_name_days] = df_dashboard[col_name_cycles] * 7

    df_dashboard['Max Hist. Scrap %'] = (df_dashboard['Scrap_Percent_Baseline'] * 100).round(2).astype(str) + '%'
    df_dashboard['Est. Annual Scrap (lbs)'] = df_dashboard['Est_Annual_Scrap_Weight_lbs'].round(0).astype(int)

    dashboard_cols = [
        'part_id', # Use snake_case
        'Max Hist. Scrap %',
        'Est. Annual Scrap (lbs)',
        '30% Cycles', '30% Days',
        '25% Cycles', '25% Days',
        '20% Cycles', '20% Days',
    ]

    df_dashboard = df_dashboard[dashboard_cols]
    df_dashboard.rename(columns={'part_id': 'Part ID'}, inplace=True) # Rename back for display

    st.dataframe(
        df_dashboard.style.background_gradient(cmap='Reds', subset=['Est. Annual Scrap (lbs)']),
        use_container_width=True,
        hide_index=True
    )

    st.caption("The 'Est. Annual Scrap (lbs)' column is used for resource prioritization. Cycles and Days columns indicate the time needed to bring THIS SPECIFIC part to the set target.")


# ==============================================================================
# TAB 2: CAUSAL PREDICTION
# ==============================================================================
with tab_pred:
    st.header("Predictive Causal Analysis (Random Forest)")
    st.markdown("A Machine Learning model trained on historical work orders to identify the **most important factors** contributing to a run having *any* scrap.")
    
    # Model Evaluation Metrics (using the calibrated model's score)
    accuracy = model.score(X_test, y_test)
    st.metric(label="Model Prediction Accuracy (on test set)", value=f"{accuracy*100:.2f}%")
    st.caption("This score reflects the model's ability to correctly predict if a work order will have scrap or not.")
    
    st.subheader("Top Causal Factors (Feature Importance)")
    
    # Feature Importance extraction from the base estimator
    importances = model.base_estimator_.feature_importances_ 
    feature_importance_df = pd.DataFrame({
        'Cause': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Clean up cause names for display (from snake_case back to Title Case)
    feature_importance_df['Cause'] = feature_importance_df['Cause'].str.replace('_rate', '').str.replace('_', ' ').str.title()
    
    st.dataframe(
        feature_importance_df.head(10).style.bar(subset=['Importance'], color='#cf5c36'),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("High Importance scores indicate these scrap causes are the strongest predictors of whether a work order fails.")

# ==============================================================================
# TAB 3: STATISTICAL VALIDATION
# ==============================================================================
with tab_val:
    st.header("Statistical Validation: High-Risk Group Confirmation")
    st.markdown("This test confirms whether the maximum scrap rates observed in the **Top 10 Worst Parts** are statistically different from the average scrap rates across **all** parts.")

    test_name, p_value, mean_all, mean_top = run_statistical_validation(df_ml, df_avg)
    
    st.subheader(f"{test_name}")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    col_stat1.metric("P-Value", f"{p_value:.4f}")
    col_stat2.metric("Mean Scrap Rate (All Runs)", f"{mean_all:.2f}%")
    col_stat3.metric("Mean Scrap Rate (Top 10 Parts)", f"{mean_top:.2f}%")

    if p_value < 0.05:
        st.success(f"**Conclusion:** With a P-Value of **{p_value:.4f}**, which is less than 0.05, we **reject the null hypothesis**. The scrap rates for the top 10 worst parts are statistically and significantly greater than the average scrap rates for all parts. The prioritization strategy is statistically justified.")
    else:
        st.warning(f"**Conclusion:** With a P-Value of **{p_value:.4f}**, we **fail to reject the null hypothesis**. There is not enough statistical evidence to say the scrap rates for the top 10 parts are significantly higher than the rest.")
        
    st.caption("The Mann-Whitney U test is used because it compares the distribution shapes of two independent, non-normally distributed samples (scrap rates).")

# ==============================================================================
# TAB 4: INDIVIDUAL PART RISK PREDICTION
# ==============================================================================
with tab_part_risk:
    st.header("🔬 Individual Part Risk Prediction")
    st.markdown("Predict the probability of a future work order scrapping for a specific part based on its historical risk profile.")
    
    # --- Input Form ---
    part_ids = sorted(df_avg['part_id'].unique().tolist()) # Use snake_case
    
    col_form1, col_form2, col_form3 = st.columns(3)
    
    selected_part_id = col_form1.selectbox(
        "Select Part ID", 
        part_ids, 
        index=0
    )
    
    # Use average weight and quantity as initial values
    if selected_part_id:
        # Use snake_case
        avg_metrics = df_avg[df_avg['part_id'] == selected_part_id].iloc[0] 
        default_weight = avg_metrics['Avg_Piece_Weight']
        default_quantity = avg_metrics['Avg_Order_Quantity']
    else:
        default_weight = 0
        default_quantity = 0

    piece_weight = col_form2.number_input(
        "Piece Weight (lbs)", 
        value=default_weight, 
        min_value=0.1, 
        format="%.2f"
    )
    
    order_quantity = col_form3.number_input(
        "Order Quantity", 
        value=int(default_quantity), 
        min_value=1, 
        step=1
    )

    # --- Prediction Calculation ---
    prob_scrap, part_drivers = predict_part_risk(model, selected_part_id, df_ml, feature_cols)
    
    # Use snake_case
    max_hist_scrap_rate = df_avg[df_avg['part_id'] == selected_part_id]['Scrap_Percent_Baseline'].iloc[0]
    
    # Expected Scrap Pieces = Order Quantity * Max Historical Rate * Prob(Scrap)
    expected_scrap_pieces = order_quantity * max_hist_scrap_rate * prob_scrap
    
    st.markdown("---")
    
    # --- Output Metrics ---
    col_out1, col_out2, col_out3 = st.columns(3)
    
    col_out1.metric(
        "Predicted Risk (Prob. of Scrap Run)", 
        f"{prob_scrap*100:.2f}%",
        help="The probability that a new work order for this part will incur ANY scrap, based on its historical defect profile."
    )
    
    col_out2.metric(
        "Expected Scrap Pieces (Estimate)", 
        f"{expected_scrap_pieces:.1f} pcs",
        help=f"Estimate based on Order Quantity * Max Historical Scrap Rate ({max_hist_scrap_rate*100:.2f}%) * Predicted Risk."
    )
    
    col_out3.metric(
        "Potential Value Loss", 
        f"${expected_scrap_pieces * piece_weight * (MATERIAL_COST_PER_LB + LABOR_OVERHEAD_COST_PER_LB) :.2f}",
        help="Estimated cost of lost material and labor for the current order."
    )
    
    st.markdown("---")

    # --- Historical Data & Drivers ---
    
    col_hist, col_drivers = st.columns(2)

    with col_drivers:
        st.subheader("Historical Cause Likelihood")
        st.markdown(f"The model's input for Part **{selected_part_id}** is based on the average likelihood of these causes being present in its past work orders.")
        
        if part_drivers is not None:
             part_drivers['Mean_Likelihood'] = part_drivers['Mean_Likelihood'].apply(lambda x: f"{x*100:.2f}%")
             part_drivers.rename(columns={'Mean_Likelihood': 'Avg. Likelihood of Presence'}, inplace=True)

             st.dataframe(
                part_drivers.head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No detailed defect data available for this part.")


    with col_hist:
        st.subheader(f"Historical Work Orders for {selected_part_id}")
        
        hist_df = df_historical[df_historical['part_id'] == selected_part_id].copy() # Use snake_case
        
        hist_df['Scrap_%'] = (hist_df['scrap_percent_hist'] * 100).round(2) # Use snake_case
        
        # Display relevant columns, using corrected names
        display_cols = [
            'work_order_id', # Use snake_case
            'order_quantity', # Use snake_case
            'pieces_scrapped', # Use snake_case
            'Scrap_%', 
            'piece_weight_lbs', # Use snake_case
            'week_ending' # Use snake_case
        ]
        
        # Rename columns back for friendly display
        hist_df_display = hist_df[display_cols].rename(columns={
            'work_order_id': 'Work Order #', # Friendly name for display
            'order_quantity': 'Order Quantity',
            'pieces_scrapped': 'Pieces Scrapped',
            'piece_weight_lbs': 'Piece Weight (lbs)',
            'week_ending': 'Week Ending'
        })
        
        st.dataframe(
            hist_df_display.sort_values(by='Week Ending', ascending=False),
            use_container_width=True,
            hide_index=True
        )
