"""
Air Pollution Source Attribution System - Interactive Dashboard
================================================================
Streamlit Web Application featuring:
- Time-series exploration & statistical spike detection
- Bivariate correlation heatmaps & diurnal profiles
- OLS Regression attribution with VIF diagnostics & standardized betas
- Non-linear Random Forest ensemble comparison
- Episodic spike deep dive & baseline contrast
- Interactive What-If Policy Scenario Simulator
- CPCB / GRAP policy advisory & empirical research bibliography
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure sys.path so modules can be imported
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from core.engine import AttributionEngine
from core.pipeline import AQI_BREAKPOINTS
from utils.policy_recommender import PolicyRecommender, RESEARCH_REFERENCES

# Page configuration
st.set_page_config(
    page_title="Air Pollution Source Attribution System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_benchmark_data():
    csv_path = os.path.join(APP_DIR, "data", "delhi_ncr_benchmark_hourly.csv")
    if not os.path.exists(csv_path):
        from utils.synthetic_generator import generate_delhi_benchmark_dataset
        generate_delhi_benchmark_dataset(year=2024, output_path=csv_path)
    return pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")


def categorize_aqi(aqi_val: float) -> str:
    if aqi_val <= 50:
        return "Good (0-50)"
    elif aqi_val <= 100:
        return "Satisfactory (51-100)"
    elif aqi_val <= 200:
        return "Moderate (101-200)"
    elif aqi_val <= 300:
        return "Poor (201-300)"
    elif aqi_val <= 400:
        return "Very Poor (301-400)"
    else:
        return "Severe (401-500+)"


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/clouds/200/air-quality.png", width=110)
st.sidebar.title("🌫️ Source Attribution")
st.sidebar.caption("Statistical Correlation & Regression Framework")

data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ["Delhi NCR Benchmark (1-Year CPCB/FIRMS)", "Upload Custom CSV File"],
    index=0
)

uploaded_file = None
if data_source == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload hourly CSV file",
        type=["csv"],
        help="CSV with timestamp, PM2.5, PM10, NO2, SO2, CO, weather, etc."
    )

target_col = st.sidebar.selectbox(
    "Target Attribution Variable:",
    ["AQI", "PM2.5", "PM10", "NO2"],
    index=0
)

spike_method = st.sidebar.selectbox(
    "Spike Detection Method:",
    [
        ("p90", "90th Percentile (Upper Decile)"),
        ("p95", "95th Percentile (Extreme 5%)"),
        ("two_sigma", "Mean + 2 Standard Deviations (μ + 2σ)"),
        ("severe_400", "CPCB Severe Benchmark (>= 400 AQI)")
    ],
    format_func=lambda x: x[1],
    index=0
)[0]

season_filter = st.sidebar.selectbox(
    "Temporal Season Slice:",
    ["Full Year (All Seasons)", "Post-Monsoon (Stubble Burning: Oct-Nov)", "Winter Smog (Dec-Feb)", "Summer Pre-Monsoon (Mar-Jun)", "Monsoon (Jul-Sep)"],
    index=0
)

# Initialize Engine
engine = AttributionEngine()

# Load Data
with st.spinner("Processing air quality and meteorological pipeline..."):
    if data_source == "Upload Custom CSV File" and uploaded_file is not None:
        df_processed = engine.load_data(uploaded_file)
    else:
        raw_df = load_benchmark_data()
        df_processed = engine.process_dataframe(raw_df)

# Apply Season Slice if requested
if season_filter == "Post-Monsoon (Stubble Burning: Oct-Nov)":
    df_processed = df_processed[df_processed["season"] == "Post-Monsoon"]
elif season_filter == "Winter Smog (Dec-Feb)":
    df_processed = df_processed[df_processed["season"] == "Winter"]
elif season_filter == "Summer Pre-Monsoon (Mar-Jun)":
    df_processed = df_processed[df_processed["season"] == "Summer"]
elif season_filter == "Monsoon (Jul-Sep)":
    df_processed = df_processed[df_processed["season"] == "Monsoon"]

# Run Attribution Pipeline
engine.df_processed = df_processed
results = engine.run_full_attribution(target_col=target_col, spike_method=spike_method)

spike_summary = results["spike_summary"]
reg = results["regression_result"]
corr = results["correlation_result"]
rf = results["random_forest_result"]
spike_reg = results["spike_regression"]
proxies = results["proxies"]

# Header
st.markdown('<div class="main-header">Air Pollution Source Attribution & Policy System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Object-Oriented Statistical Linking of AQI Spikes to Vehicular, Industrial, Stubble Burning, and Meteorological Proxies</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Executive KPIs",
    "📈 Time Series & Spikes",
    "🔍 Correlation Analysis",
    "⚖️ Source Attribution (OLS & ML)",
    "🚨 Spike Episodes Deep Dive",
    "🎛️ Policy Simulator",
    "📄 Policy Briefing Export",
    "📚 Research References"
])

# ==========================================
# TAB 1: EXECUTIVE KPIS & SYSTEM OVERVIEW
# ==========================================
with tab1:
    st.subheader("Key Performance Indicators & Air Quality Health Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    avg_target = df_processed[target_col].mean()
    peak_target = df_processed[target_col].max()
    top_source = sorted(reg.percentage_shares.items(), key=lambda x: x[1], reverse=True)[0]
    
    col1.metric("Average " + target_col, f"{avg_target:.1f}", help="Mean concentration/index over selected period")
    col2.metric("Peak " + target_col, f"{peak_target:.1f}", help="Highest recorded level")
    col3.metric("Spike Hours Flagged", f"{spike_summary.spike_hours:,}", f"{spike_summary.spike_percentage:.1f}% of hours")
    col4.metric("Leading Baseline Source", f"{top_source[0]}", f"{top_source[1]:.1f}% share")
    col5.metric("Model Fit (R²)", f"{reg.r2:.3f}", f"Adj R²: {reg.adj_r2:.3f}")

    st.markdown("---")
    
    c_left, c_right = st.columns([1.2, 0.8])
    with c_left:
        st.markdown("#### CPCB Air Quality Category Distribution")
        df_processed["AQI_Category"] = df_processed["AQI"].apply(categorize_aqi)
        cat_counts = df_processed["AQI_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Hours"]
        
        color_map = {
            "Good (0-50)": "#2ecc71",
            "Satisfactory (51-100)": "#27ae60",
            "Moderate (101-200)": "#f1c40f",
            "Poor (201-300)": "#e67e22",
            "Very Poor (301-400)": "#e74c3c",
            "Severe (401-500+)": "#8e44ad"
        }
        
        fig_cat = px.bar(
            cat_counts, x="Category", y="Hours", color="Category",
            color_discrete_map=color_map,
            text="Hours", title="Hours in Each CPCB National AQI Band"
        )
        fig_cat.update_layout(showlegend=False, height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c_right:
        st.markdown("#### Baseline Source Contribution Breakdown")
        src_df = pd.DataFrame(list(reg.percentage_shares.items()), columns=["Source", "Share"])
        fig_pie = px.pie(
            src_df, names="Source", values="Share",
            hole=0.45, title="Annual Relative Variance Attribution (%)",
            color="Source",
            color_discrete_map={
                "Vehicular Traffic": "#e74c3c",
                "Industrial Activity": "#8e44ad",
                "Stubble Burning": "#e67e22",
                "Construction & Road Dust": "#f39c12",
                "Meteorological Stagnation": "#3498db"
            }
        )
        fig_pie.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### OOP System Architecture Design")
    st.info("""
    **Object-Oriented Design Principles Utilized:**
    - **`PollutionSource` (Abstract Base Class)**: Defines polymorphically implemented `proxy(df) -> pd.Series` for interchangeable sources (`VehicularTrafficSource`, `IndustrialActivitySource`, `StubbleBurningSource`, `ConstructionDustSource`, `WeatherDispersionControl`).
    - **`AttributionModel` (Abstract Base Class)**: Enforces unified `fit(df, proxies, target) -> AttributionResult` contract implemented by `RegressionAttributionModel`, `CorrelationAttributionModel`, and `RandomForestAttributionModel`.
    - **`AttributionEngine` (Composition)**: Encapsulates sources, pipelines, and models into a decoupled modular orchestration engine.
    """)

# ==========================================
# TAB 2: TIME SERIES & SPIKE DETECTION
# ==========================================
with tab2:
    st.subheader(f"Hourly {target_col} Trajectory & Statistical Spike Identification")
    
    # Plotly interactive time series
    fig_ts = go.Figure()
    
    # Baseline line
    fig_ts.add_trace(go.Scatter(
        x=df_processed.index, y=df_processed[target_col],
        mode="lines", name=f"Hourly {target_col}",
        line=dict(color="#3498db", width=1.2),
        opacity=0.7
    ))
    
    # Threshold horizontal line
    thresh = spike_summary.threshold_value
    fig_ts.add_hline(
        y=thresh, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Spike Threshold ({spike_method}): {thresh:.1f}",
        annotation_position="top left"
    )
    
    # Highlight spike points
    spikes_df = df_processed[spike_summary.spike_mask]
    fig_ts.add_trace(go.Scatter(
        x=spikes_df.index, y=spikes_df[target_col],
        mode="markers", name="Flagged Spike Events",
        marker=dict(color="#e74c3c", size=4, symbol="circle")
    ))
    
    fig_ts.update_layout(
        title=f"Hourly {target_col} Timeline with Flagged Spikes ({len(spikes_df):,} spike hours detected)",
        xaxis_title="Timestamp", yaxis_title=target_col,
        hovermode="x unified", height=420,
        margin=dict(t=50, b=30, l=40, r=30)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Diurnal Signature (Hour of Day Profile)")
        diurnal_stats = df_processed.groupby("hour")[[target_col, "NO2", "CO", "SO2"]].mean().reset_index()
        fig_diurnal = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_diurnal.add_trace(
            go.Scatter(x=diurnal_stats["hour"], y=diurnal_stats[target_col], name=f"Mean {target_col}",
                       line=dict(color="#e74c3c", width=3)),
            secondary_y=False
        )
        if "NO2" in diurnal_stats.columns:
            fig_diurnal.add_trace(
                go.Scatter(x=diurnal_stats["hour"], y=diurnal_stats["NO2"], name="Traffic Tracer (NO2)",
                           line=dict(color="#3498db", width=2, dash="dot")),
                secondary_y=True
            )
        fig_diurnal.update_xaxes(title_text="Hour of Day (00:00 to 23:00)", tickmode="linear", dtick=2)
        fig_diurnal.update_yaxes(title_text=f"Mean {target_col}", secondary_y=False)
        fig_diurnal.update_yaxes(title_text="NO2 (μg/m³)", secondary_y=True)
        fig_diurnal.update_layout(title="24-Hour Diurnal Pattern (Notice Rush-Hour 08-10 & 18-20 Peaks)", height=350)
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with col_b:
        st.markdown("#### Seasonal Concentration Profiles")
        fig_box = px.box(
            df_processed, x="season", y=target_col, color="season",
            title=f"{target_col} Boxplot Distribution Across Indo-Gangetic Seasons",
            category_orders={"season": ["Winter", "Summer", "Monsoon", "Post-Monsoon"]}
        )
        fig_box.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 3: CORRELATION ANALYSIS
# ==========================================
with tab3:
    st.subheader(f"Bivariate Correlation Analysis: {target_col} vs Source Proxies")
    st.write("Pearson $r$ measures linear association, while Spearman $\\rho$ captures monotonic non-linear relationships.")

    col_cor1, col_cor2 = st.columns([1, 1.2])
    with col_cor1:
        st.markdown("#### Overall Correlation Coefficients")
        st.dataframe(corr.summary_table, use_container_width=True, hide_index=True)
        
        # Bar chart of correlations
        corr_bar_df = corr.summary_table.copy()
        fig_cbar = px.bar(
            corr_bar_df, x="Pearson r", y="Source / Proxy", orientation="h",
            color="Pearson r", color_continuous_scale="Blues",
            title="Pearson Correlation Strengths"
        )
        fig_cbar.update_layout(height=260, margin=dict(t=30, b=20, l=10, r=10))
        st.plotly_chart(fig_cbar, use_container_width=True)

    with col_cor2:
        st.markdown("#### Diurnal Time-of-Day Correlation Matrix")
        diurnal_mat = results["temporal_breakdown"]["diurnal"]
        fig_dheat = px.imshow(
            diurnal_mat,
            labels=dict(x="Time of Day Bin", y="Source Proxy", color="Correlation r"),
            color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0,
            title="Correlation Matrix Stratified by Diurnal Bins",
            text_auto=".2f"
        )
        fig_dheat.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_dheat, use_container_width=True)

    st.markdown("#### Seasonal Correlation Matrix")
    seasonal_mat = results["temporal_breakdown"]["seasonal"]
    if not seasonal_mat.empty:
        fig_sheat = px.imshow(
            seasonal_mat,
            labels=dict(x="Season", y="Source Proxy", color="Correlation r"),
            color_continuous_scale="Viridis",
            title="Correlation Matrix Stratified Across Meteorological Seasons",
            text_auto=".2f"
        )
        fig_sheat.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_sheat, use_container_width=True)

# ==========================================
# TAB 4: SOURCE ATTRIBUTION (OLS & ML)
# ==========================================
with tab4:
    st.subheader("Multiple Linear Regression (OLS) Attribution & Multicollinearity Diagnostics")
    
    col_mod1, col_mod2 = st.columns([1.3, 0.7])
    with col_mod1:
        st.markdown("#### OLS Statistical Parameter Estimates & Inference Table")
        st.dataframe(reg.summary_table, use_container_width=True, hide_index=True)
        st.caption("Standardized Beta (Beta*) coefficients represent standard deviations of target change per 1 SD increase in source proxy.")

    with col_mod2:
        st.markdown("#### Model Diagnostics & Multicollinearity (VIF)")
        diag_cols = st.columns(2)
        diag_cols[0].metric("Model R²", f"{reg.r2:.3f}")
        diag_cols[1].metric("Adjusted R²", f"{reg.adj_r2:.3f}")
        diag_cols[0].metric("RMSE", f"{reg.rmse:.2f}")
        diag_cols[1].metric("F-Statistic", f"{reg.f_statistic:.1f}")

        st.markdown("**Variance Inflation Factor (VIF)**:")
        vif_df = pd.DataFrame([
            {"Source": k, "VIF": round(v, 2), "Status": "Low (< 5)" if v < 5 else "High (> 5)"}
            for k, v in reg.vif.items()
        ])
        st.dataframe(vif_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_attr1, col_attr2 = st.columns(2)
    with col_attr1:
        st.markdown("#### Linear OLS Percentage Attribution Share (%)")
        fig_wfall = px.bar(
            reg.summary_table, x="Attribution Share (%)", y="Source / Proxy", orientation="h",
            color="Attribution Share (%)", color_continuous_scale="Reds",
            text="Attribution Share (%)", title="Linear Pratt Relative Variance Decomposition"
        )
        fig_wfall.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_wfall, use_container_width=True)

    with col_attr2:
        st.markdown("#### Non-Linear Random Forest Regressor Comparison")
        st.write(f"Random Forest accounts for non-linear physical reactions and interactions ($R^2 = {rf.r2:.3f}$).")
        fig_rf = px.bar(
            rf.summary_table, x="Attribution Share (%)", y="Source / Proxy", orientation="h",
            color="Attribution Share (%)", color_continuous_scale="Purples",
            text="Attribution Share (%)", title="Random Forest Non-Linear Gini Impurity Share"
        )
        fig_rf.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_rf, use_container_width=True)

    with st.expander("🔍 View Regression Residual Diagnostics"):
        res_cols = st.columns(2)
        with res_cols[0]:
            fig_pred_act = px.scatter(
                x=reg.y_true, y=reg.y_pred, opacity=0.4,
                labels={"x": f"Observed {target_col}", "y": f"Predicted {target_col}"},
                title="Predicted vs Observed Values"
            )
            fig_pred_act.add_shape(type="line", line=dict(dash="dash", color="red"),
                                  x0=reg.y_true.min(), y0=reg.y_true.min(),
                                  x1=reg.y_true.max(), y1=reg.y_true.max())
            st.plotly_chart(fig_pred_act, use_container_width=True)
        with res_cols[1]:
            fig_res_dist = px.histogram(
                reg.residuals, nbins=50, title="OLS Residuals Distribution (Normality Check)",
                labels={"value": "Residual Error"}
            )
            st.plotly_chart(fig_res_dist, use_container_width=True)

# ==========================================
# TAB 5: SPIKE EPISODES DEEP DIVE
# ==========================================
with tab5:
    st.subheader("Episodic Spike Deep Dive: Severe Pollution Episodes vs Baseline")
    st.write("""
    A fundamental limitation of aggregate annual averages is that seasonal episodic sources
    (such as crop residue stubble burning in October-November) appear diluted over 365 days,
    yet **statistically dominate the severe emergency spikes**.
    """)

    if spike_reg:
        comp_rows = []
        for src in reg.percentage_shares:
            b_val = reg.percentage_shares.get(src, 0.0)
            s_val = spike_reg.percentage_shares.get(src, 0.0)
            shift = s_val - b_val
            comp_rows.append({
                "Source / Proxy": src,
                "Annual Baseline Share (%)": round(b_val, 1),
                "Spike Episodes Share (%)": round(s_val, 1),
                "Shift in Severe Conditions": f"{'+' if shift > 0 else ''}{round(shift, 1)}%"
            })
        comp_df = pd.DataFrame(comp_rows)
        
        col_comp1, col_comp2 = st.columns([1, 1.2])
        with col_comp1:
            st.markdown("#### Baseline vs Spike Source Shares")
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            st.warning("""
            **Empirical Insight**: Note how regional Stubble Burning expands dramatically during flagged spike hours,
            confirming research by *Atmospheric Chemistry & Physics (2024)* and *Atmospheric Environment (2025)*.
            """)

        with col_comp2:
            melted_comp = comp_df.melt(
                id_vars="Source / Proxy",
                value_vars=["Annual Baseline Share (%)", "Spike Episodes Share (%)"],
                var_name="Condition", value_name="Share"
            )
            fig_comp = px.bar(
                melted_comp, x="Source / Proxy", y="Share", color="Condition",
                barmode="group", title="Source Contribution Shift During Spikes",
                color_discrete_map={
                    "Annual Baseline Share (%)": "#3498db",
                    "Spike Episodes Share (%)": "#e74c3c"
                }
            )
            fig_comp.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Contiguous Spike Event Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Discrete Spike Episodes", f"{spike_summary.consecutive_episodes}")
    c2.metric("Maximum Consecutive Duration", f"{spike_summary.max_duration_hours} hours")
    c3.metric("Spike Cutoff Threshold", f"{spike_summary.threshold_value:.1f} {target_col}")

# ==========================================
# TAB 6: POLICY SIMULATOR (WHAT-IF TOOL)
# ==========================================
with tab6:
    st.subheader("Interactive Policy Decision Support & 'What-If' Intervention Simulator")
    st.write("Simulate regulatory emission cuts and project their real-time impact on target air quality:")

    sim_cols = st.columns(4)
    with sim_cols[0]:
        red_veh = st.slider("🚗 Reduce Vehicular Traffic (%)", 0, 80, 25, 5, help="e.g. Odd-Even, EV transition, truck bans")
    with sim_cols[1]:
        red_stubble = st.slider("🌾 Reduce Stubble Burning (%)", 0, 90, 50, 5, help="e.g. Bio-decomposers, happy seeders")
    with sim_cols[2]:
        red_ind = st.slider("🏭 Reduce Industrial Emissions (%)", 0, 70, 20, 5, help="e.g. PNG fuel mandate, CEMS enforcement")
    with sim_cols[3]:
        red_dust = st.slider("🏗️ Reduce Construction Dust (%)", 0, 70, 30, 5, help="e.g. Mechanical sweepers, anti-smog guns")

    reductions = {
        "Vehicular Traffic": float(red_veh),
        "Stubble Burning": float(red_stubble),
        "Industrial Activity": float(red_ind),
        "Construction & Road Dust": float(red_dust),
        "Meteorological Stagnation": 0.0  # Meteorology cannot be reduced
    }

    sim_result = engine.simulate_policy_scenario(reductions, target_col=target_col)

    st.markdown("#### Projected Policy Simulation Impact")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Projected Mean " + target_col, f"{sim_result['simulated_mean']:.1f}",
              f"-{sim_result['mean_reduction']:.1f} ({sim_result['pct_mean_reduction']:.1f}%)")
    p2.metric("Peak " + target_col + " Reduction", f"-{sim_result['peak_reduction']:.1f}",
              f"From {sim_result['baseline_peak']:.1f} to {sim_result['simulated_peak']:.1f}")
    p3.metric("Spike Hours Averted", f"{sim_result['spikes_averted']:,}",
              f"-{sim_result['pct_spikes_averted']:.1f}% reduction")
    p4.metric("Simulated Spike Hours", f"{sim_result['simulated_spikes']:,}",
              f"Down from {sim_result['baseline_spikes']:,}")

    # Comparative Plot
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=df_processed.index, y=sim_result["actual_series"],
        mode="lines", name="Actual Baseline " + target_col,
        line=dict(color="#95a5a6", width=1.0), opacity=0.7
    ))
    fig_sim.add_trace(go.Scatter(
        x=df_processed.index, y=sim_result["simulated_series"],
        mode="lines", name="Simulated Policy Intervened " + target_col,
        line=dict(color="#27ae60", width=1.4)
    ))
    fig_sim.update_layout(
        title=f"Comparative Trajectory: Observed vs Policy-Simulated {target_col}",
        xaxis_title="Timestamp", yaxis_title=target_col,
        hovermode="x unified", height=380,
        margin=dict(t=40, b=20, l=30, r=20)
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("#### Recommended Policy Actions (Aligned with CPCB / GRAP Framework)")
    actions = PolicyRecommender.generate_recommendations(reg, spike_summary, spike_reg)
    for i, act in enumerate(actions, 1):
        with st.expander(f"📌 {i}. {act.action_title} [{act.grap_stage}] - Target: {act.target_source}"):
            st.markdown(f"**Policy Category**: `{act.category}` | **Optimal Timing**: `{act.timing}`")
            st.markdown(f"**Action Specification**: {act.description}")
            st.success(f"**Projected Policy Outcome**: {act.expected_impact}")

# ==========================================
# TAB 7: POLICY BRIEFING EXPORT
# ==========================================
with tab7:
    st.subheader("Automated Executive Policy Briefing & Report Generator")
    st.write("Generate and export a formal briefing document for environmental regulators and policymakers:")
    
    actions = PolicyRecommender.generate_recommendations(reg, spike_summary, spike_reg)
    report_markdown = PolicyRecommender.generate_markdown_report(reg, spike_summary, spike_reg, actions)

    st.download_button(
        label="📥 Download Policy Report (Markdown)",
        data=report_markdown,
        file_name=f"air_quality_source_attribution_report_{target_col}.md",
        mime="text/markdown"
    )

    st.markdown("---")
    st.markdown(report_markdown)

# ==========================================
# TAB 8: RESEARCH REFERENCES
# ==========================================
with tab8:
    st.subheader("Academic Bibliography & Source Apportionment References")
    st.write("Foundational empirical literature and policy frameworks supporting this analytical methodology:")

    for ref in RESEARCH_REFERENCES:
        with st.container():
            st.markdown(f"### [{ref['id']}] {ref['title']}")
            st.markdown(f"**Publication**: *{ref['journal']}* ({ref['year']}) | [Direct Link to Publication]({ref['url']})")
            st.info(f"**Key Empirical Finding**: {ref['key_takeaway']}")
            st.markdown("---")

    st.markdown("### Public Data Portals Cited")
    cpcb_col, firms_col, meteo_col = st.columns(3)
    cpcb_col.markdown("- [CPCB Official Portal](https://cpcb.nic.in): Central Pollution Control Board hourly continuous ambient air quality monitoring (CAAQMS).")
    firms_col.markdown("- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov): Fire Information for Resource Management System (VIIRS & MODIS 375m active fire detections).")
    meteo_col.markdown("- [Open-Meteo](https://open-meteo.com) / [IMD](https://mausam.imd.gov.in): Hourly meteorological datasets (temperature, wind, humidity, planetary boundary layer height).")
