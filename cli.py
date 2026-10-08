"""
Command-Line Interface for Air Pollution Source Attribution System
===================================================================
Run attribution analysis on any dataset from the command line:
    py -3.14 cli.py --data data/delhi_ncr_benchmark_hourly.csv --target AQI
"""

import os
import sys
import argparse
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure module path is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import AttributionEngine
from utils.policy_recommender import PolicyRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Air Pollution Source Attribution System - Statistical & ML Analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/delhi_ncr_benchmark_hourly.csv",
        help="Path to hourly pollution CSV dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="AQI",
        help="Target pollutant or index column to attribute (e.g. AQI, PM2.5)"
    )
    parser.add_argument(
        "--spike-method",
        type=str,
        default="p90",
        choices=["p90", "p95", "two_sigma", "severe_400"],
        help="Spike detection threshold method"
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional path to output generated markdown report file"
    )

    args = parser.parse_args()

    # Resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        sys.exit(1)

    print("=" * 78)
    print("      AIR POLLUTION SOURCE ATTRIBUTION SYSTEM - ANALYSIS ENGINE")
    print("=" * 78)
    print(f"Loading dataset: {data_path}")
    print(f"Target Column:   {args.target}")
    print(f"Spike Method:    {args.spike_method}")
    print("-" * 78)

    engine = AttributionEngine()
    df_processed = engine.load_data(data_path)
    print(f"Dataset loaded and cleaned: {len(df_processed):,} hourly records.")
    print("Running statistical and machine learning attribution pipeline...")

    results = engine.run_full_attribution(target_col=args.target, spike_method=args.spike_method)

    spike_summary = results["spike_summary"]
    reg = results["regression_result"]
    corr = results["correlation_result"]
    rf = results["random_forest_result"]
    spike_reg = results["spike_regression"]

    print("\n" + "=" * 78)
    print(" 1. SPIKE DETECTION SUMMARY")
    print("=" * 78)
    print(f"Total Hours Monitored:   {spike_summary.total_hours:,}")
    print(f"Spike Threshold Value:   {spike_summary.threshold_value:.2f}")
    print(f"Flagged Spike Hours:     {spike_summary.spike_hours:,} ({spike_summary.spike_percentage:.1f}%)")
    print(f"Contiguous Episodes:     {spike_summary.consecutive_episodes}")
    print(f"Max Episode Duration:    {spike_summary.max_duration_hours} consecutive hours")

    print("\n" + "=" * 78)
    print(" 2. MULTIPLE LINEAR REGRESSION (OLS) ATTRIBUTION RESULTS")
    print("=" * 78)
    print(f"Model R²: {reg.r2:.3f} | Adj R²: {reg.adj_r2:.3f} | RMSE: {reg.rmse:.2f} | MAE: {reg.mae:.2f}")
    print(f"F-Statistic: {reg.f_statistic:.2f} (p-value: {reg.p_value_f:.2e})")
    print("-" * 78)
    print(reg.summary_table.to_string(index=False))

    print("\n" + "=" * 78)
    print(" 3. NON-LINEAR RANDOM FOREST REGRESSOR IMPORTANCES")
    print("=" * 78)
    print(f"Model R²: {rf.r2:.3f} | RMSE: {rf.rmse:.2f}")
    print("-" * 78)
    print(rf.summary_table.to_string(index=False))

    if spike_reg:
        print("\n" + "=" * 78)
        print(" 4. EPISODIC SPIKE CONTRAST (Severe Smog vs Annual Baseline)")
        print("=" * 78)
        contrast_data = []
        for src in reg.percentage_shares:
            b_pct = reg.percentage_shares[src]
            s_pct = spike_reg.percentage_shares.get(src, 0.0)
            delta = s_pct - b_pct
            contrast_data.append({
                "Source / Proxy": src,
                "Baseline Share (%)": f"{b_pct:.1f}%",
                "Spike Share (%)": f"{s_pct:.1f}%",
                "Episode Shift": f"{'+' if delta > 0 else ''}{delta:.1f}%"
            })
        print(pd.DataFrame(contrast_data).to_string(index=False))

    print("\n" + "=" * 78)
    print(" 5. TARGETED REGULATORY POLICY ACTIONS")
    print("=" * 78)
    actions = PolicyRecommender.generate_recommendations(reg, spike_summary, spike_reg)
    for i, act in enumerate(actions, 1):
        print(f"\n[{i}] {act.action_title} ({act.grap_stage})")
        print(f"    Target Source:    {act.target_source}")
        print(f"    Operational Time: {act.timing}")
        print(f"    Intervention:     {act.description}")
        print(f"    Expected Impact:  {act.expected_impact}")

    if args.report_out:
        report_text = PolicyRecommender.generate_markdown_report(reg, spike_summary, spike_reg, actions)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[Success] Report saved to: {args.report_out}")

    print("\n" + "=" * 78)
    print("Analysis complete. Run 'streamlit run app.py' to launch interactive UI.")
    print("=" * 78)


if __name__ == "__main__":
    main()
