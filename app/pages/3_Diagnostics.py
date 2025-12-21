"""Diagnostics page for Streamlit app."""

import streamlit as st
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

st.header("Model Diagnostics")

# Find runs directory (relative to project root)
project_root = Path(__file__).parent.parent.parent
runs_dir = project_root / "runs"
if not runs_dir.exists():
    st.warning("No runs directory found. Run experiments first.")
    st.stop()

# List available runs
run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)

if not run_folders:
    st.warning("No experiment runs found.")
    st.stop()

# Select run
run_names = [f.name for f in run_folders]
selected_run = st.selectbox("Select Run", run_names)

if selected_run:
    run_path = runs_dir / selected_run

    # Check if this is a multi-method run
    subdirs = [d for d in run_path.iterdir() if d.is_dir()]
    methods = ["active_learning", "random", "greedy"]
    
    is_multi_method = any(m in [d.name for d in subdirs] for m in methods)

    if is_multi_method:
        # Multi-method run - show diagnostics for each method
        st.subheader("Select Method")
        selected_method = st.selectbox("Choose method to view diagnostics", methods)
        run_path = run_path / selected_method
    
    # Check for learning curve
    lc_file = run_path / "learning_curve.csv"
    candidates_file = run_path / "selected_candidates.csv"

    if lc_file.exists():
        st.subheader("Model Performance Over Time")
        lc = pd.read_csv(lc_file)

        # Check which metrics are available
        has_test_mae = "test_mae" in lc.columns
        has_test_r2 = "test_r2" in lc.columns
        has_best_found = "best_found" in lc.columns

        if has_test_mae or has_test_r2:
            fig = go.Figure()

            if has_test_mae:
                fig.add_trace(
                    go.Scatter(
                        x=lc["iteration"],
                        y=lc["test_mae"],
                        mode="lines+markers",
                        name="Test MAE",
                        line=dict(width=2, color="red"),
                    )
                )

            if has_test_r2:
                fig.add_trace(
                    go.Scatter(
                        x=lc["iteration"],
                        y=lc["test_r2"],
                        mode="lines+markers",
                        name="Test R²",
                        yaxis="y2" if has_test_mae else "y",
                        line=dict(width=2, color="blue"),
                    )
                )

            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Iteration",
                yaxis_title="Test MAE" if has_test_mae else "Test R²",
                yaxis2=dict(title="Test R²", overlaying="y", side="right") if (has_test_mae and has_test_r2) else None,
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            cols = []
            if has_test_mae:
                cols.append(("Final MAE", f"{lc['test_mae'].iloc[-1]:.4f}"))
            if has_test_r2:
                cols.append(("Final R²", f"{lc['test_r2'].iloc[-1]:.4f}"))
            if has_best_found:
                cols.append(("Best Found", f"{lc['best_found'].max():.4f}"))
            
            if cols:
                col_widgets = st.columns(len(cols))
                for i, (label, value) in enumerate(cols):
                    with col_widgets[i]:
                        st.metric(label, value)
        else:
            st.info("Performance metrics (test_mae, test_r2) not available in this run.")

    # Uncertainty calibration (only for active learning runs)
    if candidates_file.exists():
        st.subheader("Uncertainty Calibration")
        df = pd.read_csv(candidates_file)

        if "predicted_mean" in df.columns and "predicted_std" in df.columns and "true_value" in df.columns:
            # Compute prediction errors
            df["error"] = abs(df["predicted_mean"] - df["true_value"])

            # Scatter plot: uncertainty vs error
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["predicted_std"],
                    y=df["error"],
                    mode="markers",
                    name="Predictions",
                    marker=dict(size=5, opacity=0.6),
                )
            )

            # Add diagonal line (perfect calibration)
            max_val = max(df["predicted_std"].max(), df["error"].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(dash="dash", color="red"),
                )
            )

            fig.update_layout(
                title="Uncertainty Calibration: Predicted Std vs Actual Error",
                xaxis_title="Predicted Standard Deviation",
                yaxis_title="Absolute Prediction Error",
                hovermode="closest",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Correlation
            correlation = df["predicted_std"].corr(df["error"])
            st.metric("Uncertainty-Error Correlation", f"{correlation:.4f}")

            if correlation > 0.5:
                st.success("✅ Good uncertainty calibration (high correlation)")
            elif correlation > 0.3:
                st.warning("⚠️ Moderate uncertainty calibration")
            else:
                st.error("❌ Poor uncertainty calibration (low correlation)")
        else:
            st.info("ℹ️ Uncertainty calibration is only available for active learning runs. Baseline methods (random, greedy) don't include uncertainty estimates.")
            if "true_value" in df.columns:
                st.write(f"Found {len(df)} selected candidates in this run.")
