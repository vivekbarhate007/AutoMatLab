"""Run Explorer page for Streamlit app."""

import streamlit as st
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

st.header("Run Explorer")

# Find runs directory (relative to project root)
# When Streamlit runs pages, we need to go up to project root
project_root = Path(__file__).parent.parent.parent
runs_dir = project_root / "runs"
if not runs_dir.exists():
    st.warning("No runs directory found. Run experiments first.")
    st.code("python -m automatlabs.run --method all")
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

    if any(m in [d.name for d in subdirs] for m in methods):
        # Multi-method run
        st.subheader("Compare Methods")
        learning_curves = {}
        metrics = {}

        for method in methods:
            method_path = run_path / method
            lc_file = method_path / "learning_curve.csv"
            metrics_file = method_path / "metrics.json"

            if lc_file.exists():
                learning_curves[method] = pd.read_csv(lc_file)
            if metrics_file.exists():
                import json

                with open(metrics_file) as f:
                    metrics[method] = json.load(f)

        if learning_curves:
            # Plot learning curves
            fig = go.Figure()

            for method, lc in learning_curves.items():
                fig.add_trace(
                    go.Scatter(
                        x=lc["iteration"],
                        y=lc["best_found"],
                        mode="lines+markers",
                        name=method.replace("_", " ").title(),
                        line=dict(width=2),
                    )
                )

            fig.update_layout(
                title="Best Found Value vs Iteration",
                xaxis_title="Iteration",
                yaxis_title="Best Found Value",
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Metrics table
            if metrics:
                metrics_df = pd.DataFrame(metrics).T
                st.subheader("Final Metrics")
                st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No learning curves found.")
    else:
        # Single method run
        lc_file = run_path / "learning_curve.csv"
        metrics_file = run_path / "metrics.json"

        if lc_file.exists():
            lc = pd.read_csv(lc_file)

            # Plot learning curve
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=lc["iteration"],
                    y=lc["best_found"],
                    mode="lines+markers",
                    name="Best Found",
                    line=dict(width=2, color="blue"),
                )
            )

            if "test_mae" in lc.columns:
                fig.add_trace(
                    go.Scatter(
                        x=lc["iteration"],
                        y=lc["test_mae"],
                        mode="lines+markers",
                        name="Test MAE",
                        yaxis="y2",
                        line=dict(width=2, color="red"),
                    )
                )

            fig.update_layout(
                title="Learning Curve",
                xaxis_title="Iteration",
                yaxis_title="Best Found Value",
                yaxis2=dict(title="Test MAE", overlaying="y", side="right")
                if "test_mae" in lc.columns
                else None,
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display metrics
            if metrics_file.exists():
                import json

                with open(metrics_file) as f:
                    metrics = json.load(f)
                st.subheader("Metrics")
                st.json(metrics)

        else:
            st.warning("Learning curve file not found.")
