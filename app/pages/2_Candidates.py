"""Candidates page for Streamlit app."""

import streamlit as st
from pathlib import Path

import pandas as pd

st.header("Discovered Candidates")

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

    # Check for top-k file
    topk_file = run_path / "final_topk.csv"
    candidates_file = run_path / "selected_candidates.csv"

    if topk_file.exists():
        df = pd.read_csv(topk_file)
        st.subheader("Top-K Discovered Candidates")

        # Property filter
        if "band_gap" in df.columns:
            prop_col = "band_gap"
        else:
            prop_col = df.columns[-1]  # Assume last column is target

        min_val, max_val = float(df[prop_col].min()), float(df[prop_col].max())
        prop_range = st.slider(
            f"Filter by {prop_col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
        )

        filtered_df = df[
            (df[prop_col] >= prop_range[0]) & (df[prop_col] <= prop_range[1])
        ]

        st.dataframe(filtered_df, use_container_width=True)

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered candidates as CSV",
            data=csv,
            file_name="filtered_candidates.csv",
            mime="text/csv",
        )

    elif candidates_file.exists():
        df = pd.read_csv(candidates_file)
        st.subheader("All Selected Candidates")

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            if "iteration" in df.columns:
                iterations = sorted(df["iteration"].unique())
                selected_iterations = st.multiselect(
                    "Filter by iteration", iterations, default=iterations
                )
                df = df[df["iteration"].isin(selected_iterations)]

        with col2:
            if "true_value" in df.columns:
                min_val, max_val = float(df["true_value"].min()), float(df["true_value"].max())
                value_range = st.slider(
                    "Filter by true value",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                df = df[
                    (df["true_value"] >= value_range[0])
                    & (df["true_value"] <= value_range[1])
                ]

        st.dataframe(df, use_container_width=True)

        # Statistics
        if "true_value" in df.columns:
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Value", f"{df['true_value'].mean():.4f}")
            with col2:
                st.metric("Max Value", f"{df['true_value'].max():.4f}")
            with col3:
                st.metric("Count", len(df))
    else:
        st.warning("No candidate files found in this run.")
