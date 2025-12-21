"""Main Streamlit application."""

import streamlit as st

st.set_page_config(
    page_title="AutoMatLab Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 AutoMatLab Dashboard")
st.markdown(
    """
    **End-to-end closed-loop materials discovery system with active learning**
    
    This dashboard allows you to explore experiment results, compare methods, and visualize discovered materials candidates.
    """
)

# Streamlit automatically handles multi-page apps via the pages/ directory
# The numbered files (1_Run_Explorer.py, etc.) are automatically added to navigation
# This main page serves as the Overview/Home page

# This is the main/home page (Overview)
# Streamlit automatically creates navigation for files in pages/ directory
# Pages are accessible via the sidebar navigation

st.header("Overview")
st.markdown(
    """
    ## How It Works
    
    AutoMatLab implements a closed-loop active learning system for materials discovery:
    
    1. **Property Predictor**: ML model (Random Forest with uncertainty estimation) predicts target properties
    2. **Acquisition Policy**: Active learning selects promising candidates using UCB (Upper Confidence Bound)
    3. **Oracle**: Simulates experiments by revealing ground-truth labels from dataset
    4. **Baselines**: Compare against random search and greedy exploitation
    
    ## Active Learning Loop
    
    ```
    ┌─────────────────┐
    │  Labeled Set    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐      ┌──────────────┐
    │  Train Model    │─────▶│   Predict    │
    └─────────────────┘      └──────┬───────┘
                                     │
                                     ▼
    ┌─────────────────┐      ┌──────────────┐
    │  Unlabeled Pool │◀─────│ Acquisition  │
    └────────┬────────┘      └──────┬───────┘
             │                       │
             │                       │
             ▼                       │
    ┌─────────────────┐             │
    │  Query Oracle   │◀────────────┘
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Add to Labeled │
    └─────────────────┘
    ```
    
    ## Why Active Learning?
    
    - **Random Search**: Explores randomly, inefficient use of budget
    - **Greedy**: Exploits current best predictions, may get stuck in local optima
    - **Active Learning (UCB)**: Balances exploration (high uncertainty) and exploitation (high predicted value)
    
    ## Getting Started
    
    1. Run experiments: `python -m automatlabs.run --method all`
    2. Explore results in the **Run Explorer** page (see sidebar)
    3. View top candidates in the **Candidates** page (see sidebar)
    4. Check model diagnostics in the **Diagnostics** page (see sidebar)
    
    ## Navigation
    
    Use the sidebar to navigate between pages:
    - **Run Explorer**: Compare methods and view learning curves
    - **Candidates**: Browse discovered materials candidates
    - **Diagnostics**: Model performance and uncertainty calibration
    """
)

