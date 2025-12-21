# Quick Start Guide

## Installation

```bash
cd AutoMatLab
pip install -e .
```

## Run Experiments

### Single Method
```bash
# Active learning
python -m automatlabs.run --config configs/default.yaml --method active_learning

# Random baseline
python -m automatlabs.run --config configs/default.yaml --method random

# Greedy baseline
python -m automatlabs.run --config configs/default.yaml --method greedy
```

### All Methods (Comparison)
```bash
python -m automatlabs.run --config configs/default.yaml --method all
```

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Expected Outputs

Each experiment creates a directory under `runs/` with:
- `config.yaml` - Configuration used
- `learning_curve.csv` - Performance metrics over iterations
- `selected_candidates.csv` - All queried candidates
- `final_topk.csv` - Top-K discovered candidates
- `metrics.json` - Final performance metrics

## Example Output

```
runs/
└── active_learning_20251221_093606/
    ├── config.yaml
    ├── learning_curve.csv
    ├── selected_candidates.csv
    ├── final_topk.csv
    └── metrics.json
```

## Configuration

Edit `configs/default.yaml` to customize:
- Dataset: `dataset_name: "sample_csv"` (or `"matbench_mp_gap"` if installed)
- Budget: `budget_iterations: 5`, `batch_size: 2`
- Model: `model_type: "random_forest"`, `n_estimators: 100`
- Acquisition: `acquisition_type: "ucb"`, `ucb_kappa: 2.0`

## Troubleshooting

### Matbench not available
The default config uses `sample_csv` which works out of the box. To use Matbench datasets:
```bash
pip install -e ".[matbench]"
```

### Feature consistency errors
Ensure `feature_type: "composition_only"` in config (Magpie features require matminer).

## Next Steps

1. Run experiments: `python -m automatlabs.run --method all`
2. Explore results in Streamlit dashboard
3. Compare methods using learning curves
4. View top discovered candidates

