# AutoMatLab 🔬

**End-to-end closed-loop materials discovery system with active learning**

AutoMatLab is a complete system for discovering promising materials candidates using machine learning and active learning. It combines property prediction, uncertainty estimation, acquisition functions, and baselines in a reproducible, dashboard-driven workflow.

## Overview

AutoMatLab implements a closed-loop active learning system that:

1. **Predicts properties** using ML models with uncertainty estimation
2. **Selects candidates** using acquisition functions (UCB, Expected Improvement)
3. **Simulates experiments** via an oracle that reveals ground-truth labels
4. **Compares baselines** (random search, greedy exploitation)
5. **Visualizes results** in an interactive Streamlit dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoMatLab System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Data       │───▶│  Features    │───▶│   Model      │ │
│  │  Loading     │    │ Engineering  │    │  Training    │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                  │         │
│  ┌──────────────┐    ┌──────────────┐          │         │
│  │   Oracle     │◀───│ Acquisition  │◀──────────┘         │
│  │  (Ground     │    │  Function    │                    │
│  │   Truth)     │    │   (UCB/EI)   │                    │
│  └──────────────┘    └──────────────┘                    │
│         │                                                  │
│         └──────────────────────────────────────────────────┘
│                           │
│                           ▼
│                  ┌──────────────┐
│                  │   Results    │
│                  │  Dashboard   │
│                  └──────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

### Installation

```bash
# Clone repository
cd AutoMatLab

# Install dependencies
make setup
# OR
pip install -e ".[dev]"
```

### Run Experiment

```bash
# Run all methods (active learning + baselines)
make run-experiment
# OR
python -m automatlabs.run --config configs/default.yaml --method all

# Run specific method
python -m automatlabs.run --method active_learning
python -m automatlabs.run --method random
python -m automatlabs.run --method greedy
```

### Launch Dashboard

```bash
make app
# OR
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
AutoMatLab/
├── configs/              # Configuration files
│   └── default.yaml
├── data/                  # Datasets (sample CSV included)
│   └── sample.csv
├── src/automatlabs/       # Core package
│   ├── config.py         # Configuration management
│   ├── data.py           # Data loading & splitting
│   ├── features.py       # Feature engineering
│   ├── models.py         # ML models with uncertainty
│   ├── acquisition.py    # Acquisition functions
│   ├── oracle.py         # Ground-truth oracle
│   ├── loop.py           # Active learning loop
│   ├── baselines.py      # Baseline methods
│   ├── evaluation.py     # Metrics computation
│   └── utils.py          # Utilities
├── scripts/               # Utility scripts
│   ├── download_data.py
│   └── run_experiment.py
├── app/                   # Streamlit dashboard
│   ├── streamlit_app.py
│   └── pages/
│       ├── run_explorer.py
│       ├── candidates.py
│       └── diagnostics.py
├── tests/                 # Unit tests
├── runs/                  # Experiment outputs (generated)
└── artifacts/             # Processed data/models (generated)
```

## How It Works

### Active Learning Loop

1. **Initialization**: Start with a small labeled seed set
2. **Training**: Train ML model on labeled data
3. **Prediction**: Predict mean and uncertainty for unlabeled pool
4. **Acquisition**: Select candidates using UCB (mean + κ × std)
5. **Query**: Oracle reveals ground-truth labels
6. **Update**: Add queried samples to labeled set
7. **Repeat**: Iterate until budget exhausted

### Why Active Learning?

- **Random Search**: Explores randomly, inefficient use of budget
- **Greedy**: Exploits current best predictions, may get stuck in local optima
- **Active Learning (UCB)**: Balances exploration (high uncertainty) and exploitation (high predicted value)

Active learning typically finds better candidates faster by intelligently selecting informative samples.

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
dataset_name: "matbench_mp_gap"  # or "sample_csv"
target_property: "band_gap"
seed_size: 50
budget_iterations: 20
batch_size: 5
model_type: "random_forest"
acquisition_type: "ucb"
ucb_kappa: 2.0
```

## Datasets

### Matbench (Default)

AutoMatLab supports Matbench datasets via the `matbench` package:

- `matbench_mp_gap`: Band gap prediction (default)
- `matbench_jdft2d`: 2D material properties
- `matbench_steels`: Steel properties

Datasets are loaded on-demand. No manual download required.

### Sample CSV

A small sample dataset (`data/sample.csv`) is included for quick testing. Generate more samples:

```bash
python scripts/download_data.py --generate-sample --n-samples 1000
```

## Outputs

Each experiment run creates a directory under `runs/` with:

- `config.yaml`: Configuration used
- `learning_curve.csv`: Best found value and metrics per iteration
- `selected_candidates.csv`: All queried candidates with predictions
- `final_topk.csv`: Top-K discovered candidates
- `metrics.json`: Final performance metrics

## Adding a New Acquisition Function

1. Add function to `src/automatlabs/acquisition.py`:

```python
def my_acquisition(mean: np.ndarray, std: np.ndarray, **kwargs) -> np.ndarray:
    """My custom acquisition function."""
    return mean + kwargs.get("alpha", 1.0) * std**2
```

2. Update `AcquisitionFunction.compute()` to handle new type
3. Add to config options in `config.py` and `default.yaml`
4. Test in `tests/test_acquisition.py`

## Reproducing Results

Experiments are fully reproducible with fixed random seeds:

```bash
# Same config + seed = same results
python -m automatlabs.run --config configs/default.yaml --method all
```

All runs save their configuration, ensuring reproducibility.

## Extending the System

### Graph Neural Networks (GNNs)

To add GNN support:

1. Create `src/automatlabs/models_gnn.py` with GNN model wrapper
2. Update `UncertaintyModel` to support GNN backend
3. Add structure-based features in `features.py`
4. Update config to include `model_type: "gnn"`

### Multi-Objective Optimization

To support multiple objectives:

1. Extend `AcquisitionFunction` to handle vectorized objectives
2. Implement Pareto-optimal acquisition (e.g., Expected Hypervolume Improvement)
3. Update dashboard to visualize Pareto fronts
4. Modify `loop.py` to track multiple objectives

### Custom Datasets

To use your own dataset:

1. Format CSV with columns: `id`, `composition`, `<target_property>`
2. Place in `data/` directory
3. Update config: `dataset_name: "sample_csv"` (or add loader in `data.py`)

## Development

### Running Tests

```bash
make test
# OR
pytest tests/ -v
```

### Code Formatting

```bash
make format
# OR
black src/ tests/ scripts/ app/
ruff check --fix src/ tests/ scripts/ app/
```

### Linting

```bash
make lint
# OR
ruff check src/ tests/ scripts/
```

## Docker

Run in container:

```bash
docker build -t automatlab .
docker run -p 8501:8501 automatlab
```

## CI/CD

GitHub Actions workflow runs on push/PR:
- Linting (ruff)
- Format checking (black)
- Tests (pytest)
- Coverage reporting

## Citation

If you use AutoMatLab, please cite:

- **Matbench**: Dunn, A., et al. (2020). "Benchmarking materials property prediction methods: the Matbench test set for automated machine learning." *npj Computational Materials*, 6(1), 1-13.
- **Matminer**: Ward, L., et al. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." *npj Computational Materials*, 2(1), 1-7.

## License

MIT License - see LICENSE file for details.

## Screenshots

*[Placeholder for Streamlit dashboard screenshots]*

- **Overview**: System architecture and explanation
- **Run Explorer**: Compare methods, view learning curves
- **Candidates**: Browse discovered materials, filter by properties
- **Diagnostics**: Model performance, uncertainty calibration

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on GitHub.

