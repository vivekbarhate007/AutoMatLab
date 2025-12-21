"""Main entry point for running experiments."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from automatlabs.baselines import BaselineRunner
from automatlabs.config import ExperimentConfig
from automatlabs.data import load_dataset, split_data, validate_dataset
from automatlabs.loop import ActiveLearningLoop
from automatlabs.oracle import Oracle
from automatlabs.utils import ensure_dir, save_json, set_random_seed, setup_logging

logger = setup_logging()


def run_experiment(config_path: Path, method: str, output_dir: Optional[Path] = None) -> None:
    """Run a single experiment.

    Args:
        config_path: Path to configuration YAML file
        method: Method to run ("active_learning", "random", "greedy", or "all")
        output_dir: Optional output directory (defaults to runs/<timestamp>)
    """
    # Load configuration
    config = ExperimentConfig.from_yaml(config_path)
    set_random_seed(config.random_seed)

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / f"{method}_{timestamp}"
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    # Save configuration
    import yaml

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    logger.info(f"Starting experiment: method={method}, output_dir={output_dir}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    df = load_dataset(config.dataset_name)

    # Validate dataset
    validate_dataset(df, config.target_property)

    # Split data
    seed_df, pool_df, test_df = split_data(
        df,
        target_col=config.target_property,
        seed_size=config.seed_size,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )

    # Create oracle (uses full dataset for ground truth)
    oracle = Oracle(df, target_col=config.target_property)

    # Run experiment based on method
    if method == "active_learning":
        loop = ActiveLearningLoop(seed_df, pool_df, test_df, oracle, config, output_dir)
        results = loop.run()
    elif method in ["random", "greedy"]:
        from automatlabs.features import engineer_features
        from automatlabs.models import UncertaintyModel

        # Initialize model and scaler
        model = UncertaintyModel(
            model_type=config.model_type,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_bootstrap_models=config.n_bootstrap_models,
            random_seed=config.random_seed,
        )

        X_seed, scaler, _ = engineer_features(
            seed_df,
            composition_col="composition",
            feature_type=config.feature_type,
            normalize=config.normalize_features,
            scaler=None,
            all_elements=None,
        )

        runner = BaselineRunner(
            method=method,
            seed_df=seed_df,
            pool_df=pool_df,
            test_df=test_df,
            oracle=oracle,
            model=model,
            feature_scaler=scaler,
            target_col=config.target_property,
            composition_col="composition",
            feature_type=config.feature_type,
            batch_size=config.batch_size,
            budget_iterations=config.budget_iterations,
            random_seed=config.random_seed,
        )
        results = runner.run()

        # Save results
        results["learning_curve"].to_csv(output_dir / "learning_curve.csv", index=False)
        results["selected_candidates"].to_csv(
            output_dir / "selected_candidates.csv", index=False
        )

        # Top-k candidates
        top_k = config.top_k
        top_candidates = (
            results["final_labeled"]
            .nlargest(top_k, config.target_property)[["composition", config.target_property]]
            .copy()
        )
        top_candidates.reset_index(inplace=True)
        top_candidates.rename(columns={"index": "id"}, inplace=True)
        top_candidates.to_csv(output_dir / "final_topk.csv", index=False)

        # Metrics
        final_metrics = {
            "final_best_found": float(
                results["final_labeled"][config.target_property].max()
            ),
            "final_n_labeled": len(results["final_labeled"]),
        }
        save_json(final_metrics, output_dir / "metrics.json")

    elif method == "all":
        # Run all methods
        methods = ["active_learning", "random", "greedy"]
        for m in methods:
            method_output_dir = output_dir / m
            run_experiment(config_path, m, method_output_dir)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Experiment completed. Results saved to {output_dir}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run AutoMatLab experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="active_learning",
        choices=["active_learning", "random", "greedy", "all"],
        help="Method to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to runs/<timestamp>)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    run_experiment(args.config, args.method, args.output_dir)


if __name__ == "__main__":
    main()

