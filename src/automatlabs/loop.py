"""Active learning loop implementation."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from automatlabs.acquisition import AcquisitionFunction, select_candidates
from automatlabs.evaluation import compute_metrics
from automatlabs.features import engineer_features
from automatlabs.models import UncertaintyModel
from automatlabs.oracle import Oracle
from automatlabs.utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """Active learning loop with uncertainty-based acquisition."""

    def __init__(
        self,
        seed_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        oracle: Oracle,
        config,
        output_dir: Path,
    ):
        """Initialize active learning loop.

        Args:
            seed_df: Initial labeled seed set
            pool_df: Unlabeled pool
            test_df: Test set for evaluation
            oracle: Oracle instance
            config: Experiment configuration
            output_dir: Output directory for results
        """
        self.seed_df = seed_df.copy()
        self.pool_df = pool_df.copy()
        self.test_df = test_df.copy()
        self.oracle = oracle
        self.config = config
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

        # Initialize model
        self.model = UncertaintyModel(
            model_type=config.model_type,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_bootstrap_models=config.n_bootstrap_models,
            random_seed=config.random_seed,
        )

        # Initialize acquisition function
        self.acquisition = AcquisitionFunction(
            acquisition_type=config.acquisition_type,
            ucb_kappa=config.ucb_kappa,
        )

        # Track progress
        self.labeled_df = self.seed_df.copy()
        self.learning_curve = []
        self.selected_candidates = []
        self.feature_scaler = None
        self.all_elements = None

    def run(self) -> dict:
        """Run active learning loop.

        Returns:
            Dictionary with results
        """
        logger.info("Starting active learning loop")

        # Fit scaler on all data (seed + pool + test) to ensure consistent feature space
        all_df = pd.concat([self.labeled_df, self.pool_df, self.test_df], ignore_index=False)
        _, self.feature_scaler, self.all_elements = engineer_features(
            all_df,
            composition_col="composition",
            feature_type=self.config.feature_type,
            normalize=self.config.normalize_features,
            scaler=None,
            all_elements=None,
        )

        # Engineer features for test set (for evaluation)
        X_test, _, _ = engineer_features(
            self.test_df,
            composition_col="composition",
            feature_type=self.config.feature_type,
            normalize=self.config.normalize_features,
            scaler=self.feature_scaler,
            all_elements=self.all_elements,
        )
        y_test = self.test_df[self.config.target_property].values

        # Initial model training
        X_seed, _, _ = engineer_features(
            self.labeled_df,
            composition_col="composition",
            feature_type=self.config.feature_type,
            normalize=self.config.normalize_features,
            scaler=self.feature_scaler,
            all_elements=self.all_elements,
        )
        y_seed = self.labeled_df[self.config.target_property].values
        self.model.fit(X_seed, y_seed)

        # Initial evaluation
        mean_test, _ = self.model.predict(X_test)
        metrics = compute_metrics(y_test, mean_test)
        best_found = self.labeled_df[self.config.target_property].max()

        self.learning_curve.append(
            {
                "iteration": 0,
                "best_found": float(best_found),
                "test_mae": float(metrics["mae"]),
                "test_rmse": float(metrics["rmse"]),
                "test_r2": float(metrics["r2"]),
                "n_labeled": len(self.labeled_df),
            }
        )

        logger.info(
            f"Initial: best_found={best_found:.4f}, test_mae={metrics['mae']:.4f}"
        )

        # Active learning iterations
        for iteration in range(self.config.budget_iterations):
            if len(self.pool_df) == 0:
                logger.warning("Pool exhausted. Stopping early.")
                break

            # Predict on pool
            X_pool, _, _ = engineer_features(
                self.pool_df,
                composition_col="composition",
                feature_type=self.config.feature_type,
                normalize=self.config.normalize_features,
                scaler=self.feature_scaler,
                all_elements=self.all_elements,
            )
            mean_pool, std_pool = self.model.predict(X_pool)

            # Compute acquisition scores
            best_observed = self.oracle.get_best_observed(
                ids=self.labeled_df.index.tolist()
            )
            if best_observed is None:
                best_observed = self.labeled_df[self.config.target_property].max()

            acquisition_scores = self.acquisition.compute(
                mean_pool, std_pool, best_observed=best_observed
            )

            # Select candidates
            selected_indices = select_candidates(
                acquisition_scores, self.config.batch_size, method="top"
            )
            selected_df = self.pool_df.iloc[selected_indices].copy()
            selected_ids = selected_df.index.tolist()

            # Query oracle
            true_values = self.oracle.query(selected_ids)

            # Add to labeled set
            selected_df[self.config.target_property] = true_values
            self.labeled_df = pd.concat([self.labeled_df, selected_df], ignore_index=False)
            self.pool_df = self.pool_df.drop(selected_df.index)

            # Retrain model
            X_labeled, _, _ = engineer_features(
                self.labeled_df,
                composition_col="composition",
                feature_type=self.config.feature_type,
                normalize=self.config.normalize_features,
                scaler=self.feature_scaler,
                all_elements=self.all_elements,
            )
            y_labeled = self.labeled_df[self.config.target_property].values
            self.model.fit(X_labeled, y_labeled)

            # Evaluate on test set
            mean_test, _ = self.model.predict(X_test)
            metrics = compute_metrics(y_test, mean_test)
            best_found = self.labeled_df[self.config.target_property].max()

            self.learning_curve.append(
                {
                    "iteration": iteration + 1,
                    "best_found": float(best_found),
                    "test_mae": float(metrics["mae"]),
                    "test_rmse": float(metrics["rmse"]),
                    "test_r2": float(metrics["r2"]),
                    "n_labeled": len(self.labeled_df),
                }
            )

            # Track selected candidates
            for i, (idx, row) in enumerate(selected_df.iterrows()):
                self.selected_candidates.append(
                    {
                        "id": idx,
                        "composition": row["composition"],
                        "predicted_mean": float(mean_pool[selected_indices[i]]),
                        "predicted_std": float(std_pool[selected_indices[i]]),
                        "true_value": float(true_values[i]),
                        "acquisition_score": float(acquisition_scores[selected_indices[i]]),
                        "iteration": iteration + 1,
                    }
                )

            logger.info(
                f"Iteration {iteration + 1}: best_found={best_found:.4f}, "
                f"test_mae={metrics['mae']:.4f}, test_r2={metrics['r2']:.4f}"
            )

        # Save results
        self._save_results()

        return {
            "learning_curve": pd.DataFrame(self.learning_curve),
            "selected_candidates": pd.DataFrame(self.selected_candidates),
            "final_labeled": self.labeled_df,
            "metrics": metrics,
        }

    def _save_results(self) -> None:
        """Save experiment results."""
        # Learning curve
        learning_curve_df = pd.DataFrame(self.learning_curve)
        learning_curve_df.to_csv(self.output_dir / "learning_curve.csv", index=False)

        # Selected candidates
        candidates_df = pd.DataFrame(self.selected_candidates)
        candidates_df.to_csv(self.output_dir / "selected_candidates.csv", index=False)

        # Top-k candidates
        top_k = self.config.top_k
        top_candidates = (
            self.labeled_df.nlargest(top_k, self.config.target_property)[
                ["composition", self.config.target_property]
            ]
            .copy()
        )
        top_candidates.reset_index(inplace=True)
        top_candidates.rename(columns={"index": "id"}, inplace=True)
        top_candidates.to_csv(self.output_dir / "final_topk.csv", index=False)

        # Metrics
        final_metrics = {
            "final_best_found": float(self.labeled_df[self.config.target_property].max()),
            "final_n_labeled": len(self.labeled_df),
            "final_iteration": len(self.learning_curve) - 1,
        }
        save_json(final_metrics, self.output_dir / "metrics.json")

        logger.info(f"Results saved to {self.output_dir}")


