"""Baseline methods for comparison."""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def random_baseline(
    pool_df: pd.DataFrame,
    batch_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Random baseline: randomly select candidates.

    Args:
        pool_df: Unlabeled pool DataFrame
        batch_size: Number of candidates to select
        random_seed: Random seed

    Returns:
        DataFrame of selected candidates
    """
    np.random.seed(random_seed)
    if len(pool_df) == 0:
        return pd.DataFrame(columns=pool_df.columns)

    n_select = min(batch_size, len(pool_df))
    selected = pool_df.sample(n=n_select, random_state=random_seed)
    return selected


def greedy_baseline(
    pool_df: pd.DataFrame,
    mean_predictions: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    """Greedy baseline: select candidates with highest predicted mean.

    Args:
        pool_df: Unlabeled pool DataFrame
        mean_predictions: Mean predictions for pool samples
        batch_size: Number of candidates to select

    Returns:
        DataFrame of selected candidates
    """
    if len(pool_df) == 0:
        return pd.DataFrame(columns=pool_df.columns)

    if len(mean_predictions) != len(pool_df):
        raise ValueError("mean_predictions length must match pool_df length")

    n_select = min(batch_size, len(pool_df))
    top_indices = np.argsort(mean_predictions)[::-1][:n_select]
    selected = pool_df.iloc[top_indices].copy()
    return selected


class BaselineRunner:
    """Runner for baseline methods."""

    def __init__(
        self,
        method: Literal["random", "greedy"],
        seed_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        oracle,
        model,
        feature_scaler,
        target_col: str,
        composition_col: str = "composition",
        feature_type: str = "magpie",
        batch_size: int = 5,
        budget_iterations: int = 20,
        random_seed: int = 42,
    ):
        """Initialize baseline runner.

        Args:
            method: Baseline method ("random" or "greedy")
            seed_df: Initial labeled seed set
            pool_df: Unlabeled pool
            test_df: Test set for evaluation
            oracle: Oracle instance
            model: Model instance
            feature_scaler: Feature scaler
            target_col: Target column name
            composition_col: Composition column name
            feature_type: Feature type
            batch_size: Batch size
            budget_iterations: Number of iterations
            random_seed: Random seed
        """
        self.method = method
        self.seed_df = seed_df.copy()
        self.pool_df = pool_df.copy()
        self.test_df = test_df.copy()
        self.oracle = oracle
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_col = target_col
        self.composition_col = composition_col
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.budget_iterations = budget_iterations
        self.random_seed = random_seed

        # Track progress
        self.labeled_df = self.seed_df.copy()
        self.learning_curve = []
        self.selected_candidates = []

    def run(self) -> dict:
        """Run baseline experiment.

        Returns:
            Dictionary with results
        """
        logger.info(f"Running {self.method} baseline")

        from automatlabs.features import engineer_features

        # Fit scaler on all data (seed + pool + test) to ensure consistent feature space
        all_df = pd.concat([self.seed_df, self.pool_df, self.test_df], ignore_index=False)
        _, self.feature_scaler, all_elements = engineer_features(
            all_df,
            composition_col=self.composition_col,
            feature_type=self.feature_type,
            normalize=True,
            scaler=None,
            all_elements=None,
        )

        # Initial evaluation
        X_test, _, _ = engineer_features(
            self.test_df,
            composition_col=self.composition_col,
            feature_type=self.feature_type,
            normalize=True,
            scaler=self.feature_scaler,
            all_elements=all_elements,
        )
        y_test = self.test_df[self.target_col].values

        for iteration in range(self.budget_iterations):
            if len(self.pool_df) == 0:
                logger.warning("Pool exhausted. Stopping early.")
                break

            # Train model on labeled set
            X_labeled, _, _ = engineer_features(
                self.labeled_df,
                composition_col=self.composition_col,
                feature_type=self.feature_type,
                normalize=True,
                scaler=self.feature_scaler,
                all_elements=all_elements,
            )
            y_labeled = self.labeled_df[self.target_col].values
            self.model.fit(X_labeled, y_labeled)

            # Select candidates
            if self.method == "random":
                selected_df = random_baseline(
                    self.pool_df, self.batch_size, self.random_seed + iteration
                )
            elif self.method == "greedy":
                X_pool, _, _ = engineer_features(
                    self.pool_df,
                    composition_col=self.composition_col,
                    feature_type=self.feature_type,
                    normalize=True,
                    scaler=self.feature_scaler,
                    all_elements=all_elements,
                )
                mean_pred, _ = self.model.predict(X_pool)
                selected_df = greedy_baseline(self.pool_df, mean_pred, self.batch_size)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Query oracle
            selected_ids = selected_df.index.tolist()
            true_values = self.oracle.query(selected_ids)

            # Add to labeled set
            selected_df[self.target_col] = true_values
            self.labeled_df = pd.concat([self.labeled_df, selected_df], ignore_index=False)
            self.pool_df = self.pool_df.drop(selected_df.index)

            # Evaluate on test set
            mean_test, _ = self.model.predict(X_test)
            best_found = self.labeled_df[self.target_col].max()
            test_mae = np.mean(np.abs(mean_test - y_test))

            self.learning_curve.append(
                {
                    "iteration": iteration + 1,
                    "best_found": float(best_found),
                    "test_mae": float(test_mae),
                    "n_labeled": len(self.labeled_df),
                }
            )

            # Track selected candidates
            for idx, row in selected_df.iterrows():
                self.selected_candidates.append(
                    {
                        "id": idx,
                        "composition": row[self.composition_col],
                        "true_value": float(row[self.target_col]),
                        "iteration": iteration + 1,
                    }
                )

            logger.info(
                f"Iteration {iteration + 1}: best_found={best_found:.4f}, "
                f"test_mae={test_mae:.4f}"
            )

        return {
            "learning_curve": pd.DataFrame(self.learning_curve),
            "selected_candidates": pd.DataFrame(self.selected_candidates),
            "final_labeled": self.labeled_df,
        }


