"""ML models with uncertainty estimation."""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class UncertaintyModel:
    """Model wrapper with uncertainty estimation via bootstrap ensemble."""

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 20,
        n_bootstrap_models: int = 5,
        random_seed: int = 42,
    ):
        """Initialize uncertainty model.

        Args:
            model_type: Type of base model ("random_forest" or "gradient_boosting")
            n_estimators: Number of estimators per model
            max_depth: Max depth of trees
            n_bootstrap_models: Number of bootstrap models for uncertainty
            random_seed: Random seed
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bootstrap_models = n_bootstrap_models
        self.random_seed = random_seed
        self.models: list = []
        self.is_fitted = False

    def _create_base_model(self, seed: int) -> RandomForestRegressor | GradientBoostingRegressor:
        """Create a base model instance."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=seed,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit bootstrap ensemble of models.

        Args:
            X: Feature matrix
            y: Target values
        """
        self.models = []
        n_samples = len(X)

        for i in range(self.n_bootstrap_models):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Create and fit model
            model = self._create_base_model(seed=self.random_seed + i)
            model.fit(X_boot, y_boot)
            self.models.append(model)

        self.is_fitted = True
        logger.info(f"Fitted {len(self.models)} bootstrap models")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty (std) for samples.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            Dictionary of metrics
        """
        mean_pred, _ = self.predict(X)

        mae = mean_absolute_error(y, mean_pred)
        rmse = np.sqrt(mean_squared_error(y, mean_pred))
        r2 = r2_score(y, mean_pred)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }


