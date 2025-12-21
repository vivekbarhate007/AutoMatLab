"""Acquisition functions for active learning."""

import logging
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


def ucb_acquisition(
    mean: np.ndarray, std: np.ndarray, kappa: float = 2.0
) -> np.ndarray:
    """Upper Confidence Bound acquisition function.

    Args:
        mean: Mean predictions
        std: Standard deviation predictions
        kappa: Exploration-exploitation trade-off parameter

    Returns:
        Acquisition scores (higher is better)
    """
    return mean + kappa * std


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function.

    Args:
        mean: Mean predictions
        std: Standard deviation predictions
        best_observed: Best observed value so far
        xi: Exploration parameter

    Returns:
        Acquisition scores (higher is better)
    """
    z = (mean - best_observed - xi) / (std + 1e-9)
    ei = std * (z * _norm_cdf(z) + _norm_pdf(z))
    return ei


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution function of standard normal."""
    return 0.5 * (1 + np.sign(x) * (1 - np.exp(-2 * x**2 / np.pi) ** 0.5))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Probability density function of standard normal."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def select_candidates(
    acquisition_scores: np.ndarray,
    batch_size: int,
    method: Literal["top", "diverse"] = "top",
) -> np.ndarray:
    """Select candidates based on acquisition scores.

    Args:
        acquisition_scores: Acquisition scores for all candidates
        batch_size: Number of candidates to select
        method: Selection method ("top" or "diverse")

    Returns:
        Indices of selected candidates
    """
    if method == "top":
        # Simple top-k selection
        indices = np.argsort(acquisition_scores)[::-1][:batch_size]
        return indices
    elif method == "diverse":
        # Greedy diverse selection (simplified)
        selected = []
        scores_copy = acquisition_scores.copy()

        for _ in range(batch_size):
            idx = np.argmax(scores_copy)
            selected.append(idx)
            scores_copy[idx] = -np.inf  # Mark as selected

        return np.array(selected)
    else:
        raise ValueError(f"Unknown selection method: {method}")


class AcquisitionFunction:
    """Acquisition function wrapper."""

    def __init__(
        self,
        acquisition_type: Literal["ucb", "ei"] = "ucb",
        ucb_kappa: float = 2.0,
        best_observed: Optional[float] = None,
    ):
        """Initialize acquisition function.

        Args:
            acquisition_type: Type of acquisition function
            ucb_kappa: Kappa parameter for UCB
            best_observed: Best observed value (for EI)
        """
        self.acquisition_type = acquisition_type
        self.ucb_kappa = ucb_kappa
        self.best_observed = best_observed

    def compute(
        self, mean: np.ndarray, std: np.ndarray, best_observed: Optional[float] = None
    ) -> np.ndarray:
        """Compute acquisition scores.

        Args:
            mean: Mean predictions
            std: Standard deviation predictions
            best_observed: Best observed value (for EI, overrides self.best_observed)

        Returns:
            Acquisition scores
        """
        if self.acquisition_type == "ucb":
            return ucb_acquisition(mean, std, kappa=self.ucb_kappa)
        elif self.acquisition_type == "ei":
            best = best_observed if best_observed is not None else self.best_observed
            if best is None:
                raise ValueError("best_observed must be provided for Expected Improvement")
            return expected_improvement(mean, std, best)
        else:
            raise ValueError(f"Unknown acquisition_type: {self.acquisition_type}")

