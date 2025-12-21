"""Tests for acquisition functions."""

import numpy as np
import pytest

from automatlabs.acquisition import (
    AcquisitionFunction,
    expected_improvement,
    select_candidates,
    ucb_acquisition,
)


def test_ucb_acquisition():
    """Test UCB acquisition function."""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    kappa = 2.0

    scores = ucb_acquisition(mean, std, kappa)

    assert len(scores) == 3
    assert scores[2] > scores[1] > scores[0]  # Higher mean + std should give higher score
    assert np.allclose(scores, mean + kappa * std)


def test_expected_improvement():
    """Test Expected Improvement acquisition function."""
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.1, 0.2, 0.3])
    best_observed = 1.5

    scores = expected_improvement(mean, std, best_observed)

    assert len(scores) == 3
    assert scores[2] > 0  # Should be positive for values above best_observed
    assert scores[0] < scores[2]  # Higher mean should give higher EI


def test_select_candidates():
    """Test candidate selection."""
    scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    batch_size = 3

    # Top selection
    indices = select_candidates(scores, batch_size, method="top")
    assert len(indices) == batch_size
    assert indices[0] == 1  # Highest score
    assert indices[1] == 4  # Second highest
    assert indices[2] == 2  # Third highest

    # Diverse selection (simplified, should still select top)
    indices_diverse = select_candidates(scores, batch_size, method="diverse")
    assert len(indices_diverse) == batch_size


def test_acquisition_function_class():
    """Test AcquisitionFunction wrapper class."""
    # UCB
    acq_ucb = AcquisitionFunction(acquisition_type="ucb", ucb_kappa=2.0)
    mean = np.array([1.0, 2.0])
    std = np.array([0.1, 0.2])
    scores = acq_ucb.compute(mean, std)
    assert len(scores) == 2

    # EI
    acq_ei = AcquisitionFunction(acquisition_type="ei", best_observed=1.5)
    scores_ei = acq_ei.compute(mean, std)
    assert len(scores_ei) == 2

    # EI without best_observed should raise error
    acq_ei_no_best = AcquisitionFunction(acquisition_type="ei")
    with pytest.raises(ValueError):
        acq_ei_no_best.compute(mean, std)


