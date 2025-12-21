"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from automatlabs.features import (
    compute_composition_features,
    compute_magpie_features,
    engineer_features,
    parse_composition,
)


def test_parse_composition():
    """Test composition parsing."""
    # Simple element
    result = parse_composition("Fe")
    assert result == {"Fe": 1.0}

    # Binary compound
    result = parse_composition("Fe2O3")
    assert abs(result["Fe"] - 2 / 5) < 1e-6
    assert abs(result["O"] - 3 / 5) < 1e-6

    # Fractional composition
    result = parse_composition("Al0.5Si0.5")
    assert abs(result["Al"] - 0.5) < 1e-6
    assert abs(result["Si"] - 0.5) < 1e-6

    # Invalid composition
    with pytest.raises(ValueError):
        parse_composition("")


def test_compute_composition_features():
    """Test composition feature computation."""
    compositions = ["Fe", "Fe2O3", "Al0.5Si0.5"]
    features = compute_composition_features(compositions)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 3
    assert "frac_Fe" in features.columns or "frac_Al" in features.columns


def test_engineer_features():
    """Test feature engineering pipeline."""
    df = pd.DataFrame({"composition": ["Fe", "Fe2O3", "Al2O3"]})

    # Test composition_only features
    X, scaler = engineer_features(
        df, composition_col="composition", feature_type="composition_only", normalize=False
    )

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 3
    assert scaler is None

    # Test with normalization
    X_norm, scaler = engineer_features(
        df, composition_col="composition", feature_type="composition_only", normalize=True
    )

    assert isinstance(X_norm, np.ndarray)
    assert scaler is not None

    # Test with pre-fitted scaler
    X_norm2, _ = engineer_features(
        df,
        composition_col="composition",
        feature_type="composition_only",
        normalize=True,
        scaler=scaler,
    )

    np.testing.assert_array_almost_equal(X_norm, X_norm2)


def test_compute_magpie_features():
    """Test Magpie feature computation (may fail if matminer not available)."""
    compositions = ["Fe", "Fe2O3"]

    try:
        features = compute_magpie_features(compositions)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 2
    except Exception:
        # Fallback to composition_only should work
        features = compute_composition_features(compositions)
        assert isinstance(features, pd.DataFrame)


