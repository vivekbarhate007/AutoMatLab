"""Tests for oracle."""

import numpy as np
import pandas as pd
import pytest

from automatlabs.oracle import Oracle


def test_oracle_initialization():
    """Test oracle initialization."""
    df = pd.DataFrame({"id": [0, 1, 2], "band_gap": [1.0, 2.0, 3.0]})
    oracle = Oracle(df, target_col="band_gap", id_col="id")

    assert len(oracle.lookup) == 3
    assert oracle.lookup[0] == 1.0
    assert len(oracle.queried_ids) == 0


def test_oracle_query():
    """Test oracle querying."""
    df = pd.DataFrame({"id": [0, 1, 2], "band_gap": [1.0, 2.0, 3.0]})
    oracle = Oracle(df, target_col="band_gap", id_col="id")

    # Query single ID
    values = oracle.query([0])
    assert len(values) == 1
    assert values[0] == 1.0
    assert 0 in oracle.queried_ids

    # Query multiple IDs
    values = oracle.query([1, 2])
    assert len(values) == 2
    assert values[0] == 2.0
    assert values[1] == 3.0
    assert len(oracle.queried_ids) == 3

    # Query non-existent ID
    with pytest.raises(ValueError):
        oracle.query([999])


def test_oracle_get_best_observed():
    """Test getting best observed value."""
    df = pd.DataFrame({"id": [0, 1, 2], "band_gap": [1.0, 2.0, 3.0]})
    oracle = Oracle(df, target_col="band_gap", id_col="id")

    # No queries yet
    assert oracle.get_best_observed() is None

    # Query some IDs
    oracle.query([0, 2])
    best = oracle.get_best_observed()
    assert best == 3.0

    # Query with specific IDs
    best_specific = oracle.get_best_observed(ids=[0])
    assert best_specific == 1.0


