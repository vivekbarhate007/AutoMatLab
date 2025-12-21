"""Oracle that simulates experiments by revealing ground-truth labels."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Oracle:
    """Oracle that reveals ground-truth labels from dataset."""

    def __init__(self, ground_truth_df: pd.DataFrame, target_col: str, id_col: str = "id"):
        """Initialize oracle with ground truth data.

        Args:
            ground_truth_df: DataFrame containing ground truth labels
            target_col: Name of target column
            id_col: Name of ID column
        """
        self.ground_truth_df = ground_truth_df.copy()
        self.target_col = target_col
        self.id_col = id_col
        self.queried_ids = set()

        # Create lookup dictionary
        self.lookup = dict(
            zip(
                self.ground_truth_df[self.id_col],
                self.ground_truth_df[self.target_col],
            )
        )

    def query(self, ids: list) -> np.ndarray:
        """Query oracle for ground-truth labels.

        Args:
            ids: List of IDs to query

        Returns:
            Array of ground-truth values
        """
        values = []
        for id_val in ids:
            if id_val not in self.lookup:
                raise ValueError(f"ID {id_val} not found in oracle ground truth")
            values.append(self.lookup[id_val])
            self.queried_ids.add(id_val)

        logger.info(f"Oracle queried {len(ids)} samples. Total queried: {len(self.queried_ids)}")
        return np.array(values)

    def get_best_observed(self, ids: Optional[list] = None) -> Optional[float]:
        """Get best observed value.

        Args:
            ids: Optional list of IDs to consider. If None, uses all queried IDs.

        Returns:
            Best observed value, or None if no queries made
        """
        if ids is None:
            ids = list(self.queried_ids)

        if not ids:
            return None

        values = [self.lookup[id_val] for id_val in ids if id_val in self.lookup]
        if not values:
            return None

        return max(values)


