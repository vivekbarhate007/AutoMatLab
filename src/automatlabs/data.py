"""Data loading and preprocessing."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_matbench_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a Matbench dataset."""
    try:
        from matbench import MatbenchBenchmark

        mb = MatbenchBenchmark(autoload=False)
        task = mb.tasks[dataset_name]
        task.load()

        # Convert to DataFrame
        df = pd.DataFrame(task.df)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "id"}, inplace=True)

        logger.info(f"Loaded {dataset_name}: {len(df)} samples")
        return df

    except Exception as e:
        logger.error(f"Failed to load Matbench dataset {dataset_name}: {e}")
        raise


def load_sample_csv(filepath: Path) -> pd.DataFrame:
    """Load sample CSV dataset."""
    df = pd.read_csv(filepath)
    if "id" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "id"}, inplace=True)
    logger.info(f"Loaded sample CSV: {len(df)} samples")
    return df


def load_dataset(dataset_name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load dataset by name."""
    if dataset_name.startswith("matbench_"):
        return load_matbench_dataset(dataset_name)
    elif dataset_name == "sample_csv":
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        csv_path = data_dir / "sample.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Sample CSV not found at {csv_path}")
        return load_sample_csv(csv_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def split_data(
    df: pd.DataFrame,
    target_col: str,
    seed_size: int,
    test_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into seed, unlabeled pool, and test set.

    Args:
        df: Full dataset
        target_col: Name of target column
        seed_size: Size of initial labeled seed set
        test_size: Fraction for test set
        random_seed: Random seed

    Returns:
        Tuple of (seed_df, pool_df, test_df)
    """
    # First split: test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Second split: seed and pool
    if seed_size >= len(train_val_df):
        logger.warning(
            f"Seed size {seed_size} >= available data {len(train_val_df)}. "
            "Using all data as seed."
        )
        seed_df = train_val_df.copy()
        pool_df = pd.DataFrame(columns=train_val_df.columns)
    else:
        seed_df = train_val_df.sample(n=seed_size, random_state=random_seed)
        pool_df = train_val_df.drop(seed_df.index)

    logger.info(
        f"Split data: seed={len(seed_df)}, pool={len(pool_df)}, test={len(test_df)}"
    )

    return seed_df, pool_df, test_df


def validate_dataset(df: pd.DataFrame, target_col: str, composition_col: str = "composition") -> None:
    """Validate dataset has required columns and no missing values in target."""
    required_cols = [composition_col, target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[target_col].isna().any():
        n_missing = df[target_col].isna().sum()
        logger.warning(f"Found {n_missing} missing values in {target_col}. Dropping them.")
        df.dropna(subset=[target_col], inplace=True)

    if len(df) == 0:
        raise ValueError("Dataset is empty after removing missing values")


