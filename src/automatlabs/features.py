"""Feature engineering for materials compositions."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def parse_composition(composition: str) -> dict[str, float]:
    """Parse composition string to element fractions.

    Supports formats like:
    - "Fe2O3" -> {"Fe": 2/5, "O": 3/5}
    - "Al0.5Si0.5" -> {"Al": 0.5, "Si": 0.5}
    - "Fe" -> {"Fe": 1.0}

    Args:
        composition: Composition string

    Returns:
        Dictionary mapping element symbols to fractions
    """
    import re

    # Pattern to match element symbols and numbers
    pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
    matches = re.findall(pattern, composition)

    if not matches:
        raise ValueError(f"Could not parse composition: {composition}")

    elements = {}
    total = 0.0

    for element, count_str in matches:
        if count_str == "":
            count = 1.0
        else:
            count = float(count_str)
        elements[element] = count
        total += count

    # Normalize to fractions
    if total > 0:
        elements = {k: v / total for k, v in elements.items()}
    else:
        raise ValueError(f"Invalid composition: {composition}")

    return elements


def compute_magpie_features(compositions: list[str]) -> pd.DataFrame:
    """Compute Magpie features for compositions using matminer.

    Args:
        compositions: List of composition strings

    Returns:
        DataFrame with Magpie features
    """
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition

        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        comps = [Composition(comp) for comp in compositions]
        features = ep_feat.featurize_dataframe(
            pd.DataFrame({"composition": compositions}), col_id="composition"
        )
        # Remove composition column if present
        if "composition" in features.columns:
            features = features.drop(columns=["composition"])
        return features

    except Exception as e:
        logger.warning(f"Failed to compute Magpie features: {e}. Falling back to composition-only.")
        return compute_composition_features(compositions)


def compute_composition_features(
    compositions: list[str], all_elements: Optional[list[str]] = None
) -> pd.DataFrame:
    """Compute simple composition-based features (element fractions).

    Args:
        compositions: List of composition strings
        all_elements: Optional pre-determined list of all elements (for consistency)

    Returns:
        DataFrame with element fraction features
    """
    parsed_comps = []

    for comp in compositions:
        try:
            parsed = parse_composition(comp)
            parsed_comps.append(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse {comp}: {e}")
            parsed_comps.append({})

    # Determine element set
    if all_elements is None:
        # Collect all unique elements from compositions
        all_elements_set = set()
        for parsed in parsed_comps:
            all_elements_set.update(parsed.keys())
        all_elements = sorted(all_elements_set)
    # else: use provided all_elements as-is (for consistency with scaler)

    # Create feature matrix
    feature_data = []

    for parsed in parsed_comps:
        row = [parsed.get(elem, 0.0) for elem in all_elements]
        feature_data.append(row)

    feature_df = pd.DataFrame(feature_data, columns=[f"frac_{elem}" for elem in all_elements])
    return feature_df


def engineer_features(
    df: pd.DataFrame,
    composition_col: str = "composition",
    feature_type: str = "magpie",
    normalize: bool = True,
    scaler: Optional[StandardScaler] = None,
    all_elements: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[StandardScaler], Optional[list[str]]]:
    """Engineer features from compositions.

    Args:
        df: DataFrame with composition column
        composition_col: Name of composition column
        feature_type: Type of features ("magpie" or "composition_only")
        normalize: Whether to normalize features
        scaler: Optional pre-fitted scaler
        all_elements: Optional pre-determined list of all elements (for consistency)

    Returns:
        Tuple of (feature_array, scaler, all_elements)
    """
    compositions = df[composition_col].tolist()

    if feature_type == "magpie":
        feature_df = compute_magpie_features(compositions)
        # For magpie, we can't easily pre-determine features, so we'll handle it differently
        all_elements = None
    elif feature_type == "composition_only":
        feature_df = compute_composition_features(compositions, all_elements=all_elements)
        # Extract element list from feature columns for consistency
        if all_elements is None:
            all_elements = [col.replace("frac_", "") for col in feature_df.columns]
        else:
            # Ensure feature_df has exactly the columns we expect
            expected_cols = [f"frac_{elem}" for elem in all_elements]
            if set(feature_df.columns) != set(expected_cols):
                # Reorder/align columns
                new_df = pd.DataFrame(0.0, index=feature_df.index, columns=expected_cols)
                for col in feature_df.columns:
                    if col in new_df.columns:
                        new_df[col] = feature_df[col]
                feature_df = new_df
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Handle missing values
    feature_df = feature_df.fillna(0.0)

    # Ensure consistent columns if scaler is provided (for composition_only)
    if scaler is not None and feature_type == "composition_only" and all_elements is not None:
        expected_cols = [f"frac_{elem}" for elem in all_elements]
        # Create new dataframe with consistent columns
        new_df = pd.DataFrame(0.0, index=feature_df.index, columns=expected_cols)
        # Copy existing columns
        for col in feature_df.columns:
            if col in new_df.columns:
                new_df[col] = feature_df[col]
        feature_df = new_df
        logger.debug(f"Aligned columns: {len(feature_df.columns)} features (expected {len(expected_cols)})")

    # Convert to numpy array
    X = feature_df.values.astype(np.float32)

    # Normalize if requested
    if normalize:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

    logger.info(f"Engineered {X.shape[1]} features from {len(compositions)} compositions")
    return X, scaler, all_elements
