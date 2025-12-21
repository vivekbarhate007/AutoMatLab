"""Script to download or generate sample data."""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(output_path: Path, n_samples: int = 1000) -> None:
    """Generate a sample dataset for quickstart.

    Args:
        output_path: Path to save sample CSV
        n_samples: Number of samples to generate
    """
    import numpy as np

    # Simple compositions with band gaps
    compositions = [
        "Si",
        "Ge",
        "GaAs",
        "InP",
        "GaN",
        "ZnO",
        "TiO2",
        "Fe2O3",
        "Al2O3",
        "MgO",
        "CaO",
        "SrTiO3",
        "BaTiO3",
        "PbTiO3",
        "LiNbO3",
    ]

    # Generate synthetic data
    data = []
    np.random.seed(42)

    for i in range(n_samples):
        # Random composition from list or simple binary
        if np.random.random() < 0.7:
            comp = np.random.choice(compositions)
        else:
            # Generate random binary composition
            elem1 = np.random.choice(["Al", "Si", "Ga", "In", "Zn", "Ti", "Fe"])
            elem2 = np.random.choice(["O", "N", "P", "As", "S"])
            comp = f"{elem1}{elem2}"

        # Synthetic band gap (roughly correlated with composition)
        base_gap = {
            "Si": 1.1,
            "Ge": 0.67,
            "GaAs": 1.43,
            "InP": 1.35,
            "GaN": 3.4,
            "ZnO": 3.37,
            "TiO2": 3.2,
            "Fe2O3": 2.2,
            "Al2O3": 8.8,
            "MgO": 7.8,
            "CaO": 7.0,
            "SrTiO3": 3.2,
            "BaTiO3": 3.2,
            "PbTiO3": 3.4,
            "LiNbO3": 3.9,
        }.get(comp, 2.0)

        # Add noise
        band_gap = base_gap + np.random.normal(0, 0.3)
        band_gap = max(0.1, band_gap)  # Ensure positive

        data.append({"id": i, "composition": comp, "band_gap": band_gap})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} samples and saved to {output_path}")


def download_matbench_data() -> None:
    """Download Matbench datasets (placeholder - datasets load on demand)."""
    logger.info("Matbench datasets are loaded on-demand. No download needed.")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download or generate sample data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample.csv"),
        help="Output path for sample CSV",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample CSV instead of downloading",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.generate_sample:
        generate_sample_data(args.output, args.n_samples)
    else:
        download_matbench_data()
        logger.info("To generate a sample CSV, use --generate-sample flag")


if __name__ == "__main__":
    main()


