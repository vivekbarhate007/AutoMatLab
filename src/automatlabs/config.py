"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class ExperimentConfig(BaseSettings):
    """Experiment configuration model."""

    # Dataset configuration
    dataset_name: str = Field(default="matbench_mp_gap", description="Dataset name")
    target_property: str = Field(default="band_gap", description="Target property name")

    # Active learning configuration
    seed_size: int = Field(default=50, ge=1, description="Initial labeled samples")
    budget_iterations: int = Field(default=20, ge=1, description="Number of iterations")
    batch_size: int = Field(default=5, ge=1, description="Samples per iteration")
    random_seed: int = Field(default=42, description="Random seed")

    # Model configuration
    model_type: Literal["random_forest", "gradient_boosting"] = Field(
        default="random_forest", description="Model type"
    )
    n_estimators: int = Field(default=100, ge=1, description="Number of estimators")
    max_depth: int = Field(default=20, ge=1, description="Max tree depth")
    n_bootstrap_models: int = Field(
        default=5, ge=1, description="Number of bootstrap models for uncertainty"
    )

    # Acquisition function configuration
    acquisition_type: Literal["ucb", "ei"] = Field(
        default="ucb", description="Acquisition function type"
    )
    ucb_kappa: float = Field(default=2.0, ge=0.0, description="UCB exploration parameter")

    # Feature engineering
    feature_type: Literal["magpie", "composition_only"] = Field(
        default="magpie", description="Feature type"
    )
    normalize_features: bool = Field(default=True, description="Normalize features")

    # Evaluation
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Test set fraction")
    top_k: int = Field(default=10, ge=1, description="Number of top candidates")

    # Output
    output_dir: str = Field(default="runs", description="Output directory")
    save_models: bool = Field(default=True, description="Save trained models")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()

    class Config:
        """Pydantic configuration."""

        extra = "forbid"



