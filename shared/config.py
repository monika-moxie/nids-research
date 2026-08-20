"""Configuration values for the shared NIDS baseline pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and column choices for UNSW-NB15 binary detection."""

    data_dir: Path = Path("data/raw/UNSW-NB15")
    train_file: str = "UNSW_NB15_training-set.csv"
    test_file: str = "UNSW_NB15_testing-set.csv"
    target_column: str = "label"
    leakage_columns: tuple[str, ...] = ("id", "attack_cat")

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file


@dataclass(frozen=True)
class TrainingConfig:
    """Small, defensible defaults for a first deep tabular baseline."""

    random_state: int = 42
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.20
    validation_fraction: float = 0.15
