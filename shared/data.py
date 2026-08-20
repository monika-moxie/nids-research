"""Data loading and preprocessing for the shared NIDS baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.config import DatasetConfig


@dataclass
class PreparedData:
    """Preprocessed arrays and metadata used by training and evaluation."""

    x_train: object
    y_train: object
    x_test: object
    y_test: object
    preprocessor: ColumnTransformer
    feature_names: list[str]


def load_unsw_nb15(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official UNSW-NB15 train and test CSV files."""

    missing = [path for path in (config.train_path, config.test_path) if not path.exists()]
    if missing:
        expected = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing UNSW-NB15 CSV file(s):\n"
            f"{expected}\n"
            "Download the official training and testing CSVs and place them under "
            f"{config.data_dir}."
        )

    return pd.read_csv(config.train_path), pd.read_csv(config.test_path)


def split_features_and_target(
    frame: pd.DataFrame,
    target_column: str,
    leakage_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the binary target and remove leakage columns."""

    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    columns_to_drop = [target_column]
    columns_to_drop.extend(column for column in leakage_columns if column in frame.columns)

    x = frame.drop(columns=columns_to_drop)
    y = frame[target_column].astype(int)
    return x, y


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Create a tabular preprocessing pipeline fitted only on training features."""

    categorical_columns = list(x_train.select_dtypes(include=["object", "category"]).columns)
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def prepare_data(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: DatasetConfig,
) -> PreparedData:
    """Fit preprocessing on train data and transform train/test data consistently."""

    x_train_raw, y_train = split_features_and_target(
        train_frame,
        config.target_column,
        config.leakage_columns,
    )
    x_test_raw, y_test = split_features_and_target(
        test_frame,
        config.target_column,
        config.leakage_columns,
    )

    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    return PreparedData(
        x_train=x_train,
        y_train=y_train.to_numpy(),
        x_test=x_test,
        y_test=y_test.to_numpy(),
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


def prepare_from_disk(config: DatasetConfig | None = None) -> PreparedData:
    """Load official CSVs from disk and return preprocessed arrays."""

    active_config = config or DatasetConfig()
    train_frame, test_frame = load_unsw_nb15(active_config)
    return prepare_data(train_frame, test_frame, active_config)


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
