"""Division train/test con configuracion centralizada."""
from typing import NamedTuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class SplitResult(NamedTuple):
    """Resultado de la division train/test."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> SplitResult:
    """Divide un DataFrame en train/test, separando target de features."""
    if target_column not in df.columns:
        raise KeyError(f"No existe la columna objetivo: '{target_column}'")

    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size debe estar en (0, 1), recibido: {test_size}"
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    logger.info(
        f"Split completado: train={len(X_train)}, test={len(X_test)} "
        f"(test_size={test_size}, stratify={stratify})"
    )

    return SplitResult(X_train, X_test, y_train, y_test)
