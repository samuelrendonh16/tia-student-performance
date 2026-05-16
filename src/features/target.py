"""
Creación de la variable objetivo binaria.

Convierte exam_score (continuo) en aprobado (binario) usando un umbral.
Soporta dos estrategias: mediana del propio dataset o umbral fijo.
"""
from typing import Literal

import pandas as pd
from loguru import logger

Strategy = Literal["median", "fixed"]


def compute_threshold(
    series: pd.Series,
    strategy: Strategy = "median",
    fixed_threshold: float = 18.0,
) -> float:
    """
    Calcula el umbral de binarización según la estrategia.

    Parameters
    ----------
    series : pd.Series
        Serie con los puntajes (típicamente exam_score del train).
    strategy : {"median", "fixed"}
        - "median": calcula la mediana de la serie.
        - "fixed": retorna fixed_threshold sin mirar los datos.
    fixed_threshold : float
        Umbral usado cuando strategy="fixed".

    Returns
    -------
    float
        Umbral de binarización.

    Raises
    ------
    ValueError
        Si la estrategia no es válida.
    """
    if strategy == "median":
        umbral = float(series.median())
        logger.info(f"Umbral por mediana: {umbral:.2f}")
        return umbral

    if strategy == "fixed":
        logger.info(f"Umbral fijo: {fixed_threshold:.2f}")
        return float(fixed_threshold)

    raise ValueError(
        f"Estrategia inválida: '{strategy}'. Debe ser 'median' o 'fixed'."
    )


def binarize_score(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
    threshold: float,
    drop_source: bool = True,
) -> pd.DataFrame:
    """
    Crea la variable objetivo binaria a partir de una columna numérica.

    Regla:
        target = 1 si source_column >= threshold
        target = 0 en caso contrario

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    source_column : str
        Columna numérica de origen (ej. "exam_score").
    target_column : str
        Nombre de la columna objetivo a crear (ej. "aprobado").
    threshold : float
        Umbral de binarización. Debe haberse calculado con datos
        de entrenamiento (ver compute_threshold).
    drop_source : bool, default True
        Si True, elimina la columna de origen tras crear el target.
        Esto previene leakage: el modelo NO debe ver exam_score.

    Returns
    -------
    pd.DataFrame
        DataFrame con la nueva columna target. La columna source se
        elimina si drop_source=True.

    Raises
    ------
    KeyError
        Si source_column no existe.
    ValueError
        Si target_column ya existe en el DataFrame.
    """
    if source_column not in df.columns:
        raise KeyError(f"No existe la columna fuente: '{source_column}'")

    if target_column in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_column}' ya existe en el DataFrame"
        )

    df_out = df.copy()
    df_out[target_column] = (df_out[source_column] >= threshold).astype(int)

    balance = df_out[target_column].mean()
    logger.info(
        f"Target '{target_column}' creado: {balance:.1%} positivos, "
        f"{1 - balance:.1%} negativos"
    )

    if drop_source:
        df_out = df_out.drop(columns=[source_column])

    return df_out