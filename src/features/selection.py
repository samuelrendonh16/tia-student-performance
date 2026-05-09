"""
Selección de variables para el modelo.

Elimina columnas que el análisis exploratorio (TIA2) identificó como
irrelevantes para predecir el rendimiento académico.
"""
from collections.abc import Sequence

import pandas as pd
from loguru import logger


def drop_irrelevant_columns(
    df: pd.DataFrame,
    columns_to_drop: Sequence[str],
) -> pd.DataFrame:
    """
    Elimina columnas irrelevantes del DataFrame.

    Es una función pura: no modifica el DataFrame original,
    devuelve una copia con las columnas eliminadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con todas las columnas.
    columns_to_drop : Sequence[str]
        Lista de nombres de columnas a eliminar. Si alguna no existe
        en el DataFrame, se ignora silenciosamente (warning).

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame sin las columnas indicadas.
    """
    cols_existentes = [c for c in columns_to_drop if c in df.columns]
    cols_no_encontradas = set(columns_to_drop) - set(cols_existentes)

    if cols_no_encontradas:
        logger.warning(
            f"Columnas no encontradas (ignoradas): {sorted(cols_no_encontradas)}"
        )

    logger.info(
        f"Eliminando {len(cols_existentes)} columna(s): {cols_existentes}"
    )
    return df.drop(columns=cols_existentes).copy()


def select_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Selecciona explícitamente las columnas de entrada al modelo.

    Más estricta que drop: si falta alguna columna esperada, falla.
    Esto protege contra cambios silenciosos en el dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    feature_columns : Sequence[str]
        Lista exacta de columnas que deben estar presentes y ser
        seleccionadas.

    Returns
    -------
    pd.DataFrame
        DataFrame con exactamente las columnas indicadas, en ese orden.

    Raises
    ------
    KeyError
        Si alguna de las columnas esperadas no está en el DataFrame.
    """
    faltantes = set(feature_columns) - set(df.columns)
    if faltantes:
        raise KeyError(
            f"Faltan columnas requeridas para el modelo: {sorted(faltantes)}"
        )

    return df[list(feature_columns)].copy()