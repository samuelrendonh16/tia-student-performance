"""
Motor de recomendaciones.

Aplica todas las reglas de negocio sobre un DataFrame de estudiantes
y genera recomendaciones personalizadas + estadisticas de cobertura.
"""
from typing import Optional

import pandas as pd
from loguru import logger

from src.prescriptive.rules import ALL_RULES, DEFAULT_RECOMMENDATION


def apply_rules_to_row(row: pd.Series, thresholds: dict) -> str:
    """
    Aplica todas las reglas a un estudiante individual y concatena
    las recomendaciones activas.

    Parameters
    ----------
    row : pd.Series
        Una fila del DataFrame con los datos crudos del estudiante.
    thresholds : dict
        Diccionario con los umbrales de cada regla.

    Returns
    -------
    str
        Recomendaciones concatenadas con " | ". Si ninguna regla aplica,
        se devuelve la recomendacion por defecto.
    """
    recomendaciones = []
    for codigo, regla in ALL_RULES:
        resultado = regla(row, thresholds)
        if resultado is not None:
            recomendaciones.append(resultado)

    if not recomendaciones:
        return DEFAULT_RECOMMENDATION

    return " | ".join(recomendaciones)


def apply_rules_to_dataframe(
    df: pd.DataFrame,
    thresholds: dict,
    output_column: str = "recomendaciones",
) -> pd.DataFrame:
    """
    Aplica las reglas a todo un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos crudos de los estudiantes.
    thresholds : dict
        Umbrales de las reglas.
    output_column : str
        Nombre de la columna donde se guardan las recomendaciones.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con la nueva columna de recomendaciones.
    """
    df_out = df.copy()
    df_out[output_column] = df_out.apply(
        lambda row: apply_rules_to_row(row, thresholds), axis=1
    )

    logger.info(
        f"Reglas aplicadas a {len(df_out)} estudiantes. "
        f"Columna creada: '{output_column}'"
    )
    return df_out


def compute_rules_coverage(
    df: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """
    Calcula cuantos estudiantes activa cada regla.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos crudos.
    thresholds : dict
        Umbrales de las reglas.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: [codigo, descripcion, n_estudiantes, porcentaje]
        ordenado por n_estudiantes descendente.
    """
    descripciones = {
        "RN-01": "Estudio insuficiente",
        "RN-02": "Burnout alto",
        "RN-03": "Salud mental baja",
        "RN-04": "Distracciones altas",
        "RN-05": "Sueno insuficiente",
    }

    filas = []
    total = len(df)
    for codigo, regla in ALL_RULES:
        # Cuenta cuantas filas activan la regla
        activos = df.apply(lambda r: regla(r, thresholds) is not None, axis=1).sum()
        filas.append({
            "codigo": codigo,
            "descripcion": descripciones[codigo],
            "n_estudiantes": int(activos),
            "porcentaje": round(100 * activos / total, 2) if total > 0 else 0.0,
        })

    coverage_df = pd.DataFrame(filas)
    coverage_df = coverage_df.sort_values("n_estudiantes", ascending=False).reset_index(drop=True)
    return coverage_df
