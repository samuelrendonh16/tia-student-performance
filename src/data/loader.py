"""
Carga del dataset crudo de Student Performance.

Punto único de entrada para leer el CSV. Garantiza que todo el código
downstream recibe datos validados contra el schema.
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.schema import validate_raw_data


def load_raw_data(
    path: str | Path,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Carga el CSV crudo y opcionalmente lo valida contra el schema.

    Esta es una función pura: dado el mismo input, siempre produce
    el mismo output. No modifica archivos ni mantiene estado.

    Parameters
    ----------
    path : str | Path
        Ruta al CSV crudo.
    validate : bool, default True
        Si es True, valida contra RAW_SCHEMA. Si es False, solo lee.
        Útil desactivarlo para inspección manual o tests.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos crudos validados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    pandera.errors.SchemaErrors
        Si los datos no cumplen el contrato (solo si validate=True).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {path.resolve()}"
        )

    logger.info(f"Cargando dataset desde {path}")
    df = pd.read_csv(path)
    logger.info(f"  → {len(df):,} filas, {df.shape[1]} columnas")

    if validate:
        logger.info("Validando contra el schema...")
        df = validate_raw_data(df)
        logger.info("  → Schema OK")

    return df