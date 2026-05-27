"""
Ajuste de hiperparametros del modelo via GridSearchCV.

Busca la mejor combinacion de hiperparametros del arbol de decision
usando validacion cruzada, optimizando la metrica alineada con el
criterio de negocio.
"""
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.model_selection import GridSearchCV

from src.models.pipeline import build_pipeline


@dataclass
class TuningResult:
    """Resultado del ajuste de hiperparametros."""
    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    scoring: str
    n_combinations: int


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    model_config: dict,
    param_grid: dict,
    cv_folds: int = 5,
    scoring: str = "recall",
    random_state: int = 42,
) -> TuningResult:
    """
    Ejecuta GridSearchCV sobre el pipeline completo.

    El GridSearch opera sobre el pipeline entero (StandardScaler + SMOTE
    + arbol), por lo que el balanceo SMOTE se aplica correctamente dentro
    de cada fold de validacion cruzada, evitando data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    numeric_features : list[str]
        Columnas numericas para el pipeline.
    model_config : dict
        Seccion "model" del config (para construir el pipeline base).
    param_grid : dict
        Grilla de hiperparametros. Las claves usan el prefijo
        "classifier__" para apuntar al paso del arbol en el pipeline.
    cv_folds : int
        Numero de folds de validacion cruzada.
    scoring : str
        Metrica a optimizar. "recall" optimiza la clase positiva (1).
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    TuningResult
        Resultado con mejores parametros, score y tabla completa de CV.
    """
    # Construimos el pipeline base (con SMOTE y arbol)
    pipeline = build_pipeline(
        numeric_features=numeric_features,
        model_config=model_config,
        random_state=random_state,
    )

    logger.info(f"Iniciando GridSearchCV con scoring='{scoring}', cv={cv_folds}")

    # Contar combinaciones para informar al usuario
    n_combinations = 1
    for valores in param_grid.values():
        n_combinations *= len(valores)
    logger.info(
        f"Explorando {n_combinations} combinaciones "
        f"({n_combinations * cv_folds} entrenamientos en total)"
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1,        # usar todos los nucleos disponibles
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    logger.info(f"Mejor score ({scoring}): {grid.best_score_:.4f}")
    logger.info(f"Mejores parametros: {grid.best_params_}")

    cv_results = pd.DataFrame(grid.cv_results_)

    return TuningResult(
        best_params=grid.best_params_,
        best_score=float(grid.best_score_),
        cv_results=cv_results,
        scoring=scoring,
        n_combinations=n_combinations,
    )


def extract_tree_params(best_params: dict) -> dict:
    """
    Extrae los hiperparametros del arbol desde el resultado de GridSearch.

    GridSearch devuelve claves con prefijo "classifier__". Esta funcion
    las limpia para poder guardarlas en el config.yaml.

    Parameters
    ----------
    best_params : dict
        best_params_ de GridSearchCV (con prefijos "classifier__").

    Returns
    -------
    dict
        Hiperparametros sin prefijo, listos para el config.

    Example
    -------
    >>> extract_tree_params({"classifier__max_depth": 5})
    {'max_depth': 5}
    """
    prefijo = "classifier__"
    return {
        clave.replace(prefijo, ""): valor
        for clave, valor in best_params.items()
        if clave.startswith(prefijo)
    }