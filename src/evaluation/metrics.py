"""
Cálculo de métricas de clasificación.

Funciones puras que reciben y_true, y_pred (y opcionalmente y_proba)
y devuelven diccionarios serializables con las métricas.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_basic_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Calcula las métricas escalares básicas de clasificación.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.

    Returns
    -------
    dict[str, float]
        Diccionario con accuracy, precision, recall, f1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """
    Calcula la matriz de confusión en forma serializable.

    Returns
    -------
    dict
        - "matrix": lista 2D con conteos
        - "tn", "fp", "fn", "tp": valores individuales (asume binario)
    """
    cm = confusion_matrix(y_true, y_pred)
    result = {"matrix": cm.tolist()}

    # Para clasificación binaria, exponer las 4 celdas con nombre
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return result


def compute_roc_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba_positive: np.ndarray,
) -> dict[str, Any]:
    """
    Calcula AUC-ROC y los puntos de la curva ROC.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales (0 / 1).
    y_proba_positive : np.ndarray
        Probabilidad predicha de la clase positiva (clase 1).

    Returns
    -------
    dict
        - "auc": valor AUC
        - "fpr", "tpr": listas para graficar la curva
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
    auc = roc_auc_score(y_true, y_proba_positive)

    return {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def compute_classification_report_dict(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Genera el classification report como diccionario.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    class_names : list[str] | None
        Nombres legibles de las clases (en orden: 0, 1, ...).

    Returns
    -------
    dict
        Reporte por clase con precision, recall, f1, support.
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def compute_feature_importances(
    pipeline,
    feature_names: list[str],
    top_n: int | None = None,
) -> list[dict[str, float]]:
    """
    Extrae la importancia de variables del clasificador del pipeline.

    Funciona con cualquier estimador que tenga `feature_importances_`
    (árboles, random forest, gradient boosting).

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline | sklearn.pipeline.Pipeline
        Pipeline entrenado con un paso llamado "classifier".
    feature_names : list[str]
        Nombres de las features en el orden en que entran al modelo.
    top_n : int | None
        Si se especifica, retorna solo las top_n más importantes.

    Returns
    -------
    list[dict]
        Lista ordenada de mayor a menor importancia. Cada elemento:
        {"feature": str, "importance": float}
    """
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError(
            f"{type(classifier).__name__} no expone feature_importances_"
        )

    importances = classifier.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Inconsistencia: {len(importances)} importancias vs "
            f"{len(feature_names)} nombres de features"
        )

    pairs = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, importances, strict=True)
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)

    if top_n is not None:
        pairs = pairs[:top_n]

    return pairs