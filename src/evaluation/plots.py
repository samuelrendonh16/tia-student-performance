"""
Generación de gráficas de evaluación.

Cada función guarda una imagen en disco y retorna la ruta.
Usa matplotlib (no plotly) por simplicidad y reproducibilidad.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger


def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_names: list[str],
    output_path: str | Path,
    figsize: tuple[int, int] = (6, 5),
    dpi: int = 100,
) -> Path:
    """
    Guarda una matriz de confusión como heatmap.

    Parameters
    ----------
    cm : array-like 2D
        Matriz de confusión (filas = real, columnas = predicho).
    class_names : list[str]
        Nombres de las clases en orden.
    output_path : str | Path
        Ruta del PNG de salida.
    figsize : tuple
        Tamaño en pulgadas.
    dpi : int
        Resolución.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info(f"Matriz de confusión guardada: {output_path}")
    return output_path


def plot_roc_curve(
    fpr: list | np.ndarray,
    tpr: list | np.ndarray,
    auc: float,
    output_path: str | Path,
    figsize: tuple[int, int] = (7, 6),
    dpi: int = 100,
) -> Path:
    """
    Guarda la curva ROC con AUC anotado.

    Parameters
    ----------
    fpr, tpr : array-like
        Puntos de la curva ROC.
    auc : float
        Valor AUC para mostrar en la leyenda.
    output_path : str | Path
        Ruta del PNG de salida.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Azar")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info(f"Curva ROC guardada: {output_path}")
    return output_path


def plot_feature_importances(
    importances: list[dict],
    output_path: str | Path,
    figsize: tuple[int, int] = (9, 6),
    dpi: int = 100,
) -> Path:
    """
    Guarda un gráfico horizontal de importancia de variables.

    Parameters
    ----------
    importances : list[dict]
        Lista de {"feature": str, "importance": float} ya ordenada.
    output_path : str | Path
        Ruta del PNG de salida.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = [d["feature"] for d in importances]
    values = [d["importance"] for d in importances]

    fig, ax = plt.subplots(figsize=figsize)
    # Reversed para que la más importante quede arriba
    ax.barh(features[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de Variables")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info(f"Importancia de variables guardada: {output_path}")
    return output_path