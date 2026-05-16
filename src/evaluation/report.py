"""
Orquestador del reporte de evaluación.

Combina métricas + gráficas + metadatos en un único reporte JSON
y un resumen legible en Markdown.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.evaluation.metrics import (
    compute_basic_metrics,
    compute_classification_report_dict,
    compute_confusion_matrix,
    compute_feature_importances,
    compute_roc_metrics,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_roc_curve,
)


def build_full_report(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    class_names: list[str],
    output_dir: str | Path,
    figure_config: dict | None = None,
) -> dict[str, Any]:
    """
    Genera el reporte completo de evaluación.

    Calcula todas las métricas, dibuja todas las gráficas y persiste
    el resumen en JSON + Markdown.

    Parameters
    ----------
    pipeline : Pipeline entrenado.
    X_test, y_test : datos de prueba.
    feature_names : nombres de las features en orden.
    class_names : nombres legibles de las clases.
    output_dir : carpeta donde se guardarán los artefactos.
    figure_config : dict | None
        Configuración de las figuras (dpi, figsize).

    Returns
    -------
    dict
        El reporte completo (también persistido en disco).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_config = figure_config or {}
    dpi = figure_config.get("dpi", 100)

    logger.info("Calculando predicciones sobre el test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ─── Métricas ────────────────────────────────────────────────────
    logger.info("Calculando métricas...")
    basic = compute_basic_metrics(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)
    roc = compute_roc_metrics(y_test, y_proba)
    cls_report = compute_classification_report_dict(
        y_test, y_pred, class_names=class_names,
    )
    importances = compute_feature_importances(pipeline, feature_names)

    # ─── Gráficas ────────────────────────────────────────────────────
    logger.info("Generando gráficas...")
    cm_path = plot_confusion_matrix(
        cm["matrix"], class_names,
        output_dir / "confusion_matrix.png",
        figsize=tuple(figure_config.get("figsize_confusion", [6, 5])),
        dpi=dpi,
    )
    roc_path = plot_roc_curve(
        roc["fpr"], roc["tpr"], roc["auc"],
        output_dir / "roc_curve.png",
        figsize=tuple(figure_config.get("figsize_roc", [7, 6])),
        dpi=dpi,
    )
    imp_path = plot_feature_importances(
        importances,
        output_dir / "feature_importances.png",
        figsize=tuple(figure_config.get("figsize_importance", [9, 6])),
        dpi=dpi,
    )

    # ─── Reporte estructurado ────────────────────────────────────────
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_test_samples": int(len(X_test)),
            "class_names": class_names,
        },
        "metrics": basic,
        "confusion_matrix": cm,
        "roc": {"auc": roc["auc"]},   # los puntos están solo en la gráfica
        "classification_report": cls_report,
        "feature_importances": importances,
        "artifacts": {
            "confusion_matrix": str(cm_path.name),
            "roc_curve": str(roc_path.name),
            "feature_importances": str(imp_path.name),
        },
    }

    # Persistir JSON
    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Reporte JSON guardado: {json_path}")

    # Persistir Markdown legible
    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    logger.info(f"Reporte Markdown guardado: {md_path}")

    return report


def _render_markdown(report: dict) -> str:
    """Renderiza el reporte en Markdown legible."""
    m = report["metrics"]
    cm = report["confusion_matrix"]
    classes = report["metadata"]["class_names"]

    lines = [
        "# Reporte de Evaluación",
        "",
        f"**Generado:** {report['metadata']['generated_at']}",
        f"**Muestras de test:** {report['metadata']['n_test_samples']:,}",
        "",
        "## Métricas Globales",
        "",
        f"| Métrica | Valor |",
        f"|---|---|",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        f"| F1-Score | {m['f1']:.4f} |",
        f"| AUC-ROC | {report['roc']['auc']:.4f} |",
        "",
        "## Matriz de Confusión",
        "",
    ]

    if "tn" in cm:
        lines.extend([
            f"|  | Predicho: {classes[0]} | Predicho: {classes[1]} |",
            f"|---|---|---|",
            f"| **Real: {classes[0]}** | {cm['tn']} | {cm['fp']} |",
            f"| **Real: {classes[1]}** | {cm['fn']} | {cm['tp']} |",
            "",
        ])

    lines.extend([
        "## Top 10 Variables Más Importantes",
        "",
        "| # | Variable | Importancia |",
        "|---|---|---|",
    ])
    for i, item in enumerate(report["feature_importances"][:10], 1):
        lines.append(f"| {i} | {item['feature']} | {item['importance']:.4f} |")

    lines.extend([
        "",
        "## Archivos Generados",
        "",
        f"- `{report['artifacts']['confusion_matrix']}`",
        f"- `{report['artifacts']['roc_curve']}`",
        f"- `{report['artifacts']['feature_importances']}`",
        f"- `report.json` (este reporte en formato estructurado)",
        "",
    ])

    return "\n".join(lines)