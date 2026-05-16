"""
Script de evaluación del modelo entrenado.

Uso:
    python -m src.evaluation.evaluate

Carga el modelo serializado por src.models.train, reproduce el split
de test (mismas semillas → mismos datos), y genera el reporte completo.
"""
from pathlib import Path

import joblib
from loguru import logger

from src.data.loader import load_raw_data
from src.evaluation.report import build_full_report
from src.features.selection import drop_irrelevant_columns
from src.features.target import binarize_score, compute_threshold
from src.models.splitter import split_train_test
from src.utils.config import get_project_root, load_config


def evaluate(config_path: str | Path | None = None) -> dict:
    """
    Ejecuta el flujo de evaluación.

    Returns
    -------
    dict
        El reporte de evaluación.
    """
    project_root = get_project_root()
    config_path = config_path or project_root / "config" / "config.yaml"
    cfg = load_config(config_path)

    logger.info("=" * 60)
    logger.info("INICIO DE EVALUACIÓN")
    logger.info("=" * 60)

    # ─── 1. Cargar modelo serializado ────────────────────────────────
    artifact_path = (
        project_root / cfg["paths"]["models"] / cfg["model"]["artifact_name"]
    )
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {artifact_path}. "
            f"Ejecuta primero: python -m src.models.train"
        )

    logger.info(f"Cargando modelo: {artifact_path}")
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    target_column = artifact["target_column"]

    # ─── 2. Reproducir el split de test ──────────────────────────────
    # Replicamos exactamente el flujo de entrenamiento para obtener
    # el mismo test set. Esto es seguro porque todas las semillas
    # están fijas en config.yaml.
    raw_path = project_root / cfg["paths"]["data_raw"]
    df = load_raw_data(raw_path)
    df = drop_irrelevant_columns(df, cfg["features"]["drop_columns"])
    threshold = compute_threshold(
        df[cfg["target"]["source"]],
        strategy=cfg["target"]["strategy"],
        fixed_threshold=cfg["target"]["fixed_threshold"],
    )
    df = binarize_score(
        df,
        source_column=cfg["target"]["source"],
        target_column=target_column,
        threshold=threshold,
        drop_source=True,
    )

    _, X_test, _, y_test = split_train_test(
        df,
        target_column=target_column,
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
        random_state=cfg["random_seed"],
    )

    # ─── 3. Generar reporte ──────────────────────────────────────────
    output_dir = (
        project_root
        / cfg["paths"]["reports"]
        / cfg["evaluation"]["output_subdir"]
    )

    report = build_full_report(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_columns,
        class_names=cfg["evaluation"]["class_names"],
        output_dir=output_dir,
        figure_config=cfg["evaluation"].get("figures", {}),
    )

    # ─── 4. Resumen en consola ───────────────────────────────────────
    m = report["metrics"]
    logger.info("=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {m['accuracy']:.4f}")
    logger.info(f"Precision: {m['precision']:.4f}")
    logger.info(f"Recall:    {m['recall']:.4f}")
    logger.info(f"F1-Score:  {m['f1']:.4f}")
    logger.info(f"AUC-ROC:   {report['roc']['auc']:.4f}")
    logger.info(f"Reporte completo en: {output_dir}")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    evaluate()