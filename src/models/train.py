"""Script de entrenamiento del modelo."""
import sys
from datetime import datetime
from pathlib import Path

import imblearn
import joblib
import numpy
import pandas
import sklearn
from loguru import logger

from src.data.loader import load_raw_data
from src.features.selection import drop_irrelevant_columns
from src.features.target import binarize_score, compute_threshold
from src.models.pipeline import build_pipeline
from src.models.splitter import split_train_test
from src.utils.config import get_project_root, load_config


def train(config_path=None) -> dict:
    """Ejecuta el flujo completo de entrenamiento."""
    project_root = get_project_root()
    config_path = config_path or project_root / "config" / "config.yaml"
    cfg = load_config(config_path)

    logger.info("=" * 60)
    logger.info("INICIO DEL ENTRENAMIENTO")
    logger.info("=" * 60)

    # 1. Cargar datos crudos
    raw_path = project_root / cfg["paths"]["data_raw"]
    df = load_raw_data(raw_path)

    # 2. Eliminar columnas irrelevantes
    df = drop_irrelevant_columns(df, cfg["features"]["drop_columns"])

    # 3. Calcular umbral
    threshold = compute_threshold(
        df[cfg["target"]["source"]],
        strategy=cfg["target"]["strategy"],
        fixed_threshold=cfg["target"]["fixed_threshold"],
    )

    # 4. Binarizar el target
    df = binarize_score(
        df,
        source_column=cfg["target"]["source"],
        target_column=cfg["target"]["name"],
        threshold=threshold,
        drop_source=True,
    )

    # 5. Split train/test
    split_cfg = cfg["split"]
    X_train, X_test, y_train, y_test = split_train_test(
        df,
        target_column=cfg["target"]["name"],
        test_size=split_cfg["test_size"],
        stratify=split_cfg["stratify"],
        random_state=cfg["random_seed"],
    )

    # 6. Construir pipeline
    pipeline = build_pipeline(
        numeric_features=cfg["features"]["feature_columns"],
        model_config=cfg["model"],
        random_state=cfg["random_seed"],
    )

    logger.info("Pipeline construido:")
    for name, step in pipeline.steps:
        logger.info(f"  - {name}: {type(step).__name__}")

    # 7. Entrenar
    logger.info("Entrenando pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("  -> Entrenamiento completado")

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    logger.info(f"Accuracy train: {train_score:.4f}")
    logger.info(f"Accuracy test:  {test_score:.4f}")

    # 8. Serializar el modelo
    models_dir = project_root / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / cfg["model"]["artifact_name"]

    artefacto = {
        "pipeline": pipeline,
        "feature_columns": cfg["features"]["feature_columns"],
        "target_column": cfg["target"]["name"],
        "threshold": threshold,
        "random_seed": cfg["random_seed"],
        "metadata": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "imblearn": imblearn.__version__,
            "joblib": joblib.__version__,
            "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
            "modelo": "DecisionTreeClassifier",
            "tipo": "clasificacion_binaria",
            "hiperparametros": cfg["model"]["decision_tree"],
            "balanceo": "SMOTE" if cfg["model"]["smote"]["enabled"] else "ninguno",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "metricas_test": {
                "accuracy_train": float(train_score),
                "accuracy_test": float(test_score),
            },
        },
    }

    joblib.dump(artefacto, artifact_path, compress=3)
    logger.info(f"Modelo guardado en: {artifact_path}")
    logger.info(f"  -> Tamano: {artifact_path.stat().st_size / 1024:.1f} KB")

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "threshold": threshold,
        "artifact_path": artifact_path,
    }


if __name__ == "__main__":
    train()
