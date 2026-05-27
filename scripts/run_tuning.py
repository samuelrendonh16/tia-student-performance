"""
Script de ajuste de hiperparametros.

Corre GridSearchCV sobre el arbol de decision, reporta la mejor
combinacion y guarda una tabla de resultados.

Uso:
    python scripts/run_tuning.py

Salida:
    - Imprime mejores hiperparametros en consola
    - Guarda reports/v1/tuning_results.csv con la tabla completa
    - Sugiere los valores para actualizar config.yaml
"""
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import make_scorer, recall_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data
from src.features.selection import drop_irrelevant_columns
from src.features.target import binarize_score, compute_threshold
from src.models.splitter import split_train_test
from src.models.tuning import extract_tree_params, tune_decision_tree
from src.utils.config import load_config


def main():
    cfg = load_config(ROOT / "config" / "config.yaml")

    logger.info("=" * 60)
    logger.info("AJUSTE DE HIPERPARAMETROS - GridSearchCV")
    logger.info("=" * 60)

    # Cargar y preparar datos
    df = load_raw_data(ROOT / cfg["paths"]["data_raw"])
    df = drop_irrelevant_columns(df, cfg["features"]["drop_columns"])
    umbral = compute_threshold(
        df[cfg["target"]["source"]],
        strategy=cfg["target"]["strategy"],
        fixed_threshold=cfg["target"]["fixed_threshold"],
    )
    df = binarize_score(
        df,
        source_column=cfg["target"]["source"],
        target_column=cfg["target"]["name"],
        threshold=umbral,
        drop_source=True,
    )

    X_train, _, y_train, _ = split_train_test(
        df,
        target_column=cfg["target"]["name"],
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
        random_state=cfg["random_seed"],
    )

    tuning_cfg = cfg["model"]["tuning"]

    # Scorer personalizado: optimizar recall de la clase 0 (reprueba),
    # que es la clase de interes para el negocio.
    scorer = make_scorer(recall_score, pos_label=0)

    resultado = tune_decision_tree(
        X_train=X_train,
        y_train=y_train,
        numeric_features=cfg["features"]["feature_columns"],
        model_config=cfg["model"],
        param_grid=tuning_cfg["param_grid"],
        cv_folds=tuning_cfg["cv_folds"],
        scoring=scorer,
        random_state=cfg["random_seed"],
    )

    # Guardar tabla de resultados (top 10)
    output_dir = ROOT / cfg["paths"]["reports"] / cfg["evaluation"]["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cols_interes = [
        c for c in resultado.cv_results.columns
        if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")
    ]
    tabla = resultado.cv_results[cols_interes].sort_values("rank_test_score").head(10)
    csv_path = output_dir / "tuning_results.csv"
    tabla.to_csv(csv_path, index=False)

    # Reporte en consola
    logger.info("=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"Combinaciones exploradas: {resultado.n_combinations}")
    logger.info(f"Mejor recall (clase reprueba): {resultado.best_score:.4f}")
    logger.info("")
    logger.info("Mejores hiperparametros encontrados:")

    tree_params = extract_tree_params(resultado.best_params)
    for clave, valor in tree_params.items():
        logger.info(f"  {clave}: {valor}")

    logger.info("")
    logger.info("ACTUALIZA config.yaml > model > decision_tree con estos valores:")
    logger.info("")
    for clave, valor in tree_params.items():
        if isinstance(valor, str):
            logger.info(f"    {clave}: {valor}")
        else:
            logger.info(f"    {clave}: {valor}")
    logger.info("")
    logger.info(f"Tabla completa (top 10) guardada en: {csv_path}")


if __name__ == "__main__":
    main()