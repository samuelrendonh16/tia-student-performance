"""
Script de analisis prescriptivo.

Orquesta el flujo:
1. Carga el modelo entrenado
2. Identifica los estudiantes que el modelo predice como en riesgo
3. Aplica las 5 reglas de negocio para generar recomendaciones
4. Calcula la cobertura de cada regla
5. Exporta: CSV de en riesgo, grafica de cobertura, reporte Markdown

Uso:
    python -m src.prescriptive.analyze

Requiere que exista models/decision_tree_v1.joblib (ejecutar antes train).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.data.loader import load_raw_data
from src.features.selection import drop_irrelevant_columns
from src.features.target import binarize_score, compute_threshold
from src.models.splitter import split_train_test
from src.prescriptive.recommender import (
    apply_rules_to_dataframe,
    compute_rules_coverage,
)
from src.utils.config import get_project_root, load_config


def analyze(config_path=None) -> dict:
    """Ejecuta el flujo completo de analisis prescriptivo."""
    project_root = get_project_root()
    config_path = config_path or project_root / "config" / "config.yaml"
    cfg = load_config(config_path)

    logger.info("=" * 60)
    logger.info("INICIO DE ANALISIS PRESCRIPTIVO")
    logger.info("=" * 60)

    # 1. Cargar modelo
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

    # 2. Reproducir el split de test (mismas semillas)
    raw_path = project_root / cfg["paths"]["data_raw"]
    df_raw = load_raw_data(raw_path)

    # IMPORTANTE: aqui mantenemos las columnas raw para poder usar
    # gaming_hours en la regla RN-04, asi que NO usamos drop_irrelevant_columns
    # del flujo principal. En su lugar trabajamos con el df crudo + target.
    df_with_target = df_raw.copy()
    threshold = compute_threshold(
        df_with_target[cfg["target"]["source"]],
        strategy=cfg["target"]["strategy"],
        fixed_threshold=cfg["target"]["fixed_threshold"],
    )
    df_with_target = binarize_score(
        df_with_target,
        source_column=cfg["target"]["source"],
        target_column=target_column,
        threshold=threshold,
        drop_source=True,
    )

    # Hacemos split sobre el df crudo con target (para conservar gaming_hours)
    _, X_test_raw, _, y_test = split_train_test(
        df_with_target,
        target_column=target_column,
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
        random_state=cfg["random_seed"],
    )

    # Para predecir, el modelo necesita solo las feature_columns
    X_test_modelo = X_test_raw[feature_columns]
    y_pred = pipeline.predict(X_test_modelo)

    # 3. Identificar estudiantes en riesgo (prediccion = 0 = reprueba)
    df_resultados = X_test_raw.copy()
    df_resultados["prediccion"] = y_pred
    df_resultados["real"] = y_test.values

    en_riesgo = df_resultados[df_resultados["prediccion"] == 0].copy()
    logger.info(f"Estudiantes en riesgo identificados: {len(en_riesgo)}")

    # 4. Aplicar reglas
    thresholds = cfg["prescriptive"]["rules"]
    en_riesgo_con_reco = apply_rules_to_dataframe(en_riesgo, thresholds)

    # 5. Calcular cobertura
    coverage = compute_rules_coverage(en_riesgo, thresholds)
    logger.info("\n" + coverage.to_string(index=False))

    # 6. Exportar resultados
    output_dir = (
        project_root
        / cfg["paths"]["reports"]
        / cfg["prescriptive"]["output_subdir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV con estudiantes en riesgo
    csv_path = output_dir / "students_at_risk.csv"
    en_riesgo_con_reco.to_csv(csv_path, index=False)
    logger.info(f"CSV guardado: {csv_path}")

    # Grafica de cobertura
    fig_cfg = cfg["prescriptive"].get("figures", {})
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_coverage", [10, 5])))
    sns.barplot(
        data=coverage,
        x="n_estudiantes",
        y="codigo",
        hue="codigo",
        palette="Reds_r",
        legend=False,
        ax=ax,
    )
    for i, row in coverage.iterrows():
        ax.text(
            row["n_estudiantes"] + 5,
            i,
            f"{row['n_estudiantes']} ({row['porcentaje']}%)",
            va="center",
        )
    ax.set_xlabel("Estudiantes en riesgo que activan la regla")
    ax.set_ylabel("Codigo de regla")
    ax.set_title("Cobertura de las reglas de negocio")
    plt.tight_layout()
    grafica_path = output_dir / "rules_coverage.png"
    plt.savefig(grafica_path, dpi=fig_cfg.get("dpi", 100))
    plt.close()
    logger.info(f"Grafica guardada: {grafica_path}")

    # Reporte Markdown
    md_path = output_dir / "prescriptive_report.md"
    md_path.write_text(
        _render_prescriptive_markdown(
            n_test=len(df_resultados),
            n_en_riesgo=len(en_riesgo),
            coverage=coverage,
            thresholds=thresholds,
        ),
        encoding="utf-8",
    )
    logger.info(f"Reporte Markdown guardado: {md_path}")

    logger.info("=" * 60)
    logger.info("ANALISIS PRESCRIPTIVO COMPLETADO")
    logger.info("=" * 60)

    return {
        "en_riesgo": en_riesgo_con_reco,
        "coverage": coverage,
        "output_dir": output_dir,
    }


def _render_prescriptive_markdown(
    n_test: int,
    n_en_riesgo: int,
    coverage,
    thresholds: dict,
) -> str:
    """Renderiza el reporte prescriptivo en Markdown."""
    pct = (n_en_riesgo / n_test * 100) if n_test > 0 else 0.0

    lines = [
        "# Reporte Prescriptivo - TIA Student Performance",
        "",
        "## Resumen ejecutivo",
        "",
        f"- **Estudiantes en el test set:** {n_test:,}",
        f"- **Estudiantes identificados en riesgo:** {n_en_riesgo:,} ({pct:.1f}%)",
        f"- **Cobertura de recomendaciones:** 100% (fallback garantizado)",
        "",
        "## Umbrales aplicados",
        "",
        "| Regla | Umbral |",
        "|---|---|",
        f"| RN-01 (estudio insuficiente) | study_hours < {thresholds['study_hours_min']} |",
        f"| RN-02 (burnout alto) | burnout_level > {thresholds['burnout_max']} |",
        f"| RN-03 (salud mental baja) | mental_health_score < {thresholds['mental_health_min']} |",
        f"| RN-04 (distracciones altas) | social_media + gaming > {thresholds['distractions_max']} |",
        f"| RN-05 (sueno insuficiente) | sleep_hours < {thresholds['sleep_hours_min']} |",
        "",
        "## Cobertura por regla",
        "",
        "| Codigo | Descripcion | Estudiantes afectados | Porcentaje |",
        "|---|---|---|---|",
    ]
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['codigo']} | {row['descripcion']} | "
            f"{row['n_estudiantes']} | {row['porcentaje']}% |"
        )

    lines.extend([
        "",
        "## Artefactos generados",
        "",
        "- `students_at_risk.csv` - listado completo con recomendaciones",
        "- `rules_coverage.png` - grafica de cobertura",
        "- `prescriptive_report.md` - este reporte",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    analyze()
