"""
Generador del modulo prescriptivo.

Crea todos los archivos de src/prescriptive/ y los tests correspondientes
en un solo comando.

Uso:
    python scripts/generate_prescriptive_module.py
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


ARCHIVOS = {
    # ─── src/prescriptive/__init__.py ────────────────────────────────
    "src/prescriptive/__init__.py": '',

    # ─── src/prescriptive/rules.py ──────────────────────────────────
    "src/prescriptive/rules.py": '''"""
Reglas de negocio (RN-01 a RN-05).

Cada regla es una funcion pura que toma una fila de DataFrame (un
estudiante) + un diccionario de umbrales, y devuelve:
- str con la recomendacion si la regla aplica
- None si la regla no aplica

Las reglas son independientes y aditivas: un estudiante puede activar
varias a la vez, y todas las recomendaciones se concatenan al final.
"""
from typing import Optional

import pandas as pd


# ─── Reglas individuales ──────────────────────────────────────────────


def rn01_estudio_insuficiente(
    row: pd.Series, thresholds: dict
) -> Optional[str]:
    """
    RN-01: Estudio insuficiente.

    Si el estudiante dedica menos de X horas al estudio formal,
    se recomienda aumentar a un minimo razonable.
    """
    if row["study_hours"] < thresholds["study_hours_min"]:
        return "Aumentar horas de estudio a minimo 6 horas diarias"
    return None


def rn02_burnout_alto(row: pd.Series, thresholds: dict) -> Optional[str]:
    """
    RN-02: Burnout alto.

    Si el nivel de agotamiento academico supera el umbral, se recomienda
    incluir pausas activas y descanso.
    """
    if row["burnout_level"] > thresholds["burnout_max"]:
        return "Nivel de agotamiento alto - incluir pausas y descanso"
    return None


def rn03_salud_mental_baja(
    row: pd.Series, thresholds: dict
) -> Optional[str]:
    """
    RN-03: Salud mental baja.

    Si el score de salud mental esta por debajo del umbral, se recomienda
    buscar apoyo profesional.
    """
    if row["mental_health_score"] < thresholds["mental_health_min"]:
        return "Buscar apoyo en el servicio de bienestar universitario"
    return None


def rn04_distracciones_altas(
    row: pd.Series, thresholds: dict
) -> Optional[str]:
    """
    RN-04: Distracciones altas.

    Si el tiempo combinado en redes sociales + gaming supera el umbral,
    se recomienda reducirlo e invertirlo en estudio.

    NOTA: gaming_hours NO entra al modelo predictivo (su correlacion
    con exam_score es -0.05), pero SI se usa en esta regla. Las reglas
    de negocio pueden operar sobre cualquier variable del dataset crudo.
    """
    distracciones = row["social_media_hours"] + row["gaming_hours"]
    if distracciones > thresholds["distractions_max"]:
        return (
            "Reducir redes sociales y videojuegos - "
            "invertir ese tiempo en estudio"
        )
    return None


def rn05_sueno_insuficiente(
    row: pd.Series, thresholds: dict
) -> Optional[str]:
    """
    RN-05: Sueno insuficiente.

    Si duerme menos de X horas, se recomienda establecer una rutina
    de sueno saludable.
    """
    if row["sleep_hours"] < thresholds["sleep_hours_min"]:
        return "Establecer rutina de sueno de minimo 7 horas diarias"
    return None


# ─── Registro de reglas ──────────────────────────────────────────────


# Lista ordenada de (codigo, funcion). El orden controla el orden de
# aparicion en la recomendacion concatenada.
ALL_RULES = [
    ("RN-01", rn01_estudio_insuficiente),
    ("RN-02", rn02_burnout_alto),
    ("RN-03", rn03_salud_mental_baja),
    ("RN-04", rn04_distracciones_altas),
    ("RN-05", rn05_sueno_insuficiente),
]


# Recomendacion por defecto cuando ninguna regla aplica
DEFAULT_RECOMMENDATION = "Revisar habitos generales con un tutor academico"
''',

    # ─── src/prescriptive/recommender.py ────────────────────────────
    "src/prescriptive/recommender.py": '''"""
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
''',

    # ─── src/prescriptive/analyze.py ────────────────────────────────
    "src/prescriptive/analyze.py": '''"""
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
    logger.info("\\n" + coverage.to_string(index=False))

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
    return "\\n".join(lines)


if __name__ == "__main__":
    analyze()
''',

    # ─── tests/test_prescriptive_rules.py ──────────────────────────
    "tests/test_prescriptive_rules.py": '''"""Tests de las reglas de negocio individuales."""
import pandas as pd
import pytest

from src.prescriptive.rules import (
    rn01_estudio_insuficiente,
    rn02_burnout_alto,
    rn03_salud_mental_baja,
    rn04_distracciones_altas,
    rn05_sueno_insuficiente,
)


@pytest.fixture
def thresholds():
    """Umbrales standard para los tests."""
    return {
        "study_hours_min": 4,
        "burnout_max": 60,
        "mental_health_min": 5,
        "distractions_max": 4,
        "sleep_hours_min": 6,
    }


@pytest.fixture
def estudiante_ideal():
    """Estudiante con habitos optimos: ninguna regla deberia activarse."""
    return pd.Series({
        "study_hours": 8.0,
        "burnout_level": 20.0,
        "mental_health_score": 8,
        "social_media_hours": 1.0,
        "gaming_hours": 1.0,
        "sleep_hours": 8.0,
    })


@pytest.fixture
def estudiante_riesgo_total():
    """Estudiante que activa TODAS las reglas."""
    return pd.Series({
        "study_hours": 1.0,        # < 4
        "burnout_level": 85.0,      # > 60
        "mental_health_score": 2,   # < 5
        "social_media_hours": 5.0,
        "gaming_hours": 3.0,        # suma 8 > 4
        "sleep_hours": 4.0,         # < 6
    })


# ─── RN-01 ────────────────────────────────────────────────────────────


def test_rn01_se_activa_con_pocas_horas_de_estudio(thresholds):
    row = pd.Series({"study_hours": 2.0})
    resultado = rn01_estudio_insuficiente(row, thresholds)
    assert resultado is not None
    assert "estudio" in resultado.lower()


def test_rn01_no_se_activa_con_suficientes_horas(thresholds):
    row = pd.Series({"study_hours": 6.0})
    assert rn01_estudio_insuficiente(row, thresholds) is None


def test_rn01_borde_inferior_no_se_activa(thresholds):
    """Exactamente en el umbral (4.0): NO se activa porque la condicion es < 4."""
    row = pd.Series({"study_hours": 4.0})
    assert rn01_estudio_insuficiente(row, thresholds) is None


# ─── RN-02 ────────────────────────────────────────────────────────────


def test_rn02_se_activa_con_burnout_alto(thresholds):
    row = pd.Series({"burnout_level": 75.0})
    resultado = rn02_burnout_alto(row, thresholds)
    assert resultado is not None
    assert "agotamiento" in resultado.lower()


def test_rn02_no_se_activa_con_burnout_bajo(thresholds):
    row = pd.Series({"burnout_level": 30.0})
    assert rn02_burnout_alto(row, thresholds) is None


def test_rn02_borde_superior_no_se_activa(thresholds):
    """Exactamente en el umbral (60): NO se activa porque la condicion es > 60."""
    row = pd.Series({"burnout_level": 60.0})
    assert rn02_burnout_alto(row, thresholds) is None


# ─── RN-03 ────────────────────────────────────────────────────────────


def test_rn03_se_activa_con_salud_mental_baja(thresholds):
    row = pd.Series({"mental_health_score": 3})
    resultado = rn03_salud_mental_baja(row, thresholds)
    assert resultado is not None
    assert "bienestar" in resultado.lower()


def test_rn03_no_se_activa_con_salud_mental_buena(thresholds):
    row = pd.Series({"mental_health_score": 8})
    assert rn03_salud_mental_baja(row, thresholds) is None


# ─── RN-04 ────────────────────────────────────────────────────────────


def test_rn04_se_activa_con_muchas_distracciones(thresholds):
    row = pd.Series({"social_media_hours": 3.0, "gaming_hours": 3.0})
    resultado = rn04_distracciones_altas(row, thresholds)
    assert resultado is not None
    assert "videojuegos" in resultado.lower() or "redes" in resultado.lower()


def test_rn04_no_se_activa_con_pocas_distracciones(thresholds):
    row = pd.Series({"social_media_hours": 1.0, "gaming_hours": 1.0})
    assert rn04_distracciones_altas(row, thresholds) is None


def test_rn04_combina_redes_y_gaming(thresholds):
    """Cada uno por separado no activa, pero la suma si."""
    # Solo redes: 3 < 4 -> no activa
    row1 = pd.Series({"social_media_hours": 3.0, "gaming_hours": 0.0})
    assert rn04_distracciones_altas(row1, thresholds) is None

    # Solo gaming: 3 < 4 -> no activa
    row2 = pd.Series({"social_media_hours": 0.0, "gaming_hours": 3.0})
    assert rn04_distracciones_altas(row2, thresholds) is None

    # Combinacion: 3 + 3 = 6 > 4 -> SI activa
    row3 = pd.Series({"social_media_hours": 3.0, "gaming_hours": 3.0})
    assert rn04_distracciones_altas(row3, thresholds) is not None


# ─── RN-05 ────────────────────────────────────────────────────────────


def test_rn05_se_activa_con_poco_sueno(thresholds):
    row = pd.Series({"sleep_hours": 4.5})
    resultado = rn05_sueno_insuficiente(row, thresholds)
    assert resultado is not None
    assert "sueno" in resultado.lower()


def test_rn05_no_se_activa_con_suficiente_sueno(thresholds):
    row = pd.Series({"sleep_hours": 8.0})
    assert rn05_sueno_insuficiente(row, thresholds) is None


# ─── Tests sobre estudiantes completos ────────────────────────────────


def test_estudiante_ideal_no_activa_ninguna_regla(thresholds, estudiante_ideal):
    """Un estudiante con habitos optimos no deberia activar ninguna regla."""
    assert rn01_estudio_insuficiente(estudiante_ideal, thresholds) is None
    assert rn02_burnout_alto(estudiante_ideal, thresholds) is None
    assert rn03_salud_mental_baja(estudiante_ideal, thresholds) is None
    assert rn04_distracciones_altas(estudiante_ideal, thresholds) is None
    assert rn05_sueno_insuficiente(estudiante_ideal, thresholds) is None


def test_estudiante_riesgo_total_activa_todas(thresholds, estudiante_riesgo_total):
    """Un estudiante con todos los habitos malos debe activar las 5 reglas."""
    assert rn01_estudio_insuficiente(estudiante_riesgo_total, thresholds) is not None
    assert rn02_burnout_alto(estudiante_riesgo_total, thresholds) is not None
    assert rn03_salud_mental_baja(estudiante_riesgo_total, thresholds) is not None
    assert rn04_distracciones_altas(estudiante_riesgo_total, thresholds) is not None
    assert rn05_sueno_insuficiente(estudiante_riesgo_total, thresholds) is not None
''',

    # ─── tests/test_prescriptive_recommender.py ────────────────────
    "tests/test_prescriptive_recommender.py": '''"""Tests del motor de recomendaciones."""
import pandas as pd
import pytest

from src.prescriptive.recommender import (
    apply_rules_to_dataframe,
    apply_rules_to_row,
    compute_rules_coverage,
)
from src.prescriptive.rules import DEFAULT_RECOMMENDATION


@pytest.fixture
def thresholds():
    return {
        "study_hours_min": 4,
        "burnout_max": 60,
        "mental_health_min": 5,
        "distractions_max": 4,
        "sleep_hours_min": 6,
    }


@pytest.fixture
def df_estudiantes():
    """3 estudiantes: ideal, riesgo parcial, riesgo total."""
    return pd.DataFrame([
        # Estudiante ideal: no activa nada
        {
            "study_hours": 8.0,
            "burnout_level": 20.0,
            "mental_health_score": 8,
            "social_media_hours": 1.0,
            "gaming_hours": 1.0,
            "sleep_hours": 8.0,
        },
        # Riesgo parcial: solo estudio insuficiente
        {
            "study_hours": 2.0,
            "burnout_level": 30.0,
            "mental_health_score": 7,
            "social_media_hours": 1.0,
            "gaming_hours": 1.0,
            "sleep_hours": 7.0,
        },
        # Riesgo total: activa todas
        {
            "study_hours": 1.0,
            "burnout_level": 85.0,
            "mental_health_score": 2,
            "social_media_hours": 5.0,
            "gaming_hours": 3.0,
            "sleep_hours": 4.0,
        },
    ])


# ─── apply_rules_to_row ───────────────────────────────────────────────


def test_estudiante_ideal_recibe_recomendacion_por_defecto(thresholds, df_estudiantes):
    fila = df_estudiantes.iloc[0]
    resultado = apply_rules_to_row(fila, thresholds)
    assert resultado == DEFAULT_RECOMMENDATION


def test_estudiante_riesgo_parcial_recibe_una_recomendacion(thresholds, df_estudiantes):
    fila = df_estudiantes.iloc[1]
    resultado = apply_rules_to_row(fila, thresholds)
    # Solo activa RN-01
    assert "estudio" in resultado.lower()
    assert "|" not in resultado  # solo una recomendacion, sin separador


def test_estudiante_riesgo_total_recibe_varias_recomendaciones(thresholds, df_estudiantes):
    fila = df_estudiantes.iloc[2]
    resultado = apply_rules_to_row(fila, thresholds)
    # Activa las 5 reglas, deben aparecer separadas por " | "
    n_recomendaciones = resultado.count("|") + 1
    assert n_recomendaciones == 5


def test_recomendaciones_son_strings_no_vacios(thresholds, df_estudiantes):
    for i in range(len(df_estudiantes)):
        fila = df_estudiantes.iloc[i]
        resultado = apply_rules_to_row(fila, thresholds)
        assert isinstance(resultado, str)
        assert len(resultado) > 0


# ─── apply_rules_to_dataframe ────────────────────────────────────────


def test_apply_to_df_agrega_columna_recomendaciones(thresholds, df_estudiantes):
    resultado = apply_rules_to_dataframe(df_estudiantes, thresholds)
    assert "recomendaciones" in resultado.columns


def test_apply_to_df_preserva_filas_originales(thresholds, df_estudiantes):
    resultado = apply_rules_to_dataframe(df_estudiantes, thresholds)
    assert len(resultado) == len(df_estudiantes)


def test_apply_to_df_es_idempotente(thresholds, df_estudiantes):
    """Aplicar dos veces da el mismo resultado."""
    r1 = apply_rules_to_dataframe(df_estudiantes, thresholds)
    r2 = apply_rules_to_dataframe(df_estudiantes, thresholds)
    pd.testing.assert_frame_equal(r1, r2)


def test_apply_to_df_no_modifica_original(thresholds, df_estudiantes):
    columnas_orig = df_estudiantes.columns.tolist()
    apply_rules_to_dataframe(df_estudiantes, thresholds)
    assert df_estudiantes.columns.tolist() == columnas_orig


def test_cobertura_100_percent_garantizada(thresholds, df_estudiantes):
    """Todos los estudiantes deben recibir al menos una recomendacion."""
    resultado = apply_rules_to_dataframe(df_estudiantes, thresholds)
    sin_recomendacion = resultado["recomendaciones"].isna().sum()
    vacias = (resultado["recomendaciones"] == "").sum()
    assert sin_recomendacion == 0
    assert vacias == 0


# ─── compute_rules_coverage ──────────────────────────────────────────


def test_coverage_tiene_5_filas(thresholds, df_estudiantes):
    coverage = compute_rules_coverage(df_estudiantes, thresholds)
    assert len(coverage) == 5


def test_coverage_tiene_columnas_esperadas(thresholds, df_estudiantes):
    coverage = compute_rules_coverage(df_estudiantes, thresholds)
    assert set(coverage.columns) == {"codigo", "descripcion", "n_estudiantes", "porcentaje"}


def test_coverage_codigos_correctos(thresholds, df_estudiantes):
    coverage = compute_rules_coverage(df_estudiantes, thresholds)
    codigos = set(coverage["codigo"].tolist())
    assert codigos == {"RN-01", "RN-02", "RN-03", "RN-04", "RN-05"}


def test_coverage_porcentajes_son_validos(thresholds, df_estudiantes):
    coverage = compute_rules_coverage(df_estudiantes, thresholds)
    assert (coverage["porcentaje"] >= 0).all()
    assert (coverage["porcentaje"] <= 100).all()


def test_coverage_ordenado_descendente(thresholds, df_estudiantes):
    coverage = compute_rules_coverage(df_estudiantes, thresholds)
    valores = coverage["n_estudiantes"].tolist()
    assert valores == sorted(valores, reverse=True)
''',
}


def main():
    print(f"Generando modulo prescriptivo en {ROOT}")
    archivos_creados = 0
    for ruta_rel, contenido in ARCHIVOS.items():
        ruta = ROOT / ruta_rel
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(contenido, encoding="utf-8")
        size = ruta.stat().st_size
        print(f"  + {ruta_rel} ({size} bytes)")
        archivos_creados += 1

    print(f"\nOK: {archivos_creados} archivos creados.")


if __name__ == "__main__":
    main()