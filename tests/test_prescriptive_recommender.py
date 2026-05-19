"""Tests del motor de recomendaciones."""
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
