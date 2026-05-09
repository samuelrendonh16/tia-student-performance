"""Tests del módulo de creación de la variable objetivo."""
import pandas as pd
import pytest

from src.features.target import binarize_score, compute_threshold


@pytest.fixture
def df_scores():
    """DataFrame con puntajes para tests."""
    return pd.DataFrame({
        "exam_score": [10.0, 15.0, 18.0, 25.0, 40.0],
        "other": [1, 2, 3, 4, 5],
    })


# ─── Tests de compute_threshold ───────────────────────────────────────


def test_threshold_por_mediana():
    serie = pd.Series([10.0, 15.0, 18.0, 25.0, 40.0])
    assert compute_threshold(serie, strategy="median") == 18.0


def test_threshold_fijo_ignora_datos():
    """El umbral fijo no debe depender de la serie."""
    serie = pd.Series([1000.0, 2000.0])
    assert compute_threshold(
        serie, strategy="fixed", fixed_threshold=18.0
    ) == 18.0


def test_threshold_estrategia_invalida_falla():
    serie = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Estrategia inválida"):
        compute_threshold(serie, strategy="bogus")  # type: ignore[arg-type]


# ─── Tests de binarize_score ──────────────────────────────────────────


def test_binarize_crea_columna_objetivo(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=False,
    )
    assert "aprobado" in resultado.columns


def test_binarize_aplica_umbral_correcto(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=False,
    )
    # exam_score: [10, 15, 18, 25, 40] con umbral 18
    # >= 18 da: [0, 0, 1, 1, 1]
    assert resultado["aprobado"].tolist() == [0, 0, 1, 1, 1]


def test_binarize_target_es_entero(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=False,
    )
    assert resultado["aprobado"].dtype == int


def test_binarize_solo_valores_0_y_1(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=False,
    )
    assert set(resultado["aprobado"].unique()).issubset({0, 1})


def test_binarize_drop_source_true(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=True,
    )
    assert "exam_score" not in resultado.columns
    assert "aprobado" in resultado.columns


def test_binarize_drop_source_false(df_scores):
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=False,
    )
    assert "exam_score" in resultado.columns


def test_binarize_no_modifica_original(df_scores):
    columnas_originales = df_scores.columns.tolist()
    binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=18.0, drop_source=True,
    )
    assert df_scores.columns.tolist() == columnas_originales


def test_binarize_falla_si_columna_fuente_no_existe(df_scores):
    with pytest.raises(KeyError, match="No existe la columna fuente"):
        binarize_score(
            df_scores, "no_existe", "aprobado",
            threshold=18.0,
        )


def test_binarize_falla_si_target_ya_existe(df_scores):
    df_con_aprobado = df_scores.copy()
    df_con_aprobado["aprobado"] = 0
    with pytest.raises(ValueError, match="ya existe"):
        binarize_score(
            df_con_aprobado, "exam_score", "aprobado",
            threshold=18.0,
        )


# ─── Test de integración: pipeline completo ───────────────────────────


def test_pipeline_features_completo(df_scores):
    """Encadenar compute_threshold + binarize_score como en producción."""
    umbral = compute_threshold(df_scores["exam_score"], strategy="median")
    resultado = binarize_score(
        df_scores, "exam_score", "aprobado",
        threshold=umbral, drop_source=True,
    )

    # La mediana de [10, 15, 18, 25, 40] es 18.0
    assert umbral == 18.0
    assert "aprobado" in resultado.columns
    assert "exam_score" not in resultado.columns
    assert "other" in resultado.columns