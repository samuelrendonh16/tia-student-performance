"""Tests del módulo de selección de variables."""
import pandas as pd
import pytest

from src.features.selection import drop_irrelevant_columns, select_feature_columns


@pytest.fixture
def df_demo():
    """DataFrame mínimo para tests aislados."""
    return pd.DataFrame({
        "student_id": [1, 2, 3],
        "age": [20, 22, 19],
        "study_hours": [4.5, 6.0, 3.2],
        "exam_score": [25.0, 40.0, 18.5],
    })


# ─── Tests de drop_irrelevant_columns ─────────────────────────────────


def test_drop_elimina_columnas_existentes(df_demo):
    resultado = drop_irrelevant_columns(df_demo, ["student_id", "age"])
    assert "student_id" not in resultado.columns
    assert "age" not in resultado.columns
    assert "study_hours" in resultado.columns
    assert "exam_score" in resultado.columns


def test_drop_no_modifica_original(df_demo):
    """La función debe ser pura: no muta el input."""
    columnas_originales = df_demo.columns.tolist()
    drop_irrelevant_columns(df_demo, ["age"])
    assert df_demo.columns.tolist() == columnas_originales


def test_drop_ignora_columnas_inexistentes(df_demo):
    """Si una columna no existe, no debe fallar."""
    resultado = drop_irrelevant_columns(df_demo, ["columna_fantasma"])
    assert resultado.shape == df_demo.shape


def test_drop_mezcla_existentes_y_no_existentes(df_demo):
    resultado = drop_irrelevant_columns(
        df_demo, ["age", "columna_fantasma", "student_id"]
    )
    assert "age" not in resultado.columns
    assert "student_id" not in resultado.columns
    assert "study_hours" in resultado.columns


def test_drop_es_idempotente(df_demo):
    """Aplicar drop dos veces da el mismo resultado."""
    una_vez = drop_irrelevant_columns(df_demo, ["age"])
    dos_veces = drop_irrelevant_columns(una_vez, ["age"])
    pd.testing.assert_frame_equal(una_vez, dos_veces)


def test_drop_preserva_filas(df_demo):
    resultado = drop_irrelevant_columns(df_demo, ["age"])
    assert len(resultado) == len(df_demo)


# ─── Tests de select_feature_columns ──────────────────────────────────


def test_select_devuelve_columnas_correctas(df_demo):
    resultado = select_feature_columns(df_demo, ["study_hours", "age"])
    assert resultado.columns.tolist() == ["study_hours", "age"]


def test_select_preserva_orden_solicitado(df_demo):
    """Las columnas deben aparecer en el orden pedido."""
    resultado = select_feature_columns(df_demo, ["age", "study_hours"])
    assert resultado.columns.tolist() == ["age", "study_hours"]


def test_select_falla_si_falta_columna(df_demo):
    with pytest.raises(KeyError, match="Faltan columnas requeridas"):
        select_feature_columns(df_demo, ["study_hours", "no_existe"])


def test_select_no_modifica_original(df_demo):
    columnas_originales = df_demo.columns.tolist()
    select_feature_columns(df_demo, ["age"])
    assert df_demo.columns.tolist() == columnas_originales