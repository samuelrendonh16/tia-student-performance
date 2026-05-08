"""Tests del módulo de carga de datos."""
import pandas as pd
import pandera as pa
import pytest

from src.data.loader import load_raw_data
from src.data.schema import RAW_SCHEMA, validate_raw_data
from src.utils.config import load_config, get_project_root


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def config():
    """Carga la configuración del proyecto."""
    return load_config(get_project_root() / "config" / "config.yaml")


@pytest.fixture(scope="module")
def raw_path(config):
    """Ruta absoluta al CSV crudo."""
    return get_project_root() / config["paths"]["data_raw"]


@pytest.fixture(scope="module")
def df_raw(raw_path):
    """Carga el dataset una vez por módulo (cacheado)."""
    return load_raw_data(raw_path)


# ─── Tests determinísticos ────────────────────────────────────────────


def test_archivo_existe(raw_path):
    """El CSV crudo debe existir en la ruta configurada."""
    assert raw_path.exists(), f"No existe el dataset en {raw_path}"


def test_carga_retorna_dataframe(df_raw):
    """La función debe retornar un pandas DataFrame."""
    assert isinstance(df_raw, pd.DataFrame)


def test_dataset_no_vacio(df_raw, config):
    """El dataset debe tener un mínimo razonable de filas."""
    minimo = config["data"]["expected_rows_min"]
    assert len(df_raw) >= minimo, f"Solo hay {len(df_raw)} filas, esperaba >= {minimo}"


def test_numero_de_columnas(df_raw, config):
    """El dataset debe tener exactamente las columnas esperadas."""
    esperadas = config["data"]["expected_columns"]
    assert df_raw.shape[1] == esperadas, (
        f"Esperaba {esperadas} columnas, encontré {df_raw.shape[1]}"
    )


def test_columnas_obligatorias_presentes(df_raw):
    """Las columnas usadas por el modelo deben estar presentes."""
    obligatorias = {
        "student_id", "exam_score", "study_hours", "mental_health_score",
        "focus_index", "sleep_hours", "burnout_level", "social_media_hours",
        "part_time_job", "upcoming_deadline", "self_study_hours",
    }
    faltantes = obligatorias - set(df_raw.columns)
    assert not faltantes, f"Faltan columnas: {faltantes}"


def test_sin_valores_faltantes(df_raw):
    """El CSV crudo no debe tener nulos."""
    nulos = df_raw.isnull().sum().sum()
    assert nulos == 0, f"Hay {nulos} valores nulos en el dataset"


def test_student_id_unico(df_raw):
    """Cada estudiante debe tener un ID único."""
    assert df_raw["student_id"].is_unique, "Hay student_id duplicados"


# ─── Tests de idempotencia ────────────────────────────────────────────


def test_carga_es_idempotente(raw_path):
    """Cargar dos veces debe producir DataFrames idénticos."""
    df1 = load_raw_data(raw_path)
    df2 = load_raw_data(raw_path)
    pd.testing.assert_frame_equal(df1, df2)


# ─── Tests del schema ─────────────────────────────────────────────────


def test_schema_valida_dataset_real(df_raw):
    """El dataset real debe cumplir el contrato sin lanzar errores."""
    # Si esto lanza una excepción, el test falla automáticamente
    validated = validate_raw_data(df_raw)
    assert len(validated) == len(df_raw)


def test_schema_rechaza_columna_faltante(df_raw):
    """Si falta una columna obligatoria, el schema debe fallar."""
    df_corrupto = df_raw.drop(columns=["exam_score"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_data(df_corrupto)


def test_schema_rechaza_valor_fuera_de_rango(df_raw):
    """Si un valor está fuera del rango declarado, debe fallar."""
    df_corrupto = df_raw.copy()
    df_corrupto.loc[0, "age"] = 999  # imposible biológicamente
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_data(df_corrupto)


def test_schema_rechaza_categoria_invalida(df_raw):
    """Si aparece una categoría no declarada, debe fallar."""
    df_corrupto = df_raw.copy()
    df_corrupto.loc[0, "gender"] = "Unknown"  # no está en GENDER_VALUES
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_data(df_corrupto)


# ─── Tests de comportamiento ──────────────────────────────────────────


def test_archivo_inexistente_lanza_filenotfound():
    """Cargar un archivo que no existe debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/no_existe.csv")


def test_carga_sin_validacion_no_falla_con_datos_corruptos(tmp_path, df_raw):
    """Con validate=False debe cargar incluso datos que no cumplen schema."""
    df_corrupto = df_raw.copy()
    df_corrupto.loc[0, "age"] = 999
    csv_corrupto = tmp_path / "corrupto.csv"
    df_corrupto.to_csv(csv_corrupto, index=False)

    df_cargado = load_raw_data(csv_corrupto, validate=False)
    assert df_cargado.loc[0, "age"] == 999