"""Tests de las reglas de negocio individuales."""
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
