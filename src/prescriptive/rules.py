"""
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
