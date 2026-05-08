"""
Schema de validación para el dataset crudo de Student Performance.

Define el contrato de datos: columnas esperadas, tipos, rangos válidos.
Si el CSV crudo no cumple este contrato, la carga falla inmediatamente.

Los rangos fueron calibrados contra el dataset real (5000 filas).
Se usa un margen de seguridad razonable para tolerar nuevos lotes
con valores ligeramente fuera de los observados.
"""
import pandera.pandas as pa
from pandera.pandas import Column, Check


# ─── Constantes del dominio ────────────────────────────────────────────
GENDER_VALUES = ["Male", "Female", "Other"]
ACADEMIC_LEVELS = ["High School", "Undergraduate", "Postgraduate"]
INTERNET_QUALITY_VALUES = ["Poor", "Average", "Good"]


# ─── Schema del CSV crudo ──────────────────────────────────────────────
RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        # Identificación
        "student_id": Column(int, unique=True, nullable=False),

        # Demográficas
        "age": Column(int, Check.in_range(15, 30), nullable=False),
        "gender": Column(str, Check.isin(GENDER_VALUES), nullable=False),
        "academic_level": Column(
            str, Check.isin(ACADEMIC_LEVELS), nullable=False
        ),

        # Hábitos de estudio (horas por día)
        "study_hours": Column(float, Check.in_range(0, 24), nullable=False),
        "self_study_hours": Column(
            float, Check.in_range(0, 24), nullable=False
        ),
        "online_classes_hours": Column(
            float, Check.in_range(0, 24), nullable=False
        ),

        # Hábitos digitales y tiempo libre
        "social_media_hours": Column(
            float, Check.in_range(0, 24), nullable=False
        ),
        "gaming_hours": Column(float, Check.in_range(0, 24), nullable=False),
        "screen_time_hours": Column(
            float, Check.in_range(0, 24), nullable=False
        ),

        # Bienestar físico
        "sleep_hours": Column(float, Check.in_range(0, 24), nullable=False),
        "exercise_minutes": Column(
            int, Check.in_range(0, 600), nullable=False
        ),
        "caffeine_intake_mg": Column(
            int, Check.in_range(0, 2000), nullable=False
        ),

        # Contexto académico/laboral (binarias)
        "part_time_job": Column(int, Check.isin([0, 1]), nullable=False),
        "upcoming_deadline": Column(int, Check.isin([0, 1]), nullable=False),
        "internet_quality": Column(
            str, Check.isin(INTERNET_QUALITY_VALUES), nullable=False
        ),

        # Indicadores derivados (escalas observadas en el dataset)
        # mental_health_score: escala 1–10
        "mental_health_score": Column(
            int, Check.in_range(1, 10), nullable=False
        ),
        # focus_index, burnout_level, productivity_score: índices 0–100
        "focus_index": Column(float, Check.in_range(0, 100), nullable=False),
        "burnout_level": Column(
            float, Check.in_range(0, 100), nullable=False
        ),
        "productivity_score": Column(
            float, Check.in_range(0, 100), nullable=False
        ),

        # Variable objetivo: nota observada en rango 0–70
        # (en este dataset el máximo real es ~64)
        "exam_score": Column(float, Check.in_range(0, 100), nullable=False),
    },
    strict=True,        # falla si llegan columnas no declaradas
    coerce=True,        # intenta convertir tipos cuando es posible
)


def validate_raw_data(df):
    """
    Valida un DataFrame contra el schema del CSV crudo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.

    Returns
    -------
    pd.DataFrame
        El mismo DataFrame, validado.

    Raises
    ------
    pandera.errors.SchemaErrors
        Si el DataFrame no cumple el contrato. El mensaje incluye
        todas las violaciones encontradas (no solo la primera).
    """
    return RAW_SCHEMA.validate(df, lazy=True)