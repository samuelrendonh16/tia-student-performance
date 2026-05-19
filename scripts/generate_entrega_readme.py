"""Genera el README de instrucciones para el profesor."""
from pathlib import Path


CONTENIDO = """# TIA Student Performance - Entrega

**Autor:** Samuel Rendon Hincapie
**Curso:** Machine Learning - IU Pascual Bravo

---

## Contenido del paquete

| Archivo | Descripcion |
|---|---|
| `codigo_proyecto.html` | Codigo fuente completo del proyecto con syntax highlighting |
| `decision_tree_v1.joblib` | Modelo entrenado con metadata del entorno embebida |
| `requirements.txt` | Dependencias bloqueadas (versiones exactas) |
| `presentacion.pptx` | Storytelling del modelo |

## Como ejecutar / reproducir el modelo

### Opcion A - Solo cargar el modelo y predecir

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\\Scripts\\activate         # Windows
# source .venv/bin/activate      # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Cargar el modelo y hacer una prediccion
python
```

```python
import joblib
import pandas as pd
import json

# Cargar artefacto
artefacto = joblib.load("decision_tree_v1.joblib")

# Ver metadata del entorno donde se entreno
print(json.dumps(artefacto["metadata"], indent=2, default=str))

# Hacer una prediccion sobre un estudiante hipotetico
pipeline = artefacto["pipeline"]
features = artefacto["feature_columns"]

estudiante = pd.DataFrame([{
    "study_hours": 6.0,
    "self_study_hours": 3.0,
    "social_media_hours": 2.0,
    "sleep_hours": 7.5,
    "screen_time_hours": 5.0,
    "exercise_minutes": 45,
    "caffeine_intake_mg": 150,
    "part_time_job": 0,
    "upcoming_deadline": 1,
    "mental_health_score": 7,
    "focus_index": 40.0,
    "burnout_level": 35.0,
    "productivity_score": 55.0,
}])

prediccion = pipeline.predict(estudiante[features])[0]
probabilidad = pipeline.predict_proba(estudiante[features])[0][1]

print(f"Prediccion: {'Aprueba' if prediccion == 1 else 'Reprueba'}")
print(f"Probabilidad de aprobar: {probabilidad:.2%}")
```

### Opcion B - Ver el codigo completo del proyecto

Abrir `codigo_proyecto.html` en cualquier navegador (Chrome, Firefox, Edge).
El archivo es autocontenido (no necesita internet).

## Metadata del modelo

El archivo `.joblib` contiene un diccionario con:

- `pipeline`: el Pipeline entrenado (StandardScaler + SMOTE + DecisionTree)
- `feature_columns`: lista de las 13 features que espera el modelo
- `target_column`: nombre de la variable objetivo
- `threshold`: umbral de binarizacion del exam_score
- `random_seed`: semilla usada en el entrenamiento
- `metadata`: dict con versiones de Python, sklearn, numpy, pandas, fecha, etc.

## Notas tecnicas

- El modelo se entreno con **scikit-learn**, balanceado con **SMOTE** y serializado con **joblib (compress=3)** siguiendo las recomendaciones de la clase de empaquetamiento.
- La metadata del entorno esta embebida en el propio joblib, no en un archivo separado, para garantizar reproducibilidad.
- El proyecto completo migra el trabajo del TIA3 (Google Colab) a un entorno modular en VS Code con tests unitarios (~100 tests), validacion de schemas con pandera, y configuracion centralizada en YAML.
"""


def main():
    Path("entrega").mkdir(exist_ok=True)
    output = Path("entrega/README.md")
    output.write_text(CONTENIDO, encoding="utf-8")
    print(f"OK: README de entrega creado en {output.resolve()}")
    print(f"    Tamano: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()