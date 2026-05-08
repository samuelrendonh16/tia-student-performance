# TIA — Student Performance

Modelo predictivo de aprobación estudiantil basado en hábitos de estudio,
salud mental y bienestar. Curso de Machine Learning, IU Pascual Bravo.

**Autor:** Samuel Rendón Hincapié

## Pregunta de negocio

¿Debería un estudiante cambiar sus hábitos de estudio y vida social
basándose en la predicción de si aprobará o reprobará el examen final?

## Reproducir desde cero

```bash
# 1. Clonar el repo
git clone https://github.com/<usuario>/tia-student-performance.git
cd tia-student-performance

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate            # Windows

# 3. Instalar dependencias
pip install pip-tools
pip-sync requirements-dev.txt

# 4. Colocar el dataset en data/raw/studentmat.csv

# 5. Entrenar el modelo
python -m src.models.train

# 6. Correr los tests
pytest
```

## Estructura del proyecto

- `src/data/` — Carga y validación de datos
- `src/features/` — Preprocesamiento y selección de variables
- `src/models/` — Entrenamiento y predicción
- `src/evaluation/` — Métricas y reportes
- `tests/` — Pruebas unitarias
- `config/` — Hiperparámetros y configuración
- `data/raw/` — Dataset original (no versionado)
- `models/` — Modelos serializados (no versionados)

## Stack

- Python 3.11
- scikit-learn, imbalanced-learn (SMOTE)
- pandas, numpy
- pandera (validación de schemas)
- pytest (testing)