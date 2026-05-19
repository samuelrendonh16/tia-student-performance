# Reporte Prescriptivo - TIA Student Performance

## Resumen ejecutivo

- **Estudiantes en el test set:** 1,000
- **Estudiantes identificados en riesgo:** 474 (47.4%)
- **Cobertura de recomendaciones:** 100% (fallback garantizado)

## Umbrales aplicados

| Regla | Umbral |
|---|---|
| RN-01 (estudio insuficiente) | study_hours < 4 |
| RN-02 (burnout alto) | burnout_level > 60 |
| RN-03 (salud mental baja) | mental_health_score < 5 |
| RN-04 (distracciones altas) | social_media + gaming > 4 |
| RN-05 (sueno insuficiente) | sleep_hours < 6 |

## Cobertura por regla

| Codigo | Descripcion | Estudiantes afectados | Porcentaje |
|---|---|---|---|
| RN-03 | Salud mental baja | 297 | 62.66% |
| RN-04 | Distracciones altas | 290 | 61.18% |
| RN-01 | Estudio insuficiente | 273 | 57.59% |
| RN-05 | Sueno insuficiente | 123 | 25.95% |
| RN-02 | Burnout alto | 117 | 24.68% |

## Artefactos generados

- `students_at_risk.csv` - listado completo con recomendaciones
- `rules_coverage.png` - grafica de cobertura
- `prescriptive_report.md` - este reporte
