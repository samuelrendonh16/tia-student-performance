# Reporte de Evaluación

**Generado:** 2026-05-27T10:36:14
**Muestras de test:** 1,000

## Métricas Globales

| Métrica | Valor |
|---|---|
| Accuracy | 0.8600 |
| Precision | 0.8629 |
| Recall | 0.8560 |
| F1-Score | 0.8594 |
| AUC-ROC | 0.9438 |

## Matriz de Confusión

|  | Predicho: Reprueba | Predicho: Aprueba |
|---|---|---|
| **Real: Reprueba** | 432 | 68 |
| **Real: Aprueba** | 72 | 428 |

## Top 10 Variables Más Importantes

| # | Variable | Importancia |
|---|---|---|
| 1 | productivity_score | 0.8927 |
| 2 | burnout_level | 0.0732 |
| 3 | focus_index | 0.0294 |
| 4 | social_media_hours | 0.0025 |
| 5 | caffeine_intake_mg | 0.0023 |
| 6 | study_hours | 0.0000 |
| 7 | self_study_hours | 0.0000 |
| 8 | sleep_hours | 0.0000 |
| 9 | screen_time_hours | 0.0000 |
| 10 | exercise_minutes | 0.0000 |

## Archivos Generados

- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importances.png`
- `report.json` (este reporte en formato estructurado)
