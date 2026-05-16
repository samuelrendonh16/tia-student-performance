# Reporte de Evaluación

**Generado:** 2026-05-16T16:26:47
**Muestras de test:** 1,000

## Métricas Globales

| Métrica | Valor |
|---|---|
| Accuracy | 0.8420 |
| Precision | 0.8251 |
| Recall | 0.8680 |
| F1-Score | 0.8460 |
| AUC-ROC | 0.9329 |

## Matriz de Confusión

|  | Predicho: Reprueba | Predicho: Aprueba |
|---|---|---|
| **Real: Reprueba** | 408 | 92 |
| **Real: Aprueba** | 66 | 434 |

## Top 10 Variables Más Importantes

| # | Variable | Importancia |
|---|---|---|
| 1 | productivity_score | 0.9035 |
| 2 | burnout_level | 0.0463 |
| 3 | focus_index | 0.0236 |
| 4 | exercise_minutes | 0.0059 |
| 5 | screen_time_hours | 0.0049 |
| 6 | sleep_hours | 0.0035 |
| 7 | caffeine_intake_mg | 0.0034 |
| 8 | self_study_hours | 0.0034 |
| 9 | mental_health_score | 0.0027 |
| 10 | study_hours | 0.0022 |

## Archivos Generados

- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importances.png`
- `report.json` (este reporte en formato estructurado)
