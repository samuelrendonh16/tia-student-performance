# Contexto de Negocio - TIA Student Performance

**Proyecto:** Modelo predictivo de aprobacion estudiantil
**Curso:** Machine Learning - IU Pascual Bravo
**Autor:** Samuel Rendon Hincapie
**Metodologia:** CRISP-DM

---

## 1. Caso de estudio

El rendimiento academico universitario es un problema de impacto directo:
afecta la permanencia del estudiante, el prestigio de la institucion y la
empleabilidad futura. Tradicionalmente, las universidades intervienen
**despues** de que un estudiante reprueba, cuando la solucion es mas costosa
y limitada.

Este proyecto explora si es posible **anticipar** la aprobacion o reprobacion
del examen final a partir de habitos de estudio, salud mental y bienestar
fisico. Si el modelo puede identificar a tiempo a los estudiantes en riesgo,
la universidad puede activar planes de acompanamiento personalizados antes
de que el resultado sea irreversible.

El dataset utilizado contiene **5,000 estudiantes** con 21 variables que
describen sus habitos diarios (horas de estudio, sueno, redes sociales,
gaming), su bienestar (salud mental, burnout, productividad) y su contexto
academico (nivel educativo, deadlines proximos, trabajo de medio tiempo).

---

## 2. Pregunta de negocio

### Pregunta principal

> **Deberia un estudiante cambiar sus habitos de estudio y vida social
> basandose en la prediccion de si aprobara o reprobara el examen final?**

### Justificacion

El rendimiento academico no depende solo de la capacidad intelectual.
Factores como las horas de estudio, la salud mental, el tiempo en redes
sociales y el nivel de agotamiento impactan directamente en la nota final.

Un modelo que prediga si un estudiante aprobara o reprobara le permitiria
tomar decisiones a tiempo: aumentar horas de estudio, reducir distracciones
o buscar apoyo en salud mental.

### Preguntas secundarias (analitica descriptiva)

Estas preguntas se respondieron durante el EDA (TIA2) y guiaron la
seleccion de variables del modelo:

| # | Pregunta | Respuesta encontrada |
|---|---|---|
| P1 | Que tan comun es reprobar en este dataset? | La nota media es 18.8 sobre 100 y el maximo real es 64. Reprobar es MAS comun que aprobar. |
| P2 | Que variables tienen mayor relacion con la nota? | focus_index (r=0.75), mental_health_score (r=0.55), study_hours (r=0.51), burnout_level (r=-0.41). |
| P3 | Las variables demograficas (genero, edad) afectan la nota? | NO. Genero, edad, nivel academico e internet_quality no muestran diferencia entre grupos. |
| P4 | Las redes sociales y el gaming afectan el rendimiento? | Redes sociales: ligero impacto negativo (r=-0.11). Gaming: practicamente nulo (r=-0.05). |
| P5 | Hay datos faltantes o duplicados? | NO. El dataset esta completo: 0 nulos, 0 duplicados. |

---

## 3. Reglas de negocio

Las reglas operan sobre los **estudiantes que el modelo predice como en
riesgo** (prediccion = reprueba). Para cada estudiante, se evaluan las
5 reglas y se generan recomendaciones personalizadas.

| Codigo | Condicion | Recomendacion generada |
|---|---|---|
| **RN-01** | study_hours < 4 | Aumentar horas de estudio a minimo 6 horas diarias |
| **RN-02** | burnout_level > 60 | Nivel de agotamiento alto - incluir pausas y descanso |
| **RN-03** | mental_health_score < 5 | Buscar apoyo en el servicio de bienestar universitario |
| **RN-04** | social_media_hours + gaming_hours > 4 | Reducir redes sociales y videojuegos - invertir ese tiempo en estudio |
| **RN-05** | sleep_hours < 6 | Establecer rutina de sueno de minimo 7 horas diarias |
| (default) | Ninguna regla activada | Revisar habitos generales con un tutor academico |

### Notas sobre las reglas

- Las reglas son **independientes y aditivas**: un estudiante puede activar
  varias a la vez. La recomendacion final concatena todas las que apliquen.
- Los umbrales fueron calibrados con los **percentiles del dataset** y con
  recomendaciones generales de la literatura academica.
- La RN-04 usa gaming_hours aunque NO entre al modelo predictivo. **Una
  regla de negocio puede usar variables que el modelo no usa**, porque opera
  sobre los datos crudos y tiene logica propia. Es lo mismo que en un
  sistema de credito: el modelo puede excluir la edad (sesgo legal) pero
  una regla de negocio si puede decir solo mayores de 18.

---

## 4. Funciones del modelo

El proyecto produce **dos funciones complementarias**:

### 4.1 Funcion predictiva

| Aspecto | Detalle |
|---|---|
| Tipo | Clasificacion binaria supervisada |
| Algoritmo | DecisionTreeClassifier(max_depth=5) con balanceo SMOTE |
| Entrada | 13 features: habitos de estudio, salud mental, contexto |
| Salida | 0 = Reprueba / 1 = Aprueba + probabilidad asociada |
| Rol en el negocio | Identificar a tiempo a los estudiantes en riesgo |

### 4.2 Funcion prescriptiva

| Aspecto | Detalle |
|---|---|
| Tipo | Sistema basado en reglas (no ML) |
| Mecanismo | Evaluacion secuencial de RN-01 a RN-05 |
| Entrada | DataFrame con los datos crudos de los estudiantes en riesgo |
| Salida | DataFrame con recomendaciones personalizadas + reporte de cobertura |
| Rol en el negocio | Traducir la prediccion en acciones concretas para el estudiante |

---

## 5. Criterio de negocio para evaluar el resultado

> **NOTA IMPORTANTE:** estas son metricas de **negocio**, no metricas del
> modelo. Las metricas del modelo (accuracy, precision, recall, F1, AUC)
> estan en reports/v1/report.md.

### Criterio 1 - Cobertura de la prediccion

**A cuantos estudiantes en riesgo logramos identificar?**

- Meta: identificar al menos el 80% de los estudiantes que efectivamente
  reprueban (esto se traduce en recall >= 0.80 para la clase 0).
- Resultado actual: el modelo identifica al **86.8%** de los estudiantes que
  reprueban (recall = 0.868).

### Criterio 2 - Calidad de la prediccion

**De los identificados, cuantos realmente reprobarian sin intervencion?**

- Meta: que al menos el 75% de las alertas sean verdaderas (precision >= 0.75).
- Resultado actual: precision del **82.5%**, lo que evita saturar al equipo
  de bienestar con falsos positivos.

### Criterio 3 - Accionabilidad de la recomendacion

**Cada estudiante en riesgo recibe al menos una recomendacion concreta?**

- Meta: 100% de los estudiantes en riesgo deben tener al menos una accion
  recomendada (gracias al fallback revisar con tutor academico).
- Resultado actual: cobertura del **100%** garantizada por el fallback.

### Criterio 4 - Distribucion de las reglas activadas

**Las reglas distinguen perfiles distintos o todas dicen lo mismo?**

- Meta: que ninguna regla individual cubra mas del 90% de los casos, para
  garantizar diversidad de recomendaciones.
- Este criterio se verifica en reports/v1/rules_coverage.png.

---

## 6. Variables del modelo

### 6.1 Variables de entrada (features)

Las 13 features que entran al modelo, agrupadas tematicamente:

#### Habitos de estudio
| Variable | Tipo | Rango observado | Descripcion |
|---|---|---|---|
| study_hours | float | 0 a 11.8 h | Horas diarias dedicadas a clases formales |
| self_study_hours | float | 0 a 10 h | Horas diarias de estudio independiente |
| social_media_hours | float | 0 a 8 h | Horas diarias en redes sociales |
| screen_time_hours | float | 0 a 14 h | Horas totales frente a pantallas |

#### Bienestar fisico
| Variable | Tipo | Rango observado | Descripcion |
|---|---|---|---|
| sleep_hours | float | 4 a 10 h | Horas de sueno diarias |
| exercise_minutes | int | 0 a 180 min | Minutos diarios de ejercicio |
| caffeine_intake_mg | int | 0 a 499 mg | Consumo diario de cafeina |

#### Contexto academico/laboral
| Variable | Tipo | Valores | Descripcion |
|---|---|---|---|
| part_time_job | int | 0/1 | Tiene trabajo de medio tiempo? |
| upcoming_deadline | int | 0/1 | Tiene entregas proximas? |

#### Indicadores derivados (vienen pre-calculados en el dataset)
| Variable | Tipo | Rango | Descripcion |
|---|---|---|---|
| mental_health_score | int | 1 a 10 | Score auto-reportado de salud mental |
| focus_index | float | 0 a 100 | Indice de concentracion |
| burnout_level | float | 0 a 100 | Nivel de agotamiento academico |
| productivity_score | float | 0 a 100 | Score de productividad percibida |

### 6.2 Variables excluidas del modelo

Estas variables existen en el dataset pero **NO entran al modelo** porque el
EDA mostro que no aportan informacion predictiva:

| Variable | Razon de exclusion | Se usa en reglas? |
|---|---|---|
| student_id | Solo es identificador | No |
| age | Correlacion casi nula (-0.01) con exam_score | No |
| gender | Sin diferencia entre categorias (boxplots) | No |
| academic_level | Sin diferencia entre categorias | No |
| internet_quality | Sin impacto en la nota | No |
| online_classes_hours | Correlacion practicamente nula (0.00) | No |
| **gaming_hours** | **Correlacion muy baja (-0.05)** | **SI, en RN-04** |

### 6.3 Variable objetivo

| Aspecto | Detalle |
|---|---|
| Variable original | exam_score (continua, rango observado 1 a 64) |
| Variable derivada | aprobado (binaria, 0/1) |
| Regla de binarizacion | aprobado = 1 si exam_score >= mediana(exam_score) |
| Umbral calculado | 18.01 (la mediana del dataset) |
| Distribucion | 50% aprueba / 50% reprueba (balanceado por construccion) |

---

## 7. Limitaciones y consideraciones eticas

- **El modelo NO sustituye al juicio humano.** Las recomendaciones son
  apoyo a la decision, no reemplazo del acompanamiento de un tutor.
- **Sesgo del auto-reporte.** Variables como mental_health_score o
  focus_index dependen de auto-evaluacion del estudiante; pueden estar
  sesgadas en ambos sentidos.
- **Causalidad vs correlacion.** El modelo identifica patrones, no causas.
  Que focus_index correlacione con la nota no significa que mejorando
  el focus se garantice aprobar.
- **El umbral de aprobacion (mediana) es estadistico, no academico.** La
  universidad real tendria un umbral fijo (ej. nota >= 30) que podriamos
  parametrizar en config.yaml.

---

## 8. Flujo CRISP-DM aplicado

1. **Comprension del negocio:** pregunta de negocio + reglas RN-01 a RN-05
2. **Comprension de los datos:** EDA del TIA2 (5000 filas, 21 vars, 0 nulos)
3. **Preparacion de los datos:** eliminar 7 vars irrelevantes, binarizar exam_score
4. **Modelado:** DecisionTreeClassifier(max_depth=5) con Pipeline + SMOTE
5. **Evaluacion:** Accuracy 0.84, Precision 0.83, Recall 0.87, AUC 0.93
6. **Implementacion:** modelo serializado + reglas prescriptivas + reportes
