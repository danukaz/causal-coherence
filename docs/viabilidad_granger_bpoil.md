# Viabilidad de la prueba de Granger condicional sobre bpoil

Documento generado por `scripts/make_informe_viabilidad_v2.py` a partir de los CSV de `data/`; todas las cifras salen de esos archivos o de fórmulas calculadas en el script. Las secciones 2 a 7 corresponden a la estimación semanal y se conservan sin cambios de la versión anterior del informe (`scripts/make_informe_viabilidad.py`, commit `6e29f00`); las secciones 8 a 13 agregan los trabajos posteriores, hasta la estimación con la grilla de 2 días y su paquete de validez.

## 1. Resumen

Con bpoil a resolución semanal (23 ventanas) el procedimiento de Granger condicional no tiene poder: un enlace plantado que elimina el 55% del error de predicción se detecta en 3% de las réplicas. Con una grilla de 2 días (53 ventanas consecutivas, del 26 de abril de 2010 al 9 de agosto de 2010, 52 observaciones) y los 10 tópicos, el poder confirmado es de 84% para un enlace de efecto 0.8 y de 53% para uno de efecto 0.4, con falsos positivos de 2.7% a 7.7% en sustitutos con volumen exógeno. La estimación real no rechazó ninguna hipótesis en las 8 familias (p mínimo 0.0017; efecto necesario 0.28 frente a un máximo observado de 0.21 en la residualizada; en las otras tres versiones, original 0.30 frente a 0.12, diferenciada 0.31 frente a 0.20, detendenciada 0.30 frente a 0.14). El resultado nulo es informativo para enlaces que eliminan más del 50% del error y no descarta efectos débiles. Las sensibilidades con K = 10 (retardo 2, grilla de 3 días, recortes) no rechazan nada; la única configuración con un enlace declarado (3 días, K = 5, control compartido) no fue apta en la calibración. Se reportan las dos estimaciones sobre bpoil y las 15 configuraciones evaluadas.

## 2. Configuración fijada

| Parámetro | Valor |
|---|---|
| Ventanas (T) | 23 |
| Observaciones por regresión | 22 |
| Tópicos (K) | 10 |
| Retardo | 1 |
| Parámetros por ecuación (K rezagos, control, intercepto) | 12 |
| m (original, diferenciada, detendenciada) | 90 |
| m (residualizada) | 72 |
| q (Benjamini y Hochberg) | 0.05 |
| B (remuestreo salvaje) | 9999 |
| Fecha de fijación | 5 de octubre de 2026 (sección 6 de Bitacora_Semana6) |

## 3. Resultado sobre los datos reales

| Control | Versión | m | Rechazos BH | p_remuestreo mínimo | Mayor efecto | Mayor F |
|---|---|---|---|---|---|---|
| propio | original | 90 | 0 | 0.0189 | 0.424 | 5.281 |
| propio | diferenciada | 90 | 0 | 0.0238 | 0.573 | 6.964 |
| propio | detendenciada | 90 | 0 | 0.0568 | 0.416 | 5.160 |
| propio | residualizada | 72 | 0 | 0.0863 | 0.339 | 4.439 |
| compartido | original | 90 | 0 | 0.0256 | 0.412 | 5.095 |
| compartido | diferenciada | 90 | 0 | 0.0112 | 0.590 | 7.240 |
| compartido | detendenciada | 90 | 0 | 0.0539 | 0.392 | 4.795 |
| compartido | residualizada | 72 | 0 | 0.0737 | 0.359 | 4.757 |

Efecto y F necesarios para que el menor p de la familia quede por debajo del primer umbral de Benjamini y Hochberg (q/m), con F_req = f.isf(q/m, 1, df_den), efecto_req = ln(1 + F_req/df_den) y reducción del error = 1 - exp(-efecto_req), frente al mayor efecto observado:

| Versión | m | q/m | df_den | F_req | efecto_req | Reducción del error requerida (%) | Mayor efecto observado (propio) | Mayor F observado (propio) |
|---|---|---|---|---|---|---|---|---|
| original | 90 | 0.000556 | 10 | 24.77 | 1.246 | 71.2 | 0.424 | 5.281 |
| diferenciada | 90 | 0.000556 | 9 | 27.16 | 1.391 | 75.1 | 0.573 | 6.964 |
| detendenciada | 90 | 0.000556 | 10 | 24.77 | 1.246 | 71.2 | 0.416 | 5.160 |
| residualizada | 72 | 0.000694 | 11 | 21.71 | 1.090 | 66.4 | 0.339 | 4.439 |

Comprobación de las dos rutas de significancia (remuestreo y F asintótica):

| Control | Versión | Fracción de pares con 0.1 <= r <= 10 | Mediana de r | Pasa |
|---|---|---|---|---|
| propio | original | 1.000 | 0.998 | sí |
| propio | diferenciada | 1.000 | 0.986 | sí |
| propio | detendenciada | 1.000 | 1.004 | sí |
| propio | residualizada | 1.000 | 1.001 | sí |
| compartido | original | 1.000 | 0.998 | sí |
| compartido | diferenciada | 1.000 | 0.998 | sí |
| compartido | detendenciada | 1.000 | 1.008 | sí |
| compartido | residualizada | 1.000 | 1.002 | sí |

## 4. Calibración con sustitutos

| Familia | Versión | Sustitutos con al menos un rechazo (de 1000) | Fracción | Umbral |
|---|---|---|---|---|
| ar_diagonal | original | 16 | 0.016 | 68 |
| ar_diagonal | diferenciada | 8 | 0.008 | 68 |
| ar_diagonal | detendenciada | 22 | 0.022 | 68 |
| ar_diagonal | residualizada | 19 | 0.019 | 68 |
| fase_aleatoria | original | 16 | 0.016 | 68 |
| fase_aleatoria | diferenciada | 11 | 0.011 | 68 |
| fase_aleatoria | detendenciada | 18 | 0.018 | 68 |
| fase_aleatoria | residualizada | 22 | 0.022 | 68 |

## 5. Control positivo

Detección del par plantado con la regla de 2 versiones, por tamaño de efecto:

| Efecto nominal | Detección (regla) | Detección sin no estimables | Réplicas no estimables |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0 |
| 0.4 | 0.003 | 0.003 | 0 |
| 0.8 | 0.027 | 0.027 | 1 |
| 1.2 | 0.030 | 0.030 | 0 |
| 1.6 | 0.057 | 0.057 | 2 |

Detección del par plantado por Benjamini y Hochberg y efecto observado medio, por tamaño y versión:

| Efecto nominal | Versión | Detección (BH) | Efecto observado medio |
|---|---|---|---|
| 0.0 | original | 0.000 | 0.121 |
| 0.0 | diferenciada | 0.000 | 0.132 |
| 0.0 | detendenciada | 0.000 | 0.129 |
| 0.0 | residualizada | 0.000 | 0.108 |
| 0.4 | original | 0.003 | 0.465 |
| 0.4 | diferenciada | 0.007 | 0.416 |
| 0.4 | detendenciada | 0.000 | 0.403 |
| 0.4 | residualizada | 0.003 | 0.159 |
| 0.8 | original | 0.043 | 0.705 |
| 0.8 | diferenciada | 0.023 | 0.577 |
| 0.8 | detendenciada | 0.030 | 0.602 |
| 0.8 | residualizada | 0.010 | 0.216 |
| 1.2 | original | 0.063 | 0.833 |
| 1.2 | diferenciada | 0.023 | 0.714 |
| 1.2 | detendenciada | 0.030 | 0.686 |
| 1.2 | residualizada | 0.007 | 0.252 |
| 1.6 | original | 0.090 | 0.920 |
| 1.6 | diferenciada | 0.040 | 0.766 |
| 1.6 | detendenciada | 0.060 | 0.714 |
| 1.6 | residualizada | 0.010 | 0.233 |

## 6. Diagnóstico de dilución

V0: análisis original; V1: sin control de volumen; V2: sin piso en 0 en la serie de la regresión; V3: V1 y V2. Falsos positivos: máximo, sobre las 4 versiones, de réplicas con al menos un rechazo con efecto 0.0 (de 200).

| Variante | Ruta | Detección a efecto 1.6 (regla) | Falsos positivos a efecto 0.0 (máximo) | Umbral | Recupera poder válido |
|---|---|---|---|---|---|
| V0 | rem | 0.050 | 6 | 18 | no |
| V0 | teo | 0.285 | 17 | 18 | no |
| V1 | rem | 0.040 | 8 | 18 | no |
| V1 | teo | 0.250 | 19 | 18 | no |
| V2 | rem | 0.200 | 5 | 18 | no |
| V2 | teo | 0.655 | 19 | 18 | no |
| V3 | rem | 0.120 | 8 | 18 | no |
| V3 | teo | 0.510 | 18 | 18 | sí |

V3 con la ruta teórica cumple el criterio fijado, pero es un caso límite y no un procedimiento aplicable: no incluye el control de volumen que exige la guía y usa el p teórico, que supone residuos que las series no tienen.

## 7. Auditoría independiente

| Prueba | Qué verifica | Resultado |
|---|---|---|
| A1 | cadena de datos, del corpus T17 a la serie | PASA |
| A2 | F y p recalculados con statsmodels y scipy (error relativo máximo 0.0000000000413) | PASA |
| A3 | remuestreo vectorizado contra bucle explícito | PASA |
| A4 | realismo de los sustitutos (criterio corregido) | PASA |
| A5 (a) | tamaño del test, ruta F, por par | FALLA |
| A5 (b) | tamaño del test, ruta F, conjunto (fracción con p <= 0.05: 0.0705; remuestreo: 0.0490) | FALLA |
| A6 (i) | la detección no decrece al aumentar T ni el efecto | PASA |
| A6 (ii) | a T = 92, detección >= 0.80 para efectos 0.8 y 1.6 | PASA |

A5 falla en la ruta F por una propiedad conocida del F de un VAR con pocas observaciones, que rechaza de más; el remuestreo, que es el método de la guía, queda en 4.9% a nivel 5%.

Detección del par plantado (topic_0 -> topic_1) por Benjamini y Hochberg con la ruta de remuestreo en el banco de respuesta conocida (VAR gaussiano con la dinámica de bpoil, sin piso ni control):

| K | T | Observaciones | Efecto | Detección (remuestreo) |
|---|---|---|---|---|
| 10 | 23 | 22 | 0.4 | 0.01 |
| 10 | 23 | 22 | 0.8 | 0.03 |
| 10 | 23 | 22 | 1.6 | 0.26 |
| 10 | 46 | 45 | 0.4 | 0.29 |
| 10 | 46 | 45 | 0.8 | 0.75 |
| 10 | 46 | 45 | 1.6 | 1.00 |
| 10 | 92 | 91 | 0.4 | 0.96 |
| 10 | 92 | 91 | 0.8 | 1.00 |
| 10 | 92 | 91 | 1.6 | 1.00 |
| 5 | 23 | 22 | 0.4 | 0.11 |
| 5 | 23 | 22 | 0.8 | 0.44 |
| 5 | 23 | 22 | 1.6 | 0.86 |

## 8. Control positivo con series largas

Pipeline completo de la estimación semanal (control propio, piso en 0, remuestreo salvaje, Benjamini y Hochberg por versión y regla de 2 versiones) con la dinámica semanal de bpoil y series más largas, con 100 réplicas por celda. Detección del par plantado con la regla de 2 versiones:

| T | Efecto | Detección (regla) | Réplicas no estimables |
|---|---|---|---|
| 46 | 0.0 | 0.00 | 0 |
| 46 | 0.4 | 0.44 | 0 |
| 46 | 0.8 | 0.62 | 0 |
| 92 | 0.0 | 0.00 | 0 |
| 92 | 0.4 | 0.96 | 0 |
| 92 | 0.8 | 0.99 | 0 |

Comparación con el pipeline a T = 23 (sección 5, regla de 2 versiones) y con el banco gaussiano de A6 (sección 7, remuestreo):

| Efecto | Pipeline T = 23 | Pipeline T = 46 | Pipeline T = 92 | A6 T = 23 | A6 T = 46 | A6 T = 92 |
|---|---|---|---|---|---|---|
| 0.4 | 0.003 | 0.440 | 0.960 | 0.010 | 0.290 | 0.960 |
| 0.8 | 0.027 | 0.620 | 0.990 | 0.030 | 0.750 | 1.000 |

Fracción de réplicas con al menos un rechazo sin enlace plantado (efecto 0.0), por versión; esta fracción, por encima del 5% nominal, motivó el diagnóstico de la sección 9:

| T | Original | Diferenciada | Detendenciada | Residualizada |
|---|---|---|---|---|
| 46 | 0.09 | 0.09 | 0.10 | 0.05 |
| 92 | 0.24 | 0.08 | 0.23 | 0.03 |

## 9. Diagnóstico factorial de falsos positivos

Sin enlace plantado, T = 92 y K = 10, se cruzaron tres factores: C, control de volumen (C0 propio, el actual; C1 sin control; C2 control exógeno fijo); P, piso en 0 (P0 aplicado, P1 sin piso); I, innovaciones (I0 filas reales, I1 gaussianas). Etapa 1 (ruta F): fracción media de réplicas con algún rechazo en la versión original por nivel de cada factor; un factor está implicado si su nivel limpio reduce esa fracción en al menos 8 puntos:

| Factor | Nivel | Fracción media (original) | Reducción respecto del nivel actual | Implicado |
|---|---|---|---|---|
| C | C0 | 0.270 | nivel actual |  |
| C | C1 | 0.069 | 0.201 | sí |
| C | C2 | 0.077 | 0.193 | sí |
| P | P0 | 0.137 | nivel actual |  |
| P | P1 | 0.140 | -0.003 | no |
| I | I0 | 0.150 | nivel actual |  |
| I | I1 | 0.128 | 0.022 | no |

Etapa 2 (remuestreo, 100 réplicas por variante): réplicas con algún rechazo en las versiones original y detendenciada; la variante restaura la calibración si ambas quedan en o bajo el umbral binom.isf(0.05 / 8, 100, 0.05) = 11:

| Variante | Réplicas con rechazo (original) | Réplicas con rechazo (detendenciada) | Fracción con algún enlace (regla) | Restaura la calibración |
|---|---|---|---|---|
| C0-P0-I0 | 21 | 26 | 0.17 | no |
| C1-P0-I0 | 6 | 8 | 0.05 | sí |
| C0-P1-I0 | 23 | 25 | 0.19 | no |
| C0-P0-I1 | 20 | 22 | 0.18 | no |
| C1-P1-I1 | 9 | 10 | 0.08 | sí |

El factor implicado es el control de volumen (C): las dos variantes sin control (C1-P0-I0 y C1-P1-I1) restauran la calibración y las tres con el control propio no.

## 10. Factibilidad de grillas

Ventana de análisis de bpoil, con al menos 5 documentos por ventana; la serie de análisis es la mayor racha de ventanas válidas consecutivas (sin imputar), y las observaciones utilizables con retardo 1 son el largo de la racha menos 1. Corpus viejo (1009 documentos) y nuevo (1158 documentos). Mínimos de la calculadora de poder: 30 observaciones para K = 3, 45 para K = 5 y 90 para K = 10.

| Grilla | Ventanas válidas (viejo) | Racha (viejo) | Observaciones (viejo) | Período de la racha (viejo) | Racha (nuevo) | Observaciones (nuevo) | K que alcanzan el mínimo |
|---|---|---|---|---|---|---|---|
| semanal | 23 | 23 | 22 | 19 de abril de 2010 al 26 de septiembre de 2010 | 23 | 22 | ninguno |
| cada 2 días | 63 | 53 | 52 | 26 de abril de 2010 al 9 de agosto de 2010 | 53 | 52 | 3, 5 |
| cada 3 días | 45 | 36 | 35 | 26 de abril de 2010 al 11 de agosto de 2010 | 41 | 40 | 3 |
| cada 4 días | 36 | 28 | 27 | 20 de abril de 2010 al 9 de agosto de 2010 | 28 | 27 | ninguno |
| cada 5 días | 30 | 26 | 25 | 20 de abril de 2010 al 27 de agosto de 2010 | 26 | 25 | ninguno |
| diaria | 86 | 20 | 19 |  | 27 | 26 | ninguno |

La fila diaria sale de `data/factibilidad_corpus.csv` y `data/factibilidad_corpus_nuevo.csv`; las demás, de `data/factibilidad_grillas.csv`. Ninguna grilla alcanza el mínimo de K = 10; la de 2 días es la única que alcanza K = 5. Las rachas de 2 a 5 días terminan en agosto: la cola de agosto y septiembre queda fuera.

## 11. Configuraciones candidatas

15 combinaciones (5 candidatas de grilla y K por 3 controles: C0 propio, C1 sin control, C2 compartido con el conteo real de documentos), evaluadas con sustitutos AR(1) diagonales con la dinámica de cada serie: 100 réplicas de control positivo por efecto y 300 sustitutos sin enlace. Una combinación es APTA si la detección con la regla de 2 versiones a efecto 0.8 es de al menos 0.80 y las réplicas sin enlace con algún rechazo en la original y la detendenciada quedan en o bajo binom.isf(0.05 / 15, 300, 0.05) = 26.

| Candidata | K | Control | Detección 0.8 (regla) | Detección 0.4 (regla) | Réplicas con rechazo (original) | Réplicas con rechazo (detendenciada) | APTA |
|---|---|---|---|---|---|---|---|
| 2D_K5 | 5 | C0 (propio) | 0.99 | 0.79 | 34 | 34 | no |
| 2D_K5 | 5 | C1 (sin control) | 1.00 | 0.76 | 8 | 14 | sí |
| 2D_K5 | 5 | C2 (compartido) | 1.00 | 0.71 | 15 | 14 | sí |
| 2D_K8 | 8 | C0 (propio) | 0.88 | 0.54 | 25 | 23 | sí |
| 2D_K8 | 8 | C1 (sin control) | 0.84 | 0.51 | 16 | 14 | sí |
| 2D_K8 | 8 | C2 (compartido) | 0.81 | 0.51 | 16 | 15 | sí |
| 2D_K10 | 10 | C0 (propio) | 0.81 | 0.53 | 14 | 19 | sí |
| 2D_K10 | 10 | C1 (sin control) | 0.82 | 0.52 | 10 | 11 | sí |
| 2D_K10 | 10 | C2 (compartido) | 0.80 | 0.41 | 7 | 10 | sí |
| 3D_K5 | 5 | C0 (propio) | 0.84 | 0.48 | 34 | 35 | no |
| 3D_K5 | 5 | C1 (sin control) | 0.81 | 0.42 | 23 | 24 | sí |
| 3D_K5 | 5 | C2 (compartido) | 0.75 | 0.40 | 21 | 23 | no |
| semanal_K5 | 5 | C0 (propio) | 0.36 | 0.13 | 18 | 20 | no |
| semanal_K5 | 5 | C1 (sin control) | 0.28 | 0.12 | 18 | 24 | no |
| semanal_K5 | 5 | C2 (compartido) | 0.28 | 0.08 | 15 | 20 | no |

Regla de elección fijada antes de correr: entre las APTAS, la de mayor K y, en empate, la de mayor detección con efecto 0.4. Resultado: 2D_K10 con C0 (propio). La segunda estimación se fijó con la grilla de 2 días, K = 10 y el control propio como principal (compartido como sensibilidad); el control C1 quedó excluido porque la guía exige controlar el volumen.

## 12. Confirmación de poder y estimación real (2 días, K = 10)

Confirmación con 300 réplicas nuevas por efecto, control propio. Detección del par plantado por Benjamini y Hochberg en cada versión y con la regla de 2 versiones, y réplicas sin enlace con algún rechazo:

| Versión | Detección 0.8 | Detección 0.4 | Efecto observado medio (0.8) | Réplicas sin enlace con rechazo (de 300) | Fracción |
|---|---|---|---|---|---|
| original | 0.840 | 0.557 | 0.729 | 22 | 0.073 |
| diferenciada | 0.623 | 0.260 | 0.543 | 18 | 0.060 |
| detendenciada | 0.860 | 0.547 | 0.712 | 19 | 0.063 |
| residualizada | 0.227 | 0.123 | 0.250 | 14 | 0.047 |
| regla 2 versiones | 0.843 | 0.533 |  | 15 | 0.050 |

Regla de respaldo fijada: se adopta K = 10 si la detección confirmada con efecto 0.8 (regla) es de al menos 0.78. Resultado: 0.843; se adopta 2D_K10, con 22 y 19 réplicas con rechazo (umbral 26). La configuración de respaldo (K = 8, control propio) habría tenido 35 y 39. Combinando con las 100 réplicas de la sección 11 (solo informativo), la detección con efecto 0.8 es 0.835 y con 0.4, 0.532.

Estimación real, ejecutada una sola vez (B = 9999, q = 0.05 por versión). Efecto y F necesarios para que el menor p quede por debajo del primer umbral de Benjamini y Hochberg, con las mismas fórmulas de la sección 3:

| Control | Versión | m | Rechazos BH | p mínimo | Mayor efecto | Mayor F | df_den | F_req | efecto_req |
|---|---|---|---|---|---|---|---|---|---|
| propio | original | 90 | 0 | 0.0276 | 0.118 | 4.99 | 40 | 14.08 | 0.302 |
| propio | diferenciada | 90 | 0 | 0.0017 | 0.203 | 8.78 | 39 | 14.15 | 0.309 |
| propio | detendenciada | 90 | 0 | 0.0279 | 0.129 | 5.51 | 40 | 14.08 | 0.302 |
| propio | residualizada | 72 | 0 | 0.0114 | 0.196 | 8.89 | 41 | 13.46 | 0.284 |
| compartido | original | 90 | 0 | 0.0199 | 0.123 | 5.25 | 40 | 14.08 | 0.302 |
| compartido | diferenciada | 90 | 0 | 0.0025 | 0.197 | 8.49 | 39 | 14.15 | 0.309 |
| compartido | detendenciada | 90 | 0 | 0.0129 | 0.139 | 5.96 | 40 | 14.08 | 0.302 |
| compartido | residualizada | 72 | 0 | 0.0093 | 0.206 | 9.37 | 41 | 13.46 | 0.284 |

Enlaces declarados por la regla de 2 versiones: 0 con el control propio y 0 con el compartido. Comprobación de las dos rutas (criterio de la sección 3):

| Control | Versión | Fracción de pares con 0.1 <= r <= 10 | Mediana de r | Pasa |
|---|---|---|---|---|
| propio | original | 1.000 | 0.984 | sí |
| propio | diferenciada | 0.989 | 0.993 | sí |
| propio | detendenciada | 1.000 | 1.010 | sí |
| propio | residualizada | 1.000 | 0.957 | sí |
| compartido | original | 1.000 | 0.987 | sí |
| compartido | diferenciada | 1.000 | 0.987 | sí |
| compartido | detendenciada | 1.000 | 1.007 | sí |
| compartido | residualizada | 1.000 | 0.958 | sí |

Comparación con la estimación semanal (mismos pares): correlación de Spearman de los efectos y de los p, pares en común entre los 10 de menor p de cada estimación, y enlaces declarados en cada una:

| Control | Versión | Spearman de efectos | Spearman de p | Top 10 en común | Enlaces (semanal) | Enlaces (2 días) |
|---|---|---|---|---|---|---|
| propio | original | 0.05 | -0.04 | 2 | 0 | 0 |
| propio | diferenciada | 0.00 | 0.01 | 1 | 0 | 0 |
| propio | detendenciada | 0.22 | 0.15 | 1 | 0 | 0 |
| propio | residualizada | -0.04 | -0.12 | 2 | 0 | 0 |
| compartido | original | 0.09 | 0.01 | 2 | 0 | 0 |
| compartido | diferenciada | 0.11 | 0.12 | 0 | 0 | 0 |
| compartido | detendenciada | 0.22 | 0.15 | 2 | 0 | 0 |
| compartido | residualizada | 0.07 | -0.01 | 2 | 0 | 0 |

Las dos estimaciones comparten documentos y período, así que no son evidencia independiente; los efectos se comparan por orden porque dependen del número de observaciones.

## 13. Paquete de validez de la serie de 2 días

Estacionariedad (A1), con nivel 0.05; ambas pruebas sesgan hacia "no estacionaria" con 53 puntos:

| Versión | Tópicos | ADF rechaza la raíz unitaria | KPSS rechaza la estacionariedad |
|---|---|---|---|
| original | 10 | 5 | 4 |
| diferenciada | 10 | 10 | 1 |
| detendenciada | 10 | 8 | 0 |
| residualizada | 9 | 7 | 4 |

Componente común (A2): la primera componente explica el 39.0% de la varianza estandarizada. Documentos por ventana (A3): mínimo 5, mediana 16 y máximo 41; la correlación de Spearman entre el valor absoluto del residuo AR(1) y n_t va de -0.09 a 0.19 (p mínimo 0.18).

Calibración con un sustituto realista (bloque B): volumen exógeno con los n_t reales de las 53 ventanas (incluido su ciclo semanal), proporciones simuladas por un AR(1) del log-ratio centrado, sin enlaces, 300 sustitutos. Pasa si las réplicas con algún rechazo en la original y la detendenciada quedan en o bajo binom.isf(0.05 / 4, 300, 0.05) = 24:

| Control | original | diferenciada | detendenciada | residualizada | Regla de 2 versiones | Pasa |
|---|---|---|---|---|---|---|
| propio | 19 (6.3%) | 19 (6.3%) | 17 (5.7%) | 8 (2.7%) | 12 (4.0%) | sí |
| compartido | 22 (7.3%) | 23 (7.7%) | 21 (7.0%) | 11 (3.7%) | 16 (5.3%) | sí |

Sensibilidades sobre los datos reales (bloque C, sin declarar enlaces): rechazos de Benjamini y Hochberg sumados sobre las 4 versiones, menor p, enlaces de la regla y, para K = 10, cuántos de los 10 pares de menor p de la estimación principal (propio, original) están entre los 10 de la sensibilidad:

| Sensibilidad | Ventanas | Retardo | K | Rechazos (propio) | p mínimo (propio) | Enlaces (propio) | Rechazos (compartido) | p mínimo (compartido) | Enlaces (compartido) | Top 10 en común |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 retardo 2 | 53 | 2 | 10 | 0 | 0.0076 | 0 | 0 | 0.0071 | 0 | 3 |
| C2 2D K8 | 53 | 1 | 8 | 0 | 0.0015 | 0 | 0 | 0.0012 | 0 | no comparable |
| C2 2D K5 | 53 | 1 | 5 | 1 | 0.0017 | 0 | 1 | 0.0024 | 0 | no comparable |
| C3 3D K10 | 36 | 1 | 10 | 0 | 0.0038 | 0 | 0 | 0.0031 | 0 | 3 |
| C3 3D K5 | 36 | 1 | 5 | 3 | 0.0011 | 0 | 5 | 0.0002 | 1 | no comparable |
| C4 sin 5 primeras | 48 | 1 | 10 | 0 | 0.0055 | 0 | 0 | 0.0067 | 0 | 7 |
| C4 sin 5 ultimas | 48 | 1 | 10 | 0 | 0.0102 | 0 | 0 | 0.0097 | 0 | 7 |
| C5 min docs 8 | 16 |  |  |  |  |  |  |  |  | no se estima (racha menor que 30) |

Orden de retardo (bloque D, control compartido, mismas filas para p = 1 y p = 2): ecuaciones en que gana cada orden:

| Versión | Ecuaciones | AIC: gana p = 1 | AIC: gana p = 2 | BIC: gana p = 1 | BIC: gana p = 2 |
|---|---|---|---|---|---|
| original | 10 | 4 | 6 | 10 | 0 |
| diferenciada | 10 | 7 | 3 | 9 | 1 |
| detendenciada | 10 | 3 | 7 | 10 | 0 |
| residualizada | 9 | 6 | 3 | 9 | 0 |

## 14. Qué se afirma y qué no

Se afirma: con la grilla de 2 días y 10 tópicos el procedimiento detecta enlaces fuertes (84%) y moderados (53%) con falsos positivos dentro de la tolerancia fijada (hasta 7.7%), y bpoil no muestra enlaces de ese tamaño en la fase de crisis. No se afirma: que no existan relaciones débiles; que el resultado valga para agosto y septiembre (la serie cubre solo 106 días); que el poder medido sea igual en datos reales (los sustitutos no reproducen relaciones no lineales); ni que otro corpus se comporte igual. La estimación sobre datos reales se ejecutó dos veces (semanal y 2 días), y ambas se reportan.

## 15. Archivos y commits de origen

| Archivo | Script productor | Commit | MD5 |
|---|---|---|---|
| `data/topic_series_volumen.csv` | `scripts/build_volumen_series.py` | `2147655`, `9b06eba` | `5be53e06e5e4f92a8939860c17e93f19` |
| `data/resultados_reales.csv` | `scripts/run_resultados_reales.py` | `e15a9c7` | `891aaf5da18770499cce1bdf13559943` |
| `data/rutas.csv` | `scripts/run_resultados_reales.py` | `e15a9c7` | `f23e7bba86707d52249dbe639148c2b3` |
| `data/calibracion/calibracion_ar_diagonal.csv` | `scripts/run_calibracion.py` | `b2a047b` | `060082788f20398b78886d724c76d0a5` |
| `data/calibracion/calibracion_fase_aleatoria.csv` | `scripts/run_calibracion.py` | `b2a047b` | `afc413469c915238114f1785ab0b3c38` |
| `data/control_positivo/resumen_por_version.csv` | `scripts/run_control_positivo.py` | `07bccd1`, `e5c3e85` | `c967e13587ebd8b5744b91f8ef698926` |
| `data/control_positivo/resumen_por_tamano.csv` | `scripts/run_control_positivo.py` | `07bccd1`, `e5c3e85` | `d5767c0d8807cdbf1d28440d2c9e6bbf` |
| `data/diagnostico_dilucion/resumen_por_version.csv` | `scripts/run_diagnostico_dilucion.py` | `c475fff` | `b5cdc12605e287f913e04bbdf0e56216` |
| `data/diagnostico_dilucion/resumen_por_variante.csv` | `scripts/run_diagnostico_dilucion.py` | `c475fff` | `4d6ea6017187f2d6051cfd887d34c473` |
| `data/auditoria/A2.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `a36c3a8ba947b2d3702694fec783cb72` |
| `data/auditoria/A3.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `2f9f5da270402fbd5d55b6d79cf665ce` |
| `data/auditoria/A4_resumen.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `ae3d2b3e509eece5ff99b9183db7c76f` |
| `data/auditoria/A5_por_par.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `ce9d7a2f9af5378a33d31d4872570817` |
| `data/auditoria/A5_fracciones.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `ef26edcf3495431cd2f08cc887262b5f` |
| `data/auditoria/A5.npz` | `scripts/auditoria_independiente.py` | `2ae02d7` | `4ad48dfb529efa000ae73e579ac734f4` |
| `data/auditoria/A6_celdas.csv` | `scripts/auditoria_independiente.py` | `2ae02d7` | `a57e27f2858c03cbb612b6fa02268181` |
| `data/control_positivo_largo/resumen_por_celda.csv` | `scripts/run_control_positivo_largo.py` | `8744f72` | `83e4d4687e1f7028f4cde8e0544b12a3` |
| `data/control_positivo_largo/resumen_por_version.csv` | `scripts/run_control_positivo_largo.py` | `8744f72` | `a2dcdc092f8eaad3810f66c549de02cc` |
| `data/control_positivo_largo/comparacion.csv` | `scripts/run_control_positivo_largo.py` | `8744f72` | `4f80a0710f2c7ea53a86d0041d5e2bcb` |
| `data/diagnostico_fp/resumen_etapa1_efectos_principales.csv` | `scripts/diagnostico_falsos_positivos.py` | `0d9f838` | `ddd1032584fb24f471ba0eda96fea408` |
| `data/diagnostico_fp/resumen_criterio_a.csv` | `scripts/diagnostico_falsos_positivos.py` | `0d9f838` | `4a996b8b9c206a0e1111c508249321b4` |
| `data/diagnostico_fp/resumen_etapa2_por_version.csv` | `scripts/diagnostico_falsos_positivos.py` | `0d9f838` | `27e1ea56577a06879daac3ffe6baf668` |
| `data/diagnostico_fp/resumen_etapa2_regla.csv` | `scripts/diagnostico_falsos_positivos.py` | `0d9f838` | `547dcb8999e274593d6272fcf9b3a0bb` |
| `data/diagnostico_fp/resumen_criterio_b.csv` | `scripts/diagnostico_falsos_positivos.py` | `0d9f838` | `01e9e63bfe671b52878b3a89d2a98519` |
| `data/factibilidad_corpus.csv` | `scripts/factibilidad_corpus.py` | `be30272` | `1dd4ec61276027d6f80f3d548bb95400` |
| `data/factibilidad_corpus_nuevo.csv` | `scripts/factibilidad_corpus.py` | `be30272` | `256b67da229bab3b841ffc5f61552c33` |
| `data/factibilidad_grillas.csv` | `scripts/factibilidad_grillas.py` | `1282e81` | `9c1046b61d32f0c02bb944adca8b8756` |
| `data/candidatas/resumen_aptitud.csv` | `scripts/candidatas_grillas.py` | `d3b0c49`, `077cc8a` | `688e9e3751c5a8d711c6ec718e87f861` |
| `data/candidatas/resumen_c_calibracion.csv` | `scripts/candidatas_grillas.py` | `d3b0c49`, `077cc8a` | `0c6f56e3c027e4818f3449334eba01b4` |
| `data/candidatas/2D_K10_C0_e0.8.csv` | `scripts/candidatas_grillas.py` | `d3b0c49`, `077cc8a` | `32a349ea96eeebb06784c3f285d42dce` |
| `data/confirmacion/resumen_decision.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `ad3be9c190d60195ea6c7c4551c682c6` |
| `data/confirmacion/resumen_deteccion.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `4658f21303e4e061a69198e263972743` |
| `data/confirmacion/resumen_falsos_positivos.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `735fb06542e37f1a212c1e126183012d` |
| `data/confirmacion/resumen_combinado_informativo.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `a896b7168df638b3e4ac04c787a5df27` |
| `data/resultados_reales_2d/resultados.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `25aed235f762f0465226b732908b5c84` |
| `data/resultados_reales_2d/por_version.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `c7db720136c9b607c7fe8efbcaaadc49` |
| `data/resultados_reales_2d/rutas.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `bc71607496464de1632619acff65bcba` |
| `data/resultados_reales_2d/comparacion_resumen.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `2ba6585e6130caa17a265244eaf85651` |
| `data/resultados_reales_2d/enlaces_propio.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `07d1e9f215c17d546160e573eb0aefc3` |
| `data/resultados_reales_2d/enlaces_compartido.csv` | `scripts/confirmacion_y_real.py` | `15e5ed7` | `07d1e9f215c17d546160e573eb0aefc3` |
| `data/validez/A1_estacionariedad.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `7377a563621e3308681fea0eaed0d845` |
| `data/validez/A2_pca_varianza.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `8dd69b56c62698899413391bde5b58b2` |
| `data/validez/A3_documentos.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `55f3925cde8c5085f1f1020e5110fd22` |
| `data/validez/A3_spearman.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `f3e6a508dfe2e343fd05dad7902bcac0` |
| `data/validez/B_calibracion.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `40cb81dc22033ffba47832d71c672fa3` |
| `data/validez/B_criterio.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `3e837263c69947ce0417305d6457254a` |
| `data/validez/B_sustitutos.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `aa7a892d7e8ba0f3234ba5f6f6b4501b` |
| `data/validez/C_resumen.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `1fbf4dfdaee1af17cdbcf2be9afda9e0` |
| `data/validez/C_traslape.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `27d742ddff282b05e1d860a27a0b9446` |
| `data/validez/D_resumen.csv` | `scripts/paquete_validez.py` | `ba24e7a` | `ad2fa9d297f2d4e77e2ba9b88dc7887a` |

A1 de la auditoría (sección 7) no deja archivo: su resultado se obtiene al generar este documento, corriendo `prueba_a1` de `scripts/auditoria_independiente.py` a través del v1. `data/factibilidad_corpus_nuevo.csv` se generó con un script auxiliar no versionado que importa `medir` y `GRILLAS` de `scripts/factibilidad_corpus.py` (commit `be30272`, sin modificarlo) y los aplica al corpus nuevo (`Datasets/t17_timeline17/t17`); su MD5 coincide con el registrado en la sección 20 de Bitacora_Semana8.md.
