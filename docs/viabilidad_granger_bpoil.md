# Viabilidad de la prueba de Granger condicional sobre bpoil

Documento generado por `scripts/make_informe_viabilidad.py` a partir de los CSV de `data/`; todas las cifras salen de esos archivos o de fórmulas calculadas en el script.

## 1. Resumen

Este documento reúne los resultados de la prueba de Granger condicional entre tópicos sobre bpoil (serie de volumen semanal): la corrida sobre los datos reales, la calibración con sustitutos, el control positivo, el diagnóstico de dilución y la auditoría independiente de la implementación.

Con bpoil a resolución semanal (23 ventanas, 22 observaciones por regresión) y la configuración fijada, el procedimiento no tiene poder para detectar enlaces entre tópicos. Un enlace plantado que elimina el 55% del error de predicción (efecto 0.8) se detecta en 3% de las réplicas con la ruta de remuestreo en el banco de respuesta conocida, y uno que elimina el 33% (efecto 0.4), en 1%. Sobre los datos reales no se rechazó ninguna de las 8 familias de hipótesis; el p mínimo fue 0.0189 frente a un umbral de 0.000556. El resultado nulo no distingue entre ausencia de enlaces y falta de poder. El mismo código, con 92 semanas, detecta el 96% de los enlaces de efecto 0.4, de modo que la limitación es la cantidad de puntos en el tiempo y no la implementación. En consecuencia, bpoil a resolución semanal no es apto como corpus principal para estimar causalidad condicional entre tópicos.

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

## 8. Qué se afirma y qué no

Se afirma: con esta configuración y este corpus la prueba no tiene resolución para efectos moderados o fuertes; la implementación se verificó con una librería independiente y con un banco de respuesta conocida. No se afirma: que no existan relaciones entre tópicos en bpoil; que bpoil sea inadecuado para las demás partes del proyecto (línea base, juicio, E2 a E4); que otro corpus o resolución resuelva el problema, lo cual no se ha medido; ni que los resultados del banco gaussiano se trasladen sin cambios a otros corpus, porque usa la dinámica de bpoil. Opciones previstas por la guía ante este caso: corpus de mayor densidad temporal con comparación entre ambos, reducir la familia de hipótesis, otra familia de referencia, o tamaños de efecto con intervalos de confianza en lugar de enlaces con umbral.

## 9. Archivos y commits de origen

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

A1 no deja archivo: su resultado se obtiene al generar este documento, corriendo `prueba_a1` de `scripts/auditoria_independiente.py`.
