"""
diagnostico.py

Diagnóstico de dilución del control positivo (Bitacora_Semana7.md,
sección 8). Las series son las del control positivo (planted.py, mismas
claves y mismos beta); cambia solo el análisis:

  V0: análisis original (control propio, piso en 0).
  V1: sin la columna del control de volumen en la regresión.
  V2: la serie que entra a la regresión es la simulada sin piso; el control
      propio se calcula con las masas corregidas por el piso.
  V3: V1 y V2 a la vez.

Cada variante se lee por dos rutas con los mismos diseños: p por remuestreo
salvaje (clave ("diag_remuestreo", variante, efecto, índice, versión,
origen, destino)) y p teórico (F asintótica).
"""

import numpy as np
import pandas as pd

from causal_coherence import granger, planted, surrogates

# variante: (con control de volumen, con piso en la serie de la regresión)
VARIANTES = {"V0": (True, True), "V1": (False, True), "V2": (True, False), "V3": (False, False)}
EFECTOS = ("0.0", "1.6")
RUTAS = ("rem", "teo")


def replica_sin_piso(efecto: str, indice: int, real: pd.DataFrame,
                     betas: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str], float]:
    """
    Los mismos pasos de planted.generar_replica (par, beta, generador y
    simulación) sin aplicar el piso.
    """
    columnas = list(real.columns)
    origen, destino = planted.sortear_par(efecto, indice, columnas)
    fila = betas[(betas["efecto"] == efecto) & (betas["origen"] == origen) & (betas["destino"] == destino)]
    assert len(fila) == 1, f"{efecto} {origen}->{destino}: {len(fila)} betas"
    beta = float(fila["beta"].iloc[0])
    c, a, residuos = surrogates.ajustar_ar1(real)
    i_A = columnas.index(origen)
    rng = granger.generador("plantado_generacion", efecto, indice)
    X = planted.simular_cruzado(c, a, residuos, real.to_numpy().mean(axis=0), len(real),
                                i_A, columnas.index(destino), beta, planted.media_proceso(c, a, i_A), rng)
    return pd.DataFrame(X, index=real.index, columns=real.columns), (origen, destino), beta


def versiones_variante(x_sin_piso: pd.DataFrame, x_con_piso: pd.DataFrame,
                       con_piso: bool) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Por versión: (serie de la regresión, control propio). El control propio
    sale siempre de la serie con piso; la serie, de la que indique con_piso.
    """
    con = surrogates.construir_versiones(x_con_piso)
    if con_piso:
        return {v: (serie, propio) for v, (serie, _, propio) in con.items()}
    sin = series_versiones(x_sin_piso)
    return {v: (sin[v], con[v][2]) for v in surrogates.VERSIONES}


def series_versiones(series: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Solo las series de las 4 versiones, con las funciones y la composición de
    surrogates.construir_versiones. Se usa para la serie sin piso, cuyas masas
    pueden ser negativas y no admiten los controles de construir_versiones.
    """
    return {
        "original": series,
        "diferenciada": surrogates.diferenciar(series),
        "detendenciada": surrogates.detendenciar(series),
        "residualizada": surrogates.residualizar(series).drop(columns=surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA),
    }


def disenos(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str,
            con_control: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """granger.armar_diseno completo y restringido; sin control se elimina esa columna (la penúltima)."""
    X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
    X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
    if not con_control:
        X_f = np.delete(X_f, -2, axis=1)
        X_r = np.delete(X_r, -2, axis=1)
    return X_f, X_r, y


def prueba(X_f: np.ndarray, X_r: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Efecto, F, grados de libertad y p teórico con las fórmulas de granger.estadistico_f."""
    rss_f, params_f = granger.ajustar(X_f, y)
    rss_r, params_r = granger.ajustar(X_r, y)
    n_obs = X_f.shape[0]
    df_num = params_f - params_r
    df_den = n_obs - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    return {"efecto": float(np.log(rss_r / rss_f)), "F": F, "df_num": df_num, "df_den": df_den,
            "p_teo": granger.p_teorico(F, df_num, df_den)}
