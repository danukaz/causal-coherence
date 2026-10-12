"""
planted.py

Control positivo (Bitacora_Semana7.md, sección 6): AR diagonal de los
sustitutos (mismos c, a y filas de residuos reales de
surrogates.ajustar_ar1) más un término cruzado unidireccional,
    x_B,t = c_B' + a_B x_B,t-1 + beta x_A,t-1 + e_B,t,   c_B' = c_B - beta media_A,
con media_A = c_A / (1 - a_A), la media estacionaria del AR(1) de A (así se
conserva la media de B), y la ecuación de A no incluye a B.

beta se calibra por par para cada tamaño de efecto, con
efecto = ln(RSS_restringido / RSS_completo) del modelo con los K rezagos e
intercepto (sin control), medido sobre una simulación larga (T = 20000, 100
semanas de calentamiento, sin piso) con el generador
granger.generador("plantado_calibracion", efecto, origen, destino), el mismo
en todas las iteraciones de la bisección. Con esos números aleatorios
comunes, x_B depende linealmente de beta:
    x_B(beta) = x_B(0) + beta z,   z_t = a_B z_{t-1} + (x_A,t-1 - media_A),  z_{-1} = 0,
así que la serie base se simula una sola vez por par.

Cada réplica: par sorteado con granger.generador("plantado_par", efecto,
índice) entre los 72 pares sin topic_5, serie simulada con
granger.generador("plantado_generacion", efecto, índice) y piso en 0 sobre
la salida final (surrogates.aplicar_piso).
"""

import itertools

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from causal_coherence import granger, surrogates

TAMANOS = ("0.0", "0.4", "0.8", "1.2", "1.6")
T_LARGA = 20000
ERROR_RELATIVO_MAX = 0.01
MAX_ITERACIONES = 200


def media_proceso(c: np.ndarray, a: np.ndarray, i: int) -> float:
    """Media estacionaria del AR(1) del tópico i: c_i / (1 - a_i)."""
    return float(c[i] / (1 - a[i]))


def pares_plantables(columnas: list[str]) -> list[tuple[str, str]]:
    """Pares ordenados que no involucran la columna descartada en la residualizada (72)."""
    sin_descartada = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return list(itertools.permutations(sin_descartada, 2))


def simular_cruzado(c: np.ndarray, a: np.ndarray, residuos: np.ndarray, x0: np.ndarray, T: int,
                    i_A: int, i_B: int, beta: float, media_A: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    T semanas tras surrogates.CALENTAMIENTO descartadas, con
    c_B' = c_B - beta * media_A. Con beta = 0 hace las mismas operaciones, en
    el mismo orden y con los mismos sorteos, que surrogates.simular_ar_diagonal.
    """
    c_ajustado = c.copy()
    c_ajustado[i_B] = c[i_B] - beta * media_A
    filas = rng.integers(0, residuos.shape[0], size=surrogates.CALENTAMIENTO + T)
    x = x0
    simulada = np.empty((surrogates.CALENTAMIENTO + T, len(c)))
    for t, fila in enumerate(filas):
        x_previo = x
        x = c_ajustado + a * x_previo + residuos[fila]
        x[i_B] += beta * x_previo[i_A]
        simulada[t] = x
    return simulada[-T:]


def efecto_poblacional(X: np.ndarray, i_A: int, i_B: int) -> float:
    """ln(RSS_r / RSS_f) de la ecuación de B con los K rezagos e intercepto, sin control."""
    y = X[1:, i_B]
    completo = np.hstack([X[:-1], np.ones((len(y), 1))])
    restringido = np.delete(completo, i_A, axis=1)
    rss_f, _ = granger.ajustar(completo, y)
    rss_r, _ = granger.ajustar(restringido, y)
    return float(np.log(rss_r / rss_f))


def calibrar_beta(efecto: str, origen: str, destino: str, real: pd.DataFrame) -> dict[str, object]:
    """Bisección de beta en [0, cota] hasta error relativo < ERROR_RELATIVO_MAX."""
    objetivo = float(efecto)
    columnas = list(real.columns)
    i_A, i_B = columnas.index(origen), columnas.index(destino)
    c, a, residuos = surrogates.ajustar_ar1(real)
    x0 = real.to_numpy().mean(axis=0)
    if objetivo == 0.0:
        return {"efecto": efecto, "origen": origen, "destino": destino, "beta": 0.0,
                "efecto_poblacional": np.nan, "error_relativo": np.nan, "iteraciones": 0}

    # Serie base completa (con calentamiento) y z, con los mismos sorteos.
    rng = granger.generador("plantado_calibracion", efecto, origen, destino)
    filas = rng.integers(0, residuos.shape[0], size=surrogates.CALENTAMIENTO + T_LARGA)
    base = np.empty((len(filas), len(c)))
    x = x0
    for t, fila in enumerate(filas):
        x = c + a * x + residuos[fila]
        base[t] = x
    media_A = media_proceso(c, a, i_A)
    x_A_previo = np.concatenate([[x0[i_A]], base[:-1, i_A]])
    z = lfilter([1.0], [1.0, -a[i_B]], x_A_previo - media_A)

    def medir(beta: float) -> float:
        X = base.copy()
        X[:, i_B] = base[:, i_B] + beta * z
        return efecto_poblacional(X[-T_LARGA:], i_A, i_B)

    bajo, alto = 0.0, 1.0
    while medir(alto) < objetivo:
        alto *= 2
    for iteracion in range(1, MAX_ITERACIONES + 1):
        beta = (bajo + alto) / 2
        valor = medir(beta)
        error = abs(valor - objetivo) / objetivo
        if error < ERROR_RELATIVO_MAX:
            return {"efecto": efecto, "origen": origen, "destino": destino, "beta": beta,
                    "efecto_poblacional": valor, "error_relativo": error, "iteraciones": iteracion}
        if valor < objetivo:
            bajo = beta
        else:
            alto = beta
    raise RuntimeError(f"la bisección no convergió: {efecto} {origen}->{destino}")


def calibrar_todos(real: pd.DataFrame) -> pd.DataFrame:
    """Una fila por tamaño y par plantable."""
    pares = pares_plantables(list(real.columns))
    return pd.DataFrame([calibrar_beta(e, o, d, real) for e in TAMANOS for o, d in pares])


def sortear_par(efecto: str, indice: int, columnas: list[str]) -> tuple[str, str]:
    pares = pares_plantables(columnas)
    rng = granger.generador("plantado_par", efecto, indice)
    return pares[int(rng.integers(0, len(pares)))]


def generar_replica(efecto: str, indice: int, real: pd.DataFrame,
                    betas: pd.DataFrame) -> tuple[pd.DataFrame, int, tuple[str, str], float]:
    """Serie con el par plantado y el piso aplicado, celdas corregidas, par y beta."""
    columnas = list(real.columns)
    origen, destino = sortear_par(efecto, indice, columnas)
    fila = betas[(betas["efecto"] == efecto) & (betas["origen"] == origen) & (betas["destino"] == destino)]
    assert len(fila) == 1, f"{efecto} {origen}->{destino}: {len(fila)} betas"
    beta = float(fila["beta"].iloc[0])
    c, a, residuos = surrogates.ajustar_ar1(real)
    rng = granger.generador("plantado_generacion", efecto, indice)
    i_A = columnas.index(origen)
    X = simular_cruzado(c, a, residuos, real.to_numpy().mean(axis=0), len(real),
                        i_A, columnas.index(destino), beta, media_proceso(c, a, i_A), rng)
    X, corregidas = surrogates.aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas, (origen, destino), beta
