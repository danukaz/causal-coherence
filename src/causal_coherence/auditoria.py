"""
auditoria.py

Auditoría independiente (Bitacora_Semana7.md, sección 10): VAR(1) gaussiano
para los bancos de respuesta conocida A5 y A6.

  x_t = c' + a * x_{t-1} + u_t,   u_t ~ N(0, Sigma),
con c y a del ajuste AR(1) real (surrogates.ajustar_ar1), Sigma = np.cov de
los residuos reales (ddof=1), 100 semanas de calentamiento desde la media
del proceso c / (1 - a), sin piso, sin control y sin masas. Con enlace
plantado A -> B: x_B,t recibe además beta * x_A,t-1 y c_B' = c_B - beta *
c_A / (1 - a_A), como en planted.py.

El modelo de prueba tiene los K rezagos más intercepto (sin control).
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from causal_coherence import granger, planted, surrogates

T_LARGA = planted.T_LARGA
ERROR_RELATIVO_MAX = planted.ERROR_RELATIVO_MAX
MAX_ITERACIONES = planted.MAX_ITERACIONES


def parametros_var(real: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """c, a y Sigma (np.cov de los residuos AR(1), ddof=1) de las columnas dadas."""
    c, a, residuos = surrogates.ajustar_ar1(real)
    return c, a, np.cov(residuos, rowvar=False, ddof=1)


def simular_var(c: np.ndarray, a: np.ndarray, sigma: np.ndarray, T: int, rng: np.random.Generator,
                i_A: int = 0, i_B: int = 1, beta: float = 0.0, descartar: bool = True) -> np.ndarray:
    """
    T semanas tras surrogates.CALENTAMIENTO descartadas (con descartar=False
    devuelve también el calentamiento); beta = 0 es el VAR sin enlace.
    """
    media = c / (1 - a)
    c_ajustado = c.copy()
    c_ajustado[i_B] = c[i_B] - beta * media[i_A]
    total = surrogates.CALENTAMIENTO + T
    u = rng.standard_normal((total, len(c))) @ np.linalg.cholesky(sigma).T
    x = media
    simulada = np.empty((total, len(c)))
    for t in range(total):
        x_previo = x
        x = c_ajustado + a * x_previo + u[t]
        x[i_B] += beta * x_previo[i_A]
        simulada[t] = x
    return simulada[-T:] if descartar else simulada


def calibrar_beta(c: np.ndarray, a: np.ndarray, sigma: np.ndarray, efecto: str,
                  rng_clave: tuple[str | int, ...], i_A: int = 0, i_B: int = 1) -> dict[str, float | int]:
    """
    Bisección de beta en [0, cota] (cota duplicada hasta superar el objetivo)
    hasta error relativo < 1%, con el efecto de planted.efecto_poblacional
    sobre T_LARGA semanas. Números aleatorios comunes: x_B(beta) = x_B(0) +
    beta z, con z_t = a_B z_{t-1} + (x_A,t-1 - media_A).
    """
    objetivo = float(efecto)
    media = c / (1 - a)
    # Trayectoria base completa (calentamiento + T_LARGA), con los mismos
    # sorteos que simular_var(..., T_LARGA) con la misma clave.
    base = simular_var(c, a, sigma, T_LARGA, granger.generador(*rng_clave), i_A, i_B, 0.0, descartar=False)
    x_A_previo = np.concatenate([[media[i_A]], base[:-1, i_A]])
    z = lfilter([1.0], [1.0, -a[i_B]], x_A_previo - media[i_A])

    def medir(beta: float) -> float:
        X = base.copy()
        X[:, i_B] = base[:, i_B] + beta * z
        return planted.efecto_poblacional(X[-T_LARGA:], i_A, i_B)

    bajo, alto = 0.0, 1.0
    while medir(alto) < objetivo:
        alto *= 2
    for iteracion in range(1, MAX_ITERACIONES + 1):
        beta = (bajo + alto) / 2
        valor = medir(beta)
        error = abs(valor - objetivo) / objetivo
        if error < ERROR_RELATIVO_MAX:
            return {"beta": beta, "efecto_poblacional": valor, "error_relativo": error, "iteraciones": iteracion}
        if valor < objetivo:
            bajo = beta
        else:
            alto = beta
    raise RuntimeError(f"la bisección no convergió: efecto {efecto}")


def p_ruta_f(X: np.ndarray, pares: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """F y p teórico de cada par (origen, destino) con los K rezagos e intercepto."""
    n = X.shape[0] - 1
    completo = np.hstack([X[:-1], np.ones((n, 1))])
    rss_f = {}
    F = np.empty(len(pares))
    p = np.empty(len(pares))
    for j, (i_A, i_B) in enumerate(pares):
        y = X[1:, i_B]
        if i_B not in rss_f:
            rss_f[i_B] = granger.ajustar(completo, y)
        rss_completo, params_f = rss_f[i_B]
        rss_restringido, params_r = granger.ajustar(np.delete(completo, i_A, axis=1), y)
        df_num, df_den = params_f - params_r, n - params_f
        F[j] = ((rss_restringido - rss_completo) / df_num) / (rss_completo / df_den)
        p[j] = granger.p_teorico(F[j], df_num, df_den)
    return F, p


def p_ruta_remuestreo(X: np.ndarray, pares: list[tuple[int, int]], F: np.ndarray, B: int,
                      claves: list[tuple[str | int, ...]]) -> np.ndarray:
    """p por remuestreo salvaje de cada par, con el generador de su clave."""
    n = X.shape[0] - 1
    completo = np.hstack([X[:-1], np.ones((n, 1))])
    p = np.empty(len(pares))
    for j, ((i_A, i_B), clave) in enumerate(zip(pares, claves)):
        X_r = np.delete(completo, i_A, axis=1)
        p[j] = granger.remuestreo_salvaje(completo, X_r, X[1:, i_B], F[j], 1, n - completo.shape[1],
                                          B, granger.generador(*clave))["p"]
    return p
