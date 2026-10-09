"""
granger.py

Prueba de Granger condicional con p=1 y su valor p por remuestreo salvaje
(guía, sección 8, ecuaciones 3 y 7 a 9).

armar_diseno, ajustar y estadistico_f reproducen la lógica de
scripts/run_granger_sweep.py (control contemporáneo, intercepto, F con
df_num = parámetros_f - parámetros_r y df_den = n_obs - parámetros_f).

Remuestreo salvaje con diseño fijo:
    M = I - Q Q^T,   RSS = ||M y||^2                          (ecuación 7)
    y*_t = y_hat_r_t + eta_t * eps_r_t,  eta de Rademacher       (ecuación 8)
    p = (1 + #{b : F*_b >= F_obs}) / (B + 1)                      (ecuación 9)
M_f y M_r se calculan una sola vez por par y las B réplicas se evalúan
juntas como una matriz B x n.

Generadores por identidad: SeedSequence(entropy=42, spawn_key=clave), con
la clave armada desde la identidad. El propósito va siempre primero:
    generación de un sustituto:   (generacion, familia, índice)
    remuestreo de datos reales:   (remuestreo, versión, origen, destino)
    remuestreo de un sustituto:   (remuestreo, familia, índice, versión, origen, destino)
Cada texto se convierte con zlib.crc32 de su UTF-8 (estable, no depende de
PYTHONHASHSEED) y cada índice entra tal cual. Así cada p queda fijo por sí
solo y no depende del orden en que se recorren los pares.
"""

import zlib

import numpy as np
import pandas as pd
from scipy import stats

SEMILLA_BASE = 42


def armar_diseno(series: pd.DataFrame, control_series: pd.Series, destino: str,
                 excluir: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Rezagos de los tópicos (sin `excluir`), control contemporáneo e intercepto."""
    topicos = [c for c in series.columns if c != excluir]
    n = len(series)
    X_rezagado = series[topicos].iloc[:-1].to_numpy()
    control_contemporaneo = control_series.iloc[1:].to_numpy().reshape(-1, 1)
    intercepto = np.ones((n - 1, 1))
    X = np.hstack([X_rezagado, control_contemporaneo, intercepto])
    y = series[destino].iloc[1:].to_numpy()
    return X, y


def ajustar(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """RSS por mínimos cuadrados (lstsq) y cantidad de parámetros."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = np.sum((y - X @ coef) ** 2)
    return rss, X.shape[1]


def estadistico_f(series: pd.DataFrame, control_series: pd.Series, origen: str,
                  destino: str) -> dict[str, float | int]:
    """F observado, grados de libertad y RSS del completo y del restringido."""
    X_f, y = armar_diseno(series, control_series, destino, excluir=None)
    rss_f, params_f = ajustar(X_f, y)
    X_r, _ = armar_diseno(series, control_series, destino, excluir=origen)
    rss_r, params_r = ajustar(X_r, y)

    n_obs = X_f.shape[0]
    df_num = params_f - params_r
    df_den = n_obs - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    return {"n_obs": n_obs, "F": F, "df_num": df_num, "df_den": df_den,
            "rss_f": rss_f, "rss_r": rss_r}


def p_teorico(F: float, df_num: int, df_den: int) -> float:
    """Valor p de la F asintótica, igual que en run_granger_sweep.py."""
    return 1 - stats.f.cdf(F, df_num, df_den)


def proyector_residual(X: np.ndarray) -> np.ndarray:
    """M = I - Q Q^T con Q de la descomposición QR delgada de X (ecuación 7)."""
    Q, _ = np.linalg.qr(X, mode="reduced")
    return np.eye(X.shape[0]) - Q @ Q.T


def f_replicas(M_f: np.ndarray, M_r: np.ndarray, y_hat_r: np.ndarray, eps_r: np.ndarray,
               eta: np.ndarray, df_num: int, df_den: int) -> np.ndarray:
    """F*_b para cada fila de eta (B x n), con y* = y_hat_r + eta * eps_r (ecuación 8)."""
    Y = y_hat_r + eta * eps_r                 # B x n, una réplica por fila
    rss_f = np.sum((Y @ M_f) ** 2, axis=1)    # M es simétrica: (M y*)^T = y*^T M
    rss_r = np.sum((Y @ M_r) ** 2, axis=1)
    return ((rss_r - rss_f) / df_num) / (rss_f / df_den)


def multiplicadores_rademacher(B: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Matriz B x n de +1 y -1 con probabilidad 1/2 cada uno."""
    return rng.integers(0, 2, size=(B, n)) * 2.0 - 1.0


def remuestreo_salvaje(X_f: np.ndarray, X_r: np.ndarray, y: np.ndarray, F_obs: float,
                       df_num: int, df_den: int, B: int,
                       rng: np.random.Generator) -> dict[str, object]:
    """Valor p por remuestreo salvaje con B réplicas (ecuaciones 7 a 9)."""
    M_f = proyector_residual(X_f)
    M_r = proyector_residual(X_r)
    eps_r = M_r @ y
    y_hat_r = y - eps_r
    eta = multiplicadores_rademacher(B, len(y), rng)
    F_star = f_replicas(M_f, M_r, y_hat_r, eps_r, eta, df_num, df_den)
    n_superan = int(np.sum(F_star >= F_obs))
    return {"p": (1 + n_superan) / (B + 1), "n_superan": n_superan, "F_star": F_star}


def clave_semilla(*identidad: str | int) -> tuple[int, ...]:
    """Textos a zlib.crc32 de su UTF-8; enteros no negativos tal cual."""
    clave = []
    for componente in identidad:
        if isinstance(componente, str):
            clave.append(zlib.crc32(componente.encode("utf-8")))
        else:
            entero = int(componente)
            assert entero >= 0, f"componente negativo en la identidad: {identidad}"
            clave.append(entero)
    return tuple(clave)


def generador(*identidad: str | int) -> np.random.Generator:
    """Generator con SeedSequence(entropy=SEMILLA_BASE, spawn_key=clave_semilla(identidad))."""
    semilla = np.random.SeedSequence(entropy=SEMILLA_BASE, spawn_key=clave_semilla(*identidad))
    return np.random.default_rng(semilla)


def benjamini_hochberg(p: np.ndarray, q: float, m: int) -> np.ndarray:
    """
    Máscara de rechazos de Benjamini y Hochberg: ordena p_(1) <= ... <= p_(m),
    toma k* = max{k : p_(k) <= k q / m} y rechaza las k* menores. La familia m
    se declara como argumento y no se infiere del largo de p (guía, sección 4).
    """
    assert len(p) == m, f"se declararon m={m} hipótesis pero llegaron {len(p)} valores p"
    orden = np.argsort(p, kind="stable")
    bajo_umbral = p[orden] <= q * np.arange(1, m + 1) / m
    rechazos = np.zeros(m, dtype=bool)
    if bajo_umbral.any():
        k = int(np.nonzero(bajo_umbral)[0].max()) + 1
        rechazos[orden[:k]] = True
    return rechazos
