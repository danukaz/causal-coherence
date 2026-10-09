"""
surrogates.py

Corpus sustitutos para calibrar la etapa de significancia (guía, criterio
E1): series con la forma de data/topic_series_volumen.csv (T=23, K=10,
x = log(1 + masa)) en las que la hipótesis nula vale por construcción.

Familias:
  - ar_diagonal: AR(1) por tópico ajustado por MCO a la serie real,
    x_t = c + a*x_{t-1} + e_t. Se simula desde la media muestral de cada
    serie con 100 semanas de calentamiento descartadas; en cada semana se
    suma una fila completa de residuos reales (los 10 residuos de una misma
    semana juntos), elegida al azar con reemplazo. Se conservan las últimas
    23 semanas.
  - fase_aleatoria: por tópico, rfft, se conserva la amplitud de cada
    coeficiente y se reemplaza la fase por una uniforme independiente (el
    coeficiente 0 queda tal cual; con T=23 no hay frecuencia de Nyquist),
    irfft de largo 23. Fases independientes entre tópicos.

Piso en 0 sobre la salida final de ambas familias (x < 0 pasa a 0, es decir
masa >= 0). Con las masas del sustituto, masa = expm1(x), se recalculan
n_t = suma de las 10 masas, el control compartido log1p(n_t) y el propio
log1p(n_t - masa_j).

construir_versiones reproduce la lógica de scripts/build_series_variants.py
y de scripts/run_granger_sweep.py: diferenciada y detendenciada se aplican
también a los dos controles; la residualizada usa los controles sin
transformar y descarta topic_5 después del PCA sobre las 10 columnas.

Cada sustituto sale de granger.generador("generacion", familia, índice).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from causal_coherence import granger

FAMILIAS = ("ar_diagonal", "fase_aleatoria")
CALENTAMIENTO = 100
COLUMNA_DESCARTADA_RESIDUALIZADA = "topic_5"
VERSIONES = ("original", "diferenciada", "detendenciada", "residualizada")


def ajustar_ar1(series: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """c y a por tópico (MCO con intercepto) y la matriz de residuos (T-1) x K."""
    X = series.to_numpy()
    c = np.empty(X.shape[1])
    a = np.empty(X.shape[1])
    residuos = np.empty((X.shape[0] - 1, X.shape[1]))
    for k in range(X.shape[1]):
        D = np.column_stack([np.ones(X.shape[0] - 1), X[:-1, k]])
        coef, *_ = np.linalg.lstsq(D, X[1:, k], rcond=None)
        c[k], a[k] = coef
        residuos[:, k] = X[1:, k] - D @ coef
    return c, a, residuos


def simular_ar_diagonal(series: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """T semanas de la AR(1) diagonal, tras CALENTAMIENTO semanas descartadas."""
    c, a, residuos = ajustar_ar1(series)
    T = len(series)
    filas = rng.integers(0, residuos.shape[0], size=CALENTAMIENTO + T)
    x = series.to_numpy().mean(axis=0)
    simulada = np.empty((CALENTAMIENTO + T, series.shape[1]))
    for t, fila in enumerate(filas):
        x = c + a * x + residuos[fila]
        simulada[t] = x
    return simulada[-T:]


def simular_fase_aleatoria(series: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Misma amplitud espectral por tópico, fases uniformes independientes."""
    X = series.to_numpy()
    T = X.shape[0]
    assert T % 2 == 1, "con T par habría frecuencia de Nyquist, que este código no trata"
    coef = np.fft.rfft(X, axis=0)                       # (T//2 + 1) x K
    fases = rng.uniform(0.0, 2 * np.pi, size=(coef.shape[0] - 1, X.shape[1]))
    nuevo = coef.copy()
    nuevo[1:] = np.abs(coef[1:]) * np.exp(1j * fases)   # el coeficiente 0 queda igual
    return np.fft.irfft(nuevo, n=T, axis=0)


def aplicar_piso(X: np.ndarray) -> tuple[np.ndarray, int]:
    """x < 0 pasa a 0. Devuelve la serie y la cantidad de celdas corregidas."""
    negativas = X < 0
    return np.where(negativas, 0.0, X), int(negativas.sum())


def generar_sustituto(familia: str, indice: int, real: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sustituto con el índice y las columnas de la serie real, y celdas corregidas por el piso."""
    rng = granger.generador("generacion", familia, indice)
    if familia == "ar_diagonal":
        X = simular_ar_diagonal(real, rng)
    elif familia == "fase_aleatoria":
        X = simular_fase_aleatoria(real, rng)
    else:
        raise ValueError(f"familia desconocida: {familia}")
    X, corregidas = aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas


def controles(series: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Control compartido log1p(n_t) y propio log1p(n_t - masa_j), con n_t = suma de masas."""
    masa = np.expm1(series)
    n_t = masa.sum(axis=1)
    compartido = np.log1p(n_t).rename("control_volumen")
    propio = np.log1p(masa.rsub(n_t, axis=0))  # n_t - masa_j, columna a columna
    return compartido, propio


def diferenciar(df: pd.DataFrame) -> pd.DataFrame:
    """build_differenced de build_series_variants.py."""
    return df.diff().dropna()


def detendenciar(df: pd.DataFrame) -> pd.DataFrame:
    """build_detrended de build_series_variants.py."""
    t = np.arange(len(df))
    detrended = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        y = df[col].to_numpy()
        b, a = np.polyfit(t, y, deg=1)  # y = a + b*t
        detrended[col] = y - (a + b * t)
    return detrended


def residualizar(series: pd.DataFrame) -> pd.DataFrame:
    """build_residualized de build_series_variants.py (sin la varianza explicada)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(series.to_numpy())
    pca = PCA()
    scores = pca.fit_transform(X)
    X_residual = X - scores[:, [0]] @ pca.components_[[0], :]
    return pd.DataFrame(X_residual * scaler.scale_, index=series.index, columns=series.columns)


def construir_versiones(series: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """
    Por versión: (serie, control compartido, control propio con una columna
    por tópico destino), alineados en el mismo índice.
    """
    compartido, propio = controles(series)
    compartido_df = compartido.to_frame()
    versiones = {
        "original": (series, compartido, propio),
        "diferenciada": (diferenciar(series), diferenciar(compartido_df).iloc[:, 0], diferenciar(propio)),
        "detendenciada": (detendenciar(series), detendenciar(compartido_df).iloc[:, 0], detendenciar(propio)),
        "residualizada": (residualizar(series).drop(columns=COLUMNA_DESCARTADA_RESIDUALIZADA),
                          compartido, propio),
    }
    for version, (serie, comp, prop) in versiones.items():
        assert serie.index.equals(comp.index) and serie.index.equals(prop.index), \
            f"{version}: serie y controles no calzan"
    return versiones
