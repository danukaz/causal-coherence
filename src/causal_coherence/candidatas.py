"""
candidatas.py

Semana 8 (Bitacora_Semana8.md, sección 21): piezas para evaluar
configuraciones candidatas (grilla, K) con su propia dinámica.

Generaliza, sin cambiar los módulos existentes:
  - construir_versiones_k: la composición de surrogates.construir_versiones
    (controles, diferenciar, detendenciar, residualizar) con la columna
    descartada en la residualizada como parámetro;
  - simular: las operaciones de planted.simular_cruzado con el largo del
    calentamiento como parámetro;
  - calibrar_beta: la bisección de planted.calibrar_beta (efecto poblacional
    con planted.efecto_poblacional, números aleatorios comunes y
    descomposición lineal en beta) con la serie, el calentamiento y la clave
    de la candidata.
"""

import math

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from causal_coherence import granger, planted, surrogates

DIAS_SEMANA_EQUIVALENTE = 700  # 100 semanas


def calentamiento(dias_por_ventana: int) -> int:
    """100 semanas-equivalentes en ventanas de la grilla: ceil(700 / días por ventana)."""
    return math.ceil(DIAS_SEMANA_EQUIVALENTE / dias_por_ventana)


def masas_por_ventana(theta: np.ndarray, fechas: pd.Series, freq: str) -> tuple[pd.DataFrame, pd.Series]:
    """Suma de theta (filas normalizadas) y conteo de documentos por ventana, con el resample de la grilla."""
    tabla = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(theta.shape[1])])
    tabla["date"] = pd.to_datetime(fechas).to_numpy()
    agrupado = tabla.set_index("date").resample(freq)
    return agrupado.sum(), agrupado.size()


def construir_versiones_k(series: pd.DataFrame, descartada: str) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """surrogates.construir_versiones con la columna descartada de la residualizada como parámetro."""
    compartido, propio = surrogates.controles(series)
    compartido_df = compartido.to_frame()
    versiones = {
        "original": (series, compartido, propio),
        "diferenciada": (surrogates.diferenciar(series), surrogates.diferenciar(compartido_df).iloc[:, 0],
                         surrogates.diferenciar(propio)),
        "detendenciada": (surrogates.detendenciar(series), surrogates.detendenciar(compartido_df).iloc[:, 0],
                          surrogates.detendenciar(propio)),
        "residualizada": (surrogates.residualizar(series).drop(columns=descartada), compartido, propio),
    }
    for version, (serie, comp, prop) in versiones.items():
        assert serie.index.equals(comp.index) and serie.index.equals(prop.index), f"{version}: serie y controles no calzan"
    return versiones


def versiones_control_fijo(control: pd.Series) -> dict[str, pd.Series]:
    """Un control fijo (por ejemplo log1p del conteo real) por los mismos constructores que la serie."""
    df = control.to_frame()
    return {"original": control, "diferenciada": surrogates.diferenciar(df).iloc[:, 0],
            "detendenciada": surrogates.detendenciar(df).iloc[:, 0], "residualizada": control}


def simular(c: np.ndarray, a: np.ndarray, residuos: np.ndarray, x0: np.ndarray, T: int, calent: int,
            i_A: int, i_B: int, beta: float, media_A: float, rng: np.random.Generator,
            descartar: bool = True) -> np.ndarray:
    """planted.simular_cruzado con el calentamiento como parámetro (con descartar=False incluye el calentamiento)."""
    c_ajustado = c.copy()
    c_ajustado[i_B] = c[i_B] - beta * media_A
    filas = rng.integers(0, residuos.shape[0], size=calent + T)
    x = x0
    simulada = np.empty((calent + T, len(c)))
    for t, fila in enumerate(filas):
        x_previo = x
        x = c_ajustado + a * x_previo + residuos[fila]
        x[i_B] += beta * x_previo[i_A]
        simulada[t] = x
    return simulada[-T:] if descartar else simulada


def calibrar_beta(real: pd.DataFrame, efecto: str, i_A: int, i_B: int, calent: int,
                  clave: tuple[str | int, ...]) -> dict[str, float | int]:
    """La bisección de planted.calibrar_beta sobre el AR(1) diagonal de la serie de la candidata."""
    objetivo = float(efecto)
    c, a, residuos = surrogates.ajustar_ar1(real)
    x0 = real.to_numpy().mean(axis=0)
    media_A = planted.media_proceso(c, a, i_A)
    base = simular(c, a, residuos, x0, planted.T_LARGA, calent, i_A, i_B, 0.0, media_A,
                   granger.generador(*clave), descartar=False)
    x_A_previo = np.concatenate([[x0[i_A]], base[:-1, i_A]])
    z = lfilter([1.0], [1.0, -a[i_B]], x_A_previo - media_A)

    def medir(beta: float) -> float:
        X = base.copy()
        X[:, i_B] = base[:, i_B] + beta * z
        return planted.efecto_poblacional(X[-planted.T_LARGA:], i_A, i_B)

    bajo, alto = 0.0, 1.0
    while medir(alto) < objetivo:
        alto *= 2
    for iteracion in range(1, planted.MAX_ITERACIONES + 1):
        beta = (bajo + alto) / 2
        valor = medir(beta)
        error = abs(valor - objetivo) / objetivo
        if error < planted.ERROR_RELATIVO_MAX:
            return {"beta": beta, "efecto_poblacional": valor, "error_relativo": error, "iteraciones": iteracion}
        if valor < objetivo:
            bajo = beta
        else:
            alto = beta
    raise RuntimeError(f"la bisección no convergió: efecto {efecto}, columnas {i_A}->{i_B}")
