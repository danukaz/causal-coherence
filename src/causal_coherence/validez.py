"""
validez.py

Semana 8, paso 4 (Bitacora_Semana8.md, sección 26): piezas del paquete de
validez de la serie de 2 días.

  - armar_diseno_p: granger.armar_diseno con p rezagos y la primera fila de y
    como parámetro (con p = 1 e inicio = 1 es la misma matriz);
  - prueba_p: efecto, F (df_num = parámetros del completo - del restringido) y
    p teórico, con granger.ajustar y granger.p_teorico;
  - clr y softmax para el sustituto de volumen exógeno del bloque B.
"""

import numpy as np
import pandas as pd

from causal_coherence import granger


def armar_diseno_p(series: pd.DataFrame, control: pd.Series, destino: str, excluir: str | None, p: int,
                   inicio: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rezagos 1..p de los tópicos (sin `excluir`), control contemporáneo e intercepto, desde la fila `inicio`."""
    inicio = p if inicio is None else inicio
    assert inicio >= p, "la primera fila debe tener sus p rezagos"
    topicos = [c for c in series.columns if c != excluir]
    n = len(series)
    rezagos = [series[topicos].iloc[inicio - l:n - l].to_numpy() for l in range(1, p + 1)]
    control_contemporaneo = control.iloc[inicio:].to_numpy().reshape(-1, 1)
    X = np.hstack([*rezagos, control_contemporaneo, np.ones((n - inicio, 1))])
    return X, series[destino].iloc[inicio:].to_numpy()


def prueba_p(X_f: np.ndarray, X_r: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    rss_f, params_f = granger.ajustar(X_f, y)
    rss_r, params_r = granger.ajustar(X_r, y)
    df_num, df_den = params_f - params_r, X_f.shape[0] - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    return {"efecto": float(np.log(rss_r / rss_f)), "F": F, "df_num": df_num, "df_den": df_den,
            "p_teo": granger.p_teorico(F, df_num, df_den), "rss_f": rss_f, "params_f": params_f}


def reemplazo_multiplicativo(proporciones: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    """
    Ceros a delta = la mitad de la menor proporción positiva, reescalando las
    demás de la misma fila para que sigan sumando 1 (reemplazo multiplicativo).
    Devuelve las proporciones, la cantidad de celdas reemplazadas y delta.
    """
    valores = proporciones.to_numpy(dtype=float)
    ceros = valores == 0
    delta = float(valores[~ceros].min() / 2)
    suma_ceros = (ceros * delta).sum(axis=1, keepdims=True)
    nuevas = np.where(ceros, delta, valores * (1 - suma_ceros))
    return pd.DataFrame(nuevas, index=proporciones.index, columns=proporciones.columns), int(ceros.sum()), delta


def clr(proporciones: pd.DataFrame) -> pd.DataFrame:
    """Log-ratio centrado por fila."""
    logs = np.log(proporciones)
    return logs.sub(logs.mean(axis=1), axis=0)


def softmax_filas(z: np.ndarray) -> np.ndarray:
    """Proporciones a partir de un clr (recentrado por fila antes de exponenciar)."""
    z = z - z.mean(axis=1, keepdims=True)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
