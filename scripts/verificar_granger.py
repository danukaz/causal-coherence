"""
verificar_granger.py

Semana 6, primer paso: verificar a mano las ecuaciones 1-3 del
formulario para UN SOLO par de tópicos, antes de escalar a los 90
pares posibles.

Par elegido: topic_3 (causas/investigación) -> topic_8 (consecuencias
financieras), p=1, sobre la serie "original" (volumen).

Incluye el control de volumen como regresor exógeno CONTEMPORÁNEO
(de la misma semana que se predice, no rezagado) en ambas ecuaciones
por igual -- decisión de diseño ya tomada, no algo que el script
decida solo.
"""

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR

ORIGEN = "topic_3"
DESTINO = "topic_8"


def cargar_datos():
    series = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    control = pd.read_csv(OUTPUT_DIR / "control_volumen.csv", index_col=0, parse_dates=True)
    assert (series.index == control.index).all(), "Las fechas de series y control no calzan"
    return series, control["control_volumen"]


def armar_diseno(series: pd.DataFrame, control: pd.Series, excluir: str | None):
    """
    Arma la matriz de diseño para p=1: predictores = valores rezagados
    (t-1) de todos los tópicos (menos 'excluir', si se da) + el control
    de volumen CONTEMPORÁNEO (t, no t-1) + una columna de intercepto.
    """
    topicos = [c for c in series.columns if c != excluir]
    n = len(series)

    X_rezagado = series[topicos].iloc[:-1].to_numpy()   # filas 0..n-2 -> predicen fila 1..n-1
    control_contemporaneo = control.iloc[1:].to_numpy().reshape(-1, 1)  # fila 1..n-1
    intercepto = np.ones((n - 1, 1))

    X = np.hstack([X_rezagado, control_contemporaneo, intercepto])
    y = series[DESTINO].iloc[1:].to_numpy()  # fila 1..n-1

    return X, y, topicos


def ajustar(X, y):
    """OLS por mínimos cuadrados; devuelve RSS y cantidad de parámetros."""
    coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    rss = np.sum((y - y_hat) ** 2)
    return rss, X.shape[1]  # RSS, cantidad de parámetros (incluye intercepto)


def main():
    series, control = cargar_datos()
    T = len(series)
    print(f"T (ventanas totales en la serie): {T}")
    print(f"Par: {ORIGEN} -> {DESTINO}, p=1")
    print(f"Control de volumen: contemporáneo (misma semana que se predice)\n")

    # --- Ecuación 1: modelo completo (incluye el origen) ---
    X_f, y_f, topicos_f = armar_diseno(series, control, excluir=None)
    rss_f, params_f = ajustar(X_f, y_f)

    # --- Ecuación 2: modelo restringido (sin el origen) ---
    X_r, y_r, topicos_r = armar_diseno(series, control, excluir=ORIGEN)
    rss_r, params_r = ajustar(X_r, y_r)

    n_obs = X_f.shape[0]
    print(f"Observaciones usadas (T-1, se pierde 1 por el rezago): {n_obs}")
    print(f"Parámetros modelo completo (incluye {ORIGEN}, control, intercepto): {params_f}")
    print(f"Parámetros modelo restringido (sin {ORIGEN}): {params_r}")
    print(f"\nRSS completo:     {rss_f:.6f}")
    print(f"RSS restringido:  {rss_r:.6f}")

    # --- Ecuación 3: tamaño de efecto y estadístico F ---
    tamano_efecto = np.log(rss_r / rss_f)

    p = 1  # orden de retardo
    df_num = params_f - params_r  # = p, si solo cambia el origen
    df_den = n_obs - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)

    print(f"\nTamaño de efecto (Geweke) F_{{{ORIGEN}->{DESTINO}}}: {tamano_efecto:.4f}")
    print(f"Grados de libertad: numerador={df_num}, denominador={df_den}")
    print(f"Estadístico F: {F:.4f}")

    from scipy import stats
    p_valor_asintotico = 1 - stats.f.cdf(F, df_num, df_den)
    print(f"\np-valor (referencia F asintótica, NO es el p-valor final --")
    print(f"la semana 7 calibra esto con subrogados, esto es solo orientativo): {p_valor_asintotico:.4f}")


if __name__ == "__main__":
    main()