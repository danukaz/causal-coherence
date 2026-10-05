"""
check_retardo.py

Semana 7: estimabilidad del orden de retardo p en {1, 2, 3} para el modelo
completo de run_granger_sweep.py (rezagos de los K tópicos de la versión +
control de volumen contemporáneo + intercepto), en las 4 versiones de la
serie. T y K se leen de los CSV de cada versión (K=10, y K=9 en la
residualizada).

Por versión y orden p:
  - observaciones usadas: T - p
  - parámetros por ecuación: K*p + 1 (control) + 1 (intercepto)
  - grados de libertad del denominador: (T - p) - parámetros
  - estimable: df > 0

Solo para los órdenes estimables, AIC y BIC del modelo completo con
destino topic_8 y el control compartido de la versión (el mismo emparejado
que en run_granger_sweep.py), con la log-verosimilitud gaussiana de MCO:
    llf = -n/2 * (ln(2*pi) + ln(RSS/n) + 1),
    AIC = -2*llf + 2*k,  BIC = -2*llf + k*ln(n),
con k los parámetros por ecuación y n las observaciones usadas. Cada orden
se ajusta sobre sus propias T - p observaciones, así que n cambia entre
filas.

Este script no elige el orden: solo reporta los números.

Resultado: data/retardo.csv.
"""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR

ORDENES = (1, 2, 3)
DESTINO = "topic_8"
OUTPUT_PATH = OUTPUT_DIR / "retardo.csv"

# Mismas versiones, lectura y ajuste que el barrido de Granger.
_sweep = runpy.run_path(str(Path(__file__).resolve().parent / "run_granger_sweep.py"))
VERSIONES = _sweep["VERSIONES"]
leer = _sweep["leer"]
ajustar = _sweep["ajustar"]


def armar_diseno_p(series: pd.DataFrame, control_series: pd.Series, destino: str,
                   p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    armar_diseno de run_granger_sweep.py con p rezagos: las columnas son
    los rezagos 1..p de todos los tópicos, el control contemporáneo y el
    intercepto. Con p=1 es la misma matriz que en el barrido.
    """
    n = len(series)
    rezagos = [series.iloc[p - l:n - l].to_numpy() for l in range(1, p + 1)]
    control_contemporaneo = control_series.iloc[p:].to_numpy().reshape(-1, 1)
    intercepto = np.ones((n - p, 1))
    X = np.hstack([*rezagos, control_contemporaneo, intercepto])
    y = series[destino].iloc[p:].to_numpy()
    return X, y


def criterios(rss: float, n: int, k: int) -> tuple[float, float]:
    """AIC y BIC con la log-verosimilitud gaussiana de MCO."""
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return -2 * llf + 2 * k, -2 * llf + k * np.log(n)


def main() -> None:
    rows = []
    for version, (archivo_serie, archivo_control) in VERSIONES.items():
        series = leer(archivo_serie)
        compartido = leer(archivo_control).iloc[:, 0]
        assert series.index.equals(compartido.index), f"{version}: serie y control compartido no calzan"
        T, K = series.shape
        for p in ORDENES:
            n_obs = T - p
            params = K * p + 1 + 1
            df_den = n_obs - params
            fila: dict[str, object] = {"version": version, "T": T, "K": K, "p": p, "n_obs": n_obs,
                                       "params_por_ecuacion": params, "df_den": df_den,
                                       "estimable": df_den > 0, "aic": np.nan, "bic": np.nan}
            if df_den > 0:
                X, y = armar_diseno_p(series, compartido, DESTINO, p)
                assert X.shape == (n_obs, params), f"{version}, p={p}: diseño {X.shape}"
                assert np.linalg.matrix_rank(X) == params, f"{version}, p={p}: diseño sin rango completo"
                rss, k = ajustar(X, y)
                fila["aic"], fila["bic"] = criterios(rss, n_obs, k)
            rows.append(fila)

    tabla = pd.DataFrame(rows)
    print("Parámetros por ecuación = K*p + 1 (control) + 1 (intercepto); df_den = (T - p) - parámetros.")
    print(f"AIC y BIC: modelo completo, destino {DESTINO}, control compartido; "
          f"solo órdenes estimables. Cada fila usa sus propias T - p observaciones.\n")
    print(tabla.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
