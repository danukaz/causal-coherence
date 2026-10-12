"""
comparar_reales_sustitutos.py

Semana 7: compara, de forma solo descriptiva, los p_remuestreo reales
(data/resultados_reales.csv, control propio) con los p de los sustitutos de
la calibración (data/calibracion/p_<familia>.npz). No se usa para declarar
enlaces.

Para cada versión y familia, y para cada estadístico (fracción de pares con
p <= 0.05, <= 0.01 y <= 0.001, y p mínimo), calculado por sustituto y sobre
los datos reales: el valor real, la media y los percentiles 2.5 y 97.5 sobre
los 1000 sustitutos (numpy.percentile con su interpolación lineal por
defecto), y el porcentaje de sustitutos con un valor igual o más extremo que
el real (fracción >= la real; p mínimo <= el real).

Resultado: data/comparacion_reales_sustitutos.csv.
"""

import numpy as np
import pandas as pd

from causal_coherence import surrogates
from causal_coherence.config import OUTPUT_DIR

UMBRALES = (0.05, 0.01, 0.001)
CONTROL = "propio"
ENTRADA_SUSTITUTOS = OUTPUT_DIR / "calibracion"
ENTRADA_REALES = OUTPUT_DIR / "resultados_reales.csv"
OUTPUT_PATH = OUTPUT_DIR / "comparacion_reales_sustitutos.csv"


def estadisticos(p: np.ndarray) -> dict[str, np.ndarray]:
    """Por fila de p (un sustituto o los datos reales): fracciones bajo cada umbral y p mínimo."""
    salida = {f"frac_p<={u}": np.mean(p <= u, axis=-1) for u in UMBRALES}
    salida["p_min"] = p.min(axis=-1)
    return salida


def p_reales(reales: pd.DataFrame, version: str, pares: np.ndarray) -> np.ndarray:
    """p_remuestreo reales de la versión, en el mismo orden de pares que el npz."""
    g = reales[(reales["control_tipo"] == CONTROL) & (reales["version"] == version)]
    por_par = g.set_index(g["origen"] + "->" + g["destino"])["p_remuestreo"]
    assert len(por_par) == len(pares) and set(por_par.index) == set(pares), f"{version}: pares distintos"
    return por_par.loc[pares].to_numpy()


def main() -> None:
    reales = pd.read_csv(ENTRADA_REALES, float_precision="round_trip")
    filas = []
    for version in surrogates.VERSIONES:
        for familia in surrogates.FAMILIAS:
            with np.load(ENTRADA_SUSTITUTOS / f"p_{familia}.npz") as z:
                p_sust = z[f"p_{version}"]
                pares = z[f"pares_{version}"]
            est_sust = estadisticos(p_sust)
            est_real = estadisticos(p_reales(reales, version, pares))
            for nombre, valores in est_sust.items():
                real = float(est_real[nombre])
                extremos = valores <= real if nombre == "p_min" else valores >= real
                filas.append({"version": version, "familia": familia, "estadistico": nombre,
                              "valor_real": real, "media_sustitutos": float(valores.mean()),
                              "p2.5_sustitutos": float(np.percentile(valores, 2.5)),
                              "p97.5_sustitutos": float(np.percentile(valores, 97.5)),
                              "n_sustitutos": len(valores),
                              "pct_sustitutos_igual_o_mas_extremo": 100 * float(extremos.mean())})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUTPUT_PATH, index=False)
    print("Control propio. Extremo: fracción >= la real; p mínimo <= el real.\n")
    with pd.option_context("display.width", 220, "display.float_format", "{:.6g}".format):
        print(tabla.to_string(index=False))
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
