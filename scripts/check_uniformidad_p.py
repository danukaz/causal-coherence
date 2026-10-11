"""
check_uniformidad_p.py

Semana 7: distribución de los valores p de la calibración. Lee
data/calibracion/p_<familia>.npz (escritos por run_calibracion.py) y, para
cada familia y versión, reporta sobre todos los p (sustitutos x pares):
la cantidad de valores, la fracción menor o igual a 0.05, 0.01 y 0.001, y
el p mínimo global.

No escribe archivos.
"""

import numpy as np
import pandas as pd

from causal_coherence import surrogates
from causal_coherence.config import OUTPUT_DIR

UMBRALES = (0.05, 0.01, 0.001)
ENTRADA = OUTPUT_DIR / "calibracion"


def resumen_familia(familia: str) -> list[dict[str, object]]:
    filas = []
    with np.load(ENTRADA / f"p_{familia}.npz") as z:
        for version in surrogates.VERSIONES:
            p = z[f"p_{version}"]
            fila: dict[str, object] = {"familia": familia, "version": version,
                                       "sustitutos": p.shape[0], "pares": p.shape[1], "n_p": p.size}
            for u in UMBRALES:
                fila[f"frac_p<={u}"] = float(np.mean(p <= u))
            fila["p_min"] = float(p.min())
            filas.append(fila)
    return filas


def main() -> None:
    tabla = pd.DataFrame([f for familia in surrogates.FAMILIAS for f in resumen_familia(familia)])
    with pd.option_context("display.width", 200, "display.float_format", "{:.6g}".format):
        print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
