"""
check_p_floor.py

Semana 7, antes de calibrar nada: factibilidad de la corrección por
comparaciones múltiples (guía, ecuaciones 10 y 11).

El conjunto de referencia fija un piso del valor p que ninguna cantidad de
datos baja:
  - desplazamiento circular exhaustivo: p_min = 1/T (T traslaciones,
    incluida la identidad)
  - remuestreo con B réplicas: p_min = 1/(B+1)

Con Benjamini-Hochberg sobre m hipótesis al nivel q, la cantidad de
hipótesis que tendrían que empatar en ese piso para que haya algún rechazo
es (ecuación 11):

    hipótesis que deben empatar en el piso = ceil(m * p_min / q)

Si esa cantidad supera m, ninguna configuración de los datos produce un
rechazo.

T es la cantidad de ventanas de la serie original y m = K(K-1) los pares
ordenados de cada versión, ambos leídos de los CSV de data/. El cálculo se
hace con fracciones exactas para que el ceil no dependa del redondeo de
punto flotante.

Resultado: data/p_floor.csv. No interpreta los números.
"""

import math
from fractions import Fraction

import pandas as pd

from causal_coherence.config import OUTPUT_DIR

Q = Fraction(5, 100)
B_VALORES = (999, 4999, 9999)
OUTPUT_PATH = OUTPUT_DIR / "p_floor.csv"

SERIE_ORIGINAL = "topic_series_volumen.csv"
SERIES = {
    "original": "topic_series_volumen.csv",
    "diferenciada": "topic_series_volumen_diferenciada.csv",
    "detendenciada": "topic_series_volumen_detendenciada.csv",
    "residualizada": "topic_series_volumen_residualizada.csv",
}


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / nombre, index_col=0, parse_dates=True)


def hipotesis_por_m() -> dict[int, list[str]]:
    """m = K(K-1) pares ordenados, con las versiones que tienen ese m."""
    por_m: dict[int, list[str]] = {}
    for version, archivo in SERIES.items():
        K = leer(archivo).shape[1]
        por_m.setdefault(K * (K - 1), []).append(version)
    return por_m


def pisos(T: int) -> list[tuple[str, str, Fraction]]:
    """(método, parámetro del conjunto de referencia, p_min), ecuación 10."""
    filas = [("desplazamiento circular exhaustivo", f"T={T}", Fraction(1, T))]
    filas += [("remuestreo", f"B={B}", Fraction(1, B + 1)) for B in B_VALORES]
    return filas


def empates_necesarios(m: int, p_min: Fraction, q: Fraction) -> int:
    """Ecuación 11: ceil(m * p_min / q), en aritmética exacta."""
    return math.ceil(m * p_min / q)


def main() -> None:
    T = len(leer(SERIE_ORIGINAL))
    por_m = hipotesis_por_m()
    print(f"T = {T} ventanas ({SERIE_ORIGINAL}), q = {float(Q)}")
    for m, versiones in por_m.items():
        print(f"m = {m}: {', '.join(versiones)}")

    rows = []
    for m in por_m:
        for metodo, parametro, p_min in pisos(T):
            necesarias = empates_necesarios(m, p_min, Q)
            rows.append({
                "metodo": metodo,
                "parametro": parametro,
                "p_min_fraccion": f"{p_min.numerator}/{p_min.denominator}",
                "p_min": float(p_min),
                "m": m,
                "q": float(Q),
                "m_p_min_sobre_q": float(m * p_min / Q),
                "hipotesis_que_deben_empatar": necesarias,
                "supera_m": necesarias > m,
            })

    tabla = pd.DataFrame(rows)
    print()
    print(tabla.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
