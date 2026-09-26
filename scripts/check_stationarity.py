"""
check_stationarity.py

Semana 5: pruebas de estacionariedad sobre las 4 versiones de la serie de
tópicos (original, diferenciada, detendenciada, residualizada), para cada
una de sus 9 columnas:

  - ADF (statsmodels.tsa.stattools.adfuller), regression="c".
    Nula: hay raíz unitaria (no estacionaria).
  - KPSS (statsmodels.tsa.stattools.kpss), regression="c".
    Nula: la serie es estacionaria.

KPSS interpola su p-valor en una tabla y lo satura en los extremos (0.01 y
0.1); cuando pasa, statsmodels emite un InterpolationWarning. El p-valor se
guarda tal como lo entrega statsmodels, y la columna kpss_fuera_de_tabla
marca las filas donde hubo ese aviso.
"""

import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from causal_coherence.config import OUTPUT_DIR

SERIES = {
    "original": "topic_series_log_ratio.csv",
    "diferenciada": "topic_series_diferenciada.csv",
    "detendenciada": "topic_series_detendenciada.csv",
    "residualizada": "topic_series_residualizada.csv",
}
OUTPUT_PATH = OUTPUT_DIR / "stationarity_results.csv"


def test_column(x: pd.Series) -> dict:
    adf_stat, adf_p, adf_lags, adf_nobs, _, _ = adfuller(x, regression="c", result_object=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, _ = kpss(x, regression="c", result_object=False)
    return {
        "n": len(x),
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_lags": adf_lags,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "kpss_lags": kpss_lags,
        "kpss_fuera_de_tabla": any(issubclass(w.category, InterpolationWarning) for w in caught),
    }


def main():
    rows = []
    for version, filename in SERIES.items():
        series = pd.read_csv(OUTPUT_DIR / filename, index_col=0, parse_dates=True)
        for topic in series.columns:
            rows.append({"serie": version, "topico": topic, **test_column(series[topic].dropna())})

    results = pd.DataFrame(rows)
    results["serie"] = pd.Categorical(results["serie"], categories=list(SERIES), ordered=True)
    results = results.sort_values(["serie", "topico"]).reset_index(drop=True)

    print(f"Pruebas de estacionariedad: {len(results)} combinaciones serie x tópico "
          f"(ADF y KPSS, regression='c')\n")
    print(results.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
