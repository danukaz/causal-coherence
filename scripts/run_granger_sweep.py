"""
run_granger_sweep.py

Semana 6: escala la verificación de Granger de verificar_granger_v2.py (un
par, topic_3 -> topic_8) a todos los pares ordenados (origen, destino) de
los tópicos presentes en cada versión de la serie de volumen, con las dos
formas del control de volumen:

  - compartido: log(1+n_t), el mismo para todas las ecuaciones; se
    diferencia y detendencia en paralelo con la serie (control_volumen_*.csv)
    y en la residualizada se usa el original.
  - propio: log(1 + n_t - masa_cruda_j) para el tópico destino j, con la
    masa cruda invertida con expm1 desde topic_series_volumen.csv y n_t
    desde control_volumen.csv. Se diferencia y detendencia con las mismas
    funciones de build_series_variants.py; en la residualizada se usa la
    forma original, igual que el compartido.

Ecuaciones 1-3 del formulario, p=1, control contemporáneo, igual que en
verificar_granger_v2.py: modelo completo (rezagos de todos los tópicos de la
versión + control + intercepto) contra el restringido (sin el origen).

El p-valor es orientativo (F asintótica): NO es el valor final; la semana 7
lo calibra con subrogados.

Resultado: data/granger_results.csv.
"""

import itertools
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence.config import OUTPUT_DIR

P = 1
OUTPUT_PATH = OUTPUT_DIR / "granger_results.csv"

VERSIONES = {
    "original": ("topic_series_volumen.csv", "control_volumen.csv"),
    "diferenciada": ("topic_series_volumen_diferenciada.csv", "control_volumen_diferenciado.csv"),
    "detendenciada": ("topic_series_volumen_detendenciada.csv", "control_volumen_detendenciado.csv"),
    "residualizada": ("topic_series_volumen_residualizada.csv", "control_volumen.csv"),
}

# Las mismas transformaciones que se aplican al control compartido.
_variants = runpy.run_path(str(Path(__file__).resolve().parent / "build_series_variants.py"))
build_differenced = _variants["build_differenced"]
build_detrended = _variants["build_detrended"]


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / nombre, index_col=0, parse_dates=True)


def controles_propios() -> dict[str, pd.DataFrame]:
    """
    log(1 + n_t - masa_cruda_j) para los 10 tópicos (una columna por tópico
    destino), en la forma que corresponde a cada versión.
    """
    masa_cruda = np.expm1(leer("topic_series_volumen.csv"))  # deshace el log1p
    n_t = np.expm1(leer("control_volumen.csv").iloc[:, 0])
    propio = np.log1p(masa_cruda.rsub(n_t, axis=0))  # n_t - masa_j, columna a columna
    return {
        "original": propio,
        "diferenciada": build_differenced(propio),
        "detendenciada": build_detrended(propio),
        "residualizada": propio,
    }


def armar_diseno(series: pd.DataFrame, control_series: pd.Series, destino: str,
                 excluir: str | None) -> tuple[np.ndarray, np.ndarray]:
    """armar_diseno de verificar_granger_v2.py, con el destino como parámetro."""
    topicos = [c for c in series.columns if c != excluir]
    n = len(series)
    X_rezagado = series[topicos].iloc[:-1].to_numpy()
    control_contemporaneo = control_series.iloc[1:].to_numpy().reshape(-1, 1)
    intercepto = np.ones((n - 1, 1))
    X = np.hstack([X_rezagado, control_contemporaneo, intercepto])
    y = series[destino].iloc[1:].to_numpy()
    return X, y


def ajustar(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """ajustar de verificar_granger_v2.py, sin cambios."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = np.sum((y - X @ coef) ** 2)
    return rss, X.shape[1]


def probar(series: pd.DataFrame, control_series: pd.Series, origen: str, destino: str) -> dict:
    """Las ecuaciones de probar() de verificar_granger_v2.py, devolviendo los valores."""
    X_f, y_f = armar_diseno(series, control_series, destino, excluir=None)
    rss_f, params_f = ajustar(X_f, y_f)
    X_r, y_r = armar_diseno(series, control_series, destino, excluir=origen)
    rss_r, params_r = ajustar(X_r, y_r)

    n_obs = X_f.shape[0]
    tamano_efecto = np.log(rss_r / rss_f)
    df_num = params_f - params_r
    df_den = n_obs - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    p_valor = 1 - stats.f.cdf(F, df_num, df_den)
    return {"n_obs": n_obs, "efecto": tamano_efecto, "F": F, "df_num": df_num, "df_den": df_den,
            "p_valor_orientativo": p_valor, "_rss_f": rss_f, "_rss_r": rss_r}


def main() -> None:
    propios = controles_propios()
    rows = []
    for version, (archivo_serie, archivo_control) in VERSIONES.items():
        series = leer(archivo_serie)
        compartido = leer(archivo_control).iloc[:, 0]
        assert series.index.equals(compartido.index), f"{version}: serie y control compartido no calzan"
        propio = propios[version].loc[series.index]
        assert propio.index.equals(series.index), f"{version}: serie y control propio no calzan"
        for origen, destino in itertools.permutations(series.columns, 2):
            for control_tipo, control in (("compartido", compartido), ("propio", propio[destino])):
                rows.append({"version": version, "control_tipo": control_tipo,
                             "origen": origen, "destino": destino,
                             **probar(series, control, origen, destino)})

    results = pd.DataFrame(rows)
    rss_negativo = (results["_rss_f"] < 0) | (results["_rss_r"] < 0)
    columnas = ["version", "control_tipo", "origen", "destino", "n_obs", "efecto", "F",
                "df_num", "df_den", "p_valor_orientativo"]
    numericas = results[["efecto", "F", "p_valor_orientativo", "_rss_f", "_rss_r"]]
    problematicas = results[numericas.isna().any(axis=1) | np.isinf(numericas).any(axis=1) | rss_negativo]

    results[columnas].to_csv(OUTPUT_PATH, index=False)
    print(f"Filas calculadas: {len(results)}")
    print(results.groupby(["version", "control_tipo"]).size().rename("filas").to_string())
    print(f"\nFilas con NaN, infinito o RSS negativo: {len(problematicas)}")
    if len(problematicas):
        print(problematicas.to_string())
    print(f"\nGuardado en {OUTPUT_PATH}")
    print("p_valor_orientativo: F asintótica, NO es el p-valor final (la semana 7 lo calibra con subrogados).")


if __name__ == "__main__":
    main()
