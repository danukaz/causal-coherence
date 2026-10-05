"""
comparar_controles.py

Compara los resultados del barrido de Granger entre las dos formas del
control de volumen (compartido y propio): imprime los pares mas
sensibles al control, los que entran o salen del top 10 de cada
version, y guarda un grafico de dispersion en data/graficos/.

Ademas, guarda en data/comparacion_controles.csv:
  - por version: correlacion de rangos de Spearman entre el efecto con
    control compartido y con control propio (sobre todos los pares), y
    cantidad de pares en comun entre los dos top 10.
  - por topico, serie original: media y desviacion estandar (ddof=1) de la
    brecha log(1+n_t) - log(1+n_t - masa_j), es decir, control compartido
    menos control propio. El control propio es el de controles_propios()
    de run_granger_sweep.py.
  - serie original: correlacion de Pearson de cada topico y de la primera
    componente principal con control_volumen. La componente se calcula
    como en build_series_variants.py (PCA sobre las 10 series
    estandarizadas); su signo es el que entrega sklearn, asi que el signo
    de esa correlacion es arbitrario.

La parte del barrido solo lee data/granger_results.csv; la brecha y las
correlaciones leen las series de data/. No recalcula Granger.
"""

import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from causal_coherence.config import OUTPUT_DIR

TOPICO_SENSIBLE = "topic_6"
VERSIONES = ["original", "diferenciada", "detendenciada", "residualizada"]
TOP_N = 10
CSV_SALIDA = OUTPUT_DIR / "comparacion_controles.csv"

# Mismo control propio y misma lectura que el barrido de Granger.
_sweep = runpy.run_path(str(Path(__file__).resolve().parent / "run_granger_sweep.py"))
controles_propios = _sweep["controles_propios"]
leer = _sweep["leer"]


def pares_por_control(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Una fila por par, con el efecto bajo cada control y su diferencia."""
    g = df[df["version"] == version]
    w = g.pivot_table(
        index=["origen", "destino"], columns="control_tipo", values="efecto"
    )
    w["diferencia"] = (w["compartido"] - w["propio"]).abs()
    return w


def imprimir_resumen(w: pd.DataFrame, version: str) -> None:
    print(f"\n=== {version} ===")
    print("Los 5 pares mas sensibles al control:")
    print(w.nlargest(5, "diferencia").round(4).to_string())

    top_c = set(w["compartido"].nlargest(10).index)
    top_p = set(w["propio"].nlargest(10).index)
    print(f"Solo en el top 10 del compartido: {sorted(top_c - top_p)}")
    print(f"Solo en el top 10 del propio:     {sorted(top_p - top_c)}")


def graficar(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, version in zip(axes.ravel(), VERSIONES):
        w = pares_por_control(df, version)
        con_t6 = [TOPICO_SENSIBLE in par for par in w.index]
        ax.scatter(
            w.loc[[not x for x in con_t6], "compartido"],
            w.loc[[not x for x in con_t6], "propio"],
            s=18, color="#4F81BD", label="resto de los pares",
        )
        ax.scatter(
            w.loc[con_t6, "compartido"], w.loc[con_t6, "propio"],
            s=22, color="#C0392B", label=f"pares con {TOPICO_SENSIBLE}",
        )
        tope = max(w["compartido"].max(), w["propio"].max()) * 1.05
        ax.plot([0, tope], [0, tope], color="#666666", lw=1, ls="--")
        ax.set_title(version)
        ax.set_xlabel("efecto, control compartido")
        ax.set_ylabel("efecto, control propio")
        ax.legend(fontsize=8)
    fig.suptitle("Efecto de Granger bajo los dos controles de volumen\n"
                 "(sobre la diagonal punteada, ambos controles coinciden)")
    fig.tight_layout()
    salida = OUTPUT_DIR / "graficos" / "comparacion_controles.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"\nGrafico guardado en {salida}")


def fila(seccion: str, version: str, item: str, estadistico: str, valor: float) -> dict[str, object]:
    return {"seccion": seccion, "version": version, "item": item,
            "estadistico": estadistico, "valor": valor}


def concordancia_rangos(df: pd.DataFrame) -> list[dict[str, object]]:
    """Spearman entre los dos efectos y pares en comun del top 10, por version."""
    rows = []
    for version in VERSIONES:
        w = pares_por_control(df, version)
        rho = stats.spearmanr(w["compartido"], w["propio"]).statistic
        top_c = set(w["compartido"].nlargest(TOP_N).index)
        top_p = set(w["propio"].nlargest(TOP_N).index)
        rows.append(fila("rangos", version, "compartido vs propio", "n_pares", len(w)))
        rows.append(fila("rangos", version, "compartido vs propio", "spearman_rho", rho))
        rows.append(fila("rangos", version, "compartido vs propio", f"pares_en_comun_top{TOP_N}",
                         len(top_c & top_p)))
    return rows


def brecha_controles() -> list[dict[str, object]]:
    """log(1+n_t) - log(1+n_t - masa_j) por topico, serie original."""
    compartido = leer("control_volumen.csv").iloc[:, 0]
    propio = controles_propios()["original"]
    brecha = propio.rsub(compartido, axis=0)  # compartido - propio_j, columna a columna
    rows = []
    for topico in brecha.columns:
        rows.append(fila("brecha", "original", topico, "media", brecha[topico].mean()))
        rows.append(fila("brecha", "original", topico, "desv_est", brecha[topico].std()))
    return rows


def correlacion_con_control() -> list[dict[str, object]]:
    """Pearson de cada topico y de la componente 1 con control_volumen, serie original."""
    series = leer("topic_series_volumen.csv")
    control = leer("control_volumen.csv").iloc[:, 0]
    assert series.index.equals(control.index), "serie y control no calzan"
    X = StandardScaler().fit_transform(series.to_numpy())
    pc1 = pd.Series(PCA().fit_transform(X)[:, 0], index=series.index)
    rows = []
    for topico in series.columns:
        rows.append(fila("correlacion_control", "original", topico, "pearson_r",
                         series[topico].corr(control)))
    rows.append(fila("correlacion_control", "original", "componente_1", "pearson_r",
                     pc1.corr(control)))
    return rows


def imprimir_tabla(tabla: pd.DataFrame, seccion: str, titulo: str) -> None:
    print(f"\n=== {titulo} ===")
    t = tabla[tabla["seccion"] == seccion]
    ancha = t.pivot(index=["version", "item"], columns="estadistico", values="valor")
    ancha = ancha.reindex(index=t.set_index(["version", "item"]).index.unique(),
                          columns=t["estadistico"].unique())
    print(ancha.round(4).to_string())


def main() -> None:
    df = pd.read_csv(OUTPUT_DIR / "granger_results.csv")
    for version in VERSIONES:
        imprimir_resumen(pares_por_control(df, version), version)
    graficar(df)

    tabla = pd.DataFrame(concordancia_rangos(df) + brecha_controles() + correlacion_con_control())
    imprimir_tabla(tabla, "rangos",
                   f"Spearman compartido vs. propio y pares en comun del top {TOP_N}")
    imprimir_tabla(tabla, "brecha",
                   "Brecha log(1+n_t) - log(1+n_t - masa_j), serie original (desv_est con ddof=1)")
    imprimir_tabla(tabla, "correlacion_control",
                   "Correlacion de Pearson con control_volumen, serie original")
    tabla.to_csv(CSV_SALIDA, index=False)
    print(f"\nGuardado en {CSV_SALIDA}")


if __name__ == "__main__":
    main()