"""
make_graficos.py

Gráficos de apoyo de la semana 5, guardados en data/graficos/:

  1. componente_comun.png: 3 tópicos en la serie de volumen original y en
     la residualizada respecto de la componente común.
  2. barrido_K.png: T/parámetros y varianza de la componente 1 según K
     (data/barrido_K/resumen.csv, de sweep_topic_k.py).
  3. estacionariedad.png: clasificación ADF + KPSS de cada serie x tópico
     (data/stationarity_results.csv, de check_stationarity.py).

Clasificación de estacionariedad (nivel 0.05):
  - acuerdo: estacionaria      ADF rechaza (p < 0.05) y KPSS no rechaza (p >= 0.05)
  - acuerdo: no estacionaria   ADF no rechaza (p >= 0.05) y KPSS rechaza (p < 0.05)
  - ambiguo                    ninguna de las dos rechaza su nula
  - contradicción              las dos rechazan su nula
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from causal_coherence.config import OUTPUT_DIR  # noqa: E402

OUT_DIR = OUTPUT_DIR / "graficos"
ALPHA = 0.05
TOPICS_FIG1 = ["topic_0", "topic_4", "topic_8"]
VERSIONS = ["original", "diferenciada", "detendenciada", "residualizada"]
CLASSES = ["acuerdo: estacionaria", "acuerdo: no estacionaria", "ambiguo", "contradicción"]
CLASS_COLORS = ["#4a9a5c", "#c0504d", "#d9d9d9", "#e3a857"]
LINE_COLORS = ["#1f4e79", "#7f7f7f", "#b9772f"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def classify(row: pd.Series) -> str:
    adf_rejects = row["adf_pvalue"] < ALPHA
    kpss_rejects = row["kpss_pvalue"] < ALPHA
    if adf_rejects and not kpss_rejects:
        return CLASSES[0]
    if not adf_rejects and kpss_rejects:
        return CLASSES[1]
    if not adf_rejects and not kpss_rejects:
        return CLASSES[2]
    return CLASSES[3]


def stationarity_table() -> pd.DataFrame:
    r = pd.read_csv(OUTPUT_DIR / "stationarity_results.csv")
    r["clase"] = r.apply(classify, axis=1)
    return r


def fig_componente_comun() -> None:
    orig = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    resid = pd.read_csv(OUTPUT_DIR / "topic_series_volumen_residualizada.csv", index_col=0, parse_dates=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    for topic, color in zip(TOPICS_FIG1, LINE_COLORS):
        ax1.plot(orig.index, orig[topic], color=color, marker="o", ms=3, label=topic)
        ax2.plot(resid.index, resid[topic], color=color, marker="o", ms=3, label=topic)
    ax1.set_title("Serie original: los tópicos suben y bajan juntos")
    ax1.set_ylabel("log(1 + masa del tópico)")
    ax2.set_title("Sin la componente común: queda lo propio de cada tópico")
    ax2.set_ylabel("residuo")
    ax2.axhline(0, color="black", lw=0.6)
    ax1.legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Movimiento compartido entre tópicos, antes y después de quitar la componente común")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "componente_comun.png", dpi=150)
    plt.close(fig)


def fig_barrido_k() -> None:
    r = pd.read_csv(OUTPUT_DIR / "barrido_K" / "resumen.csv")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    l1, = ax1.plot(r["K"], r["T_sobre_params"], color=LINE_COLORS[0], marker="o", label="T / parámetros por ecuación")
    l2, = ax2.plot(r["K"], r["varianza_PC1"] * 100, color=LINE_COLORS[2], marker="s", label="Varianza de la componente 1 (%)")
    ax1.axhspan(0, 1.5, color="#f2f2f2", zorder=0)
    ax1.axhline(3, color="grey", ls=":", lw=1)
    ax1.axhline(1.5, color="grey", ls=":", lw=1)
    ax1.text(9.0, 2.94, "umbral OK (T/parámetros = 3)", va="top", fontsize=8, color="grey")
    ax1.text(4.6, 1.56, "umbral ajustado (1.5); abajo: insuficiente", va="bottom", fontsize=8, color="grey")
    ax1.set_xticks(r["K"])
    ax1.set_xlim(4.4, 15.6)
    ax1.set_xlabel("Cantidad de tópicos (K)")
    ax1.set_ylabel("T / parámetros", color=LINE_COLORS[0])
    ax2.set_ylabel("Varianza explicada (%)", color=LINE_COLORS[2])
    ax1.set_ylim(0, 4.5)
    ax2.set_ylim(0, 100)
    ax1.legend(handles=[l1, l2], frameon=False, loc="lower left")
    ax1.set_title("Barrido de K: menos datos por parámetro, componente común siempre dominante")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "barrido_K.png", dpi=150)
    plt.close(fig)


def fig_estacionariedad(r: pd.DataFrame) -> None:
    topics = sorted(r["topico"].unique(), key=lambda t: int(t.split("_")[1]))
    grid = pd.DataFrame(np.nan, index=VERSIONS, columns=topics)
    for _, row in r.iterrows():
        grid.loc[row["serie"], row["topico"]] = CLASSES.index(row["clase"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(grid.to_numpy()), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(topics)), topics, rotation=45, ha="right")
    ax.set_yticks(range(len(VERSIONS)), VERSIONS)
    ax.set_xticks(np.arange(-0.5, len(topics)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(VERSIONS)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(VERSIONS):
        for j, t in enumerate(topics):
            if np.isnan(grid.loc[v, t]):
                ax.text(j, i, "no aplica", ha="center", va="center", fontsize=7, color="grey")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASSES)],
              frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8)
    ax.set_title("Estacionariedad por versión y tópico (ADF + KPSS, nivel 0.05)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "estacionariedad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = stationarity_table()
    counts = pd.crosstab(pd.Categorical(r["serie"], VERSIONS), pd.Categorical(r["clase"], CLASSES), dropna=False)
    counts.index.name, counts.columns.name = "versión", "clase"
    print("Clasificación de estacionariedad por versión:")
    print(counts.to_string())
    print(f"\nTotal: {len(r)} filas")
    print("\ntopic_6 en cada versión:")
    print(r[r["topico"] == "topic_6"][["serie", "adf_pvalue", "kpss_pvalue", "clase"]].to_string(index=False))

    fig_componente_comun()
    fig_barrido_k()
    fig_estacionariedad(r)
    print(f"\nGráficos guardados en {OUT_DIR}:")
    for name in ("componente_comun.png", "barrido_K.png", "estacionariedad.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
