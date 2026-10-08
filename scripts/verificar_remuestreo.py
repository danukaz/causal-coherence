"""
verificar_remuestreo.py

Semana 7: verifica el remuestreo salvaje de causal_coherence.granger sobre
dos pares de la serie original, con el control propio del tópico destino
(el mismo de run_granger_sweep.py). Para cada par imprime F_obs, cuántas
réplicas lo igualan o superan, el p por remuestreo (B=9999), el p teórico
(F asintótica del barrido) y la razón entre ambos.

Verificaciones:
  a) matriz de diseño idéntica a la de run_granger_sweep.py
  b) F_obs idéntico al F de data/granger_results.csv
  c) RSS con el proyector M igual al de lstsq (tolerancia 1e-10)
  d) con todos los eta = +1, F*_b igual a F_obs (tolerancia 1e-10)
  e) dos corridas con la semilla 42 dan exactamente el mismo p
  f) tiempo de las 9999 réplicas por par

Un solo generador numpy.random.default_rng(42) se crea en main y se pasa
explícitamente; los pares lo consumen en el orden de PARES.

No escribe archivos.
"""

import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence import granger
from causal_coherence.config import OUTPUT_DIR

PARES = [("topic_3", "topic_8"), ("topic_7", "topic_4")]
VERSION = "original"
B = 9999
SEMILLA = 42
TOLERANCIA = 1e-10

# Mismo control propio, lectura y diseño que el barrido de Granger.
_sweep = runpy.run_path(str(Path(__file__).resolve().parent / "run_granger_sweep.py"))
leer = _sweep["leer"]
controles_propios = _sweep["controles_propios"]
armar_diseno_barrido = _sweep["armar_diseno"]
archivo_serie = _sweep["VERSIONES"][VERSION][0]


def marca(ok: bool) -> str:
    return "OK" if ok else "FALLA"


def correr_pares(series: pd.DataFrame, propio: pd.DataFrame,
                 rng: np.random.Generator) -> list[dict[str, object]]:
    """Remuestreo salvaje de todos los pares, consumiendo rng en orden."""
    salidas = []
    for origen, destino in PARES:
        control = propio[destino]
        obs = granger.estadistico_f(series, control, origen, destino)
        X_f, y = granger.armar_diseno(series, control, destino, excluir=None)
        X_r, _ = granger.armar_diseno(series, control, destino, excluir=origen)
        t0 = time.perf_counter()
        rem = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B, rng)
        segundos = time.perf_counter() - t0
        salidas.append({"origen": origen, "destino": destino, **obs, **rem, "segundos": segundos})
    return salidas


def main() -> None:
    series = leer(archivo_serie)
    propio = controles_propios()[VERSION].loc[series.index]
    assert propio.index.equals(series.index), "serie y control propio no calzan"
    csv = pd.read_csv(OUTPUT_DIR / "granger_results.csv", float_precision="round_trip")

    print(f"Versión {VERSION}, control propio, B={B}, semilla {SEMILLA}, tolerancia {TOLERANCIA}\n")
    print("=== Verificaciones a) a d), por par ===")
    for origen, destino in PARES:
        control = propio[destino]
        print(f"\n{origen} -> {destino}")

        # a) matriz de diseño
        for nombre, excluir in (("completo", None), ("restringido", origen)):
            X_n, y_n = granger.armar_diseno(series, control, destino, excluir)
            X_b, y_b = armar_diseno_barrido(series, control, destino, excluir)
            ok = np.array_equal(X_n, X_b) and np.array_equal(y_n, y_b)
            print(f"  a) diseño {nombre:<11} X {X_n.shape}, np.array_equal con el barrido: {marca(ok)}")

        # b) F_obs contra el CSV
        obs = granger.estadistico_f(series, control, origen, destino)
        fila = csv[(csv["version"] == VERSION) & (csv["control_tipo"] == "propio")
                   & (csv["origen"] == origen) & (csv["destino"] == destino)]
        assert len(fila) == 1, f"{origen} -> {destino}: {len(fila)} filas en el CSV"
        F_csv = fila["F"].iloc[0]
        print(f"  b) F_obs = {obs['F']!r}, F del CSV = {F_csv!r}, idénticos: {marca(obs['F'] == F_csv)}")
        print(f"     df_num = {obs['df_num']} (CSV {fila['df_num'].iloc[0]}), "
              f"df_den = {obs['df_den']} (CSV {fila['df_den'].iloc[0]})")

        # c) RSS con el proyector contra lstsq
        X_f, y = granger.armar_diseno(series, control, destino, excluir=None)
        X_r, _ = granger.armar_diseno(series, control, destino, excluir=origen)
        M_f = granger.proyector_residual(X_f)
        M_r = granger.proyector_residual(X_r)
        for nombre, M, rss_lstsq in (("completo", M_f, obs["rss_f"]), ("restringido", M_r, obs["rss_r"])):
            rss_M = float(np.sum((M @ y) ** 2))
            dif = abs(rss_M - rss_lstsq)
            print(f"  c) RSS {nombre:<11} proyector = {rss_M!r}, lstsq = {rss_lstsq!r}, "
                  f"|dif| = {dif:.3e}: {marca(dif <= TOLERANCIA)}")

        # d) identidad: eta = +1 reproduce y
        eps_r = M_r @ y
        y_hat_r = y - eps_r
        unos = np.ones((1, len(y)))
        F_id = granger.f_replicas(M_f, M_r, y_hat_r, eps_r, unos, obs["df_num"], obs["df_den"])[0]
        dif = abs(F_id - obs["F"])
        print(f"  d) eta = +1: F* = {F_id!r}, F_obs = {obs['F']!r}, "
              f"|dif| = {dif:.3e}: {marca(dif <= TOLERANCIA)}")

    # e) reproducibilidad: dos corridas completas, cada una con su default_rng(42)
    corrida_1 = correr_pares(series, propio, np.random.default_rng(SEMILLA))
    corrida_2 = correr_pares(series, propio, np.random.default_rng(SEMILLA))
    print("\n=== e) Reproducibilidad (dos corridas con default_rng(42)) ===")
    for r1, r2 in zip(corrida_1, corrida_2):
        mismo_p = r1["p"] == r2["p"]
        mismos_F = np.array_equal(r1["F_star"], r2["F_star"])
        print(f"  {r1['origen']} -> {r1['destino']}: p1 = {r1['p']!r}, p2 = {r2['p']!r}, "
              f"mismo p: {marca(mismo_p)}; mismos F*: {marca(mismos_F)}")

    # f) tiempo
    print(f"\n=== f) Tiempo de las {B} réplicas (incluye QR de M_f y M_r) ===")
    for i, corrida in enumerate((corrida_1, corrida_2), start=1):
        for r in corrida:
            print(f"  corrida {i}, {r['origen']} -> {r['destino']}: {r['segundos']:.4f} s")

    print(f"\n=== Resultados (corrida 1) ===")
    tabla = pd.DataFrame([{
        "par": f"{r['origen']} -> {r['destino']}",
        "F_obs": r["F"],
        "df_num": r["df_num"],
        "df_den": r["df_den"],
        "replicas_F*>=F_obs": r["n_superan"],
        "p_remuestreo": r["p"],
        "p_teorico": granger.p_teorico(r["F"], r["df_num"], r["df_den"]),
    } for r in corrida_1])
    tabla["razon_rem_sobre_teo"] = tabla["p_remuestreo"] / tabla["p_teorico"]
    with pd.option_context("display.float_format", "{:.6g}".format, "display.width", 200):
        print(tabla.to_string(index=False))

    for r in corrida_1:
        fila = csv[(csv["version"] == VERSION) & (csv["control_tipo"] == "propio")
                   & (csv["origen"] == r["origen"]) & (csv["destino"] == r["destino"])]
        p_csv = fila["p_valor_orientativo"].iloc[0]
        p_teo = granger.p_teorico(r["F"], r["df_num"], r["df_den"])
        print(f"  p teórico {r['origen']} -> {r['destino']} idéntico al p_valor_orientativo del CSV: "
              f"{marca(p_teo == p_csv)}")


if __name__ == "__main__":
    main()
