"""
confirmacion_y_real.py

Semana 8, paso 3 (Bitacora_Semana8.md, sección 23): confirmación de poder de
(2 días, K = 10, control propio) y (2 días, K = 8, control propio), regla de
respaldo y estimación sobre los datos reales de bpoil con la configuración que
resulte, UNA sola vez.

Confirmación: 300 réplicas por efecto (0.0, 0.4 y 0.8) por candidata, con el
simulador y la lógica de la sección 21 (scripts/candidatas_grillas.py) y las
claves "conf_"; los beta se reutilizan de data/candidatas/betas.csv. Regla de
respaldo: K = 10 si la detección con la regla de 2 versiones y efecto 0.8 es
>= 0.78; si no, K = 8. Criterio de falsos positivos: conteo de réplicas con
algún rechazo <= binom.isf(0.05 / 15, 300, 0.05) en original y detendenciada.

Estimación real (--real): con la configuración adoptada, controles propio y
compartido, remuestreo salvaje B = 9999 con la clave ("real2_remuestreo",
versión, origen, destino), BH a q = 0.05 por versión y control, regla de 2
versiones, comprobación de las dos rutas (sección 4 de Bitacora_Semana7.md),
efecto y F necesarios para rechazar, y comparación con la estimación semanal
(data/resultados_reales.csv). Solo corre si la confirmación está completa y no
hay una estimación previa.

Uso:
  python scripts/confirmacion_y_real.py --verificar   verificaciones a) a e)
  python scripts/confirmacion_y_real.py --medir 3     tiempo; no guarda nada
  python scripts/confirmacion_y_real.py               confirmación (reanuda)
  python scripts/confirmacion_y_real.py --resumen     tablas de la confirmación y decisión
  python scripts/confirmacion_y_real.py --real        estimación real, una sola vez
"""

import argparse
import itertools
import math
import os
import runpy
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.signal import detrend

from causal_coherence import candidatas, diagnostico, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

CONFIRMAR = ("2D_K10", "2D_K8")
EFECTOS = {"0.0": 300, "0.4": 300, "0.8": 300}
CONTROL = "C0"
B = 9999
Q = 0.05
CADA = 25
MIN_VERSIONES = 2
UMBRAL_RESPALDO = 0.78
UMBRAL_FP = stats.binom.isf(0.05 / 15, 300, 0.05)
SALIDA_CONF = OUTPUT_DIR / "confirmacion"
SALIDA_REAL = OUTPUT_DIR / "resultados_reales_2d"
TOL_C = 1e-8
PARES_FIJOS_T5 = [("topic_5", "topic_0"), ("topic_3", "topic_5")]
VERSIONES = list(surrogates.VERSIONES)

_cand = runpy.run_path(str(Path(__file__).resolve().parent / "candidatas_grillas.py"))
CANDIDATAS: dict = _cand["CANDIDATAS"]
DESCARTE: dict = _cand["DESCARTE"]
serie_real: Callable = _cand["serie_real"]
pares_por_version: Callable = _cand["pares_por_version"]
leer_csv: Callable = _cand["leer_csv"]
RUTA_BETAS: Path = _cand["RUTA_BETAS"]


# ---------------------------------------------------------------------------
# Réplicas (la lógica de candidatas_grillas.generar y procesar, con el prefijo de las claves como parámetro)
# ---------------------------------------------------------------------------

def generar(prefijo: str, cand: str, efecto: str, indice: int, betas: pd.DataFrame) -> tuple[pd.DataFrame, int, tuple[str, str] | None]:
    _, _, dias, _, K = CANDIDATAS[cand]
    real, _ = serie_real(cand)
    columnas = list(real.columns)
    c, a, residuos = surrogates.ajustar_ar1(real)
    par: tuple[str, str] | None = None
    i_A, i_B, beta = 0, 1, 0.0
    if float(efecto) > 0:
        plantables = list(itertools.permutations([col for col in columnas if col != DESCARTE[K]], 2))
        par = plantables[int(granger.generador(f"{prefijo}_par", cand, efecto, str(indice)).integers(0, len(plantables)))]
        fila = betas[(betas["candidata"] == cand) & (betas["efecto"] == efecto)
                     & (betas["origen"] == par[0]) & (betas["destino"] == par[1])]
        assert len(fila) == 1, f"{cand} {efecto} {par}: {len(fila)} betas"
        beta = float(fila["beta"].iloc[0])
        i_A, i_B = columnas.index(par[0]), columnas.index(par[1])
    X = candidatas.simular(c, a, residuos, real.to_numpy().mean(axis=0), len(real), candidatas.calentamiento(dias),
                           i_A, i_B, beta, planted.media_proceso(c, a, i_A),
                           granger.generador(f"{prefijo}_generacion", cand, efecto, str(indice)))
    X, corregidas = surrogates.aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas, par


def procesar(prefijo: str, cand: str, control: str, efecto: str, indice: int, betas: pd.DataFrame
             ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    K = CANDIDATAS[cand][4]
    real, n_t = serie_real(cand)
    pares = pares_por_version(list(real.columns), DESCARTE[K])
    x, corregidas, par = generar(prefijo, cand, efecto, indice, betas)
    versiones = candidatas.construir_versiones_k(x, DESCARTE[K])
    c2 = candidatas.versiones_control_fijo(np.log1p(n_t).rename("control_c2"))
    fila: dict[str, object] = {"candidata": cand, "control": control, "efecto": efecto, "indice": indice,
                               "origen": par[0] if par else "", "destino": par[1] if par else "",
                               "celdas_piso": corregidas}
    arreglos: dict[str, np.ndarray] = {}
    veces: dict[tuple[str, str], int] = {}
    n_versiones = 0
    for v, (serie, _, propio) in versiones.items():
        m = len(pares[v])
        val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_F", "p_rem")}
        disenos = [diagnostico.disenos(serie, propio[d] if control == "C0" else c2[v], o, d, control != "C1")
                   for o, d in pares[v]]
        no_estimable = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for X_f, _, _ in disenos))
        if not no_estimable:
            for j, ((o, d), (X_f, X_r, y)) in enumerate(zip(pares[v], disenos)):
                obs = diagnostico.prueba(X_f, X_r, y)
                val["efecto"][j], val["F"][j], val["p_F"][j] = obs["efecto"], obs["F"], obs["p_teo"]
                rng = granger.generador(f"{prefijo}_remuestreo", cand, control, efecto, str(indice), v, o, d)
                val["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"],
                                                             B, rng)["p"]
        r = np.zeros(m, dtype=bool) if no_estimable else granger.benjamini_hochberg(val["p_rem"], Q, m)
        for par_vd, rr in zip(pares[v], r):
            veces[par_vd] = veces.get(par_vd, 0) + int(rr)
        fila[f"no_estimable_{v}"] = no_estimable
        fila[f"rechazos_{v}"] = int(r.sum())
        if par is not None:
            j_pl = pares[v].index(par)
            fila[f"rechaza_plantado_{v}"] = int(r[j_pl])
            fila[f"efecto_plantado_{v}"] = float(val["efecto"][j_pl])
            n_versiones += int(r[j_pl])
        for k, arr in val.items():
            arreglos[f"{k}_{v}"] = arr
    fila["enlaces_regla"] = sum(1 for n in veces.values() if n >= MIN_VERSIONES)
    if par is not None:
        fila["n_versiones_plantado"] = n_versiones
        fila["detectado_regla"] = int(n_versiones >= MIN_VERSIONES)
    return fila, arreglos


# ---------------------------------------------------------------------------
# Confirmación
# ---------------------------------------------------------------------------

def rutas_conf(cand: str, e: str) -> tuple[Path, Path]:
    return SALIDA_CONF / f"{cand}_e{e}.csv", SALIDA_CONF / f"{cand}_e{e}.npz"


def correr() -> None:
    betas = leer_csv(RUTA_BETAS)
    t0 = time.perf_counter()
    for cand in CONFIRMAR:
        for e, n in EFECTOS.items():
            ruta_csv, ruta_npz = rutas_conf(cand, e)
            filas: list[dict[str, object]] = []
            arreglos: dict[str, list[np.ndarray]] = {}
            if ruta_csv.exists():
                tabla = leer_csv(ruta_csv)
                assert tabla["indice"].tolist() == list(range(len(tabla))), f"{ruta_csv.name}: índices no consecutivos"
                filas = tabla.to_dict("records")
                with np.load(ruta_npz) as z:
                    arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
            if len(filas) < n:
                print(f"{cand} efecto {e}: {len(filas)} guardadas, faltan {n - len(filas)}", flush=True)
            for i in range(len(filas), n):
                fila, arr = procesar("conf", cand, CONTROL, e, i, betas)
                filas.append(fila)
                for k, val in arr.items():
                    arreglos.setdefault(k, []).append(val)
                if (i + 1) % CADA == 0 or i + 1 == n:
                    SALIDA_CONF.mkdir(parents=True, exist_ok=True)
                    tmp = ruta_npz.with_name(ruta_npz.stem + ".tmp.npz")
                    np.savez(tmp, **{k: np.array(v) for k, v in arreglos.items()})
                    os.replace(tmp, ruta_npz)
                    tmp = ruta_csv.with_name(ruta_csv.stem + ".tmp.csv")
                    pd.DataFrame(filas).to_csv(tmp, index=False)
                    os.replace(tmp, ruta_csv)
            print(f"  {cand} efecto {e}: {n}/{n} ({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def medir(n: int) -> None:
    betas = leer_csv(RUTA_BETAS)
    total = 0.0
    for cand in CONFIRMAR:
        for e, reps in EFECTOS.items():
            tiempos = []
            for i in range(n):
                t0 = time.perf_counter()
                procesar("conf", cand, CONTROL, e, i, betas)
                tiempos.append(time.perf_counter() - t0)
            total += float(np.mean(tiempos)) * reps
            print(f"  {cand} efecto {e}: {np.mean(tiempos):.3f} s por réplica x {reps} = "
                  f"{np.mean(tiempos) * reps / 60:.1f} min", flush=True)
    print(f"Total de la confirmación ({len(CONFIRMAR) * sum(EFECTOS.values())} réplicas): {total / 60:.1f} min")


def decision() -> dict[str, object]:
    """Regla de respaldo y criterio de falsos positivos, solo con las réplicas de la confirmación."""
    det = {cand: float(leer_csv(rutas_conf(cand, "0.8")[0])["detectado_regla"].mean()) for cand in CONFIRMAR}
    adoptada = "2D_K10" if det["2D_K10"] >= UMBRAL_RESPALDO else "2D_K8"
    nulos = leer_csv(rutas_conf(adoptada, "0.0")[0])
    fp = {v: int((nulos[f"rechazos_{v}"] >= 1).sum()) for v in ("original", "detendenciada")}
    return {"deteccion_regla_0.8_2D_K10": det["2D_K10"], "deteccion_regla_0.8_2D_K8": det["2D_K8"],
            "umbral_respaldo": UMBRAL_RESPALDO, "adoptada": adoptada, "fp_original": fp["original"],
            "fp_detendenciada": fp["detendenciada"], "umbral_fp": UMBRAL_FP,
            "fp_cumple": all(c <= UMBRAL_FP for c in fp.values())}


def resumen() -> None:
    filas, fp = [], []
    for cand in CONFIRMAR:
        for e, n in EFECTOS.items():
            t = leer_csv(rutas_conf(cand, e)[0])
            assert len(t) == n, f"{cand} {e} incompleta"
            if e == "0.0":
                for v in VERSIONES:
                    fp.append({"candidata": cand, "version": v, "replicas_con_rechazo": int((t[f"rechazos_{v}"] >= 1).sum()),
                               "fraccion": float((t[f"rechazos_{v}"] >= 1).mean())})
                fp.append({"candidata": cand, "version": "regla_2_versiones",
                           "replicas_con_rechazo": int((t["enlaces_regla"] >= 1).sum()),
                           "fraccion": float((t["enlaces_regla"] >= 1).mean())})
                continue
            for v in VERSIONES:
                filas.append({"candidata": cand, "efecto": e, "version": v, "replicas": len(t),
                              "deteccion_BH": float(t[f"rechaza_plantado_{v}"].mean()),
                              "efecto_observado_medio": float(t[f"efecto_plantado_{v}"].mean()),
                              "n_no_estimables": int(t[f"no_estimable_{v}"].sum())})
            filas.append({"candidata": cand, "efecto": e, "version": "regla_2_versiones", "replicas": len(t),
                          "deteccion_BH": float(t["detectado_regla"].mean()), "efecto_observado_medio": np.nan,
                          "n_no_estimables": np.nan})
    combinado = []
    for cand in CONFIRMAR:
        for e in ("0.4", "0.8"):
            nuevo = leer_csv(rutas_conf(cand, e)[0])
            previo = leer_csv(_cand["rutas"](cand, CONTROL, e)[0])
            juntos = pd.concat([nuevo["detectado_regla"], previo["detectado_regla"]])
            combinado.append({"candidata": cand, "efecto": e, "replicas_confirmacion": len(nuevo),
                              "deteccion_confirmacion": float(nuevo["detectado_regla"].mean()),
                              "replicas_seccion_21": len(previo), "deteccion_seccion_21": float(previo["detectado_regla"].mean()),
                              "replicas_combinadas": len(juntos), "deteccion_combinada": float(juntos.mean())})
    tablas = {"deteccion": pd.DataFrame(filas), "falsos_positivos": pd.DataFrame(fp),
              "combinado_informativo": pd.DataFrame(combinado), "decision": pd.DataFrame([decision()])}
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for nombre, tabla in tablas.items():
            tabla.to_csv(SALIDA_CONF / f"resumen_{nombre}.csv", index=False)
            print(f"\n=== {nombre} (data/confirmacion/resumen_{nombre}.csv) ===")
            print(tabla.to_string(index=False))


# ---------------------------------------------------------------------------
# Estimación real
# ---------------------------------------------------------------------------

def versiones_reales(cand: str) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    real, _ = serie_real(cand)
    return candidatas.construir_versiones_k(real, DESCARTE[CANDIDATAS[cand][4]])


def requisito(m: int, df_den: int) -> dict[str, float]:
    F_req = float(stats.f.isf(Q / m, 1, df_den))
    efecto_req = math.log(1 + F_req / df_den)
    return {"q_sobre_m": Q / m, "F_req": F_req, "efecto_req": efecto_req, "reduccion_req": 1 - math.exp(-efecto_req)}


def enlaces(tabla: pd.DataFrame, control: str) -> pd.DataFrame:
    t = tabla[tabla["control_tipo"] == control]
    filas = []
    for (o, d), g in t.groupby(["origen", "destino"], sort=False):
        g = g.set_index("version")
        rech = [v for v in VERSIONES if v in g.index and g.loc[v, "rechaza_BH"] == 1]
        if len(rech) < MIN_VERSIONES:
            continue
        fila: dict[str, object] = {"origen": o, "destino": d, "versiones_rechazado": ";".join(rech),
                                   "n_versiones_rechazado": len(rech), "n_versiones_disponibles": len(g)}
        for v in VERSIONES:
            existe = v in g.index
            fila[f"efecto_{v}"] = g.loc[v, "efecto"] if existe else np.nan
            fila[f"p_remuestreo_{v}"] = g.loc[v, "p_remuestreo"] if existe else np.nan
            fila[f"rechaza_{v}"] = int(g.loc[v, "rechaza_BH"]) if existe else pd.NA
        filas.append(fila)
    columnas = ["origen", "destino", "versiones_rechazado", "n_versiones_rechazado", "n_versiones_disponibles"] + \
        [f"{k}_{v}" for v in VERSIONES for k in ("efecto", "p_remuestreo", "rechaza")]
    return pd.DataFrame(filas, columns=columnas)


def real() -> None:
    ruta_resultados = SALIDA_REAL / "resultados.csv"
    if ruta_resultados.exists():
        print(f"Ya existe {ruta_resultados}: la estimación real se ejecuta una sola vez. No se repite.")
        return
    for cand in CONFIRMAR:
        for e, n in EFECTOS.items():
            ruta = rutas_conf(cand, e)[0]
            assert ruta.exists() and len(leer_csv(ruta)) == n, f"confirmación incompleta: {ruta.name}"
    dec = decision()
    print("Decisión de la regla de respaldo:", dec)
    if not dec["fp_cumple"]:
        print("El criterio de falsos positivos no se cumple en la configuración adoptada: no se estima.")
        return
    cand = str(dec["adoptada"])
    real_s, _ = serie_real(cand)
    pares = pares_por_version(list(real_s.columns), DESCARTE[CANDIDATAS[cand][4]])
    versiones = versiones_reales(cand)
    filas, arreglos = [], {}
    for control in ("propio", "compartido"):
        for v, (serie, compartido, propio) in versiones.items():
            m = len(pares[v])
            val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_teo", "p_rem")}
            for j, (o, d) in enumerate(pares[v]):
                ctrl = propio[d] if control == "propio" else compartido
                obs = granger.estadistico_f(serie, ctrl, o, d)
                X_f, y = granger.armar_diseno(serie, ctrl, d, excluir=None)
                X_r, _ = granger.armar_diseno(serie, ctrl, d, excluir=o)
                assert np.linalg.matrix_rank(X_f) == X_f.shape[1], f"{control} {v} {o}->{d}: diseño sin rango completo"
                rem = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B,
                                                 granger.generador("real2_remuestreo", v, o, d))
                val["efecto"][j] = float(np.log(obs["rss_r"] / obs["rss_f"]))
                val["F"][j], val["p_rem"][j] = obs["F"], rem["p"]
                val["p_teo"][j] = granger.p_teorico(obs["F"], obs["df_num"], obs["df_den"])
                filas.append({"configuracion": cand, "control_tipo": control, "version": v, "origen": o, "destino": d,
                              "n_obs": obs["n_obs"], "efecto": val["efecto"][j], "F": obs["F"], "df_num": obs["df_num"],
                              "df_den": obs["df_den"], "p_teorico": val["p_teo"][j], "p_remuestreo": rem["p"], "B": B})
            for k, arr in val.items():
                arreglos[f"{control}_{k}_{v}"] = arr
    tabla = pd.DataFrame(filas)
    tabla["rechaza_BH"] = 0
    for (control, v), g in tabla.groupby(["control_tipo", "version"], sort=False):
        tabla.loc[g.index, "rechaza_BH"] = granger.benjamini_hochberg(g["p_remuestreo"].to_numpy(), Q, len(g)).astype(int)
    SALIDA_REAL.mkdir(parents=True, exist_ok=True)
    np.savez(SALIDA_REAL / "p_todos_los_pares.npz", **arreglos,
             **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in VERSIONES})
    tabla.to_csv(ruta_resultados, index=False)

    por_version, rutas_r, req = [], [], []
    for (control, v), g in tabla.groupby(["control_tipo", "version"], sort=False):
        por_version.append({"control_tipo": control, "version": v, "m": len(g), "rechazos_BH": int(g["rechaza_BH"].sum()),
                            "p_remuestreo_min": float(g["p_remuestreo"].min()), "efecto_max": float(g["efecto"].max()),
                            "F_max": float(g["F"].max()), "df_den": int(g["df_den"].iloc[0]),
                            **requisito(len(g), int(g["df_den"].iloc[0]))})
        r = g["p_remuestreo"] / np.maximum(g["p_teorico"], 1 / (B + 1))
        frac, med = float(((r >= 0.1) & (r <= 10)).mean()), float(r.median())
        rutas_r.append({"control_tipo": control, "version": v, "n_pares": len(g), "fraccion_r_en_0.1_10": frac,
                        "mediana_r": med, "cumple_orden_magnitud": frac >= 0.95,
                        "discrepancia_sistematica": not (1 / 3 <= med <= 3), "pasa": frac >= 0.95 and 1 / 3 <= med <= 3})
    salidas = {"por_version": pd.DataFrame(por_version), "rutas": pd.DataFrame(rutas_r),
               "enlaces_propio": enlaces(tabla, "propio"), "enlaces_compartido": enlaces(tabla, "compartido")}

    # Comparación con la estimación semanal (mismos pares).
    semanal = pd.read_csv(OUTPUT_DIR / "resultados_reales.csv", float_precision="round_trip")
    claves = ["control_tipo", "version", "origen", "destino"]
    comp = semanal[claves + ["efecto", "F", "p_remuestreo", "rechaza_BH"]].merge(
        tabla[claves + ["efecto", "F", "p_remuestreo", "rechaza_BH"]], on=claves, suffixes=("_semanal", "_2d"), how="inner")
    salidas["comparacion_pares"] = comp
    resumen_comp, top = [], []
    enl = {"semanal": {}, "2d": {}}
    for control in ("propio", "compartido"):
        for etiqueta, t in (("semanal", semanal), ("2d", tabla)):
            tt = t[t["control_tipo"] == control]
            veces = tt.groupby(["origen", "destino"])["rechaza_BH"].sum()
            enl[etiqueta][control] = int((veces >= MIN_VERSIONES).sum())
    for (control, v), g in comp.groupby(["control_tipo", "version"], sort=False):
        top_2d = set(g.nsmallest(10, "p_remuestreo_2d")[["origen", "destino"]].itertuples(index=False, name=None))
        top_sem = set(g.nsmallest(10, "p_remuestreo_semanal")[["origen", "destino"]].itertuples(index=False, name=None))
        resumen_comp.append({"control_tipo": control, "version": v, "n_pares": len(g),
                             "spearman_efecto": float(stats.spearmanr(g["efecto_semanal"], g["efecto_2d"]).statistic),
                             "spearman_p": float(stats.spearmanr(g["p_remuestreo_semanal"], g["p_remuestreo_2d"]).statistic),
                             "top10_en_comun": len(top_2d & top_sem),
                             "enlaces_regla_semanal": enl["semanal"][control], "enlaces_regla_2d": enl["2d"][control]})
        if control == "propio":
            for fila in g.nsmallest(10, "p_remuestreo_2d").itertuples(index=False):
                top.append({"version": v, "origen": fila.origen, "destino": fila.destino,
                            "p_remuestreo_2d": fila.p_remuestreo_2d, "efecto_2d": fila.efecto_2d,
                            "p_remuestreo_semanal": fila.p_remuestreo_semanal, "efecto_semanal": fila.efecto_semanal})
    salidas["comparacion_resumen"] = pd.DataFrame(resumen_comp)
    salidas["comparacion_top10_propio"] = pd.DataFrame(top)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(f"\nConfiguración estimada: {cand}")
        for nombre, t in salidas.items():
            t.to_csv(SALIDA_REAL / f"{nombre}.csv", index=False)
            if nombre == "comparacion_pares":
                print(f"\n=== {nombre} ({len(t)} filas, data/resultados_reales_2d/{nombre}.csv) ===")
                continue
            print(f"\n=== {nombre} (data/resultados_reales_2d/{nombre}.csv) ===")
            print(t.to_string(index=False) if len(t) else "  sin filas: ningún enlace cumple la regla")


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------

def control_propio_independiente(masas_log: pd.DataFrame, n_t: pd.Series, version: str) -> pd.DataFrame:
    """log1p(n_t - masa_j) con pandas y scipy, transformado según la versión."""
    masa = np.expm1(masas_log)
    propio = np.log1p(masa.rsub(n_t.astype(float), axis=0))
    if version == "diferenciada":
        return propio.diff().iloc[1:]
    if version == "detendenciada":
        return pd.DataFrame(detrend(propio.to_numpy(), axis=0, type="linear"), index=propio.index, columns=propio.columns)
    return propio


def f_statsmodels(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> tuple[float, float]:
    rezagos = serie.shift(1).add_suffix("_rezago")
    datos = pd.concat([serie[destino].rename("y"), rezagos, control.rename("control")], axis=1).dropna()
    X_f = sm.add_constant(datos.drop(columns="y"))
    completo = sm.OLS(datos["y"], X_f).fit()
    restringido = sm.OLS(datos["y"], X_f.drop(columns=f"{origen}_rezago")).fit()
    F, _, df_num = completo.compare_f_test(restringido)
    return float(F), float(stats.f.sf(F, df_num, completo.df_resid))


def claves_previas_y_candidatas() -> list[tuple[str | int, ...]]:
    columnas10 = list(serie_real("2D_K10")[0].columns)
    ids = _cand["claves_previas"](columnas10)
    betas = leer_csv(RUTA_BETAS)
    ids += [("cand_calibracion", f.candidata, f.efecto, f.origen, f.destino) for f in betas.itertuples(index=False)]
    for cand, control, e in _cand["unidades"]():
        K = CANDIDATAS[cand][4]
        pares = pares_por_version(list(serie_real(cand)[0].columns), DESCARTE[K])
        for i in range(_cand["EFECTOS"][e]):
            if control == "C0":
                ids.append(("cand_generacion", cand, e, str(i)))
                if float(e) > 0:
                    ids.append(("cand_par", cand, e, str(i)))
            for v, lista in pares.items():
                ids += [("cand_remuestreo", cand, control, e, str(i), v, o, d) for o, d in lista]
    return ids


def verificar() -> bool:
    res: dict[str, bool] = {}
    betas = leer_csv(RUTA_BETAS)

    print("=== a) Serie (2 días, K = 10) ===")
    real, n_t = serie_real("2D_K10")
    inicios = real.index
    consecutivas = bool((pd.Series(inicios).diff().dropna() == pd.Timedelta(days=2)).all())
    dif = float(np.abs(np.expm1(real).sum(axis=1).to_numpy() - n_t.to_numpy()).max())
    sin_ceros = bool((np.expm1(real).sum(axis=1) > 0).all()) and bool((real.abs().sum(axis=1) > 0).all())
    fin = inicios[-1] + pd.Timedelta(days=1)
    res["a"] = (len(real) == 53 and inicios[0] == pd.Timestamp("2010-04-26") and fin == pd.Timestamp("2010-08-09")
                and dif <= 1e-9 and bool((n_t >= 5).all()) and consecutivas and sin_ceros)
    print(f"  {len(real)} ventanas, {inicios[0].date()} a {fin.date()}; max |suma de masas - n_t| = {dif:.3e}; "
          f"n_t >= 5 en todas: {bool((n_t >= 5).all())} (mínimo {int(n_t.min())}); consecutivas cada 2 días: "
          f"{consecutivas}; sin filas en cero: {sin_ceros}: {'OK' if res['a'] else 'FALLA'}")

    print("\n=== Beta usados (filas de data/candidatas/betas.csv) ===")
    ok_b = True
    for cand in CONFIRMAR:
        K = CANDIDATAS[cand][4]
        cols = list(serie_real(cand)[0].columns)
        for e in ("0.4", "0.8"):
            for o, d in itertools.permutations([c for c in cols if c != DESCARTE[K]], 2):
                n = len(betas[(betas["candidata"] == cand) & (betas["efecto"] == e) & (betas["origen"] == o)
                              & (betas["destino"] == d)])
                ok_b &= n == 1
    print(f"  una fila exacta por candidata, efecto y par plantable (2D_K10: 72 x 2, 2D_K8: 42 x 2): {'OK' if ok_b else 'FALLA'}")
    res["betas"] = ok_b

    print("\n=== b) Reproducción de la sección 21 (2D_K10, C0, efecto 0.8, réplicas 0 a 4) ===")
    guardado = leer_csv(_cand["rutas"]("2D_K10", "C0", "0.8")[0]).set_index("indice")
    ok = True
    for i in range(5):
        fila, _ = procesar("cand", "2D_K10", "C0", "0.8", i, betas)
        cols = [f"rechaza_plantado_{v}" for v in VERSIONES] + ["detectado_regla", "origen", "destino"]
        ok &= all(fila[c] == guardado.loc[i, c] for c in cols)
    res["b"] = ok
    print(f"  detección por versión, regla y par idénticos en las 5 réplicas: {'OK' if ok else 'FALLA'}")

    print("\n=== c) F y p_teorico con statsmodels (control propio) ===")
    versiones = versiones_reales("2D_K10")
    pares = pares_por_version(list(real.columns), DESCARTE[10])
    rng = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(zlib.crc32(b"real2_c"),)))
    errores = []
    for v in VERSIONES:
        serie, _, propio = versiones[v]
        Fs = {(o, d): granger.estadistico_f(serie, propio[d], o, d)["F"] for o, d in pares[v]}
        top = sorted(Fs, key=Fs.get, reverse=True)[:5]
        fijos = [p for p in PARES_FIJOS_T5 if p in Fs]
        resto = [p for p in pares[v] if p not in top and p not in fijos]
        sorteo = [resto[k] for k in np.sort(rng.choice(len(resto), size=5, replace=False))]
        ctrl_ind = control_propio_independiente(real, n_t, v).loc[serie.index]
        for o, d in top + sorteo + fijos:
            obs = granger.estadistico_f(serie, propio[d], o, d)
            p_codigo = granger.p_teorico(obs["F"], obs["df_num"], obs["df_den"])
            F_sm, p_sm = f_statsmodels(serie, ctrl_ind[d], o, d)
            errores.append({"version": v, "par": f"{o}->{d}", "grupo": "mayor F" if (o, d) in top else
                            ("fijo topic_5" if (o, d) in fijos else "sorteado"),
                            "error_rel_F": abs(F_sm - obs["F"]) / abs(obs["F"]), "error_rel_p": abs(p_sm - p_codigo) / abs(p_codigo)})
    tabla_c = pd.DataFrame(errores)
    error_max = float(tabla_c[["error_rel_F", "error_rel_p"]].to_numpy().max())
    res["c"] = error_max <= TOL_C
    print(f"  {len(tabla_c)} pares ({', '.join(f'{v}: {(tabla_c.version == v).sum()}' for v in VERSIONES)}); "
          f"error relativo máximo {error_max:.3e} <= {TOL_C}: {'OK' if res['c'] else 'FALLA'}")

    print("\n=== d) Misma réplica dos veces (conf, 2D_K10, efecto 0.8, réplica 0) ===")
    (f1, a1), (f2, a2) = (procesar("conf", "2D_K10", CONTROL, "0.8", 0, betas) for _ in range(2))
    res["d"] = f1 == f2 and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
    print(f"  {'OK' if res['d'] else 'FALLA'}")

    print("\n=== e) Unicidad de las claves de semilla ===")
    previas = claves_previas_y_candidatas()
    nuevas: list[tuple[str | int, ...]] = []
    for cand in CONFIRMAR:
        K = CANDIDATAS[cand][4]
        pares_c = pares_por_version(list(serie_real(cand)[0].columns), DESCARTE[K])
        for e, n in EFECTOS.items():
            for i in range(n):
                nuevas.append(("conf_generacion", cand, e, str(i)))
                if float(e) > 0:
                    nuevas.append(("conf_par", cand, e, str(i)))
                for v, lista in pares_c.items():
                    nuevas += [("conf_remuestreo", cand, CONTROL, e, str(i), v, o, d) for o, d in lista]
    reales = {(v, o, d) for cand in CONFIRMAR for v, lista in
              pares_por_version(list(serie_real(cand)[0].columns), DESCARTE[CANDIDATAS[cand][4]]).items() for o, d in lista}
    nuevas += [("real2_remuestreo", v, o, d) for v, o, d in sorted(reales)]
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    res["e"] = distintas == len(todas)
    print(f"  {len(todas)} ({len(previas)} previas + {len(nuevas)} nuevas), distintas: {distintas}: {'OK' if res['e'] else 'FALLA'}")

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true")
    grupo.add_argument("--medir", type=int, metavar="N")
    grupo.add_argument("--resumen", action="store_true")
    grupo.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.verificar:
        verificar()
    elif args.medir:
        medir(args.medir)
    elif args.resumen:
        resumen()
    elif args.real:
        real()
    else:
        correr()


if __name__ == "__main__":
    main()
