"""
paquete_validez.py

Semana 8, paso 4 (Bitacora_Semana8.md, sección 26): paquete de validez de la
serie de 2 días (K = 10) sobre bpoil.

  A: R1 sobre la serie: ADF y KPSS (lógica de check_stationarity.py), PCA
     estandarizado, documentos por ventana y Spearman de |residuo AR(1)| con
     n_t y 1/n_t.
  B: calibración con un sustituto de volumen exógeno: n_t reales, AR(1)
     diagonal del clr de las proporciones, softmax; 300 sustitutos, remuestreo
     B = 9999, BH por versión, regla de 2 versiones, controles propio y
     compartido. Pasa si el conteo en original y detendenciada es
     <= binom.isf(0.05 / 4, 300, 0.05) para cada control.
  C: sensibilidades sobre datos reales (no declaran enlaces): retardo 2;
     K = 8 y 5; 3 días con K = 10 y 5; sin las 5 primeras o las 5 últimas
     ventanas; MIN_DOCS = 8.
  D: AIC y BIC por ecuación con p = 1 y p = 2 sobre las mismas filas.

Salidas en data/validez/.

Uso:
  python scripts/paquete_validez.py --verificar
  python scripts/paquete_validez.py --medir 3          tiempo del bloque B; no guarda nada
  python scripts/paquete_validez.py                    bloques A, D, C y B, en ese orden (B reanuda)
  python scripts/paquete_validez.py --bloque A|B|C|D   un bloque
  python scripts/paquete_validez.py --resumen          tablas desde lo guardado
"""

import argparse
import itertools
import math
import os
import runpy
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from causal_coherence import candidatas, granger, surrogates, validez
from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import T17_DIR

SCRIPTS = Path(__file__).resolve().parent
SALIDA = OUTPUT_DIR / "validez"
VERSIONES = list(surrogates.VERSIONES)
CONTROLES = ("propio", "compartido")
B = 9999
Q = 0.05
CADA = 25
MIN_VERSIONES = 2
N_SUSTITUTOS = 300
CALENT_B = math.ceil(700 / 2)
UMBRAL_B = stats.binom.isf(0.05 / 4, N_SUSTITUTOS, 0.05)
TOP = 5
TOP_TRASLAPE = 10
LARGO_MINIMO_C5 = 30

_cand = runpy.run_path(str(SCRIPTS / "candidatas_grillas.py"))
_conf = runpy.run_path(str(SCRIPTS / "confirmacion_y_real.py"))
_estac = runpy.run_path(str(SCRIPTS / "check_stationarity.py"))
_retardo = runpy.run_path(str(SCRIPTS / "check_retardo.py"))
serie_real: Callable = _cand["serie_real"]
theta_K: Callable = _cand["theta_K"]
fechas_entrenamiento: Callable = _cand["fechas_entrenamiento"]
ventanas_recortadas: Callable = _cand["ventanas_recortadas"]
rachas: Callable = _cand["rachas"]
fechas_bpoil: Callable = _cand["fechas_bpoil"]
DESCARTE: dict = _cand["DESCARTE"]
test_column: Callable = _estac["test_column"]
criterios: Callable = _retardo["criterios"]


def pares_por_version(columnas: list[str], descartada: str) -> dict[str, list[tuple[str, str]]]:
    sin = [c for c in columnas if c != descartada]
    return {v: list(itertools.permutations(sin if v == "residualizada" else columnas, 2)) for v in VERSIONES}


def serie_grilla(nombre_grilla: str, freq: str, derecha: bool, K: int, min_docs: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """log1p de las masas por ventana de la mayor racha y n_t, como candidatas_grillas.serie_real pero con K y min_docs libres."""
    masa, conteo = candidatas.masas_por_ventana(theta_K(K), fechas_entrenamiento(), freq)
    v = ventanas_recortadas(fechas_bpoil(T17_DIR), nombre_grilla, min_docs)
    pos, largo = max(rachas(v["valida"].to_numpy()), key=lambda r: r[1])
    tramo = v.iloc[pos:pos + largo]
    etiquetas = tramo["fin"] if derecha else tramo["inicio"]
    return np.log1p(masa.loc[etiquetas]), conteo.loc[etiquetas].rename("n_t")


def versiones_propio(propio: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"original": propio, "diferenciada": surrogates.diferenciar(propio),
            "detendenciada": surrogates.detendenciar(propio), "residualizada": propio}


def estimar(series: pd.DataFrame, descartada: str, p: int, clave: Callable[[str, str, str], tuple],
            controles: dict[str, Callable[[str, str], pd.Series]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """F, efecto, p_teo y p_rem de todos los pares, por control y versión, con BH y m = número de pares de la versión."""
    pares = pares_por_version(list(series.columns), descartada)
    versiones = candidatas.construir_versiones_k(series, descartada)
    filas, arreglos = [], {}
    for control, control_de in controles.items():
        for v, (serie, _, _) in versiones.items():
            m = len(pares[v])
            val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_teo", "p_rem")}
            disenos = [(o, d, *validez.armar_diseno_p(serie, control_de(v, d), d, None, p),
                        validez.armar_diseno_p(serie, control_de(v, d), d, o, p)[0]) for o, d in pares[v]]
            no_estimable = any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for _, _, X_f, _, _ in disenos)
            if not no_estimable:
                for j, (o, d, X_f, y, X_r) in enumerate(disenos):
                    obs = validez.prueba_p(X_f, X_r, y)
                    val["efecto"][j], val["F"][j], val["p_teo"][j] = obs["efecto"], obs["F"], obs["p_teo"]
                    val["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B,
                                                                 granger.generador(*clave(v, o, d)))["p"]
            r = np.zeros(m, dtype=bool) if no_estimable else granger.benjamini_hochberg(val["p_rem"], Q, m)
            for j, (o, d) in enumerate(pares[v]):
                filas.append({"control_tipo": control, "version": v, "origen": o, "destino": d,
                              "efecto": val["efecto"][j], "F": val["F"][j], "p_teorico": val["p_teo"][j],
                              "p_remuestreo": val["p_rem"][j], "rechaza_BH": int(r[j]), "no_estimable": int(no_estimable)})
            for k, arr in val.items():
                arreglos[f"{control}_{k}_{v}"] = arr
    return pd.DataFrame(filas), arreglos


def controles_reales(series: pd.DataFrame, descartada: str) -> dict[str, Callable[[str, str], pd.Series]]:
    versiones = candidatas.construir_versiones_k(series, descartada)
    return {"propio": lambda v, d: versiones[v][2][d], "compartido": lambda v, d: versiones[v][1]}


def enlaces_regla(tabla: pd.DataFrame) -> dict[str, int]:
    return {c: int((g.groupby(["origen", "destino"])["rechaza_BH"].sum() >= MIN_VERSIONES).sum())
            for c, g in tabla.groupby("control_tipo")}


# ---------------------------------------------------------------------------
# Bloque A
# ---------------------------------------------------------------------------

def bloque_a() -> None:
    real, n_t = serie_real("2D_K10")
    versiones = candidatas.construir_versiones_k(real, DESCARTE[10])
    filas = [{"serie": v, "topico": t, **test_column(versiones[v][0][t].dropna())} for v in VERSIONES
             for t in versiones[v][0].columns]
    a1 = pd.DataFrame(filas)
    scaler = StandardScaler()
    pca = PCA().fit(scaler.fit_transform(real.to_numpy()))
    a2_var = pd.DataFrame({"componente": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                           "varianza_explicada": pca.explained_variance_ratio_,
                           "acumulada": np.cumsum(pca.explained_variance_ratio_)})
    a2_cargas = pd.DataFrame({"topico": real.columns, "carga_componente_1": pca.components_[0]})
    docs = n_t.to_numpy()
    cortes = list(range(5, int(docs.max()) + 5, 5))
    hist = pd.cut(pd.Series(docs), bins=cortes, right=False).value_counts().sort_index()
    a3_hist = pd.DataFrame({"documentos": [f"{int(i.left)} a {int(i.right) - 1}" for i in hist.index],
                            "ventanas": hist.to_numpy(), "barra": ["#" * int(c) for c in hist.to_numpy()]})
    _, _, residuos = surrogates.ajustar_ar1(real)
    n_res = n_t.to_numpy()[1:].astype(float)
    a3_sp = pd.DataFrame([{"topico": t,
                           "spearman_abs_residuo_n_t": float(stats.spearmanr(np.abs(residuos[:, k]), n_res).statistic),
                           "p_n_t": float(stats.spearmanr(np.abs(residuos[:, k]), n_res).pvalue),
                           "spearman_abs_residuo_inv_n_t": float(stats.spearmanr(np.abs(residuos[:, k]), 1 / n_res).statistic)}
                          for k, t in enumerate(real.columns)])
    a3_docs = pd.DataFrame([{"ventanas": len(docs), "minimo": int(docs.min()), "mediana": float(np.median(docs)),
                             "maximo": int(docs.max())}])
    guardar_tablas({"A1_estacionariedad": a1, "A2_pca_varianza": a2_var, "A2_pca_cargas": a2_cargas,
                    "A3_documentos": a3_docs, "A3_histograma": a3_hist, "A3_spearman": a3_sp})


# ---------------------------------------------------------------------------
# Bloque B
# ---------------------------------------------------------------------------

def preparar_b() -> dict[str, object]:
    real, n_t = serie_real("2D_K10")
    proporciones = np.expm1(real).div(n_t.astype(float), axis=0)
    # Ceros (masa 0 de un tópico en una ventana): reemplazo multiplicativo solo para el clr del sustituto.
    sin_ceros, n_ceros, delta = validez.reemplazo_multiplicativo(proporciones)
    z = validez.clr(sin_ceros)
    c, a, residuos = surrogates.ajustar_ar1(z)
    return {"real": real, "n_t": n_t, "proporciones": proporciones, "clr": z, "c": c, "a": a, "residuos": residuos,
            "ceros_reemplazados": n_ceros, "delta": delta}


def sustituto_b(prep: dict[str, object], indice: int) -> tuple[pd.DataFrame, np.ndarray]:
    real, n_t, z = prep["real"], prep["n_t"], prep["clr"]
    Z = candidatas.simular(prep["c"], prep["a"], prep["residuos"], z.to_numpy().mean(axis=0), len(real), CALENT_B,
                           0, 1, 0.0, 0.0, granger.generador("val_generacion", str(indice)))
    s = validez.softmax_filas(Z)
    masa = s * n_t.to_numpy(dtype=float)[:, None]
    return pd.DataFrame(np.log1p(masa), index=real.index, columns=real.columns), s


def procesar_b_replica(prep: dict[str, object], indice: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fila por réplica (rechazos por control y versión, enlaces por la regla) y p de todos los pares."""
    x, _ = sustituto_b(prep, indice)
    n_t = prep["n_t"].astype(float)
    masa = np.expm1(x)
    propio = versiones_propio(np.log1p(masa.rsub(n_t, axis=0)))
    compartido = candidatas.versiones_control_fijo(np.log1p(n_t).rename("control"))
    fila: dict[str, object] = {"indice": indice}
    arreglos: dict[str, np.ndarray] = {}
    for control, control_de in (("propio", lambda v, d: propio[v][d]), ("compartido", lambda v, d: compartido[v])):
        tabla, arr = estimar(x, DESCARTE[10], 1, lambda v, o, d, c=control: ("val_remuestreo", c, str(indice), v, o, d),
                             {control: control_de})
        for v in VERSIONES:
            fila[f"rechazos_{control}_{v}"] = int(tabla[tabla["version"] == v]["rechaza_BH"].sum())
        fila[f"enlaces_regla_{control}"] = enlaces_regla(tabla).get(control, 0)
        arreglos.update(arr)
    return fila, arreglos


def bloque_b() -> None:
    prep = preparar_b()
    ruta_csv, ruta_npz = SALIDA / "B_sustitutos.csv", SALIDA / "B_sustitutos.npz"
    filas: list[dict[str, object]] = []
    arreglos: dict[str, list[np.ndarray]] = {}
    if ruta_csv.exists():
        tabla = pd.read_csv(ruta_csv)
        assert tabla["indice"].tolist() == list(range(len(tabla))), "B: índices no consecutivos"
        filas = tabla.to_dict("records")
        with np.load(ruta_npz) as z:
            arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
    print(f"B: {len(filas)} sustitutos guardados, faltan {N_SUSTITUTOS - len(filas)}", flush=True)
    t0 = time.perf_counter()
    for i in range(len(filas), N_SUSTITUTOS):
        fila, arr = procesar_b_replica(prep, i)
        filas.append(fila)
        for k, val in arr.items():
            arreglos.setdefault(k, []).append(val)
        if (i + 1) % CADA == 0 or i + 1 == N_SUSTITUTOS:
            SALIDA.mkdir(parents=True, exist_ok=True)
            tmp = SALIDA / "B_sustitutos.tmp.npz"
            np.savez(tmp, **{k: np.array(v) for k, v in arreglos.items()})
            os.replace(tmp, ruta_npz)
            tmp = SALIDA / "B_sustitutos.tmp.csv"
            pd.DataFrame(filas).to_csv(tmp, index=False)
            os.replace(tmp, ruta_csv)
            print(f"  B: {i + 1}/{N_SUSTITUTOS} ({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen_b()


def resumen_b() -> None:
    t = pd.read_csv(SALIDA / "B_sustitutos.csv")
    assert len(t) == N_SUSTITUTOS, f"B incompleto: {len(t)}"
    filas, criterio = [], []
    for control in CONTROLES:
        for v in VERSIONES:
            filas.append({"control_tipo": control, "version": v, "replicas_con_rechazo": int((t[f"rechazos_{control}_{v}"] >= 1).sum()),
                          "fraccion": float((t[f"rechazos_{control}_{v}"] >= 1).mean()),
                          "rechazos_medios": float(t[f"rechazos_{control}_{v}"].mean())})
        filas.append({"control_tipo": control, "version": "regla_2_versiones",
                      "replicas_con_rechazo": int((t[f"enlaces_regla_{control}"] >= 1).sum()),
                      "fraccion": float((t[f"enlaces_regla_{control}"] >= 1).mean()),
                      "rechazos_medios": float(t[f"enlaces_regla_{control}"].mean())})
        cuentas = {v: int((t[f"rechazos_{control}_{v}"] >= 1).sum()) for v in ("original", "detendenciada")}
        criterio.append({"control_tipo": control, "original": cuentas["original"], "detendenciada": cuentas["detendenciada"],
                         "umbral": UMBRAL_B, "pasa": all(c <= UMBRAL_B for c in cuentas.values())})
    guardar_tablas({"B_calibracion": pd.DataFrame(filas), "B_criterio": pd.DataFrame(criterio)})


def medir(n: int) -> None:
    prep = preparar_b()
    tiempos = []
    for i in range(n):
        t0 = time.perf_counter()
        procesar_b_replica(prep, i)
        tiempos.append(time.perf_counter() - t0)
    b = float(np.mean(tiempos)) * N_SUSTITUTOS
    t0 = time.perf_counter()
    real, _ = serie_real("2D_K10")
    estimar(real, DESCARTE[10], 1, lambda v, o, d: ("sens_medicion", v, o, d), controles_reales(real, DESCARTE[10]))
    una = time.perf_counter() - t0
    print(f"B: {np.mean(tiempos):.3f} s por sustituto x {N_SUSTITUTOS} = {b / 60:.1f} min")
    print(f"C: una estimación con K = 10, 2 controles, {una:.1f} s; 9 sensibilidades, a lo sumo {9 * una * 2 / 60:.1f} min")
    print(f"Total estimado (A y D, menos de 1 min): {(b + 9 * una * 2) / 60 + 1:.1f} min")


# ---------------------------------------------------------------------------
# Bloque C
# ---------------------------------------------------------------------------

def sensibilidades() -> dict[str, tuple[pd.DataFrame, str, int] | int]:
    """Nombre -> (serie, columna descartada, retardo), o el largo de la racha si no se estima."""
    real10, _ = serie_real("2D_K10")
    salida: dict[str, tuple[pd.DataFrame, str, int] | int] = {
        "C1_retardo_2": (real10, DESCARTE[10], 2),
        "C2_2D_K8": (serie_real("2D_K8")[0], DESCARTE[8], 1),
        "C2_2D_K5": (serie_real("2D_K5")[0], DESCARTE[5], 1),
        "C3_3D_K10": (serie_grilla("cada 3 días", "3D", False, 10)[0], DESCARTE[10], 1),
        "C3_3D_K5": (serie_real("3D_K5")[0], DESCARTE[5], 1),
        "C4_sin_5_primeras": (real10.iloc[5:], DESCARTE[10], 1),
        "C4_sin_5_ultimas": (real10.iloc[:-5], DESCARTE[10], 1),
    }
    serie8, _ = serie_grilla("cada 2 días", "2D", False, 10, min_docs=8)
    salida["C5_min_docs_8"] = (serie8, DESCARTE[10], 1) if len(serie8) >= LARGO_MINIMO_C5 else len(serie8)
    return salida


def bloque_c() -> None:
    principal = pd.read_csv(OUTPUT_DIR / "resultados_reales_2d" / "resultados.csv", float_precision="round_trip")
    top_principal = set(principal[(principal["control_tipo"] == "propio") & (principal["version"] == "original")]
                        .nsmallest(TOP_TRASLAPE, "p_remuestreo")[["origen", "destino"]].itertuples(index=False, name=None))
    resumen, tops, traslape = [], [], []
    for nombre, spec in sensibilidades().items():
        if isinstance(spec, int):
            resumen.append({"sensibilidad": nombre, "ventanas": spec, "estimada": False})
            traslape.append({"sensibilidad": nombre, "comparable": False, "top10_en_comun": np.nan,
                             "nota": f"racha de {spec} ventanas, menos de {LARGO_MINIMO_C5}: no se estima"})
            continue
        serie, descartada, p = spec
        tabla, arreglos = estimar(serie, descartada, p, lambda v, o, d, s=nombre: ("sens_remuestreo", s, v, o, d),
                                  controles_reales(serie, descartada))
        SALIDA.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(SALIDA / f"C_{nombre}.csv", index=False)
        np.savez(SALIDA / f"C_{nombre}.npz", **arreglos)
        enl = enlaces_regla(tabla)
        for (control, v), g in tabla.groupby(["control_tipo", "version"], sort=False):
            resumen.append({"sensibilidad": nombre, "ventanas": len(serie), "retardo": p, "K": serie.shape[1],
                            "estimada": True, "control_tipo": control, "version": v, "m": len(g),
                            "rechazos_BH": int(g["rechaza_BH"].sum()), "p_remuestreo_min": float(g["p_remuestreo"].min()),
                            "efecto_max": float(g["efecto"].max()), "enlaces_regla": enl.get(control, 0)})
            for fila in g.nsmallest(TOP, "p_remuestreo").itertuples(index=False):
                tops.append({"sensibilidad": nombre, "control_tipo": control, "version": v, "origen": fila.origen,
                             "destino": fila.destino, "p_remuestreo": fila.p_remuestreo, "efecto": fila.efecto})
        comparable = serie.shape[1] == 10
        g = tabla[(tabla["control_tipo"] == "propio") & (tabla["version"] == "original")]
        top_s = set(g.nsmallest(TOP_TRASLAPE, "p_remuestreo")[["origen", "destino"]].itertuples(index=False, name=None))
        traslape.append({"sensibilidad": nombre, "comparable": comparable,
                         "top10_en_comun": len(top_s & top_principal) if comparable else np.nan,
                         "nota": "" if comparable else f"K = {serie.shape[1]}: tópicos de otro modelo, rótulos no comparables"})
        print(f"  C {nombre}: estimada ({len(serie)} ventanas, retardo {p}, K = {serie.shape[1]})", flush=True)
    guardar_tablas({"C_resumen": pd.DataFrame(resumen), "C_top5": pd.DataFrame(tops), "C_traslape": pd.DataFrame(traslape)})


# ---------------------------------------------------------------------------
# Bloque D
# ---------------------------------------------------------------------------

def bloque_d() -> None:
    real, _ = serie_real("2D_K10")
    versiones = candidatas.construir_versiones_k(real, DESCARTE[10])
    filas = []
    for v, (serie, compartido, _) in versiones.items():
        for d in serie.columns:
            fila: dict[str, object] = {"version": v, "destino": d}
            for p in (1, 2):
                X, y = validez.armar_diseno_p(serie, compartido, d, None, p, inicio=2)
                rss, k = granger.ajustar(X, y)
                aic, bic = criterios(rss, len(y), k)
                fila.update({f"n_obs_p{p}": len(y), f"params_p{p}": k, f"aic_p{p}": aic, f"bic_p{p}": bic})
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    resumen = tabla.groupby("version", sort=False).apply(lambda g: pd.Series({
        "ecuaciones": len(g), "aic_medio_p1": g["aic_p1"].mean(), "aic_medio_p2": g["aic_p2"].mean(),
        "bic_medio_p1": g["bic_p1"].mean(), "bic_medio_p2": g["bic_p2"].mean(),
        "gana_p1_aic": int((g["aic_p1"] < g["aic_p2"]).sum()), "gana_p2_aic": int((g["aic_p2"] < g["aic_p1"]).sum()),
        "gana_p1_bic": int((g["bic_p1"] < g["bic_p2"]).sum()), "gana_p2_bic": int((g["bic_p2"] < g["bic_p1"]).sum())}),
        include_groups=False).reset_index()
    guardar_tablas({"D_criterios": tabla, "D_resumen": resumen})


# ---------------------------------------------------------------------------
# Salidas, verificaciones y resumen
# ---------------------------------------------------------------------------

def guardar_tablas(tablas: dict[str, pd.DataFrame]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        for nombre, tabla in tablas.items():
            tabla.to_csv(SALIDA / f"{nombre}.csv", index=False)
            print(f"\n=== {nombre} (data/validez/{nombre}.csv) ===")
            print(tabla.to_string(index=False))


def verificar() -> bool:
    res: dict[str, bool] = {}
    print("=== a) Serie y control contra la estimación principal (data/resultados_reales_2d/resultados.csv) ===")
    real, _ = serie_real("2D_K10")
    principal = pd.read_csv(OUTPUT_DIR / "resultados_reales_2d" / "resultados.csv", float_precision="round_trip")
    versiones = candidatas.construir_versiones_k(real, DESCARTE[10])
    ok_a, iguales_diseno = True, True
    for fila in principal.itertuples(index=False):
        serie, compartido, propio = versiones[fila.version]
        ctrl = propio[fila.destino] if fila.control_tipo == "propio" else compartido
        X_f, y = validez.armar_diseno_p(serie, ctrl, fila.destino, None, 1)
        X_r, _ = validez.armar_diseno_p(serie, ctrl, fila.destino, fila.origen, 1)
        Xg, yg = granger.armar_diseno(serie, ctrl, fila.destino, excluir=None)
        iguales_diseno &= np.array_equal(X_f, Xg) and np.array_equal(y, yg)
        obs = validez.prueba_p(X_f, X_r, y)
        ok_a &= obs["F"] == fila.F and obs["efecto"] == fila.efecto
    res["a"] = ok_a and iguales_diseno
    print(f"  {len(principal)} filas: F y efecto idénticos {ok_a}; armar_diseno_p(p = 1) igual a granger.armar_diseno "
          f"{iguales_diseno}: {'OK' if res['a'] else 'FALLA'}")

    print("\n=== b) Sustituto del bloque B: masas y proporciones ===")
    prep = preparar_b()
    print(f"  ceros reemplazados en las proporciones reales (solo para el clr): {prep['ceros_reemplazados']}, "
          f"delta = {prep['delta']:.6g}")
    dif_n, dif_s = 0.0, 0.0
    for i in range(5):
        x, s = sustituto_b(prep, i)
        dif_n = max(dif_n, float(np.abs(np.expm1(x).sum(axis=1).to_numpy() - prep["n_t"].to_numpy()).max()))
        dif_s = max(dif_s, float(np.abs(s.sum(axis=1) - 1).max()))
    res["b"] = dif_n <= 1e-9 and dif_s <= 1e-12
    print(f"  5 sustitutos: max |suma de masas - n_t| = {dif_n:.3e}; max |suma de proporciones - 1| = {dif_s:.3e}: "
          f"{'OK' if res['b'] else 'FALLA'}")

    print("\n=== c) Sin dinámica y n_t constante: media de las proporciones ===")
    z = prep["clr"].to_numpy()
    centrado = z - z.mean(axis=0)
    K = z.shape[1]
    Zc = candidatas.simular(z.mean(axis=0), np.zeros(K), centrado, z.mean(axis=0), 20000, CALENT_B, 0, 1, 0.0, 0.0,
                            granger.generador("val_verificacion", "c"))
    s_sim = validez.softmax_filas(Zc)
    n_cte = float(prep["n_t"].mean())
    masa_c = s_sim * n_cte
    dif_c = float(np.abs((masa_c / n_cte).mean(axis=0) - prep["proporciones"].to_numpy().mean(axis=0)).max())
    res["c"] = dif_c <= 0.02
    print(f"  max |media simulada - media real| por tópico = {dif_c:.4f} <= 0.02: {'OK' if res['c'] else 'FALLA'}")

    print("\n=== d) Misma réplica dos veces (bloque B, sustituto 0) ===")
    (f1, a1), (f2, a2) = (procesar_b_replica(prep, 0) for _ in range(2))
    res["d"] = f1 == f2 and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
    print(f"  {'OK' if res['d'] else 'FALLA'}")

    print("\n=== e) Unicidad de las claves de semilla ===")
    previas = _conf["claves_previas_y_candidatas"]()
    for cand in _conf["CONFIRMAR"]:
        K_ = _cand["CANDIDATAS"][cand][4]
        pares_c = pares_por_version(list(serie_real(cand)[0].columns), DESCARTE[K_])
        for e, n in _conf["EFECTOS"].items():
            for i in range(n):
                previas.append(("conf_generacion", cand, e, str(i)))
                if float(e) > 0:
                    previas.append(("conf_par", cand, e, str(i)))
                for v, lista in pares_c.items():
                    previas += [("conf_remuestreo", cand, "C0", e, str(i), v, o, d) for o, d in lista]
    pares10 = pares_por_version(list(real.columns), DESCARTE[10])
    previas += [("real2_remuestreo", v, o, d) for v, lista in pares10.items() for o, d in lista]
    nuevas: list[tuple[str | int, ...]] = [("val_verificacion", "c")]
    for i in range(N_SUSTITUTOS):
        nuevas.append(("val_generacion", str(i)))
        for control in CONTROLES:
            nuevas += [("val_remuestreo", control, str(i), v, o, d) for v, lista in pares10.items() for o, d in lista]
    for nombre, spec in sensibilidades().items():
        if isinstance(spec, int):
            continue
        serie, descartada, _ = spec
        for v, lista in pares_por_version(list(serie.columns), descartada).items():
            nuevas += [("sens_remuestreo", nombre, v, o, d) for o, d in lista]
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    res["e"] = distintas == len(todas)
    print(f"  {len(todas)} ({len(previas)} previas + {len(nuevas)} nuevas), distintas: {distintas}: {'OK' if res['e'] else 'FALLA'}")

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


def resumen() -> None:
    nombres = ["A1_estacionariedad", "A2_pca_varianza", "A2_pca_cargas", "A3_documentos", "A3_histograma", "A3_spearman",
               "D_resumen", "C_resumen", "C_top5", "C_traslape", "B_calibracion", "B_criterio"]
    guardar_tablas({n: pd.read_csv(SALIDA / f"{n}.csv") for n in nombres})


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true")
    grupo.add_argument("--medir", type=int, metavar="N")
    grupo.add_argument("--resumen", action="store_true")
    grupo.add_argument("--bloque", choices=["A", "B", "C", "D"])
    args = parser.parse_args()
    if args.verificar:
        verificar()
    elif args.medir:
        medir(args.medir)
    elif args.resumen:
        resumen()
    elif args.bloque:
        {"A": bloque_a, "B": bloque_b, "C": bloque_c, "D": bloque_d}[args.bloque]()
    else:
        for bloque in (bloque_a, bloque_d, bloque_c, bloque_b):
            bloque()


if __name__ == "__main__":
    main()
