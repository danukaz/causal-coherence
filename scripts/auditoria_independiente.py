"""
auditoria_independiente.py

Semana 7: auditoría independiente de la implementación (diseño y criterios
en Bitacora_Semana7.md, sección 10). Una función por prueba; cada una
imprime su resultado y PASA o FALLA.

  A1  cadena de datos, del corpus T17 a la serie de 23 x 10
  A2  recálculo independiente de F con statsmodels
  A3  remuestreo vectorizado contra un bucle explícito
  A4  realismo de los sustitutos
  A5  VAR gaussiano sin enlace: tamaño del test (reanudable)
  A6  VAR gaussiano con enlace plantado: poder según T (reanudable)

Salidas en data/auditoria/. A5 y A6 guardan un punto de control cada 100
réplicas y siguen desde la primera que falta.

Uso:
  python scripts/auditoria_independiente.py A1 A2 A3 A4
  python scripts/auditoria_independiente.py --medir 5 A5 A6   tiempo; no guarda réplicas
  python scripts/auditoria_independiente.py A5 A6             corridas completas
  python scripts/auditoria_independiente.py --resumen A5 A6   reporte desde lo guardado
"""

import argparse
import hashlib
import inspect
import itertools
import os
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.signal import detrend

from causal_coherence import auditoria, data_loading, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

SALIDA = OUTPUT_DIR / "auditoria"
SCRIPTS = Path(__file__).resolve().parent
Q = 0.05
B = 9999
CADA = 100
VERSIONES = {
    "original": "topic_series_volumen.csv",
    "diferenciada": "topic_series_volumen_diferenciada.csv",
    "detendenciada": "topic_series_volumen_detendenciada.csv",
    "residualizada": "topic_series_volumen_residualizada.csv",
}

# A2 y A3
TOL_A2 = 1e-8
B_A3 = 999
TOL_A3 = 1e-9
# A4 (criterio corregido, Bitacora_Semana7.md, sección 10, nota de las 16:01)
MAX_FUERA_A4 = 2
TOL_A4 = 1e-9
POR_TOPICO_A4 = ("media", "desv_est", "autocorr_1")
# A5
N_A5 = 2000
N_REM_A5 = 300
INTERVALO_A5 = (0.0325, 0.0700)
Z_A5 = 3
# A6
EFECTOS_A6 = ("0.4", "0.8", "1.6")
CELDAS_A6 = [(10, T, e) for T in (23, 46, 92) for e in EFECTOS_A6] + [(5, 23, e) for e in EFECTOS_A6]
N_A6 = 1000
N_REM_A6 = 100
MARGEN_A6 = 0.03
T_PRINCIPAL_A6 = 92
DETECCION_MIN_A6 = 0.80


def marca(ok: bool) -> str:
    return "PASA" if ok else "FALLA"


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / nombre, index_col=0, parse_dates=True)


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

def prueba_a1() -> bool:
    print("=== A1. Cadena de datos: del corpus T17 a la serie de 23 x 10 ===")
    from causal_coherence.topic_model import load_topic_model, training_documents

    docs = data_loading.load_full_corpus()
    print(f"(a) documentos totales del corpus T17: {len(docs)}")
    for fuente, n in sorted(Counter(d["source"] for d in docs).items()):
        print(f"      {n:5d}  {fuente}")
    bpoil = [d for d in docs if d["source"] == data_loading.SOURCE_LABEL]
    ok_b = len(bpoil) == 1133
    print(f"(b) source == '{data_loading.SOURCE_LABEL}': {len(bpoil)} (esperado 1133): {marca(ok_b)}")
    fechas = pd.to_datetime(pd.Series([d["date"] for d in bpoil]))
    en_ventana = (fechas >= data_loading.CORPUS_START) & (fechas <= data_loading.CORPUS_END)
    ok_c = int(en_ventana.sum()) == 1009
    print(f"(c) dentro de [{data_loading.CORPUS_START.date()}, {data_loading.CORPUS_END.date()}]: "
          f"{int(en_ventana.sum())} (esperado 1009): {marca(ok_c)}")

    theta = load_topic_model().get_document_topic_matrix()
    entrenamiento = training_documents()
    etiquetas = sorted(set(entrenamiento["source"]))
    ok_d = theta.shape[0] == 1009 == len(entrenamiento) and etiquetas == [data_loading.SOURCE_LABEL]
    print(f"(d) filas de theta: {theta.shape[0]} (esperado 1009); documentos de entrenamiento: "
          f"{len(entrenamiento)}; etiquetas en el entrenamiento: {etiquetas}: {marca(ok_d)}")

    n_t = np.expm1(leer("control_volumen.csv").iloc[:, 0])
    fv = fechas[en_ventana]
    fin_semana = fv.dt.normalize() + pd.to_timedelta(6 - fv.dt.weekday, unit="D")
    conteo = fin_semana.value_counts().reindex(n_t.index)
    dif_e = float((conteo - n_t).abs().max())
    ok_e = bool(conteo.notna().all()) and dif_e <= 1e-9
    print(f"(e) documentos por semana desde las fechas crudas (semana que cierra el domingo) contra "
          f"expm1(control_volumen.csv), {len(n_t)} semanas: max |dif| = {dif_e:.3e}: {marca(ok_e)}")
    print("      " + ", ".join(f"{f.date()}: {int(c)}" for f, c in conteo.items()))

    series = leer("topic_series_volumen.csv")
    dif_f = float((np.expm1(series).sum(axis=1) - n_t).abs().max())
    ok_f = series.index.equals(n_t.index) and dif_f <= 1e-9
    print(f"(f) suma de las 10 masas (expm1) contra n_t, por semana: max |dif| = {dif_f:.3e}: {marca(ok_f)}")

    fuente_sust = inspect.getsource(surrogates)
    lecturas_sust = [p for p in ("read_csv", "np.load", "open(", "OUTPUT_DIR") if p in fuente_sust]
    lineas_calib = [linea.strip() for linea in (SCRIPTS / "run_calibracion.py").read_text(encoding="utf-8").splitlines()
                    if "read_csv(OUTPUT_DIR" in linea]
    ok_g = (series.shape == (23, 10) and not lecturas_sust and len(lineas_calib) == 1
            and "topic_series_volumen.csv" in lineas_calib[0])
    print(f"(g) shape {series.shape} (esperado (23, 10)); MD5 de topic_series_volumen.csv: "
          f"{md5(OUTPUT_DIR / 'topic_series_volumen.csv')}")
    print(f"      surrogates.py no lee archivos (busca read_csv, np.load, open(, OUTPUT_DIR): "
          f"{'ninguno' if not lecturas_sust else lecturas_sust}; recibe la serie como argumento")
    print(f"      lecturas de datos de entrada en run_calibracion.py: {lineas_calib}: {marca(ok_g)}")

    ok = ok_b and ok_c and ok_d and ok_e and ok_f and ok_g
    print(f"A1: {marca(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------

def pares_a2() -> list[tuple[str, str]]:
    """Top 5 por F (original, control propio) entre los 72 pares sin topic_5, y 5 sorteados entre los 67 restantes."""
    barrido = pd.read_csv(OUTPUT_DIR / "granger_results.csv", float_precision="round_trip")
    g = barrido[(barrido["version"] == "original") & (barrido["control_tipo"] == "propio")]
    g = g[(g["origen"] != "topic_5") & (g["destino"] != "topic_5")]
    assert len(g) == 72
    top = g.nlargest(5, "F")
    resto = g.drop(index=top.index)
    rng = np.random.default_rng(np.random.SeedSequence(42, spawn_key=(zlib.crc32(b"auditoria_A2"),)))
    sorteo = resto.iloc[np.sort(rng.choice(len(resto), size=5, replace=False))]
    return list(zip(top["origen"], top["destino"])) + list(zip(sorteo["origen"], sorteo["destino"]))


def control_propio_independiente(version: str, indice: pd.Index) -> pd.DataFrame:
    """log1p(n_t - masa_j) con n_t = suma de las masas, transformado según la versión con pandas y scipy."""
    masa = np.expm1(leer("topic_series_volumen.csv"))
    propio = np.log1p(masa.sum(axis=1).to_numpy()[:, None] - masa)
    if version == "diferenciada":
        propio = propio.diff().iloc[1:]
    elif version == "detendenciada":
        propio = pd.DataFrame(detrend(propio.to_numpy(), axis=0, type="linear"),
                              index=propio.index, columns=propio.columns)
    return propio.loc[indice]


def f_statsmodels(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> tuple[float, float, int, int]:
    """F (compare_f_test) y p (scipy.stats.f.sf) con rezagos de pandas shift, control e intercepto."""
    rezagos = serie.shift(1).add_suffix("_rezago")
    datos = pd.concat([serie[destino].rename("y"), rezagos, control.rename("control")], axis=1).dropna()
    X_f = sm.add_constant(datos.drop(columns="y"))
    X_r = X_f.drop(columns=f"{origen}_rezago")
    completo = sm.OLS(datos["y"], X_f).fit()
    restringido = sm.OLS(datos["y"], X_r).fit()
    F, _, df_num = completo.compare_f_test(restringido)
    df_den = int(completo.df_resid)
    return float(F), float(stats.f.sf(F, df_num, df_den)), int(df_num), df_den


def prueba_a2() -> bool:
    print("=== A2. Recálculo independiente de F con statsmodels ===")
    pares = pares_a2()
    print(f"Pares (5 de mayor F en la original y 5 sorteados): {[f'{o}->{d}' for o, d in pares]}")
    barrido = pd.read_csv(OUTPUT_DIR / "granger_results.csv", float_precision="round_trip")
    barrido = barrido[barrido["control_tipo"] == "propio"].set_index(["version", "origen", "destino"])
    filas = []
    for version, archivo in VERSIONES.items():
        serie = leer(archivo)
        propio = control_propio_independiente(version, serie.index)
        for origen, destino in pares:
            F, p, df_num, df_den = f_statsmodels(serie, propio[destino], origen, destino)
            ref = barrido.loc[(version, origen, destino)]
            filas.append({"version": version, "par": f"{origen}->{destino}", "F_statsmodels": F,
                          "F_barrido": ref["F"], "error_rel_F": abs(F - ref["F"]) / abs(ref["F"]),
                          "p_scipy_sf": p, "p_barrido": ref["p_valor_orientativo"],
                          "error_rel_p": abs(p - ref["p_valor_orientativo"]) / abs(ref["p_valor_orientativo"]),
                          "df": f"{df_num},{df_den}", "df_barrido": f"{int(ref['df_num'])},{int(ref['df_den'])}"})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "A2.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.10g}".format):
        print(tabla.to_string(index=False))
    error_max = float(tabla[["error_rel_F", "error_rel_p"]].to_numpy().max())
    ok = error_max <= TOL_A2 and bool((tabla["df"] == tabla["df_barrido"]).all())
    print(f"Error relativo máximo (F y p, 40 comparaciones): {error_max:.3e} <= {TOL_A2}; grados de libertad "
          f"iguales: {bool((tabla['df'] == tabla['df_barrido']).all())}")
    print(f"A2: {marca(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# A3
# ---------------------------------------------------------------------------

def prueba_a3() -> bool:
    print(f"=== A3. Remuestreo vectorizado contra bucle explícito (B = {B_A3}, versión original) ===")
    real = leer("topic_series_volumen.csv")
    serie, _, propio = surrogates.construir_versiones(real)["original"]
    filas = []
    for origen, destino in pares_a2()[:3]:
        control = propio[destino]
        obs = granger.estadistico_f(serie, control, origen, destino)
        X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
        X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
        eta = granger.multiplicadores_rademacher(B_A3, len(y), granger.generador("remuestreo", "original", origen, destino))
        M_f, M_r = granger.proyector_residual(X_f), granger.proyector_residual(X_r)
        eps_r = M_r @ y
        F_vec = granger.f_replicas(M_f, M_r, y - eps_r, eps_r, eta, obs["df_num"], obs["df_den"])
        # Bucle explícito con lstsq, independiente de los proyectores.
        coef_r, *_ = np.linalg.lstsq(X_r, y, rcond=None)
        y_hat_e = X_r @ coef_r
        eps_e = y - y_hat_e
        F_bucle = np.empty(B_A3)
        for b in range(B_A3):
            y_b = y_hat_e + eta[b] * eps_e
            rss = []
            for X in (X_f, X_r):
                cb, *_ = np.linalg.lstsq(X, y_b, rcond=None)
                rss.append(float(np.sum((y_b - X @ cb) ** 2)))
            F_bucle[b] = ((rss[1] - rss[0]) / obs["df_num"]) / (rss[0] / obs["df_den"])
        dif = float(np.max(np.abs(F_vec - F_bucle)))
        filas.append({"par": f"{origen}->{destino}", "F_obs": obs["F"], "B": B_A3,
                      "max_abs_dif_F_estrella": dif, "F_estrella_min": float(F_bucle.min()),
                      "F_estrella_max": float(F_bucle.max()), "pasa": dif <= TOL_A3})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "A3.csv", index=False)
    print(tabla.to_string(index=False))
    ok = bool(tabla["pasa"].all())
    print(f"A3: {marca(ok)} (tolerancia {TOL_A3})\n")
    return ok


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------

def estadisticos_a4(x: pd.DataFrame) -> dict[str, np.ndarray]:
    X = x.to_numpy()
    control = np.log1p(np.expm1(X).sum(axis=1))
    with np.errstate(all="ignore"):
        corr = np.corrcoef(X, rowvar=False)
        acf1 = np.array([np.corrcoef(X[1:, k], X[:-1, k])[0, 1] for k in range(X.shape[1])])
        corr_control = np.array([np.corrcoef(X[:, k], control)[0, 1] for k in range(X.shape[1])])
    return {"media": X.mean(axis=0), "desv_est": X.std(axis=0, ddof=1), "autocorr_1": acf1,
            "corr_control": corr_control,
            "corr_media_abs": np.array([np.abs(corr[np.triu_indices(X.shape[1], k=1)]).mean()])}


def prueba_a4() -> bool:
    print("=== A4. Realismo de los sustitutos (criterio corregido) ===")
    print(f"Dentro del intervalo si la distancia es <= {TOL_A4}. Por tópico {POR_TOPICO_A4} en ambas familias; "
          f"corr_control solo en ar_diagonal; corr_media_abs y corr_control de fase_aleatoria, informativos.")
    real = leer("topic_series_volumen.csv")
    est_real = estadisticos_a4(real)
    filas = []
    resumen = []
    ok = True
    for familia in surrogates.FAMILIAS:
        sust = [estadisticos_a4(surrogates.generar_sustituto(familia, i, real)[0]) for i in range(1000)]
        for nombre in est_real:
            valores = np.array([s[nombre] for s in sust])  # 1000 x (10 o 1)
            p_bajo = np.nanpercentile(valores, 2.5, axis=0)
            p_alto = np.nanpercentile(valores, 97.5, axis=0)
            n_validos = np.sum(~np.isnan(valores), axis=0)
            for k, valor in enumerate(est_real[nombre]):
                columna = valores[:, k]
                distancia = max(p_bajo[k] - valor, valor - p_alto[k], 0.0)
                filas.append({"familia": familia, "estadistico": nombre,
                              "topico": real.columns[k] if len(est_real[nombre]) > 1 else "todos",
                              "valor_real": valor, "percentil_real": 100 * float(np.nanmean(columna <= valor)),
                              "p2.5": p_bajo[k], "p97.5": p_alto[k], "n_validos": int(n_validos[k]),
                              "distancia_al_intervalo": distancia, "fuera_sin_tolerancia": distancia > 0,
                              "fuera": distancia > TOL_A4})
            fuera_n = sum(f["fuera"] for f in filas if f["familia"] == familia and f["estadistico"] == nombre)
            # Criterio corregido: los estadísticos por tópico se evalúan en ambas
            # familias; corr_control (por tópico, entre series) solo en ar_diagonal;
            # corr_media_abs (un valor) es informativo.
            evaluado = nombre in POR_TOPICO_A4 or (nombre == "corr_control" and familia == "ar_diagonal")
            cumple = (fuera_n <= MAX_FUERA_A4) if evaluado else None
            if evaluado:
                ok &= cumple
            resumen.append({"familia": familia, "estadistico": nombre, "n_items": len(est_real[nombre]),
                            "fuera": fuera_n, "criterio": "<= 2 de 10" if evaluado else "informativo",
                            "cumple": cumple})
    tabla = pd.DataFrame(filas)
    tabla_resumen = pd.DataFrame(resumen)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "A4.csv", index=False)
    tabla_resumen.to_csv(SALIDA / "A4_resumen.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 200):
        print(tabla.to_string(index=False))
        print()
        print(tabla_resumen.to_string(index=False))
    print(f"A4: {marca(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# Claves de semilla
# ---------------------------------------------------------------------------

def pares_por_version(columnas: list[str]) -> dict[str, list[tuple[str, str]]]:
    sin = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(sin if v == "residualizada" else columnas, 2)) for v in surrogates.VERSIONES}


def verificar_claves() -> bool:
    """Unicidad de las claves nuevas de A5 y A6 contra todas las ya usadas."""
    columnas = list(leer("topic_series_volumen.csv").columns)
    pares = pares_por_version(columnas)
    plantables = planted.pares_plantables(columnas)
    ids: list[tuple[str | int, ...]] = []
    for v, lista in pares.items():
        ids += [("remuestreo", v, o, d) for o, d in lista]
    for familia in surrogates.FAMILIAS:
        for i in range(1000):
            ids.append(("generacion", familia, i))
            for v, lista in pares.items():
                ids += [("remuestreo", familia, i, v, o, d) for o, d in lista]
    for e in planted.TAMANOS:
        if float(e) != 0.0:
            ids += [("plantado_calibracion", e, o, d) for o, d in plantables]
            ids += [("plantado_sesgo", e, o, d, s) for o, d in plantables for s in range(5)]
        for i in range(300):
            ids += [("plantado_par", e, i), ("plantado_generacion", e, i)]
            for v, lista in pares.items():
                ids += [("plantado_remuestreo", e, i, v, o, d) for o, d in lista]
    for variante in ("V0", "V1", "V2", "V3"):
        for e in ("0.0", "1.6"):
            for i in range(200):
                for v, lista in pares.items():
                    ids += [("diag_remuestreo", variante, e, i, v, o, d) for o, d in lista]
    n_previas = len(ids)
    todos = list(itertools.permutations(columnas, 2))
    for i in range(N_A5):
        ids.append(("auditoria_A5_generacion", i))
        if i < N_REM_A5:
            ids += [("auditoria_A5_remuestreo", i, o, d) for o, d in todos]
    for K in sorted({K for K, _, _ in CELDAS_A6}):
        ids += [("auditoria_A6_calibracion", K, e) for e in EFECTOS_A6]
    for K, T, e in CELDAS_A6:
        pares_K = list(itertools.permutations(columnas[:K], 2))
        for i in range(N_A6):
            ids.append(("auditoria_A6_generacion", K, T, e, i))
            if i < N_REM_A6:
                ids += [("auditoria_A6_remuestreo", K, T, e, i, o, d) for o, d in pares_K]
    distintas = len({granger.clave_semilla(*ident) for ident in ids})
    ok = distintas == len(ids)
    print(f"Claves de semilla: {len(ids)} ({n_previas} previas + {len(ids) - n_previas} nuevas), "
          f"distintas: {distintas}: {'OK' if ok else 'FALLA'}")
    return ok


# ---------------------------------------------------------------------------
# A5 y A6: puntos de control
# ---------------------------------------------------------------------------

def cargar(ruta: Path) -> dict[str, np.ndarray]:
    if not ruta.exists():
        return {}
    with np.load(ruta) as z:
        return {k: z[k] for k in z.files}


def guardar(ruta: Path, datos: dict[str, np.ndarray]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.stem + ".tmp.npz")
    np.savez(tmp, **datos)
    os.replace(tmp, ruta)


# ---------------------------------------------------------------------------
# A5
# ---------------------------------------------------------------------------

def replica_a5(i: int, c: np.ndarray, a: np.ndarray, sigma: np.ndarray, columnas: list[str],
               con_remuestreo: bool) -> tuple[np.ndarray, np.ndarray | None]:
    X = auditoria.simular_var(c, a, sigma, 23, granger.generador("auditoria_A5_generacion", i))
    pares = list(itertools.permutations(range(len(columnas)), 2))
    F, p_f = auditoria.p_ruta_f(X, pares)
    if not con_remuestreo:
        return p_f, None
    claves = [("auditoria_A5_remuestreo", i, columnas[o], columnas[d]) for o, d in pares]
    return p_f, auditoria.p_ruta_remuestreo(X, pares, F, B, claves)


def correr_a5(medir: int | None) -> None:
    real = leer("topic_series_volumen.csv")
    columnas = list(real.columns)
    c, a, sigma = auditoria.parametros_var(real)
    if medir:
        t_f, t_r = [], []
        for i in range(medir):
            t0 = time.perf_counter()
            replica_a5(i, c, a, sigma, columnas, False)
            t_f.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            replica_a5(i, c, a, sigma, columnas, True)
            t_r.append(time.perf_counter() - t0 - t_f[-1])
        total = np.mean(t_f) * N_A5 + np.mean(t_r) * N_REM_A5
        print(f"A5: ruta F {np.mean(t_f):.3f} s por réplica x {N_A5} = {np.mean(t_f) * N_A5 / 60:.1f} min; "
              f"remuestreo {np.mean(t_r):.3f} s por réplica x {N_REM_A5} = {np.mean(t_r) * N_REM_A5 / 60:.1f} min; "
              f"total {total / 60:.1f} min")
        return
    ruta = SALIDA / "A5.npz"
    datos = cargar(ruta)
    p_f = list(datos.get("p_f", []))
    p_rem = list(datos.get("p_rem", []))
    print(f"A5: {len(p_f)} réplicas ya guardadas, faltan {N_A5 - len(p_f)}", flush=True)
    for i in range(len(p_f), N_A5):
        pf, pr = replica_a5(i, c, a, sigma, columnas, i < N_REM_A5)
        p_f.append(pf)
        if pr is not None:
            p_rem.append(pr)
        if (i + 1) % CADA == 0:
            guardar(ruta, {"p_f": np.array(p_f), "p_rem": np.array(p_rem).reshape(-1, len(pf))})
            print(f"  A5: {i + 1}/{N_A5} guardadas", flush=True)


def resumen_a5() -> bool:
    print("=== A5. VAR gaussiano sin enlace: tamaño del test (T = 23, 22 observaciones, 11 parámetros) ===")
    datos = cargar(SALIDA / "A5.npz")
    p_f, p_rem = datos["p_f"], datos["p_rem"]
    assert p_f.shape == (N_A5, 90) and p_rem.shape == (N_REM_A5, 90), f"A5 incompleta: {p_f.shape}, {p_rem.shape}"
    filas = []
    for nombre, p in (("F", p_f), ("remuestreo", p_rem)):
        filas.append({"ruta": nombre, "replicas": p.shape[0], "pruebas": p.size,
                      **{f"frac_p<={u}": float(np.mean(p <= u)) for u in (0.05, 0.01, 0.001)}})
    tabla = pd.DataFrame(filas)
    por_par = np.mean(p_f <= 0.05, axis=0)
    fuera = int(np.sum((por_par < INTERVALO_A5[0]) | (por_par > INTERVALO_A5[1])))
    ok_a = fuera == 0
    por_replica = np.mean(p_f <= 0.05, axis=1)
    media = float(por_replica.mean())
    ee = float(por_replica.std(ddof=1) / np.sqrt(N_A5))
    ok_b = abs(media - 0.05) < Z_A5 * ee
    razon = float(np.median(p_rem / p_f[:N_REM_A5]))
    pares = list(itertools.permutations(leer("topic_series_volumen.csv").columns, 2))
    tabla_par = pd.DataFrame({"par": [f"{o}->{d}" for o, d in pares], "frac_pF<=0.05": por_par,
                              "dentro": (por_par >= INTERVALO_A5[0]) & (por_par <= INTERVALO_A5[1])})
    tabla_crit = pd.DataFrame([
        {"criterio": f"(a) las 90 fracciones por par en [{INTERVALO_A5[0]}, {INTERVALO_A5[1]}]",
         "valor": f"min {por_par.min():.4f}, max {por_par.max():.4f}, fuera {fuera} de 90", "pasa": ok_a},
        {"criterio": "(b) |fracción media - 0.05| < 3 errores estándar entre réplicas",
         "valor": f"media {media:.5f}, ee {ee:.5f}, |dif| {abs(media - 0.05):.5f}, 3 ee {Z_A5 * ee:.5f}", "pasa": ok_b},
        {"criterio": "remuestreo (informativo): mediana de p_remuestreo / p_F, 300 x 90",
         "valor": f"{razon:.4f}", "pasa": None}])
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "A5_fracciones.csv", index=False)
    tabla_par.to_csv(SALIDA / "A5_por_par.csv", index=False)
    tabla_crit.to_csv(SALIDA / "A5_criterios.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(tabla.to_string(index=False))
        print()
        print(tabla_crit.to_string(index=False))
    ok = ok_a and ok_b
    print(f"A5 (ruta F): {marca(ok)}\n")
    return ok


# ---------------------------------------------------------------------------
# A6
# ---------------------------------------------------------------------------

def betas_a6() -> pd.DataFrame:
    """Beta calibrado por K y efecto (topic_0 -> topic_1), y su verificación re-simulada."""
    ruta = SALIDA / "A6_betas.csv"
    if ruta.exists():
        return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")
    real = leer("topic_series_volumen.csv")
    filas = []
    for K in sorted({K for K, _, _ in CELDAS_A6}):
        c, a, sigma = auditoria.parametros_var(real.iloc[:, :K])
        for e in EFECTOS_A6:
            clave = ("auditoria_A6_calibracion", K, e)
            cal = auditoria.calibrar_beta(c, a, sigma, e, clave)
            X = auditoria.simular_var(c, a, sigma, auditoria.T_LARGA, granger.generador(*clave), 0, 1, cal["beta"])
            valor = planted.efecto_poblacional(X, 0, 1)
            filas.append({"K": K, "efecto": e, **cal, "efecto_resimulado": valor,
                          "error_relativo_resimulado": abs(valor - float(e)) / float(e)})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def replica_a6(K: int, T: int, e: str, i: int, beta: float, c: np.ndarray, a: np.ndarray, sigma: np.ndarray,
               columnas: list[str], con_remuestreo: bool) -> tuple[np.ndarray, np.ndarray | None]:
    X = auditoria.simular_var(c, a, sigma, T, granger.generador("auditoria_A6_generacion", K, T, e, i), 0, 1, beta)
    pares = list(itertools.permutations(range(K), 2))
    F, p_f = auditoria.p_ruta_f(X, pares)
    if not con_remuestreo:
        return p_f, None
    claves = [("auditoria_A6_remuestreo", K, T, e, i, columnas[o], columnas[d]) for o, d in pares]
    return p_f, auditoria.p_ruta_remuestreo(X, pares, F, B, claves)


def correr_a6(medir: int | None) -> None:
    real = leer("topic_series_volumen.csv")
    columnas = list(real.columns)
    t0 = time.perf_counter()
    betas = betas_a6()
    print(f"A6: betas calibrados en {time.perf_counter() - t0:.1f} s")
    print(betas.to_string(index=False))
    total = 0.0
    for K, T, e in CELDAS_A6:
        c, a, sigma = auditoria.parametros_var(real.iloc[:, :K])
        beta = float(betas[(betas["K"] == K) & (betas["efecto"] == e)]["beta"].iloc[0])
        if medir:
            t_f, t_r = [], []
            for i in range(medir):
                t1 = time.perf_counter()
                replica_a6(K, T, e, i, beta, c, a, sigma, columnas, False)
                t_f.append(time.perf_counter() - t1)
                t1 = time.perf_counter()
                replica_a6(K, T, e, i, beta, c, a, sigma, columnas, True)
                t_r.append(time.perf_counter() - t1 - t_f[-1])
            celda = np.mean(t_f) * N_A6 + np.mean(t_r) * N_REM_A6
            total += celda
            print(f"  A6 K={K} T={T} e={e}: ruta F {np.mean(t_f):.3f} s x {N_A6}, remuestreo "
                  f"{np.mean(t_r):.3f} s x {N_REM_A6}; celda {celda / 60:.1f} min", flush=True)
            continue
        ruta = SALIDA / f"A6_K{K}_T{T}_e{e}.npz"
        datos = cargar(ruta)
        p_f = list(datos.get("p_f", []))
        p_rem = list(datos.get("p_rem", []))
        print(f"A6 K={K} T={T} e={e}: {len(p_f)} réplicas ya guardadas, faltan {N_A6 - len(p_f)}", flush=True)
        for i in range(len(p_f), N_A6):
            pf, pr = replica_a6(K, T, e, i, beta, c, a, sigma, columnas, i < N_REM_A6)
            p_f.append(pf)
            if pr is not None:
                p_rem.append(pr)
            if (i + 1) % CADA == 0:
                guardar(ruta, {"p_f": np.array(p_f), "p_rem": np.array(p_rem).reshape(-1, len(pf))})
                print(f"  A6 K={K} T={T} e={e}: {i + 1}/{N_A6} guardadas", flush=True)
    if medir:
        print(f"A6: total {total / 60:.1f} min")


def resumen_a6() -> bool:
    print("=== A6. VAR gaussiano con enlace topic_0 -> topic_1: poder según T ===")
    betas = betas_a6()
    print(betas.to_string(index=False))
    filas = []
    for K, T, e in CELDAS_A6:
        datos = cargar(SALIDA / f"A6_K{K}_T{T}_e{e}.npz")
        p_f, p_rem = datos["p_f"], datos["p_rem"]
        m = K * (K - 1)
        assert p_f.shape == (N_A6, m) and p_rem.shape == (N_REM_A6, m), f"A6 K={K} T={T} e={e} incompleta"
        j = list(itertools.permutations(range(K), 2)).index((0, 1))
        det_f = float(np.mean([granger.benjamini_hochberg(p, Q, m)[j] for p in p_f]))
        det_r = float(np.mean([granger.benjamini_hochberg(p, Q, m)[j] for p in p_rem]))
        n_obs, params = T - 1, K + 1
        umbral = stats.f.isf(0.05 / m, 1, n_obs - params)
        poder = float(stats.ncf.sf(umbral, 1, n_obs - params, n_obs * (np.exp(float(e)) - 1)))
        filas.append({"K": K, "T": T, "efecto": e, "m": m, "n_obs": n_obs, "replicas_F": len(p_f),
                      "deteccion_BH_F": det_f, "replicas_remuestreo": len(p_rem), "deteccion_BH_remuestreo": det_r,
                      "poder_teorico": poder, "dif_F_menos_teorico": det_f - poder})
    tabla = pd.DataFrame(filas)
    # (i) no decrece al aumentar T (K = 10, e fijo) ni al aumentar e (K y T fijos), margen 0.03.
    det = {(r.K, r.T, r.efecto): r.deteccion_BH_F for r in tabla.itertuples(index=False)}
    violaciones = []
    for e in EFECTOS_A6:
        for T1, T2 in ((23, 46), (46, 92)):
            if det[(10, T2, e)] < det[(10, T1, e)] - MARGEN_A6:
                violaciones.append(f"K=10 e={e}: T={T1} {det[(10, T1, e)]:.3f} > T={T2} {det[(10, T2, e)]:.3f}")
    for K, T in ((10, 23), (10, 46), (10, 92), (5, 23)):
        for e1, e2 in zip(EFECTOS_A6, EFECTOS_A6[1:]):
            if det[(K, T, e2)] < det[(K, T, e1)] - MARGEN_A6:
                violaciones.append(f"K={K} T={T}: e={e1} {det[(K, T, e1)]:.3f} > e={e2} {det[(K, T, e2)]:.3f}")
    ok_i = not violaciones
    principal = {e: det[(10, T_PRINCIPAL_A6, e)] for e in ("0.8", "1.6")}
    ok_ii = all(v >= DETECCION_MIN_A6 for v in principal.values())
    tabla_crit = pd.DataFrame([
        {"criterio": "(i) detección F no decrece al aumentar T ni e (margen 0.03)",
         "valor": "; ".join(violaciones) if violaciones else "sin violaciones", "pasa": ok_i},
        {"criterio": "(ii) principal: T = 92, detección F >= 0.80 para e = 0.8 y 1.6",
         "valor": ", ".join(f"e={e}: {v:.3f}" for e, v in principal.items()), "pasa": ok_ii},
        {"criterio": "(iii) informativa: diferencia con el poder teórico por celda",
         "valor": f"rango [{tabla['dif_F_menos_teorico'].min():.3f}, {tabla['dif_F_menos_teorico'].max():.3f}]",
         "pasa": None}])
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "A6_celdas.csv", index=False)
    tabla_crit.to_csv(SALIDA / "A6_criterios.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(tabla.to_string(index=False))
        print()
        print(tabla_crit.to_string(index=False))
    ok = ok_i and ok_ii
    print(f"A6: {marca(ok)}\n")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pruebas", nargs="+", choices=["A1", "A2", "A3", "A4", "A5", "A6"])
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--medir", type=int, metavar="N", help="A5/A6: tiempo con N réplicas por celda, sin guardarlas")
    grupo.add_argument("--resumen", action="store_true", help="A5/A6: solo el reporte desde lo guardado")
    args = parser.parse_args()

    rapidas = {"A1": prueba_a1, "A2": prueba_a2, "A3": prueba_a3, "A4": prueba_a4}
    resultados: dict[str, bool] = {}
    for prueba in args.pruebas:
        if prueba in rapidas:
            resultados[prueba] = rapidas[prueba]()
    largas = [p for p in args.pruebas if p in ("A5", "A6")]
    if largas and not args.resumen:
        if not verificar_claves():
            print("Hay claves repetidas: se detiene aquí.")
            return
        for prueba in largas:
            (correr_a5 if prueba == "A5" else correr_a6)(args.medir)
    if largas and not args.medir:
        for prueba in largas:
            resultados[prueba] = (resumen_a5 if prueba == "A5" else resumen_a6)()
    if resultados:
        print("Resultado por prueba: " + ", ".join(f"{p}: {marca(ok)}" for p, ok in resultados.items()))


if __name__ == "__main__":
    main()
