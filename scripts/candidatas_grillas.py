"""
candidatas_grillas.py

Semana 8, paso 2 (Bitacora_Semana8.md, sección 21 y su nota): evaluación de
configuraciones candidatas (grilla, K) para bpoil con su propia dinámica.

Candidatas: (2 días, 5), (2 días, 8), (2 días, 10), (3 días, 5) y (semanal,
5), con el corpus viejo, la ventana de análisis, MIN_DOCS = 5 y la mayor
racha de ventanas válidas (las ventanas de scripts/factibilidad_grillas.py).
Series: log1p de la suma de theta por ventana de la racha (modelos K5, K8 y
el vigente); 4 versiones con candidatas.construir_versiones_k (residualizada
sin topic_4 en K5, topic_3 en K8 y topic_5 en K10).

Controles: C0 propio (de las masas de la réplica), C1 sin control, C2
log1p del conteo real de documentos de cada ventana de la racha, fijo.

Por candidata y control: 300 sustitutos AR diagonal con efecto 0.0 y 100
con efectos 0.4 y 0.8 (par plantado entre los pares sin la columna
descartada, beta calibrado por par y candidata), remuestreo salvaje
B = 9999, Benjamini y Hochberg a q = 0.05 por versión (m = número de pares de
la versión) y regla de 2 versiones. Las 3 variantes de control analizan las
mismas series (clave de generación sin el control).

Salidas en data/candidatas/: betas.csv, un CSV por unidad (candidata,
control, efecto) con una fila por réplica y su .npz con efecto, F, p_F y
p_rem de todos los pares (punto de control cada 25 réplicas), y las tablas
de --resumen.

Uso:
  python scripts/candidatas_grillas.py --verificar   verificaciones a) a e)
  python scripts/candidatas_grillas.py --medir 3     tiempo; no guarda réplicas
  python scripts/candidatas_grillas.py               corrida completa (reanuda)
  python scripts/candidatas_grillas.py --resumen     tablas desde lo guardado
"""

import argparse
import itertools
import os
import runpy
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence import candidatas, diagnostico, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import T17_DIR
from causal_coherence.topic_model import LDA_CONFIG, MODEL_PATH, training_documents

# candidata: (nombre de grilla en factibilidad_grillas, frecuencia, días por ventana, etiqueta derecha, K)
CANDIDATAS: dict[str, tuple[str, str, int, bool, int]] = {
    "2D_K5": ("cada 2 días", "2D", 2, False, 5),
    "2D_K8": ("cada 2 días", "2D", 2, False, 8),
    "2D_K10": ("cada 2 días", "2D", 2, False, 10),
    "3D_K5": ("cada 3 días", "3D", 3, False, 5),
    "semanal_K5": ("semanal", "W", 7, True, 5),
}
MODELOS = {5: OUTPUT_DIR / "roll_lda_bpoil_K5.pickle", 8: OUTPUT_DIR / "roll_lda_bpoil_K8.pickle", 10: MODEL_PATH}
DESCARTE = {5: "topic_4", 8: "topic_3", 10: "topic_5"}
CONTROLES = ("C0", "C1", "C2")
EFECTOS = {"0.0": 300, "0.4": 100, "0.8": 100}
LARGO_RACHA = {"cada 2 días": 53, "cada 3 días": 36, "semanal": 23}
B = 9999
Q = 0.05
CADA = 25
MIN_VERSIONES = 2
REZAGOS_ACF = 10
DETECCION_MIN = 0.80
UMBRAL_FP = stats.binom.isf(0.05 / 15, 300, 0.05)
SALIDA = OUTPUT_DIR / "candidatas"
RUTA_BETAS = SALIDA / "betas.csv"

_grillas = runpy.run_path(str(Path(__file__).resolve().parent / "factibilidad_grillas.py"))
ventanas_recortadas: Callable = _grillas["ventanas_recortadas"]
rachas: Callable = _grillas["rachas"]
fechas_bpoil: Callable = _grillas["fechas_bpoil"]


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


# ---------------------------------------------------------------------------
# Series reales de cada candidata
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def theta_K(K: int) -> np.ndarray:
    roll = RollingLDA(**{**LDA_CONFIG, "K": K})
    roll.load(str(MODELOS[K]))
    theta = roll.get_document_topic_matrix()
    return theta / theta.sum(axis=1, keepdims=True)


@lru_cache(maxsize=None)
def fechas_entrenamiento() -> pd.Series:
    return training_documents()["date"]


def tramo_racha(grilla: str) -> pd.DataFrame:
    """Ventanas de la mayor racha (corpus viejo, MIN_DOCS = 5), como en factibilidad_grillas.py."""
    v = ventanas_recortadas(fechas_bpoil(T17_DIR), grilla, 5)
    pos, largo = max(rachas(v["valida"].to_numpy()), key=lambda r: r[1])
    return v.iloc[pos:pos + largo].reset_index(drop=True)


@lru_cache(maxsize=None)
def serie_real(candidata: str) -> tuple[pd.DataFrame, pd.Series]:
    """(log1p de las masas por ventana de la racha, conteo real de documentos n_t)."""
    grilla, freq, _, derecha, K = CANDIDATAS[candidata]
    masa, conteo = candidatas.masas_por_ventana(theta_K(K), fechas_entrenamiento(), freq)
    tramo = tramo_racha(grilla)
    etiquetas = tramo["fin"] if derecha else tramo["inicio"]
    masa, conteo = masa.loc[etiquetas], conteo.loc[etiquetas]
    assert (conteo.to_numpy() == tramo["documentos"].to_numpy()).all(), f"{candidata}: conteos distintos"
    return np.log1p(masa), conteo.rename("n_t")


def pares_por_version(columnas: list[str], descartada: str) -> dict[str, list[tuple[str, str]]]:
    sin = [c for c in columnas if c != descartada]
    return {v: list(itertools.permutations(sin if v == "residualizada" else columnas, 2)) for v in surrogates.VERSIONES}


# ---------------------------------------------------------------------------
# Betas
# ---------------------------------------------------------------------------

def betas() -> pd.DataFrame:
    if RUTA_BETAS.exists():
        return leer_csv(RUTA_BETAS)
    filas = []
    for cand, (_, _, dias, _, K) in CANDIDATAS.items():
        real, _ = serie_real(cand)
        columnas = list(real.columns)
        _, a, _ = surrogates.ajustar_ar1(real)
        for e in ("0.4", "0.8"):
            for o, d in itertools.permutations([c for c in columnas if c != DESCARTE[K]], 2):
                i_A, i_B = columnas.index(o), columnas.index(d)
                cal = candidatas.calibrar_beta(real, e, i_A, i_B, candidatas.calentamiento(dias),
                                               ("cand_calibracion", cand, e, o, d))
                M = np.diag(a)
                M[i_B, i_A] = cal["beta"]
                filas.append({"candidata": cand, "efecto": e, "origen": o, "destino": d, **cal,
                              "radio_espectral": float(np.max(np.abs(np.linalg.eigvals(M))))})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_BETAS, index=False)
    return leer_csv(RUTA_BETAS)


# ---------------------------------------------------------------------------
# Réplicas
# ---------------------------------------------------------------------------

def generar(cand: str, efecto: str, indice: int, tabla_betas: pd.DataFrame) -> tuple[pd.DataFrame, int, tuple[str, str] | None]:
    """Sustituto con piso en 0; el par plantado (si efecto > 0) se sortea entre los pares sin la columna descartada."""
    _, _, dias, _, K = CANDIDATAS[cand]
    real, _ = serie_real(cand)
    columnas = list(real.columns)
    c, a, residuos = surrogates.ajustar_ar1(real)
    par: tuple[str, str] | None = None
    i_A, i_B, beta = 0, 1, 0.0
    if float(efecto) > 0:
        plantables = list(itertools.permutations([col for col in columnas if col != DESCARTE[K]], 2))
        par = plantables[int(granger.generador("cand_par", cand, efecto, str(indice)).integers(0, len(plantables)))]
        fila = tabla_betas[(tabla_betas["candidata"] == cand) & (tabla_betas["efecto"] == efecto)
                           & (tabla_betas["origen"] == par[0]) & (tabla_betas["destino"] == par[1])]
        beta = float(fila["beta"].iloc[0])
        i_A, i_B = columnas.index(par[0]), columnas.index(par[1])
    X = candidatas.simular(c, a, residuos, real.to_numpy().mean(axis=0), len(real), candidatas.calentamiento(dias),
                           i_A, i_B, beta, planted.media_proceso(c, a, i_A),
                           granger.generador("cand_generacion", cand, efecto, str(indice)))
    X, corregidas = surrogates.aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas, par


def procesar(cand: str, control: str, efecto: str, indice: int, tabla_betas: pd.DataFrame
             ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    K = CANDIDATAS[cand][4]
    real, n_t = serie_real(cand)
    pares = pares_por_version(list(real.columns), DESCARTE[K])
    x, corregidas, par = generar(cand, efecto, indice, tabla_betas)
    versiones = candidatas.construir_versiones_k(x, DESCARTE[K])
    c2 = candidatas.versiones_control_fijo(np.log1p(n_t).rename("control_c2"))
    fila: dict[str, object] = {"candidata": cand, "control": control, "efecto": efecto, "indice": indice,
                               "origen": par[0] if par else "", "destino": par[1] if par else "",
                               "celdas_piso": corregidas}
    arreglos: dict[str, np.ndarray] = {}
    veces: dict[tuple[str, str], int] = {}
    n_versiones = 0
    for v, (serie, _, propio) in versiones.items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[v], f"{v}: pares distintos"
        m = len(pares[v])
        val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_F", "p_rem")}
        disenos = [diagnostico.disenos(serie, propio[d] if control == "C0" else c2[v], o, d, control != "C1")
                   for o, d in pares[v]]
        no_estimable = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for X_f, _, _ in disenos))
        if not no_estimable:
            for j, ((o, d), (X_f, X_r, y)) in enumerate(zip(pares[v], disenos)):
                obs = diagnostico.prueba(X_f, X_r, y)
                val["efecto"][j], val["F"][j], val["p_F"][j] = obs["efecto"], obs["F"], obs["p_teo"]
                rng = granger.generador("cand_remuestreo", cand, control, efecto, str(indice), v, o, d)
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


def unidades() -> list[tuple[str, str, str]]:
    return [(cand, control, e) for cand in CANDIDATAS for control in CONTROLES for e in EFECTOS]


def rutas(cand: str, control: str, e: str) -> tuple[Path, Path]:
    return SALIDA / f"{cand}_{control}_e{e}.csv", SALIDA / f"{cand}_{control}_e{e}.npz"


def correr() -> None:
    tabla_betas = betas()
    t0 = time.perf_counter()
    for cand, control, e in unidades():
        ruta_csv, ruta_npz = rutas(cand, control, e)
        filas: list[dict[str, object]] = []
        arreglos: dict[str, list[np.ndarray]] = {}
        if ruta_csv.exists():
            tabla = leer_csv(ruta_csv)
            assert tabla["indice"].tolist() == list(range(len(tabla))), f"{ruta_csv.name}: índices no consecutivos"
            filas = tabla.to_dict("records")
            with np.load(ruta_npz) as z:
                arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
        n = EFECTOS[e]
        if len(filas) < n:
            print(f"{cand} {control} efecto {e}: {len(filas)} guardadas, faltan {n - len(filas)}", flush=True)
        for i in range(len(filas), n):
            fila, arr = procesar(cand, control, e, i, tabla_betas)
            filas.append(fila)
            for k, val in arr.items():
                arreglos.setdefault(k, []).append(val)
            if (i + 1) % CADA == 0 or i + 1 == n:
                SALIDA.mkdir(parents=True, exist_ok=True)
                tmp = ruta_npz.with_name(ruta_npz.stem + ".tmp.npz")
                np.savez(tmp, **{k: np.array(v) for k, v in arreglos.items()})
                os.replace(tmp, ruta_npz)
                tmp = ruta_csv.with_name(ruta_csv.stem + ".tmp.csv")
                pd.DataFrame(filas).to_csv(tmp, index=False)
                os.replace(tmp, ruta_csv)
        print(f"  {cand} {control} efecto {e}: {n}/{n} ({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def medir(n: int) -> None:
    tabla_betas = betas()
    total = 0.0
    por_cand: dict[str, float] = {}
    for cand, control, e in unidades():
        tiempos = []
        for i in range(n):
            t0 = time.perf_counter()
            procesar(cand, control, e, i, tabla_betas)
            tiempos.append(time.perf_counter() - t0)
        unidad = float(np.mean(tiempos)) * EFECTOS[e]
        total += unidad
        por_cand[f"{cand} {control}"] = por_cand.get(f"{cand} {control}", 0.0) + unidad
        print(f"  {cand} {control} efecto {e}: {np.mean(tiempos):.3f} s por réplica x {EFECTOS[e]} = {unidad / 60:.1f} min",
              flush=True)
    print("\nPor candidata y control:")
    for clave, s in por_cand.items():
        print(f"  {clave}: {s / 60:.1f} min")
    print(f"Total ({sum(EFECTOS.values()) * len(CANDIDATAS) * len(CONTROLES)} réplicas): {total / 60:.1f} min")


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------

def claves_previas(columnas10: list[str]) -> list[tuple[str | int, ...]]:
    """Todas las claves usadas hasta el diagnóstico factorial (sección 18)."""
    largo = runpy.run_path(str(Path(__file__).resolve().parent / "run_control_positivo_largo.py"))
    ids = largo["identidades_previas"](columnas10)
    pares = largo["pares_por_version"](columnas10)
    for T_, e in [(46, "0.0"), (46, "0.4"), (46, "0.8"), (92, "0.0"), (92, "0.4"), (92, "0.8")]:
        for i in range(100):
            ids += [("largo_par", str(T_), e, str(i)), ("largo_generacion", str(T_), e, str(i))]
            for v, lista in pares.items():
                ids += [("largo_remuestreo", str(T_), e, str(i), v, o, d) for o, d in lista]
    ids.append(("fp_verificacion", "I1"))
    for i in range(500):
        ids.append(("fp_generacion", "I1", "92", str(i)))
        if i >= 100:
            ids += [("largo_par", "92", "0.0", str(i)), ("largo_generacion", "92", "0.0", str(i))]
    for variante in ("C0-P0-I0", "C1-P0-I0", "C0-P1-I0", "C0-P0-I1", "C1-P1-I1"):
        for i in range(100):
            for v, lista in pares.items():
                ids += [("fp_remuestreo", variante, "92", str(i), v, o, d) for o, d in lista]
    return ids


def verificar() -> bool:
    res: dict[str, bool] = {}

    print("=== a) Semanal con K = 10 contra data/topic_series_volumen.csv y data/control_volumen.csv ===")
    masa, conteo = candidatas.masas_por_ventana(theta_K(10), fechas_entrenamiento(), "W")
    ref = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    ref_c = pd.read_csv(OUTPUT_DIR / "control_volumen.csv", index_col=0, parse_dates=True).iloc[:, 0]
    serie, n_t = serie_real("semanal_K5")  # solo para tomar las ventanas de la racha semanal
    nueva = np.log1p(masa.loc[serie.index])
    dif_s = float(np.abs(nueva.to_numpy() - ref.to_numpy()).max())
    dif_c = float(np.abs(np.log1p(conteo.loc[serie.index]).to_numpy() - ref_c.to_numpy()).max())
    res["a"] = nueva.index.equals(ref.index) and list(nueva.columns) == list(ref.columns) and dif_s <= 1e-9 and dif_c <= 1e-9
    print(f"  serie: max |dif| = {dif_s:.3e}; control: max |dif| = {dif_c:.3e}; mismas fechas y columnas: "
          f"{'OK' if res['a'] else 'FALLA'}")

    print("\n=== b) Suma de masas contra n_t y largo de la racha ===")
    ok_b = True
    for cand, (grilla, *_rest) in CANDIDATAS.items():
        real, n_t = serie_real(cand)
        dif = float(np.abs(np.expm1(real).sum(axis=1).to_numpy() - n_t.to_numpy()).max())
        ok = dif <= 1e-9 and len(real) == LARGO_RACHA[grilla]
        ok_b &= ok
        print(f"  {cand}: {len(real)} ventanas (esperado {LARGO_RACHA[grilla]}), {real.index[0].date()} a "
              f"{real.index[-1].date()}, max |suma de masas - n_t| = {dif:.3e}: {'OK' if ok else 'FALLA'}")
    res["b"] = ok_b

    print("\n=== Equivalencias con los módulos existentes ===")
    real10 = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    v_mod = candidatas.construir_versiones_k(real10, "topic_5")
    v_ref = surrogates.construir_versiones(real10)
    eq_v = all(v_mod[v][k].equals(v_ref[v][k]) for v in surrogates.VERSIONES for k in range(3))
    c, a, residuos = surrogates.ajustar_ar1(real10)
    x0 = real10.to_numpy().mean(axis=0)
    s1 = candidatas.simular(c, a, residuos, x0, 23, surrogates.CALENTAMIENTO, 2, 5, 1.3, 1.1, granger.generador("x", 1))
    s2 = planted.simular_cruzado(c, a, residuos, x0, 23, 2, 5, 1.3, 1.1, granger.generador("x", 1))
    eq_s = bool(np.array_equal(s1, s2))
    print(f"  construir_versiones_k(K = 10, topic_5) igual a surrogates.construir_versiones: {'OK' if eq_v else 'FALLA'}")
    print(f"  simular con calentamiento 100 igual a planted.simular_cruzado: {'OK' if eq_s else 'FALLA'}")
    res["equivalencias"] = eq_v and eq_s

    print("\n=== d) Beta calibrados ===")
    tabla_betas = betas()
    ok_d = True
    for fila in tabla_betas.itertuples(index=False):
        _, _, dias, _, K = CANDIDATAS[fila.candidata]
        real, _ = serie_real(fila.candidata)
        columnas = list(real.columns)
        c, a, residuos = surrogates.ajustar_ar1(real)
        i_A, i_B = columnas.index(fila.origen), columnas.index(fila.destino)
        X = candidatas.simular(c, a, residuos, real.to_numpy().mean(axis=0), planted.T_LARGA,
                               candidatas.calentamiento(dias), i_A, i_B, fila.beta, planted.media_proceso(c, a, i_A),
                               granger.generador("cand_calibracion", fila.candidata, fila.efecto, fila.origen, fila.destino))
        valor = planted.efecto_poblacional(X, i_A, i_B)
        ok_d &= abs(valor - float(fila.efecto)) / float(fila.efecto) < planted.ERROR_RELATIVO_MAX and fila.radio_espectral < 1
    res["d"] = ok_d
    resumen_b = tabla_betas.groupby(["candidata", "efecto"], sort=False).agg(
        pares=("beta", "size"), beta_min=("beta", "min"), beta_mediana=("beta", "median"), beta_max=("beta", "max"),
        error_max=("error_relativo", "max"), radio_max=("radio_espectral", "max"))
    print(resumen_b.to_string())
    print(f"  error relativo re-simulado < 1% y radio espectral < 1 en las {len(tabla_betas)} filas: {'OK' if ok_d else 'FALLA'}")

    print("\n=== c) Misma réplica y variante dos veces ===")
    ok_c = True
    for cand, control, e in (("2D_K8", "C0", "0.8"), ("3D_K5", "C2", "0.0"), ("semanal_K5", "C1", "0.4")):
        (f1, a1), (f2, a2) = (procesar(cand, control, e, 0, tabla_betas) for _ in range(2))
        ok = f1 == f2 and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
        ok_c &= ok
        print(f"  {cand} {control} efecto {e} réplica 0: {'OK' if ok else 'FALLA'}")
    res["c"] = ok_c

    print("\n=== e) Unicidad de las claves de semilla ===")
    columnas10 = list(real10.columns)
    previas = claves_previas(columnas10)
    nuevas: list[tuple[str | int, ...]] = []
    for fila in tabla_betas.itertuples(index=False):
        nuevas.append(("cand_calibracion", fila.candidata, fila.efecto, fila.origen, fila.destino))
    for cand, control, e in unidades():
        K = CANDIDATAS[cand][4]
        real, _ = serie_real(cand)
        pares = pares_por_version(list(real.columns), DESCARTE[K])
        for i in range(EFECTOS[e]):
            if control == "C0":  # generación y par no dependen del control
                nuevas.append(("cand_generacion", cand, e, str(i)))
                if float(e) > 0:
                    nuevas.append(("cand_par", cand, e, str(i)))
            for v, lista in pares.items():
                nuevas += [("cand_remuestreo", cand, control, e, str(i), v, o, d) for o, d in lista]
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    res["e"] = distintas == len(todas)
    print(f"  {len(todas)} ({len(previas)} previas + {len(nuevas)} nuevas), distintas: {distintas}: "
          f"{'OK' if res['e'] else 'FALLA'}")

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def resumen() -> None:
    salidas: dict[str, pd.DataFrame] = {}
    filas = []
    for cand in CANDIDATAS:
        _, n_t = serie_real(cand)
        s = np.log1p(n_t)
        filas.append({"candidata": cand, "ventanas": len(s),
                      **{f"acf_{k}": float(s.autocorr(lag=k)) for k in range(1, REZAGOS_ACF + 1)}})
    salidas["a_ciclo_semanal"] = pd.DataFrame(filas)

    cal, pos, apt = [], [], []
    for cand in CANDIDATAS:
        for control in CONTROLES:
            t0_ = leer_csv(rutas(cand, control, "0.0")[0])
            assert len(t0_) == EFECTOS["0.0"], f"{cand} {control} 0.0 incompleta"
            for v in surrogates.VERSIONES:
                cal.append({"candidata": cand, "control": control, "version": v, "n_replicas": len(t0_),
                            "n_no_estimables": int(t0_[f"no_estimable_{v}"].sum()),
                            "replicas_con_rechazo": int((t0_[f"rechazos_{v}"] >= 1).sum()),
                            "fraccion_con_rechazo": float((t0_[f"rechazos_{v}"] >= 1).mean())})
            cal.append({"candidata": cand, "control": control, "version": "regla_2_versiones", "n_replicas": len(t0_),
                        "n_no_estimables": np.nan, "replicas_con_rechazo": int((t0_["enlaces_regla"] >= 1).sum()),
                        "fraccion_con_rechazo": float((t0_["enlaces_regla"] >= 1).mean())})
            det = {}
            for e in ("0.4", "0.8"):
                t = leer_csv(rutas(cand, control, e)[0])
                assert len(t) == EFECTOS[e], f"{cand} {control} {e} incompleta"
                for v in surrogates.VERSIONES:
                    pos.append({"candidata": cand, "control": control, "efecto": e, "version": v,
                                "deteccion_BH": float(t[f"rechaza_plantado_{v}"].mean()),
                                "efecto_observado_medio": float(t[f"efecto_plantado_{v}"].mean()),
                                "n_no_estimables": int(t[f"no_estimable_{v}"].sum())})
                det[e] = float(t["detectado_regla"].mean())
                pos.append({"candidata": cand, "control": control, "efecto": e, "version": "regla_2_versiones",
                            "deteccion_BH": det[e], "efecto_observado_medio": np.nan, "n_no_estimables": np.nan})
            fp = {v: int((t0_[f"rechazos_{v}"] >= 1).sum()) for v in ("original", "detendenciada")}
            apt.append({"candidata": cand, "K": CANDIDATAS[cand][4], "control": control,
                        "deteccion_regla_0.8": det["0.8"], "deteccion_regla_0.4": det["0.4"],
                        "fp_original": fp["original"], "fp_detendenciada": fp["detendenciada"], "umbral_fp": UMBRAL_FP,
                        "apta": bool(det["0.8"] >= DETECCION_MIN and all(c <= UMBRAL_FP for c in fp.values()))})
    salidas["c_calibracion"] = pd.DataFrame(cal)
    salidas["d_control_positivo"] = pd.DataFrame(pos)
    salidas["aptitud"] = pd.DataFrame(apt)
    salidas["betas_resumen"] = betas().groupby(["candidata", "efecto"], sort=False).agg(
        pares=("beta", "size"), beta_min=("beta", "min"), beta_mediana=("beta", "median"), beta_max=("beta", "max"),
        error_max=("error_relativo", "max"), radio_max=("radio_espectral", "max")).reset_index()
    SALIDA.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 300):
        for nombre, tabla in salidas.items():
            tabla.to_csv(SALIDA / f"resumen_{nombre}.csv", index=False)
            print(f"\n=== {nombre} (resumen_{nombre}.csv) ===")
            print(tabla.to_string(index=False))
    aptas = salidas["aptitud"][salidas["aptitud"]["apta"]]
    print(f"\nUmbral de falsos positivos: binom.isf(0.05 / 15, 300, 0.05) = {UMBRAL_FP}")
    print("Combinaciones APTAS: " + (", ".join(f"{r.candidata} {r.control}" for r in aptas.itertuples()) or "ninguna"))


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true")
    grupo.add_argument("--medir", type=int, metavar="N")
    grupo.add_argument("--resumen", action="store_true")
    args = parser.parse_args()
    if args.verificar:
        verificar()
    elif args.medir:
        medir(args.medir)
    elif args.resumen:
        resumen()
    else:
        correr()


if __name__ == "__main__":
    main()
