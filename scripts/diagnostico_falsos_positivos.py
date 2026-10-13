"""
diagnostico_falsos_positivos.py

Semana 7, paso 11: diagnóstico factorial de falsos positivos del pipeline
completo (diseño en Bitacora_Semana7.md, sección 17 y su nota de las 22:17).

Factores, con efecto 0.0, T = 92 y K = 10:
  C: C0 control propio (actual), C1 sin control, C2 control exógeno fijo
     (log1p del conteo semanal real de bpoil, repetido en ciclo hasta T, que
     pasa por los mismos constructores de versiones que la serie).
  P: P0 piso en la serie de la regresión (actual), P1 serie sin piso (el
     control propio sale siempre de las masas con piso).
  I: I0 filas reales de residuos (claves del control positivo largo, T = "92",
     efecto "0.0"); I1 innovaciones gaussianas N(0, np.cov(residuos, ddof=1))
     con la clave ("fp_generacion", "I1", "92", índice).

Etapa 1: las 12 variantes, ruta F, 500 réplicas.
Etapa 2: C0-P0-I0, C1-P0-I0, C0-P1-I0, C0-P0-I1 y C1-P1-I1, remuestreo
  (B = 9999, clave ("fp_remuestreo", variante, "92", índice, versión, origen,
  destino)) y ruta F, 100 réplicas.
Etapa 3b: el pipeline del control positivo largo, con sus mismas claves, para
  (T = 92, efecto 0.4), (T = 92, efecto 0.8) y (T = 46, efecto 0.0), 100
  réplicas, guardando p, efecto y F por par.

Salidas en data/diagnostico_fp/: por unidad (variante de una etapa o celda de
la 3b), un CSV por réplica y un .npz con efecto, F y p de todos los pares
(punto de control cada 25 réplicas), y las tablas de --resumen.

Uso:
  python scripts/diagnostico_falsos_positivos.py --verificar   verificaciones a) a g)
  python scripts/diagnostico_falsos_positivos.py --medir 3     tiempo; no guarda nada
  python scripts/diagnostico_falsos_positivos.py               etapas 1, 2 y 3b (reanuda)
  python scripts/diagnostico_falsos_positivos.py --resumen     tablas desde lo guardado
"""

import argparse
import itertools
import os
import runpy
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence import diagnostico, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

T = 92
VARIANTES = [f"C{c}-P{p}-I{i}" for c in range(3) for p in range(2) for i in range(2)]
ETAPA2 = ["C0-P0-I0", "C1-P0-I0", "C0-P1-I0", "C0-P0-I1", "C1-P1-I1"]
CELDAS_3B = [(92, "0.4"), (92, "0.8"), (46, "0.0")]
N_ETAPA = {"etapa1": 500, "etapa2": 100, "etapa3b": 100}
CADA = 25
B = 9999
Q = 0.05
MIN_VERSIONES = 2
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
REDUCCION_MIN = 0.08
UMBRAL_B = stats.binom.isf(0.05 / 8, 100, 0.05)
TOL_COV = 0.02
SALIDA = OUTPUT_DIR / "diagnostico_fp"
RUTA_BETAS = OUTPUT_DIR / "control_positivo" / "betas.csv"
RUTA_LARGO = OUTPUT_DIR / "control_positivo_largo" / "control_positivo_largo.csv"

_largo = runpy.run_path(str(Path(__file__).resolve().parent / "run_control_positivo_largo.py"))
generar_replica_T: Callable = _largo["generar_replica_T"]
claves_largo: Callable = _largo["claves"]
identidades_previas: Callable = _largo["identidades_previas"]
pares_por_version: Callable = _largo["pares_por_version"]


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def leer_real() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)


def iguales(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------------------
# Generación de las series del factorial
# ---------------------------------------------------------------------------

def simular_gaussiano(c: np.ndarray, a: np.ndarray, sigma: np.ndarray, x0: np.ndarray, largo: int,
                      rng: np.random.Generator) -> np.ndarray:
    """AR diagonal con innovaciones N(0, sigma), calentamiento de surrogates.CALENTAMIENTO desde x0."""
    total = surrogates.CALENTAMIENTO + largo
    u = rng.standard_normal((total, len(c))) @ np.linalg.cholesky(sigma).T
    x = x0
    simulada = np.empty((total, len(c)))
    for t in range(total):
        x = c + a * x + u[t]
        simulada[t] = x
    return simulada[-largo:]


def serie_cruda(innovaciones: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str] | None]:
    """Serie de largo T sin piso. I0: los pasos de generar_replica_T (efecto 0.0, claves largo_*) sin el piso."""
    columnas = list(real.columns)
    c, a, residuos = surrogates.ajustar_ar1(real)
    x0 = real.to_numpy().mean(axis=0)
    if innovaciones == "I0":
        clave_par, clave_gen = claves_largo(T, "0.0", indice, False)
        pares = planted.pares_plantables(columnas)
        origen, destino = pares[int(granger.generador(*clave_par).integers(0, len(pares)))]
        fila = betas[(betas["efecto"] == "0.0") & (betas["origen"] == origen) & (betas["destino"] == destino)]
        beta = float(fila["beta"].iloc[0])
        i_A = columnas.index(origen)
        X = planted.simular_cruzado(c, a, residuos, x0, T, i_A, columnas.index(destino), beta,
                                    planted.media_proceso(c, a, i_A), granger.generador(*clave_gen))
        par: tuple[str, str] | None = (origen, destino)
    else:
        sigma = np.cov(residuos, rowvar=False, ddof=1)
        X = simular_gaussiano(c, a, sigma, x0, T, granger.generador("fp_generacion", "I1", str(T), str(indice)))
        par = None
    return pd.DataFrame(X, index=pd.RangeIndex(T), columns=real.columns), par


def control_c2(indice_filas: pd.Index) -> pd.DataFrame:
    """log1p del conteo semanal real (control_volumen.csv) repetido en ciclo hasta el largo pedido."""
    base = pd.read_csv(OUTPUT_DIR / "control_volumen.csv", index_col=0).iloc[:, 0].to_numpy()
    valores = np.tile(base, int(np.ceil(len(indice_filas) / len(base))))[:len(indice_filas)]
    return pd.DataFrame({"control_c2": valores}, index=indice_filas)


def versiones_c2(indice_filas: pd.Index) -> dict[str, pd.Series]:
    c2 = control_c2(indice_filas)
    return {"original": c2.iloc[:, 0], "diferenciada": surrogates.diferenciar(c2).iloc[:, 0],
            "detendenciada": surrogates.detendenciar(c2).iloc[:, 0], "residualizada": c2.iloc[:, 0]}


def armar_variante(variante: str, x_cruda: pd.DataFrame) -> tuple[dict[str, tuple[pd.DataFrame, Callable[[str], pd.Series]]], int, bool]:
    """Por versión: (serie de la regresión, control por destino); celdas corregidas; si lleva columna de control."""
    C, P = variante.split("-")[0], variante.split("-")[1]
    x_piso, corregidas = surrogates.aplicar_piso(x_cruda.to_numpy())
    x_piso = pd.DataFrame(x_piso, index=x_cruda.index, columns=x_cruda.columns)
    con = surrogates.construir_versiones(x_piso)
    series = {v: con[v][0] for v in surrogates.VERSIONES} if P == "P0" else diagnostico.series_versiones(x_cruda)
    c2 = versiones_c2(x_cruda.index)
    salida = {}
    for v in surrogates.VERSIONES:
        if C == "C0":
            salida[v] = (series[v], (lambda d, prop=con[v][2]: prop[d]))
        else:
            # C1 elimina la columna de control en diagnostico.disenos; el control pasado no se usa.
            salida[v] = (series[v], (lambda d, ctrl=c2[v]: ctrl))
    return salida, corregidas, C != "C1"


# ---------------------------------------------------------------------------
# Procesamiento de una réplica
# ---------------------------------------------------------------------------

def analizar(versiones: dict[str, tuple[pd.DataFrame, Callable[[str], pd.Series]]], con_control: bool,
             pares: dict[str, list[tuple[str, str]]], clave_rem: Callable[[str, str, str], tuple] | None
             ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Efecto, F, p_F (y p_rem si clave_rem) de todos los pares por versión; versión no estimable con NaN."""
    valores: dict[str, dict[str, np.ndarray]] = {}
    no_estimable: dict[str, int] = {}
    for v, (serie, control_de) in versiones.items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[v], f"{v}: pares distintos"
        m = len(pares[v])
        val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_F", "p_rem")}
        disenos = [diagnostico.disenos(serie, control_de(d), o, d, con_control) for o, d in pares[v]]
        no_estimable[v] = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for X_f, _, _ in disenos))
        if not no_estimable[v]:
            for j, ((o, d), (X_f, X_r, y)) in enumerate(zip(pares[v], disenos)):
                obs = diagnostico.prueba(X_f, X_r, y)
                val["efecto"][j], val["F"][j], val["p_F"][j] = obs["efecto"], obs["F"], obs["p_teo"]
                if clave_rem is not None:
                    val["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"],
                                                                 B, granger.generador(*clave_rem(v, o, d)))["p"]
        valores[v] = val
    return valores, no_estimable


def resumir_rechazos(valores: dict[str, dict[str, np.ndarray]], no_estimable: dict[str, int],
                     pares: dict[str, list[tuple[str, str]]], ruta: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Rechazos BH por versión con la ruta dada y enlaces con la regla de 2 versiones."""
    fila: dict[str, object] = {}
    rechazos: dict[str, np.ndarray] = {}
    veces: dict[tuple[str, str], int] = {}
    for v in surrogates.VERSIONES:
        p = valores[v][f"p_{ruta}"]
        r = np.zeros(len(pares[v]), dtype=bool) if no_estimable[v] else granger.benjamini_hochberg(p, Q, M_FAMILIA[v])
        rechazos[v] = r
        fila[f"rechazos_{ruta}_{v}"] = int(r.sum())
        for par, rr in zip(pares[v], r):
            veces[par] = veces.get(par, 0) + int(rr)
    fila[f"enlaces_regla_{ruta}"] = sum(1 for n in veces.values() if n >= MIN_VERSIONES)
    return fila, rechazos


def procesar_factorial(etapa: str, variante: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                       pares: dict[str, list[tuple[str, str]]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x_cruda, par = serie_cruda(variante.split("-")[2], indice, real, betas)
    versiones, corregidas, con_control = armar_variante(variante, x_cruda)
    clave_rem = (lambda v, o, d: ("fp_remuestreo", variante, str(T), str(indice), v, o, d)) if etapa == "etapa2" else None
    valores, no_est = analizar(versiones, con_control, pares, clave_rem)
    fila: dict[str, object] = {"variante": variante, "indice": indice, "celdas_piso": corregidas,
                               "par_origen": par[0] if par else "", "par_destino": par[1] if par else ""}
    for ruta in (("F", "rem") if etapa == "etapa2" else ("F",)):
        fila.update(resumir_rechazos(valores, no_est, pares, ruta)[0])
    for v in surrogates.VERSIONES:
        fila[f"no_estimable_{v}"] = no_est[v]
        if par is not None:
            fila[f"efecto_par_{v}"] = float(valores[v]["efecto"][pares[v].index(par)])
    arreglos = {f"{k}_{v}": valores[v][k] for v in surrogates.VERSIONES
                for k in (("efecto", "F", "p_F", "p_rem") if etapa == "etapa2" else ("efecto", "F", "p_F"))}
    return fila, arreglos


def procesar_3b(T_: int, efecto: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                pares: dict[str, list[tuple[str, str]]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """El pipeline del control positivo largo (sus claves y su generar_replica_T), guardando todos los pares."""
    x, corregidas, par, beta = generar_replica_T(T_, efecto, indice, real, betas)
    con = surrogates.construir_versiones(x)
    fila: dict[str, object] = {"T": T_, "efecto": efecto, "indice": indice, "origen": par[0], "destino": par[1],
                               "celdas_piso": corregidas}
    valores: dict[str, dict[str, np.ndarray]] = {}
    no_est: dict[str, int] = {}
    n_versiones = 0
    for v, (serie, _, propio) in con.items():
        m = len(pares[v])
        val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_rem")}
        disenos = [(o, d, *granger.armar_diseno(serie, propio[d], d, excluir=None)) for o, d in pares[v]]
        no_est[v] = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for _, _, X_f, _ in disenos))
        if not no_est[v]:
            for j, (o, d, X_f, y) in enumerate(disenos):
                obs = granger.estadistico_f(serie, propio[d], o, d)
                X_r, _ = granger.armar_diseno(serie, propio[d], d, excluir=o)
                rng = granger.generador("largo_remuestreo", str(T_), efecto, str(indice), v, o, d)
                val["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B, rng)["p"]
                val["efecto"][j], val["F"][j] = float(np.log(obs["rss_r"] / obs["rss_f"])), obs["F"]
        r = np.zeros(m, dtype=bool) if no_est[v] else granger.benjamini_hochberg(val["p_rem"], Q, M_FAMILIA[v])
        j_pl = pares[v].index(par)
        n_versiones += int(r[j_pl])
        fila.update({f"rechaza_plantado_{v}": int(r[j_pl]), f"efecto_plantado_{v}": val["efecto"][j_pl],
                     f"p_plantado_{v}": val["p_rem"][j_pl], f"rechazos_no_plantados_{v}": int(r.sum()) - int(r[j_pl]),
                     f"rechazos_total_{v}": int(r.sum())})
        valores[v] = val
    fila["n_versiones_plantado"] = n_versiones
    fila["detectado_regla"] = int(n_versiones >= MIN_VERSIONES)
    for v in surrogates.VERSIONES:
        fila[f"no_estimable_{v}"] = no_est[v]
    return fila, {f"{k}_{v}": valores[v][k] for v in surrogates.VERSIONES for k in ("efecto", "F", "p_rem")}


# ---------------------------------------------------------------------------
# Unidades, puntos de control y corrida
# ---------------------------------------------------------------------------

def unidades() -> list[tuple[str, str, Callable[..., tuple[dict[str, object], dict[str, np.ndarray]]]]]:
    """(etapa, nombre, función(índice, real, betas, pares))."""
    salida = []
    for variante in VARIANTES:
        salida.append(("etapa1", variante, lambda i, r, b, p, v=variante: procesar_factorial("etapa1", v, i, r, b, p)))
    for variante in ETAPA2:
        salida.append(("etapa2", variante, lambda i, r, b, p, v=variante: procesar_factorial("etapa2", v, i, r, b, p)))
    for T_, e in CELDAS_3B:
        salida.append(("etapa3b", f"T{T_}_e{e}", lambda i, r, b, p, t=T_, ef=e: procesar_3b(t, ef, i, r, b, p)))
    return salida


def rutas(etapa: str, nombre: str) -> tuple[Path, Path]:
    return SALIDA / f"{etapa}_{nombre}.csv", SALIDA / f"{etapa}_{nombre}.npz"


def correr() -> None:
    real, betas = leer_real(), leer_csv(RUTA_BETAS)
    pares = pares_por_version(list(real.columns))
    SALIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for etapa, nombre, funcion in unidades():
        ruta_csv, ruta_npz = rutas(etapa, nombre)
        filas: list[dict[str, object]] = []
        arreglos: dict[str, list[np.ndarray]] = {}
        if ruta_csv.exists():
            tabla = leer_csv(ruta_csv)
            assert tabla["indice"].tolist() == list(range(len(tabla))), f"{etapa} {nombre}: índices no consecutivos"
            filas = tabla.to_dict("records")
            with np.load(ruta_npz) as z:
                arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
        n = N_ETAPA[etapa]
        if len(filas) < n:
            print(f"{etapa} {nombre}: {len(filas)} guardadas, faltan {n - len(filas)}", flush=True)
        for i in range(len(filas), n):
            fila, arr = funcion(i, real, betas, pares)
            filas.append(fila)
            for k, val in arr.items():
                arreglos.setdefault(k, []).append(val)
            if (i + 1) % CADA == 0 or i + 1 == n:
                tmp_npz = SALIDA / f"{etapa}_{nombre}.tmp.npz"
                np.savez(tmp_npz, **{k: np.array(v) for k, v in arreglos.items()})
                os.replace(tmp_npz, ruta_npz)
                tmp_csv = SALIDA / f"{etapa}_{nombre}.tmp.csv"
                pd.DataFrame(filas).to_csv(tmp_csv, index=False)
                os.replace(tmp_csv, ruta_csv)
        print(f"  {etapa} {nombre}: {n}/{n} ({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def medir(n: int) -> None:
    real, betas = leer_real(), leer_csv(RUTA_BETAS)
    pares = pares_por_version(list(real.columns))
    total = {"etapa1": 0.0, "etapa2": 0.0, "etapa3b": 0.0}
    for etapa, nombre, funcion in unidades():
        tiempos = []
        for i in range(n):
            t0 = time.perf_counter()
            funcion(i, real, betas, pares)
            tiempos.append(time.perf_counter() - t0)
        unidad = float(np.mean(tiempos)) * N_ETAPA[etapa]
        total[etapa] += unidad
        print(f"  {etapa} {nombre}: {np.mean(tiempos):.3f} s por réplica x {N_ETAPA[etapa]} = {unidad / 60:.1f} min",
              flush=True)
    for etapa, segundos in total.items():
        print(f"Total {etapa}: {segundos / 60:.1f} min")
    print(f"Total etapas 1, 2 y 3b: {sum(total.values()) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------

def verificar() -> bool:
    real, betas = leer_real(), leer_csv(RUTA_BETAS)
    columnas = list(real.columns)
    pares = pares_por_version(columnas)
    largo = leer_csv(RUTA_LARGO)
    res: dict[str, bool] = {}

    print("=== f) Unicidad de las claves de semilla ===")
    previas = identidades_previas(columnas)
    for T_, e in [(46, "0.0"), (46, "0.4"), (46, "0.8"), (92, "0.0"), (92, "0.4"), (92, "0.8")]:
        for i in range(100):
            previas += [("largo_par", str(T_), e, str(i)), ("largo_generacion", str(T_), e, str(i))]
            for v, lista in pares.items():
                previas += [("largo_remuestreo", str(T_), e, str(i), v, o, d) for o, d in lista]
    nuevas: list[tuple[str | int, ...]] = [("fp_verificacion", "I1")]
    for i in range(N_ETAPA["etapa1"]):
        nuevas.append(("fp_generacion", "I1", str(T), str(i)))
        if i >= 100:  # 0 a 99 reutilizan a propósito las claves del control positivo largo
            nuevas += [("largo_par", str(T), "0.0", str(i)), ("largo_generacion", str(T), "0.0", str(i))]
    for variante in ETAPA2:
        for i in range(N_ETAPA["etapa2"]):
            for v, lista in pares.items():
                nuevas += [("fp_remuestreo", variante, str(T), str(i), v, o, d) for o, d in lista]
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    res["f"] = distintas == len(todas)
    print(f"  {len(todas)} ({len(previas)} previas + {len(nuevas)} nuevas), distintas: {distintas}: "
          f"{'OK' if res['f'] else 'FALLA'}  (la etapa 3b reutiliza claves previas a propósito)")

    print("\n=== a) C0-P0-I0 contra data/control_positivo_largo/ (T = 92, efecto 0.0, réplicas 0 a 9) ===")
    guardado = largo[(largo["T"] == 92) & (largo["efecto"] == "0.0")].set_index("indice")
    propias = [procesar_factorial("etapa1", "C0-P0-I0", i, real, betas, pares)[0] for i in range(10)]
    ok_a = all(f["celdas_piso"] == guardado.loc[i, "celdas_piso"]
               and (f["par_origen"], f["par_destino"]) == (guardado.loc[i, "origen"], guardado.loc[i, "destino"])
               for i, f in enumerate(propias))
    for v in surrogates.VERSIONES:
        media_propia = np.mean([f[f"efecto_par_{v}"] for f in propias])
        media_guardada = np.mean([float(guardado.loc[i, f"efecto_plantado_{v}"]) for i in range(10)])
        ok_v = all(iguales(f[f"efecto_par_{v}"], float(guardado.loc[i, f"efecto_plantado_{v}"])) for i, f in enumerate(propias))
        ok_a &= ok_v and media_propia == media_guardada
        print(f"  {v}: efecto observado medio {media_propia!r} contra {media_guardada!r}: {'OK' if ok_v else 'FALLA'}")
    res["a"] = ok_a
    print(f"  celdas corregidas y par idénticos, y efectos exactos: {'OK' if ok_a else 'FALLA'}")

    print("\n=== b) Covarianza de 20000 innovaciones I1 ===")
    _, _, residuos = surrogates.ajustar_ar1(real)
    sigma = np.cov(residuos, rowvar=False, ddof=1)
    u = granger.generador("fp_verificacion", "I1").standard_normal((20000, len(columnas))) @ np.linalg.cholesky(sigma).T
    dif = float(np.max(np.abs(np.cov(u, rowvar=False, ddof=1) - sigma)))
    res["b"] = dif < TOL_COV
    print(f"  max |cov empírica - objetivo| = {dif:.5f} < {TOL_COV}: {'OK' if res['b'] else 'FALLA'}")

    print("\n=== c) C2 no usa masas simuladas ===")
    x0_, _ = serie_cruda("I0", 0, real, betas)
    x1_, _ = serie_cruda("I1", 1, real, betas)
    v0, _, _ = armar_variante("C2-P0-I0", x0_)
    v1, _, _ = armar_variante("C2-P1-I1", x1_)
    res["c"] = all(v0[v][1]("topic_0").equals(v1[v][1]("topic_3")) for v in surrogates.VERSIONES)
    print(f"  control C2 idéntico entre réplicas, variantes y destinos, en las 4 versiones: {'OK' if res['c'] else 'FALLA'}")

    print("\n=== d) Parámetros por ecuación ===")
    ok_d = True
    for variante in ("C0-P0-I0", "C1-P0-I0", "C2-P0-I0"):
        versiones, _, con_control = armar_variante(variante, x0_)
        for v, (serie, control_de) in versiones.items():
            K = len(serie.columns)
            formas = {diagnostico.disenos(serie, control_de(d), o, d, con_control)[0].shape[1] for o, d in pares[v]}
            esperado = K + (1 if variante.startswith("C1") else 2)
            ok_d &= formas == {esperado}
            print(f"  {variante} {v}: K = {K}, parámetros {sorted(formas)} (esperado {esperado})")
    res["d"] = ok_d

    print("\n=== e) Misma réplica y variante dos veces ===")
    ok_e = True
    for etapa, variante in (("etapa2", "C1-P1-I1"), ("etapa1", "C2-P1-I0")):
        (f1, a1), (f2, a2) = (procesar_factorial(etapa, variante, 0, real, betas, pares) for _ in range(2))
        ok = f1.keys() == f2.keys() and all(iguales(f1[k], f2[k]) for k in f1) and \
            all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
        ok_e &= ok
        print(f"  {etapa} {variante} réplica 0: {'OK' if ok else 'FALLA'}")
    res["e"] = ok_e

    print("\n=== g) Etapa 3b contra data/control_positivo_largo/ (réplicas 0 y 1 de cada celda; completa en --resumen) ===")
    ok_g = True
    for T_, e in CELDAS_3B:
        g = largo[(largo["T"] == T_) & (largo["efecto"] == e)].set_index("indice")
        for i in (0, 1):
            fila, _ = procesar_3b(T_, e, i, real, betas, pares)
            cols = [c for c in fila if c in g.columns and c not in ("T", "efecto", "indice")]
            ok = all(iguales(fila[c] if not isinstance(fila[c], np.floating) else float(fila[c]),
                             g.loc[i, c] if not isinstance(g.loc[i, c], np.floating) else float(g.loc[i, c])) for c in cols)
            ok_g &= ok
            print(f"  T = {T_} efecto {e} réplica {i}: {len(cols)} columnas idénticas: {'OK' if ok else 'FALLA'}")
    res["g"] = ok_g

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def columnas_identicas(a: pd.Series, b: pd.Series) -> bool:
    """Igualdad exacta: numéricas con NaN igual a NaN; texto como texto."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
    return a.astype(str).equals(b.astype(str))


def resumen() -> None:
    real = leer_real()
    pares = pares_por_version(list(real.columns))
    salidas: dict[str, pd.DataFrame] = {}

    # Etapa 1
    filas = []
    for variante in VARIANTES:
        t = leer_csv(rutas("etapa1", variante)[0])
        assert len(t) == N_ETAPA["etapa1"], f"etapa1 {variante} incompleta"
        C, P, I = variante.split("-")
        for v in surrogates.VERSIONES:
            filas.append({"variante": variante, "C": C, "P": P, "I": I, "version": v, "n_replicas": len(t),
                          "n_no_estimables": int(t[f"no_estimable_{v}"].sum()),
                          "fraccion_con_rechazo_F": float((t[f"rechazos_F_{v}"] >= 1).mean()),
                          "rechazos_medios_F": float(t[f"rechazos_F_{v}"].mean())})
    e1 = pd.DataFrame(filas)
    orig = e1[e1["version"] == "original"]
    efectos = []
    for factor, niveles in (("C", ("C0", "C1", "C2")), ("P", ("P0", "P1")), ("I", ("I0", "I1"))):
        for nivel in niveles:
            efectos.append({"factor": factor, "nivel": nivel,
                            "fraccion_media_original": float(orig[orig[factor] == nivel]["fraccion_con_rechazo_F"].mean())})
    ef = pd.DataFrame(efectos)
    actual = {"C": "C0", "P": "P0", "I": "I0"}
    criterio_a = []
    for factor, limpio in (("C", "C1"), ("C", "C2"), ("P", "P1"), ("I", "I1")):
        base = float(ef[(ef["factor"] == factor) & (ef["nivel"] == actual[factor])]["fraccion_media_original"].iloc[0])
        nuevo = float(ef[(ef["factor"] == factor) & (ef["nivel"] == limpio)]["fraccion_media_original"].iloc[0])
        criterio_a.append({"factor": factor, "nivel_actual": actual[factor], "nivel_limpio": limpio,
                           "fraccion_actual": base, "fraccion_limpia": nuevo, "reduccion": base - nuevo,
                           "implicado": base - nuevo >= REDUCCION_MIN})
    salidas["etapa1_por_version"], salidas["etapa1_efectos_principales"] = e1, ef
    salidas["criterio_a"] = pd.DataFrame(criterio_a)

    # Etapa 2
    filas, regla, criterio_b = [], [], []
    for variante in ETAPA2:
        t = leer_csv(rutas("etapa2", variante)[0])
        assert len(t) == N_ETAPA["etapa2"], f"etapa2 {variante} incompleta"
        for v in surrogates.VERSIONES:
            fila: dict[str, object] = {"variante": variante, "version": v, "n_replicas": len(t),
                                       "n_no_estimables": int(t[f"no_estimable_{v}"].sum())}
            for ruta in ("rem", "F"):
                fila[f"replicas_con_rechazo_{ruta}"] = int((t[f"rechazos_{ruta}_{v}"] >= 1).sum())
                fila[f"fraccion_con_rechazo_{ruta}"] = float((t[f"rechazos_{ruta}_{v}"] >= 1).mean())
                fila[f"rechazos_medios_{ruta}"] = float(t[f"rechazos_{ruta}_{v}"].mean())
            filas.append(fila)
        regla.append({"variante": variante, **{k: v for ruta in ("rem", "F") for k, v in (
            (f"enlaces_medios_regla_{ruta}", float(t[f"enlaces_regla_{ruta}"].mean())),
            (f"fraccion_con_enlace_regla_{ruta}", float((t[f"enlaces_regla_{ruta}"] >= 1).mean())))}})
        cuentas = {v: int((t[f"rechazos_rem_{v}"] >= 1).sum()) for v in ("original", "detendenciada")}
        criterio_b.append({"variante": variante, "replicas_con_rechazo_original": cuentas["original"],
                           "replicas_con_rechazo_detendenciada": cuentas["detendenciada"], "umbral": UMBRAL_B,
                           "restaura_calibracion": all(c <= UMBRAL_B for c in cuentas.values())})
    salidas["etapa2_por_version"], salidas["etapa2_regla"] = pd.DataFrame(filas), pd.DataFrame(regla)
    salidas["criterio_b"] = pd.DataFrame(criterio_b)

    # Etapa 3b
    largo = leer_csv(RUTA_LARGO)
    verif, regla46, clasif = [], [], []
    for T_, e in CELDAS_3B:
        nombre = f"T{T_}_e{e}"
        t = leer_csv(rutas("etapa3b", nombre)[0])
        assert len(t) == N_ETAPA["etapa3b"], f"etapa3b {nombre} incompleta"
        g = largo[(largo["T"] == T_) & (largo["efecto"] == e)].reset_index(drop=True)
        cols = [c for c in t.columns if c in g.columns and c not in ("T", "efecto")]
        iguales_todo = all(columnas_identicas(t[c], g[c]) for c in cols)
        verif.append({"celda": nombre, "columnas_comparadas": len(cols), "identicas": iguales_todo,
                      "tasa_regla_3b": float(t["detectado_regla"].mean()), "tasa_regla_guardada": float(g["detectado_regla"].mean())})
        with np.load(rutas("etapa3b", nombre)[1]) as z:
            p = {v: z[f"p_rem_{v}"] for v in surrogates.VERSIONES}
        rech = {v: np.array([np.zeros(len(pares[v]), bool) if np.isnan(fila_p).any() else
                             granger.benjamini_hochberg(fila_p, Q, M_FAMILIA[v]) for fila_p in p[v]]) for v in surrogates.VERSIONES}
        if e == "0.0":
            con_enlace = 0
            for i in range(len(t)):
                veces: dict[tuple[str, str], int] = {}
                for v in surrogates.VERSIONES:
                    for par, rr in zip(pares[v], rech[v][i]):
                        veces[par] = veces.get(par, 0) + int(rr)
                con_enlace += int(any(n >= MIN_VERSIONES for n in veces.values()))
            regla46.append({"celda": nombre, "fraccion_con_enlace_regla": con_enlace / len(t)})
        else:
            for v in surrogates.VERSIONES:
                cuenta = {"inverso": 0, "misma_fuente": 0, "mismo_destino": 0, "otros": 0}
                for i in range(len(t)):
                    A, Bd = t.loc[i, "origen"], t.loc[i, "destino"]
                    for (o, d), rr in zip(pares[v], rech[v][i]):
                        if not rr or (o, d) == (A, Bd):
                            continue
                        clave = ("inverso" if (o, d) == (Bd, A) else "misma_fuente" if o == A
                                 else "mismo_destino" if d == Bd else "otros")
                        cuenta[clave] += 1
                clasif.append({"celda": nombre, "version": v, **{k: c / len(t) for k, c in cuenta.items()}})
    salidas["etapa3b_verificacion"] = pd.DataFrame(verif)
    salidas["etapa3b_regla_T46_e0.0"] = pd.DataFrame(regla46)
    salidas["etapa3b_clasificacion"] = pd.DataFrame(clasif)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        for nombre, tabla in salidas.items():
            tabla.to_csv(SALIDA / f"resumen_{nombre}.csv", index=False)
            print(f"\n=== {nombre} (resumen_{nombre}.csv) ===")
            print(tabla.to_string(index=False))
    print(f"\nCriterio (a), factores implicados: "
          f"{[f'{r.factor} ({r.nivel_limpio})' for r in salidas['criterio_a'].itertuples() if r.implicado] or 'ninguno'}")
    print(f"Criterio (b), variantes que restauran la calibración: "
          f"{[r.variante for r in salidas['criterio_b'].itertuples() if r.restaura_calibracion] or 'ninguna'}")
    print(f"Verificación g) completa: {'OK' if salidas['etapa3b_verificacion']['identicas'].all() else 'FALLA'}")


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
