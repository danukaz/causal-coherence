"""
run_control_positivo_largo.py

Semana 7, paso 10: control positivo con el pipeline completo y series más
largas (diseño en Bitacora_Semana7.md, sección 15).

Mismo simulador del control positivo (planted.simular_cruzado: AR diagonal
con filas de residuos reales, término cruzado, beta por par de
data/control_positivo/betas.csv, c_B centrado con planted.media_proceso,
calentamiento de surrogates.CALENTAMIENTO semanas, piso en 0 con
surrogates.aplicar_piso), con la serie conservada de largo T en {46, 92}.
planted.generar_replica fija T = 23 y sus claves; aquí generar_replica_T
repite sus pasos con T y las claves como parámetros, sin cambiar planted.py.

Para cada T, efecto (0.0, 0.4, 0.8) y réplica (0 a 99): versiones y
controles de surrogates.construir_versiones; en cada versión, con el control
propio, F y p por remuestreo salvaje (B = 9999) de todos los pares;
Benjamini y Hochberg a q = 0.05 por versión (m = 90, 90, 90 y 72); regla de
2 versiones. Versión con algún diseño sin rango completo: no estimable, sin
p ni rechazos.

Claves: ("largo_generacion", T, efecto, índice), ("largo_par", T, efecto,
índice) y ("largo_remuestreo", T, efecto, índice, versión, origen, destino),
con T, efecto e índice como texto.

Salidas en data/control_positivo_largo/: control_positivo_largo.csv (una fila
por réplica, punto de control cada 25), resumen_por_version.csv,
resumen_por_celda.csv y comparacion.csv (--resumen).

Uso:
  python scripts/run_control_positivo_largo.py --verificar   verificaciones a) a d)
  python scripts/run_control_positivo_largo.py --medir 3     tiempo; no guarda nada
  python scripts/run_control_positivo_largo.py               corrida completa (reanuda)
  python scripts/run_control_positivo_largo.py --resumen     tablas desde lo guardado
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from causal_coherence import granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

LARGOS = (46, 92)
EFECTOS = ("0.0", "0.4", "0.8")
N_POR_CELDA = 100
CADA = 25
B = 9999
Q = 0.05
MIN_VERSIONES = 2
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
SALIDA = OUTPUT_DIR / "control_positivo_largo"
RUTA_REPLICAS = SALIDA / "control_positivo_largo.csv"
RUTA_BETAS = OUTPUT_DIR / "control_positivo" / "betas.csv"
CP = OUTPUT_DIR / "control_positivo"


def leer_csv(ruta: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def leer_real() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)


def pares_por_version(columnas: list[str]) -> dict[str, list[tuple[str, str]]]:
    sin = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(sin if v == "residualizada" else columnas, 2)) for v in surrogates.VERSIONES}


def claves(T: int, efecto: str, indice: int, original: bool) -> tuple[tuple[str | int, ...], tuple[str | int, ...]]:
    """(clave del par, clave de generación). original=True: las del control positivo (T = 23)."""
    if original:
        return ("plantado_par", efecto, indice), ("plantado_generacion", efecto, indice)
    return ("largo_par", str(T), efecto, str(indice)), ("largo_generacion", str(T), efecto, str(indice))


def generar_replica_T(T: int, efecto: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                      original: bool = False) -> tuple[pd.DataFrame, int, tuple[str, str], float]:
    """Los pasos de planted.generar_replica con el largo T y las claves como parámetros."""
    columnas = list(real.columns)
    clave_par, clave_gen = claves(T, efecto, indice, original)
    pares = planted.pares_plantables(columnas)
    origen, destino = pares[int(granger.generador(*clave_par).integers(0, len(pares)))]  # como planted.sortear_par
    fila = betas[(betas["efecto"] == efecto) & (betas["origen"] == origen) & (betas["destino"] == destino)]
    assert len(fila) == 1, f"{efecto} {origen}->{destino}: {len(fila)} betas"
    beta = float(fila["beta"].iloc[0])
    c, a, residuos = surrogates.ajustar_ar1(real)
    i_A = columnas.index(origen)
    X = planted.simular_cruzado(c, a, residuos, real.to_numpy().mean(axis=0), T, i_A, columnas.index(destino),
                                beta, planted.media_proceso(c, a, i_A), granger.generador(*clave_gen))
    X, corregidas = surrogates.aplicar_piso(X)
    indice_filas = real.index if T == len(real) else pd.RangeIndex(T)
    return pd.DataFrame(X, index=indice_filas, columns=real.columns), corregidas, (origen, destino), beta


def efectos_plantado(x: pd.DataFrame, par: tuple[str, str], pares: dict[str, list[tuple[str, str]]]) -> dict[str, float]:
    """Efecto observado del par plantado por versión (NaN si la versión no es estimable), sin remuestreo."""
    salida = {}
    for version, (serie, _, propio) in surrogates.construir_versiones(x).items():
        no_estimable = any(np.linalg.matrix_rank(granger.armar_diseno(serie, propio[d], d, excluir=None)[0])
                           < len(serie.columns) + 2 for _, d in pares[version])
        obs = granger.estadistico_f(serie, propio[par[1]], par[0], par[1])
        salida[version] = np.nan if no_estimable else float(np.log(obs["rss_r"] / obs["rss_f"]))
    return salida


def procesar_replica(T: int, efecto: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                     pares: dict[str, list[tuple[str, str]]]) -> dict[str, object]:
    """Fila del CSV para una réplica (la lógica de run_control_positivo.procesar_replica con T y claves largo_*)."""
    x, corregidas, (pl_A, pl_B), beta = generar_replica_T(T, efecto, indice, real, betas)
    fila: dict[str, object] = {"T": T, "efecto": efecto, "indice": indice, "origen": pl_A, "destino": pl_B,
                               "beta": beta, "celdas_piso": corregidas,
                               "celdas_piso_destino": int((x[pl_B] == 0).sum())}
    n_versiones = 0
    no_estimable: dict[str, int] = {}
    for version, (serie, _, propio) in surrogates.construir_versiones(x).items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[version], f"{version}: pares distintos"
        disenos = [(o, d, *granger.armar_diseno(serie, propio[d], d, excluir=None)) for o, d in pares[version]]
        no_estimable[version] = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for _, _, X_f, _ in disenos))
        if no_estimable[version]:
            fila.update({f"rechaza_plantado_{version}": 0, f"efecto_plantado_{version}": np.nan,
                         f"p_plantado_{version}": np.nan, f"rechazos_no_plantados_{version}": 0,
                         f"rechazos_total_{version}": 0})
            continue
        p = np.empty(len(pares[version]))
        efecto_plantado = np.nan
        for j, (origen, destino, X_f, y) in enumerate(disenos):
            control = propio[destino]
            obs = granger.estadistico_f(serie, control, origen, destino)
            X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
            rng = granger.generador("largo_remuestreo", str(T), efecto, str(indice), version, origen, destino)
            p[j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B, rng)["p"]
            if (origen, destino) == (pl_A, pl_B):
                efecto_plantado = float(np.log(obs["rss_r"] / obs["rss_f"]))
        rechazos = granger.benjamini_hochberg(p, Q, M_FAMILIA[version])
        j_pl = pares[version].index((pl_A, pl_B))
        rechaza = int(rechazos[j_pl])
        n_versiones += rechaza
        fila.update({f"rechaza_plantado_{version}": rechaza, f"efecto_plantado_{version}": efecto_plantado,
                     f"p_plantado_{version}": float(p[j_pl]),
                     f"rechazos_no_plantados_{version}": int(rechazos.sum()) - rechaza,
                     f"rechazos_total_{version}": int(rechazos.sum())})
    fila["n_versiones_plantado"] = n_versiones
    fila["detectado_regla"] = int(n_versiones >= MIN_VERSIONES)
    for version in surrogates.VERSIONES:
        fila[f"no_estimable_{version}"] = no_estimable[version]
    return fila


def secuencia() -> list[tuple[int, str, int]]:
    return [(T, e, i) for T in LARGOS for e in EFECTOS for i in range(N_POR_CELDA)]


def iguales(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------

def identidades_previas(columnas: list[str]) -> list[tuple[str | int, ...]]:
    """Todas las claves ya usadas (las de calculadora_poder.verificar_claves, previas y nuevas de esa etapa)."""
    sin5 = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    pares = {v: list(itertools.permutations(sin5 if v == "residualizada" else columnas, 2)) for v in surrogates.VERSIONES}
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
    todos = list(itertools.permutations(columnas, 2))
    for i in range(2000):
        ids.append(("auditoria_A5_generacion", i))
        if i < 300:
            ids += [("auditoria_A5_remuestreo", i, o, d) for o, d in todos]
    celdas_a6 = [(10, T, e) for T in (23, 46, 92) for e in ("0.4", "0.8", "1.6")] + [(5, 23, e) for e in ("0.4", "0.8", "1.6")]
    for K in (5, 10):
        ids += [("auditoria_A6_calibracion", K, e) for e in ("0.4", "0.8", "1.6")]
    for K, T, e in celdas_a6:
        pares_K = list(itertools.permutations(columnas[:K], 2))
        for i in range(1000):
            ids.append(("auditoria_A6_generacion", K, T, e, i))
            if i < 100:
                ids += [("auditoria_A6_remuestreo", K, T, e, i, o, d) for o, d in pares_K]
    for K in (3, 5, 10):
        ids += [("calc_calibracion", K, e) for e in ("0.4", "0.8")]
    for K in (3, 5, 10):
        pares_K = list(itertools.permutations(columnas[:K], 2))
        for n in (22, 30, 45, 60, 90):
            for e, reps in (("0.0", 200), ("0.4", 100), ("0.8", 100)):
                for i in range(reps):
                    ids.append(("calc_generacion", K, n, e, i))
                    ids += [("calc_remuestreo", K, n, e, i, o, d) for o, d in pares_K]
    return ids


def verificar() -> bool:
    real = leer_real()
    columnas = list(real.columns)
    betas = leer_csv(RUTA_BETAS)
    pares = pares_por_version(columnas)
    resultados: dict[str, bool] = {}

    print("=== d) Unicidad de las claves de semilla ===")
    ids = identidades_previas(columnas)
    n_previas = len(ids)
    for T, e, i in secuencia():
        ids += [("largo_par", str(T), e, str(i)), ("largo_generacion", str(T), e, str(i))]
        for v, lista in pares.items():
            ids += [("largo_remuestreo", str(T), e, str(i), v, o, d) for o, d in lista]
    distintas = len({granger.clave_semilla(*ident) for ident in ids})
    resultados["d"] = distintas == len(ids)
    print(f"  {len(ids)} ({n_previas} previas + {len(ids) - n_previas} nuevas), distintas: {distintas}: "
          f"{'OK' if resultados['d'] else 'FALLA'}")

    print("\n=== a) T = 23 con las claves originales contra data/control_positivo/ (efecto 0.4, índices 0 a 9) ===")
    cp = leer_csv(CP / "control_positivo.csv").set_index(["efecto", "indice"])
    difs = []
    for i in range(10):
        x, corregidas, par, beta = generar_replica_T(23, "0.4", i, real, betas, original=True)
        guardada = cp.loc[("0.4", i)]
        if corregidas != guardada["celdas_piso"] or par != (guardada["origen"], guardada["destino"]):
            difs.append(f"{i}: celdas o par")
        for version, valor in efectos_plantado(x, par, pares).items():
            if not iguales(valor, float(guardada[f"efecto_plantado_{version}"])):
                difs.append(f"{i} {version}: {valor!r} != {guardada[f'efecto_plantado_{version}']!r}")
    resultados["a"] = not difs
    print(f"  efecto observado del par plantado (4 versiones) y celdas corregidas idénticos en las 10 réplicas: "
          f"{'OK' if resultados['a'] else 'FALLA'}" + (f" {difs[:5]}" if difs else ""))

    print("\n=== b) Filas y columnas de las versiones ===")
    ok_b = True
    for T in LARGOS:
        x, *_ = generar_replica_T(T, "0.4", 0, real, betas)
        formas = {v: s.shape for v, (s, _, _) in surrogates.construir_versiones(x).items()}
        esperado = {"original": (T, 10), "diferenciada": (T - 1, 10), "detendenciada": (T, 10), "residualizada": (T, 9)}
        ok_b &= formas == esperado
        print(f"  T = {T}: {formas} (esperado {esperado}): {'OK' if formas == esperado else 'FALLA'}")
    resultados["b"] = ok_b

    print("\n=== c) Misma réplica dos veces (T = 46, efecto 0.8, índice 0) ===")
    f1 = procesar_replica(46, "0.8", 0, real, betas, pares)
    f2 = procesar_replica(46, "0.8", 0, real, betas, pares)
    resultados["c"] = f1.keys() == f2.keys() and all(iguales(f1[k], f2[k]) for k in f1)
    print(f"  fila idéntica: {'OK' if resultados['c'] else 'FALLA'}")

    ok = all(resultados.values())
    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(resultados.items())))
    if not ok:
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return ok


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------

def medir(n: int) -> None:
    real, betas = leer_real(), leer_csv(RUTA_BETAS)
    pares = pares_por_version(list(real.columns))
    total = 0.0
    for T in LARGOS:
        for e in EFECTOS:
            tiempos = []
            for i in range(n):
                t0 = time.perf_counter()
                procesar_replica(T, e, i, real, betas, pares)
                tiempos.append(time.perf_counter() - t0)
            celda = float(np.mean(tiempos)) * N_POR_CELDA
            total += celda
            print(f"  T = {T} efecto {e}: {np.mean(tiempos):.3f} s por réplica x {N_POR_CELDA} = {celda / 60:.1f} min",
                  flush=True)
    print(f"Total ({len(secuencia())} réplicas): {total / 60:.1f} min")


def correr() -> None:
    real, betas = leer_real(), leer_csv(RUTA_BETAS)
    pares = pares_por_version(list(real.columns))
    filas: list[dict[str, object]] = []
    if RUTA_REPLICAS.exists():
        tabla = leer_csv(RUTA_REPLICAS)
        assert list(zip(tabla["T"], tabla["efecto"], tabla["indice"])) == secuencia()[:len(tabla)], "avance fuera de secuencia"
        filas = tabla.to_dict("records")
    print(f"{len(filas)} réplicas ya guardadas, faltan {len(secuencia()) - len(filas)}", flush=True)
    t0 = time.perf_counter()
    for T, e, i in secuencia()[len(filas):]:
        filas.append(procesar_replica(T, e, i, real, betas, pares))
        if len(filas) % CADA == 0 or len(filas) == len(secuencia()):
            SALIDA.mkdir(parents=True, exist_ok=True)
            tmp = SALIDA / "control_positivo_largo.tmp.csv"
            pd.DataFrame(filas).to_csv(tmp, index=False)
            os.replace(tmp, RUTA_REPLICAS)
            print(f"  {len(filas)}/{len(secuencia())} guardadas (T = {T}, efecto {e}, réplica {i}; "
                  f"{time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def resumen() -> None:
    tabla = leer_csv(RUTA_REPLICAS)
    assert len(tabla) == len(secuencia()), f"{len(tabla)} réplicas, faltan para {len(secuencia())}"
    por_version, por_celda = [], []
    for (T, e), g in tabla.groupby(["T", "efecto"], sort=False):
        for v in surrogates.VERSIONES:
            con_rechazo = int((g[f"rechazos_total_{v}"] >= 1).sum())
            por_version.append({"T": T, "efecto": e, "version": v, "n_replicas": len(g),
                                "n_no_estimables": int(g[f"no_estimable_{v}"].sum()),
                                "tasa_deteccion_plantado_BH": g[f"rechaza_plantado_{v}"].mean(),
                                "efecto_observado_medio": g[f"efecto_plantado_{v}"].mean(),
                                "rechazos_no_plantados_medio": g[f"rechazos_no_plantados_{v}"].mean(),
                                "replicas_con_al_menos_un_rechazo": con_rechazo,
                                "fraccion_con_al_menos_un_rechazo": con_rechazo / len(g)})
        no_est = g[[f"no_estimable_{v}" for v in surrogates.VERSIONES]].max(axis=1)
        por_celda.append({"T": T, "efecto": e, "n_replicas": len(g), "n_replicas_no_estimables": int(no_est.sum()),
                          "tasa_deteccion_regla_2_versiones": g["detectado_regla"].mean(),
                          "celdas_piso_total": int(g["celdas_piso"].sum()), "celdas_piso_media": g["celdas_piso"].mean(),
                          "celdas_piso_destino_media": g["celdas_piso_destino"].mean()})
    tabla_version, tabla_celda = pd.DataFrame(por_version), pd.DataFrame(por_celda)

    cp_tam = leer_csv(CP / "resumen_por_tamano.csv")
    a6 = leer_csv(OUTPUT_DIR / "auditoria" / "A6_celdas.csv")
    comparacion = []
    for e in ("0.4", "0.8"):
        fila: dict[str, object] = {"efecto": e,
                                   "pipeline_T23_regla": float(cp_tam.loc[cp_tam["efecto"] == e, "tasa_deteccion_regla_2_versiones"].iloc[0])}
        for T in LARGOS:
            fila[f"pipeline_T{T}_regla"] = float(tabla_celda[(tabla_celda["T"] == T) & (tabla_celda["efecto"] == e)]
                                                 ["tasa_deteccion_regla_2_versiones"].iloc[0])
        for T in (23, 46, 92):
            fila[f"A6_K10_T{T}_remuestreo"] = float(a6[(a6["K"] == 10) & (a6["T"] == T) & (a6["efecto"] == e)]
                                                    ["deteccion_BH_remuestreo"].iloc[0])
        comparacion.append(fila)
    tabla_comp = pd.DataFrame(comparacion)

    tabla_version.to_csv(SALIDA / "resumen_por_version.csv", index=False)
    tabla_celda.to_csv(SALIDA / "resumen_por_celda.csv", index=False)
    tabla_comp.to_csv(SALIDA / "comparacion.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== Por T, efecto y versión (resumen_por_version.csv) ===")
        print(tabla_version.to_string(index=False))
        print("\n=== Por T y efecto (resumen_por_celda.csv) ===")
        print(tabla_celda.to_string(index=False))
        print("\n=== Comparación (comparacion.csv): pipeline completo con la regla de 2 versiones; A6 K = 10, "
              "detección BH con remuestreo ===")
        print(tabla_comp.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true", help="verificaciones a) a d)")
    grupo.add_argument("--medir", type=int, metavar="N", help="tiempo con N réplicas por celda, sin guardarlas")
    grupo.add_argument("--resumen", action="store_true", help="tablas desde lo guardado")
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
