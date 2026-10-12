"""
run_diagnostico_dilucion.py

Semana 7: diagnóstico de dilución del control positivo (diseño en
Bitacora_Semana7.md, sección 8).

Para cada efecto (0.0 y 1.6), réplica (0 a 199) y variante (V0 a V3 de
causal_coherence.diagnostico): en cada versión, con el control propio,
efecto, F, p teórico y p por remuestreo salvaje (B = 9999) de todos los
pares; Benjamini y Hochberg a q = 0.05 por versión (m = 90, 90, 90 y 72) con
cada ruta, y regla de al menos 2 versiones para el par plantado. Si algún
diseño de una versión no tiene rango completo, la versión es no estimable:
sin p ni rechazos, y el par plantado cuenta como no detectado (sección 6).

Cada réplica se procesa con las 4 variantes a la vez; el punto de control
se escribe cada 50 réplicas y la corrida sigue desde la primera que falta.

Salidas en data/diagnostico_dilucion/:
  diagnostico_<variante>.csv   una fila por réplica
  p_diagnostico.npz            efecto, F, p_teo y p_rem de todos los pares,
                               por variante y versión, y las etiquetas
  resumen_por_version.csv      (--resumen) por variante, efecto, versión y ruta
  resumen_por_variante.csv     (--resumen) por variante, efecto y ruta, con el
                               criterio de lectura de la sección 8

Uso:
  python scripts/run_diagnostico_dilucion.py --medir 3   tiempo; no guarda nada
  python scripts/run_diagnostico_dilucion.py             corrida completa (reanuda)
  python scripts/run_diagnostico_dilucion.py --resumen   resumen y criterio de lectura
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence import diagnostico, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

B = 9999
Q = 0.05
N_POR_EFECTO = 200
CADA = 50
MIN_VERSIONES = 2
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
DETECCION_MINIMA = 0.50
NIVEL_NOMINAL = 0.05
CHEQUEOS = 8
RUTA_BETAS = OUTPUT_DIR / "control_positivo" / "betas.csv"
SALIDA = OUTPUT_DIR / "diagnostico_dilucion"
RUTA_NPZ = SALIDA / "p_diagnostico.npz"
RUTA_RESUMEN_VERSION = SALIDA / "resumen_por_version.csv"
RUTA_RESUMEN_VARIANTE = SALIDA / "resumen_por_variante.csv"
MAGNITUDES = ("efecto", "F", "p_teo", "p_rem")


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def ruta_csv(variante: str) -> Path:
    return SALIDA / f"diagnostico_{variante}.csv"


def pares_por_version(columnas: list[str]) -> dict[str, list[tuple[str, str]]]:
    sin_descartada = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(sin_descartada if v == "residualizada" else columnas, 2))
            for v in surrogates.VERSIONES}


def procesar_variante(variante: str, efecto: str, indice: int, x_sin_piso: pd.DataFrame,
                      x_con_piso: pd.DataFrame, corregidas: int, par: tuple[str, str], beta: float,
                      pares: dict[str, list[tuple[str, str]]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fila del CSV de la variante y arreglos por versión (NaN si no estimable)."""
    con_control, con_piso = diagnostico.VARIANTES[variante]
    versiones = diagnostico.versiones_variante(x_sin_piso, x_con_piso, con_piso)
    fila: dict[str, object] = {"efecto": efecto, "indice": indice, "origen": par[0], "destino": par[1],
                               "beta": beta, "celdas_piso": corregidas}
    arreglos: dict[str, np.ndarray] = {}
    n_versiones = {r: 0 for r in diagnostico.RUTAS}
    for version, (serie, propio) in versiones.items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[version], f"{version}: pares distintos"
        m = len(pares[version])
        valores = {k: np.full(m, np.nan) for k in MAGNITUDES}
        j_plantado = pares[version].index(par)
        disenos = [diagnostico.disenos(serie, propio[d], o, d, con_control) for o, d in pares[version]]
        no_estimable = int(any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for X_f, _, _ in disenos))
        fila[f"no_estimable_{version}"] = no_estimable
        fila[f"n_params_{version}"] = disenos[j_plantado][0].shape[1]
        fila[f"df_den_{version}"] = np.nan
        if not no_estimable:
            for j, ((origen, destino), (X_f, X_r, y)) in enumerate(zip(pares[version], disenos)):
                obs = diagnostico.prueba(X_f, X_r, y)
                rng = granger.generador("diag_remuestreo", variante, efecto, indice, version, origen, destino)
                valores["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"],
                                                                 obs["df_den"], B, rng)["p"]
                valores["efecto"][j], valores["F"][j], valores["p_teo"][j] = obs["efecto"], obs["F"], obs["p_teo"]
                if j == j_plantado:
                    fila[f"df_den_{version}"] = obs["df_den"]
        fila[f"efecto_plantado_{version}"] = valores["efecto"][j_plantado]
        fila[f"F_plantado_{version}"] = valores["F"][j_plantado]
        for r in diagnostico.RUTAS:
            p = valores[f"p_{r}"]
            rechazos = (np.zeros(m, dtype=bool) if no_estimable
                        else granger.benjamini_hochberg(p, Q, M_FAMILIA[version]))
            rechaza = int(rechazos[j_plantado])
            n_versiones[r] += rechaza
            fila[f"p_{r}_plantado_{version}"] = p[j_plantado]
            fila[f"rechaza_plantado_{r}_{version}"] = rechaza
            fila[f"rechazos_no_plantados_{r}_{version}"] = int(rechazos.sum()) - rechaza
            fila[f"rechazos_total_{r}_{version}"] = int(rechazos.sum())
        for k in MAGNITUDES:
            arreglos[f"{k}_{version}"] = valores[k]
    for r in diagnostico.RUTAS:
        fila[f"n_versiones_plantado_{r}"] = n_versiones[r]
        fila[f"detectado_regla_{r}"] = int(n_versiones[r] >= MIN_VERSIONES)
    return fila, arreglos


def procesar_replica(efecto: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                     pares: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[dict[str, object], dict[str, np.ndarray]]]:
    """Las 4 variantes de una réplica, con la misma serie."""
    x_con_piso, corregidas, par, beta = planted.generar_replica(efecto, indice, real, betas)
    x_sin_piso, par_sin, beta_sin = diagnostico.replica_sin_piso(efecto, indice, real, betas)
    assert par_sin == par and beta_sin == beta, f"{efecto} {indice}: par o beta distintos"
    return {v: procesar_variante(v, efecto, indice, x_sin_piso, x_con_piso, corregidas, par, beta, pares)
            for v in diagnostico.VARIANTES}


def secuencia() -> list[tuple[str, int]]:
    return [(e, i) for e in diagnostico.EFECTOS for i in range(N_POR_EFECTO)]


def cargar_avance() -> tuple[dict[str, list[dict[str, object]]], dict[str, list[np.ndarray]]]:
    """Filas y arreglos guardados. El npz se escribe antes que los CSV; se toma el mínimo común."""
    filas: dict[str, list[dict[str, object]]] = {v: [] for v in diagnostico.VARIANTES}
    arreglos: dict[str, list[np.ndarray]] = {}
    if not all(ruta_csv(v).exists() for v in diagnostico.VARIANTES):
        return filas, arreglos
    tablas = {v: leer_csv(ruta_csv(v)) for v in diagnostico.VARIANTES}
    n = min(len(t) for t in tablas.values())
    for v, t in tablas.items():
        t = t.iloc[:n]
        assert list(zip(t["efecto"], t["indice"])) == secuencia()[:n], f"{v}: avance fuera de secuencia"
        filas[v] = t.to_dict("records")
    with np.load(RUTA_NPZ) as z:
        assert len(z["indice"]) >= n, "el npz tiene menos réplicas que los CSV"
        arreglos = {k: list(z[k][:n]) for k in z.files if k.startswith("V")}
    return filas, arreglos


def guardar(filas: dict[str, list[dict[str, object]]], arreglos: dict[str, list[np.ndarray]],
            pares: dict[str, list[tuple[str, str]]]) -> None:
    """Archivos temporales y renombre: primero el npz, después los CSV."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    hechas = secuencia()[:len(filas["V0"])]
    tmp = SALIDA / "p_diagnostico.tmp.npz"
    np.savez(tmp, efecto=np.array([e for e, _ in hechas]), indice=np.array([i for _, i in hechas]),
             **{k: np.array(v) for k, v in arreglos.items()},
             **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in surrogates.VERSIONES})
    os.replace(tmp, RUTA_NPZ)
    for v, f in filas.items():
        tmp_csv = SALIDA / f"diagnostico_{v}.tmp.csv"
        pd.DataFrame(f).to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, ruta_csv(v))


def medir(n: int, real: pd.DataFrame, betas: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    """n réplicas por variante y efecto; mide cada variante por separado y extrapola. No guarda nada."""
    tiempos: dict[tuple[str, str], list[float]] = {}
    for efecto in diagnostico.EFECTOS:
        for i in range(n):
            x_con_piso, corregidas, par, beta = planted.generar_replica(efecto, i, real, betas)
            x_sin_piso, _, _ = diagnostico.replica_sin_piso(efecto, i, real, betas)
            for v in diagnostico.VARIANTES:
                t0 = time.perf_counter()
                fila, _ = procesar_variante(v, efecto, i, x_sin_piso, x_con_piso, corregidas, par, beta, pares)
                tiempos.setdefault((v, efecto), []).append(time.perf_counter() - t0)
                print(f"  {v} efecto {efecto} réplica {i}: {par[0]}->{par[1]}, "
                      f"detectado rem {fila['detectado_regla_rem']}, teo {fila['detectado_regla_teo']}, "
                      f"{tiempos[(v, efecto)][-1]:.2f} s", flush=True)
    print("\n=== Tiempo y extrapolación ===")
    total = 0.0
    for (v, efecto), t in tiempos.items():
        promedio = float(np.mean(t))
        total += promedio * N_POR_EFECTO
        print(f"  {v} efecto {efecto}: promedio {promedio:.3f} s por réplica; "
              f"{N_POR_EFECTO} réplicas = {promedio * N_POR_EFECTO / 60:.1f} min")
    print(f"  total {len(tiempos) * N_POR_EFECTO} réplicas = {total / 60:.1f} min")


def correr(real: pd.DataFrame, betas: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    filas, arreglos = cargar_avance()
    hechas = len(filas["V0"])
    pendientes = secuencia()[hechas:]
    print(f"{hechas} réplicas ya guardadas (por variante), faltan {len(pendientes)}", flush=True)
    t0 = time.perf_counter()
    for efecto, i in pendientes:
        for v, (fila, arr) in procesar_replica(efecto, i, real, betas, pares).items():
            filas[v].append(fila)
            for version, valores in arr.items():
                arreglos.setdefault(f"{v}_{version}", []).append(valores)
        hechas = len(filas["V0"])
        if hechas % CADA == 0 or hechas == len(secuencia()):
            guardar(filas, arreglos, pares)
            print(f"  {hechas}/{len(secuencia())} guardadas (efecto {efecto}, réplica {i}; "
                  f"{time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def resumen() -> None:
    umbral = stats.binom.isf(NIVEL_NOMINAL / CHEQUEOS, N_POR_EFECTO, NIVEL_NOMINAL)
    por_version = []
    por_variante = []
    for v in diagnostico.VARIANTES:
        tabla = leer_csv(ruta_csv(v))
        assert len(tabla) == len(secuencia()), f"{v}: {len(tabla)} réplicas, faltan para {len(secuencia())}"
        for efecto, g in tabla.groupby("efecto", sort=False):
            for r in diagnostico.RUTAS:
                for version in surrogates.VERSIONES:
                    con_rechazo = int((g[f"rechazos_total_{r}_{version}"] >= 1).sum())
                    por_version.append({
                        "variante": v, "efecto": efecto, "version": version, "ruta": r, "n_replicas": len(g),
                        "n_no_estimables": int(g[f"no_estimable_{version}"].sum()),
                        "tasa_deteccion_plantado_BH": g[f"rechaza_plantado_{r}_{version}"].mean(),
                        "efecto_observado_medio": g[f"efecto_plantado_{version}"].mean(),
                        "p_plantado_medio": g[f"p_{r}_plantado_{version}"].mean(),
                        "rechazos_no_plantados_medio": g[f"rechazos_no_plantados_{r}_{version}"].mean(),
                        "replicas_con_al_menos_un_rechazo": con_rechazo,
                        "fraccion_con_al_menos_un_rechazo": con_rechazo / len(g),
                        "umbral_binomial": umbral})
                por_variante.append({"variante": v, "efecto": efecto, "ruta": r, "n_replicas": len(g),
                                     "tasa_deteccion_regla_2_versiones": g[f"detectado_regla_{r}"].mean()})
    tabla_version = pd.DataFrame(por_version)
    tabla_variante = pd.DataFrame(por_variante)
    criterio = []
    for (v, r), g in tabla_variante.groupby(["variante", "ruta"], sort=False):
        deteccion = float(g.loc[g["efecto"] == "1.6", "tasa_deteccion_regla_2_versiones"].iloc[0])
        nulos = tabla_version[(tabla_version["variante"] == v) & (tabla_version["ruta"] == r)
                              & (tabla_version["efecto"] == "0.0")]
        falsos_ok = bool((nulos["replicas_con_al_menos_un_rechazo"] <= umbral).all())
        criterio.append({"variante": v, "ruta": r, "deteccion_1.6_regla": deteccion,
                         "deteccion_ge_0.50": deteccion >= DETECCION_MINIMA,
                         "falsos_positivos_0.0_dentro_del_umbral": falsos_ok,
                         "recupera_poder_valido": deteccion >= DETECCION_MINIMA and falsos_ok})
    tabla_variante = tabla_variante.merge(pd.DataFrame(criterio), on=["variante", "ruta"], how="left")
    tabla_version.to_csv(RUTA_RESUMEN_VERSION, index=False)
    tabla_variante.to_csv(RUTA_RESUMEN_VARIANTE, index=False)

    print(f"Umbral: binom.isf({NIVEL_NOMINAL} / {CHEQUEOS}, {N_POR_EFECTO}, {NIVEL_NOMINAL}) = {umbral}")
    print(f"\n=== Por variante, efecto, versión y ruta ({RUTA_RESUMEN_VERSION.name}) ===")
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(tabla_version.to_string(index=False))
    print(f"\n=== Por variante, efecto y ruta, con el criterio de lectura ({RUTA_RESUMEN_VARIANTE.name}) ===")
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(tabla_variante.to_string(index=False))
    cumplen = [f"{c['variante']} {c['ruta']}" for c in criterio if c["recupera_poder_valido"]]
    print(f"\nVariantes y rutas que cumplen el criterio de lectura de la sección 8: "
          f"{', '.join(cumplen) if cumplen else 'ninguna'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--medir", type=int, metavar="N", help="tiempo de N réplicas por variante y efecto")
    grupo.add_argument("--resumen", action="store_true", help="solo el resumen")
    args = parser.parse_args()

    if args.resumen:
        resumen()
        return
    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    pares = pares_por_version(list(real.columns))
    assert {v: len(p) for v, p in pares.items()} == M_FAMILIA, "pares por versión distintos de m"
    betas = leer_csv(RUTA_BETAS)
    if args.medir:
        medir(args.medir, real, betas, pares)
    else:
        correr(real, betas, pares)


if __name__ == "__main__":
    main()
