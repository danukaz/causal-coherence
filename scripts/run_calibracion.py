"""
run_calibracion.py

Semana 7: calibración de la etapa de significancia con corpus sustitutos
(guía, criterio E1). Las decisiones están en Bitacora_Semana7.md,
secciones 1 y 2.1.

Por sustituto (familia ar_diagonal o fase_aleatoria, índices 0 a 999), en
cada versión y con el control propio del tópico destino: F_obs y p por
remuestreo salvaje (B = 9999) para todos los pares ordenados, con el
generador granger.generador("remuestreo", familia, índice, versión, origen,
destino); después Benjamini y Hochberg a q = 0.05 dentro de la versión
(m = 90, 90, 90 y 72). Se registra por versión la cantidad de rechazos y el
p mínimo, y la cantidad de enlaces con la regla final (significativo en al
menos 2 versiones; los pares de topic_5 solo tienen 3 disponibles).

Salidas en data/calibracion/:
  calibracion_<familia>.csv  una fila por sustituto
  p_<familia>.npz            todos los p (una matriz N x m por versión) y
                             las etiquetas de los pares
Ambos se reescriben cada 50 sustitutos como punto de control; al volver a
correr, el script sigue desde el primer sustituto que falta.

Uso:
  python scripts/run_calibracion.py --medir 5   tiempo por sustituto; no guarda nada
  python scripts/run_calibracion.py             corrida completa (reanuda si hay avance)
  python scripts/run_calibracion.py --resumen   criterio de aceptación sobre lo guardado
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence import granger, surrogates
from causal_coherence.config import OUTPUT_DIR

B = 9999
Q = 0.05
N_POR_FAMILIA = 1000
CADA = 50
MIN_VERSIONES = 2
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
NIVEL_NOMINAL = 0.05
SALIDA = OUTPUT_DIR / "calibracion"


def pares_por_version(columnas: list[str]) -> dict[str, list[tuple[str, str]]]:
    sin_descartada = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(sin_descartada if v == "residualizada" else columnas, 2))
            for v in surrogates.VERSIONES}


def procesar_sustituto(familia: str, indice: int, real: pd.DataFrame,
                       pares: dict[str, list[tuple[str, str]]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fila del CSV y los p de cada versión para un sustituto."""
    x, corregidas = surrogates.generar_sustituto(familia, indice, real)
    versiones = surrogates.construir_versiones(x)
    fila: dict[str, object] = {"indice": indice, "familia": familia}
    ps: dict[str, np.ndarray] = {}
    veces_significativo: dict[tuple[str, str], int] = {}
    for version, (serie, _, propio) in versiones.items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[version], f"{version}: pares distintos"
        p = np.empty(len(pares[version]))
        for j, (origen, destino) in enumerate(pares[version]):
            control = propio[destino]
            obs = granger.estadistico_f(serie, control, origen, destino)
            X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
            X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
            assert np.linalg.matrix_rank(X_f) == X_f.shape[1], \
                f"{familia} {indice} {version} {origen}->{destino}: diseño sin rango completo"
            rng = granger.generador("remuestreo", familia, indice, version, origen, destino)
            p[j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"],
                                              B, rng)["p"]
        rechazos = granger.benjamini_hochberg(p, Q, M_FAMILIA[version])
        fila[f"rechazos_{version}"] = int(rechazos.sum())
        fila[f"p_min_{version}"] = float(p.min())
        for par, r in zip(pares[version], rechazos):
            veces_significativo[par] = veces_significativo.get(par, 0) + int(r)
        ps[version] = p
    fila["enlaces_regla_final"] = sum(1 for n in veces_significativo.values() if n >= MIN_VERSIONES)
    fila["celdas_piso"] = corregidas
    return fila, ps


def rutas(familia: str) -> tuple[Path, Path]:
    return SALIDA / f"calibracion_{familia}.csv", SALIDA / f"p_{familia}.npz"


def cargar_avance(familia: str) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    """Filas y p ya guardados. El npz se escribe antes que el CSV, así que puede tener más filas."""
    ruta_csv, ruta_npz = rutas(familia)
    if not os.path.exists(ruta_csv):
        return [], {v: [] for v in surrogates.VERSIONES}
    tabla = pd.read_csv(ruta_csv, float_precision="round_trip")
    n = len(tabla)
    assert tabla["indice"].tolist() == list(range(n)), f"{familia}: índices no consecutivos en el CSV"
    with np.load(ruta_npz) as z:
        assert z["indice"][:n].tolist() == list(range(n)), f"{familia}: el npz no calza con el CSV"
        ps = {v: list(z[f"p_{v}"][:n]) for v in surrogates.VERSIONES}
    return tabla.to_dict("records"), ps


def guardar(familia: str, filas: list[dict[str, object]], ps: dict[str, list[np.ndarray]],
            pares: dict[str, list[tuple[str, str]]]) -> None:
    """Escribe a archivos temporales y los renombra, primero el npz y después el CSV."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_csv, ruta_npz = rutas(familia)
    tmp_npz = SALIDA / f"p_{familia}.tmp.npz"
    tmp_csv = SALIDA / f"calibracion_{familia}.tmp.csv"
    np.savez(tmp_npz, indice=np.arange(len(filas)),
             **{f"p_{v}": np.array(ps[v]) for v in surrogates.VERSIONES},
             **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in surrogates.VERSIONES})
    os.replace(tmp_npz, ruta_npz)
    pd.DataFrame(filas).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, ruta_csv)


def medir(n: int, real: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    """Corre n sustitutos por familia, mide el tiempo y extrapola. No guarda nada."""
    tiempos: dict[str, list[float]] = {}
    for familia in surrogates.FAMILIAS:
        filas = []
        tiempos[familia] = []
        for i in range(n):
            t0 = time.perf_counter()
            fila, _ = procesar_sustituto(familia, i, real, pares)
            tiempos[familia].append(time.perf_counter() - t0)
            filas.append({**fila, "segundos": tiempos[familia][-1]})
        print(f"\n=== {familia}, sustitutos 0 a {n - 1} ===")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(pd.DataFrame(filas).to_string(index=False))
    print("\n=== Tiempo y extrapolación ===")
    total = 0.0
    for familia, t in tiempos.items():
        promedio = float(np.mean(t))
        total += promedio * N_POR_FAMILIA
        print(f"  {familia}: promedio {promedio:.3f} s por sustituto; "
              f"{N_POR_FAMILIA} sustitutos = {promedio * N_POR_FAMILIA / 60:.1f} min")
    print(f"  total {2 * N_POR_FAMILIA} sustitutos = {total / 60:.1f} min")


def correr(real: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    for familia in surrogates.FAMILIAS:
        filas, ps = cargar_avance(familia)
        inicio = len(filas)
        print(f"{familia}: {inicio} sustitutos ya guardados, faltan {N_POR_FAMILIA - inicio}", flush=True)
        t0 = time.perf_counter()
        for i in range(inicio, N_POR_FAMILIA):
            fila, p = procesar_sustituto(familia, i, real, pares)
            filas.append(fila)
            for v in surrogates.VERSIONES:
                ps[v].append(p[v])
            if (i + 1) % CADA == 0 or i + 1 == N_POR_FAMILIA:
                guardar(familia, filas, ps, pares)
                print(f"  {familia}: {i + 1}/{N_POR_FAMILIA} guardados "
                      f"({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def resumen() -> None:
    """Criterio de aceptación: conteo de sustitutos con al menos un rechazo contra el umbral binomial."""
    chequeos = len(surrogates.VERSIONES) * len(surrogates.FAMILIAS)
    umbral = stats.binom.isf(NIVEL_NOMINAL / chequeos, N_POR_FAMILIA, NIVEL_NOMINAL)
    print(f"\nUmbral: binom.isf({NIVEL_NOMINAL} / {chequeos}, {N_POR_FAMILIA}, {NIVEL_NOMINAL}) = {umbral}")
    filas = []
    informativo = []
    for familia in surrogates.FAMILIAS:
        tabla = pd.read_csv(rutas(familia)[0], float_precision="round_trip")
        assert len(tabla) == N_POR_FAMILIA, f"{familia}: {len(tabla)} sustitutos, faltan para {N_POR_FAMILIA}"
        for v in surrogates.VERSIONES:
            conteo = int((tabla[f"rechazos_{v}"] >= 1).sum())
            filas.append({"familia": familia, "version": v, "con_al_menos_un_rechazo": conteo,
                          "fraccion": conteo / N_POR_FAMILIA, "umbral": umbral, "supera_umbral": conteo > umbral})
        con_enlace = int((tabla["enlaces_regla_final"] >= 1).sum())
        informativo.append({"familia": familia, "con_al_menos_un_enlace_regla_final": con_enlace,
                            "fraccion": con_enlace / N_POR_FAMILIA,
                            "celdas_piso_total": int(tabla["celdas_piso"].sum())})
    resultado = pd.DataFrame(filas)
    print(resultado.to_string(index=False))
    print(f"\nAlgún conteo supera el umbral: {bool(resultado['supera_umbral'].any())}")
    print("\nInformativo (no se usa para aceptar o rechazar):")
    print(pd.DataFrame(informativo).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--medir", type=int, metavar="N", help="tiempo de N sustitutos por familia, sin guardar")
    grupo.add_argument("--resumen", action="store_true", help="solo el criterio de aceptación")
    args = parser.parse_args()

    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    pares = pares_por_version(list(real.columns))
    assert {v: len(p) for v, p in pares.items()} == M_FAMILIA, "pares por versión distintos de m"

    if args.resumen:
        resumen()
    elif args.medir:
        medir(args.medir, real, pares)
    else:
        correr(real, pares)


if __name__ == "__main__":
    main()
