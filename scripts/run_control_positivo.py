"""
run_control_positivo.py

Semana 7: control positivo (diseño en Bitacora_Semana7.md, sección 6).

Para cada tamaño de efecto (0.0, 0.4, 0.8, 1.2, 1.6) y réplica (0 a 299):
serie de planted.generar_replica (AR diagonal con el par plantado, piso en
0), versiones y controles de surrogates.construir_versiones, y para cada
versión, con el control propio, F_obs y p por remuestreo salvaje (B = 9999,
granger.generador("plantado_remuestreo", efecto, índice, versión, origen,
destino)) en todos los pares; Benjamini y Hochberg a q = 0.05 dentro de la
versión (m = 90, 90, 90 y 72) y regla de al menos 2 versiones para el par
plantado.

Salidas en data/control_positivo/:
  betas.csv                  beta calibrado por tamaño y par
  control_positivo.csv       una fila por réplica; se reescribe cada 50
                             réplicas como punto de control y la corrida
                             sigue desde la primera réplica que falta
  resumen_por_version.csv    (--resumen) por tamaño y versión
  resumen_por_tamano.csv     (--resumen) por tamaño

Uso:
  python scripts/run_control_positivo.py --medir 5   tiempo por réplica; no guarda réplicas
  python scripts/run_control_positivo.py             corrida completa (reanuda si hay avance)
  python scripts/run_control_positivo.py --resumen   resumen y efecto mínimo detectable
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence import granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

B = 9999
Q = 0.05
N_POR_TAMANO = 300
CADA = 50
MIN_VERSIONES = 2
TASA_MINIMA = 0.80
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
SALIDA = OUTPUT_DIR / "control_positivo"
RUTA_BETAS = SALIDA / "betas.csv"
RUTA_REPLICAS = SALIDA / "control_positivo.csv"
RUTA_RESUMEN_VERSION = SALIDA / "resumen_por_version.csv"
RUTA_RESUMEN_TAMANO = SALIDA / "resumen_por_tamano.csv"


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def obtener_betas(real: pd.DataFrame) -> pd.DataFrame:
    """Lee betas.csv; si no existe, calibra y lo guarda."""
    if RUTA_BETAS.exists():
        return leer_csv(RUTA_BETAS)
    betas = planted.calibrar_todos(real)
    SALIDA.mkdir(parents=True, exist_ok=True)
    betas.to_csv(RUTA_BETAS, index=False)
    return leer_csv(RUTA_BETAS)


def pares_por_version(columnas: list[str]) -> dict[str, list[tuple[str, str]]]:
    sin_descartada = [c for c in columnas if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(sin_descartada if v == "residualizada" else columnas, 2))
            for v in surrogates.VERSIONES}


def procesar_replica(efecto: str, indice: int, real: pd.DataFrame, betas: pd.DataFrame,
                     pares: dict[str, list[tuple[str, str]]]) -> dict[str, object]:
    """Fila del CSV para una réplica."""
    x, corregidas, (plantado_A, plantado_B), beta = planted.generar_replica(efecto, indice, real, betas)
    versiones = surrogates.construir_versiones(x)
    fila: dict[str, object] = {"efecto": efecto, "indice": indice, "origen": plantado_A,
                               "destino": plantado_B, "beta": beta, "celdas_piso": corregidas}
    n_versiones = 0
    for version, (serie, _, propio) in versiones.items():
        assert list(itertools.permutations(serie.columns, 2)) == pares[version], f"{version}: pares distintos"
        p = np.empty(len(pares[version]))
        efecto_plantado = np.nan
        for j, (origen, destino) in enumerate(pares[version]):
            control = propio[destino]
            obs = granger.estadistico_f(serie, control, origen, destino)
            X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
            X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
            assert np.linalg.matrix_rank(X_f) == X_f.shape[1], \
                f"{efecto} {indice} {version} {origen}->{destino}: diseño sin rango completo"
            rng = granger.generador("plantado_remuestreo", efecto, indice, version, origen, destino)
            p[j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"],
                                              B, rng)["p"]
            if (origen, destino) == (plantado_A, plantado_B):
                efecto_plantado = float(np.log(obs["rss_r"] / obs["rss_f"]))
        rechazos = granger.benjamini_hochberg(p, Q, M_FAMILIA[version])
        j_plantado = pares[version].index((plantado_A, plantado_B))
        rechaza = int(rechazos[j_plantado])
        n_versiones += rechaza
        fila[f"rechaza_plantado_{version}"] = rechaza
        fila[f"efecto_plantado_{version}"] = efecto_plantado
        fila[f"p_plantado_{version}"] = float(p[j_plantado])
        fila[f"rechazos_no_plantados_{version}"] = int(rechazos.sum()) - rechaza
    fila["n_versiones_plantado"] = n_versiones
    fila["detectado_regla"] = int(n_versiones >= MIN_VERSIONES)
    return fila


def secuencia() -> list[tuple[str, int]]:
    return [(e, i) for e in planted.TAMANOS for i in range(N_POR_TAMANO)]


def cargar_avance() -> list[dict[str, object]]:
    if not RUTA_REPLICAS.exists():
        return []
    tabla = leer_csv(RUTA_REPLICAS)
    hechas = list(zip(tabla["efecto"], tabla["indice"]))
    assert hechas == secuencia()[:len(hechas)], "el avance guardado no sigue la secuencia esperada"
    return tabla.to_dict("records")


def guardar(filas: list[dict[str, object]]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = SALIDA / "control_positivo.tmp.csv"
    pd.DataFrame(filas).to_csv(tmp, index=False)
    os.replace(tmp, RUTA_REPLICAS)


def medir(n: int, real: pd.DataFrame, betas: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    """Corre n réplicas por tamaño, mide el tiempo y extrapola. No guarda réplicas."""
    promedios = {}
    for efecto in planted.TAMANOS:
        filas = []
        for i in range(n):
            t0 = time.perf_counter()
            fila = procesar_replica(efecto, i, real, betas, pares)
            filas.append({**fila, "segundos": time.perf_counter() - t0})
        tabla = pd.DataFrame(filas)
        promedios[efecto] = float(tabla["segundos"].mean())
        print(f"\n=== efecto {efecto}, réplicas 0 a {n - 1} ===")
        columnas = ["indice", "origen", "destino", "beta", "celdas_piso",
                    *[f"rechaza_plantado_{v}" for v in surrogates.VERSIONES],
                    "n_versiones_plantado", "detectado_regla", "segundos"]
        with pd.option_context("display.width", 250):
            print(tabla[columnas].to_string(index=False))
    print("\n=== Tiempo y extrapolación ===")
    total = 0.0
    for efecto, promedio in promedios.items():
        total += promedio * N_POR_TAMANO
        print(f"  efecto {efecto}: promedio {promedio:.3f} s por réplica; "
              f"{N_POR_TAMANO} réplicas = {promedio * N_POR_TAMANO / 60:.1f} min")
    print(f"  total {N_POR_TAMANO * len(planted.TAMANOS)} réplicas = {total / 60:.1f} min")


def correr(real: pd.DataFrame, betas: pd.DataFrame, pares: dict[str, list[tuple[str, str]]]) -> None:
    filas = cargar_avance()
    pendientes = secuencia()[len(filas):]
    print(f"{len(filas)} réplicas ya guardadas, faltan {len(pendientes)}", flush=True)
    t0 = time.perf_counter()
    for efecto, i in pendientes:
        filas.append(procesar_replica(efecto, i, real, betas, pares))
        if len(filas) % CADA == 0 or len(filas) == len(secuencia()):
            guardar(filas)
            print(f"  {len(filas)}/{len(secuencia())} guardadas (efecto {efecto}, réplica {i}; "
                  f"{time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)
    resumen()


def resumen() -> None:
    tabla = leer_csv(RUTA_REPLICAS)
    assert len(tabla) == len(secuencia()), f"{len(tabla)} réplicas, faltan para {len(secuencia())}"
    por_version = []
    por_tamano = []
    for efecto, g in tabla.groupby("efecto", sort=False):
        for v in surrogates.VERSIONES:
            por_version.append({"efecto": efecto, "version": v, "n_replicas": len(g),
                                "tasa_deteccion_plantado_BH": g[f"rechaza_plantado_{v}"].mean(),
                                "efecto_observado_medio": g[f"efecto_plantado_{v}"].mean(),
                                "p_plantado_medio": g[f"p_plantado_{v}"].mean(),
                                "rechazos_no_plantados_medio": g[f"rechazos_no_plantados_{v}"].mean()})
        por_tamano.append({"efecto": efecto, "n_replicas": len(g),
                           "tasa_deteccion_regla_2_versiones": g["detectado_regla"].mean(),
                           "celdas_piso_total": int(g["celdas_piso"].sum()),
                           "celdas_piso_media": g["celdas_piso"].mean()})
    tabla_version = pd.DataFrame(por_version)
    tabla_tamano = pd.DataFrame(por_tamano)
    tabla_version.to_csv(RUTA_RESUMEN_VERSION, index=False)
    tabla_tamano.to_csv(RUTA_RESUMEN_TAMANO, index=False)

    print(f"\n=== Por tamaño y versión ({RUTA_RESUMEN_VERSION.name}) ===")
    print(tabla_version.to_string(index=False))
    print(f"\n=== Por tamaño ({RUTA_RESUMEN_TAMANO.name}) ===")
    print(tabla_tamano.to_string(index=False))
    cumplen = tabla_tamano[tabla_tamano["tasa_deteccion_regla_2_versiones"] >= TASA_MINIMA]
    if cumplen.empty:
        emd = "ninguno de los tamaños probados"
    else:
        emd = min(cumplen["efecto"], key=float)
    print(f"\nEfecto mínimo detectable (menor efecto con tasa de detección >= {TASA_MINIMA} "
          f"según la regla de 2 versiones): {emd}")


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--medir", type=int, metavar="N", help="tiempo de N réplicas por tamaño, sin guardarlas")
    grupo.add_argument("--resumen", action="store_true", help="solo el resumen")
    args = parser.parse_args()

    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    pares = pares_por_version(list(real.columns))
    assert {v: len(p) for v, p in pares.items()} == M_FAMILIA, "pares por versión distintos de m"

    if args.resumen:
        resumen()
        return
    betas = obtener_betas(real)
    if args.medir:
        medir(args.medir, real, betas, pares)
    else:
        correr(real, betas, pares)


if __name__ == "__main__":
    main()
