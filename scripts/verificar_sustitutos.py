"""
verificar_sustitutos.py

Semana 7: verificaciones previas a generar los corpus sustitutos.

  1) Unicidad de las claves de semilla: todas las claves que usará la
     calibración (remuestreo de los datos reales en las 4 versiones,
     generación de los sustitutos y remuestreo de cada par de cada
     sustituto, índices 0 a 999 de ambas familias) son distintas.
  2) Ajuste AR(1) diagonal de la serie real: c, a, si algún |a| >= 1, y la
     forma de la matriz de residuos.
  3) Pipeline end to end: la serie real pasada por
     surrogates.construir_versiones (n_t = suma de masas) reproduce los F
     de las 684 filas de data/granger_results.csv, con el control propio y
     con el compartido (np.allclose con rtol=0, atol=1e-9).
  4) Un mismo sustituto (misma familia y mismo índice) genera exactamente
     la misma tabla dos veces, en las 4 versiones y sus controles.

No escribe archivos.
"""

import itertools

import numpy as np
import pandas as pd

from causal_coherence import granger, surrogates
from causal_coherence.config import OUTPUT_DIR

N_POR_FAMILIA = 1000
TOLERANCIA = 1e-9
INDICE_PRUEBA = 0


def marca(ok: bool) -> str:
    return "OK" if ok else "FALLA"


def pares_por_version(real: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    columnas = {v: list(real.columns) for v in surrogates.VERSIONES}
    columnas["residualizada"] = [c for c in real.columns if c != surrogates.COLUMNA_DESCARTADA_RESIDUALIZADA]
    return {v: list(itertools.permutations(cols, 2)) for v, cols in columnas.items()}


def verificar_claves(pares: dict[str, list[tuple[str, str]]]) -> bool:
    identidades: list[tuple[str | int, ...]] = []
    for version, lista in pares.items():
        identidades += [("remuestreo", version, o, d) for o, d in lista]
    for familia in surrogates.FAMILIAS:
        for i in range(N_POR_FAMILIA):
            identidades.append(("generacion", familia, i))
            for version, lista in pares.items():
                identidades += [("remuestreo", familia, i, version, o, d) for o, d in lista]
    claves = [granger.clave_semilla(*ident) for ident in identidades]
    distintas = len(set(claves))
    print(f"  identidades: {len(identidades)}, claves distintas: {distintas}")
    ok = distintas == len(identidades)
    if not ok:
        vistas: dict[tuple[int, ...], tuple[str | int, ...]] = {}
        for ident, clave in zip(identidades, claves):
            if clave in vistas:
                print(f"  repetida: {ident} y {vistas[clave]} -> {clave}")
            vistas.setdefault(clave, ident)
    print(f"  sin claves repetidas: {marca(ok)}")
    return ok


def main() -> None:
    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    pares = pares_por_version(real)

    print("=== 1) Unicidad de las claves de semilla ===")
    if not verificar_claves(pares):
        print("Hay claves repetidas: se detiene aquí.")
        return

    print("\n=== 2) Ajuste AR(1) diagonal de la serie real ===")
    c, a, residuos = surrogates.ajustar_ar1(real)
    print(pd.DataFrame({"c": c, "a": a}, index=real.columns).round(6).to_string())
    print(f"  matriz de residuos: {residuos.shape}")
    print(f"  algún |a| >= 1: {bool(np.any(np.abs(a) >= 1))}")

    print("\n=== 3) Pipeline nuevo sobre la serie real contra data/granger_results.csv ===")
    csv = pd.read_csv(OUTPUT_DIR / "granger_results.csv", float_precision="round_trip")
    versiones = surrogates.construir_versiones(real)
    filas = []
    for version, (serie, compartido, propio) in versiones.items():
        for origen, destino in pares[version]:
            for control_tipo, control in (("compartido", compartido), ("propio", propio[destino])):
                F = granger.estadistico_f(serie, control, origen, destino)["F"]
                filas.append({"version": version, "control_tipo": control_tipo,
                              "origen": origen, "destino": destino, "F_nuevo": F})
    nuevo = pd.DataFrame(filas)
    unido = csv.merge(nuevo, on=["version", "control_tipo", "origen", "destino"], how="outer",
                      validate="one_to_one", indicator=True)
    print(f"  filas en el CSV: {len(csv)}, filas del pipeline nuevo: {len(nuevo)}, "
          f"emparejadas: {int((unido['_merge'] == 'both').sum())}")
    unido["dif"] = (unido["F_nuevo"] - unido["F"]).abs()
    todo_ok = True
    for control_tipo in ("propio", "compartido"):
        g = unido[unido["control_tipo"] == control_tipo]
        ok = len(g) == 342 and np.allclose(g["F_nuevo"], g["F"], rtol=0, atol=TOLERANCIA)
        todo_ok &= ok
        print(f"  control {control_tipo}: {len(g)} filas, max |dif F| = {g['dif'].max():.3e}, "
              f"np.allclose(rtol=0, atol={TOLERANCIA}): {marca(ok)}")
        if not ok:
            print(g[g["dif"] > TOLERANCIA].sort_values("dif", ascending=False).head(10).to_string())
    if not todo_ok:
        print("Los F no coinciden: se detiene aquí.")
        return

    print(f"\n=== 4) Mismo sustituto dos veces (índice {INDICE_PRUEBA}) ===")
    for familia in surrogates.FAMILIAS:
        x1, corr1 = surrogates.generar_sustituto(familia, INDICE_PRUEBA, real)
        x2, corr2 = surrogates.generar_sustituto(familia, INDICE_PRUEBA, real)
        v1 = surrogates.construir_versiones(x1)
        v2 = surrogates.construir_versiones(x2)
        iguales = x1.equals(x2) and corr1 == corr2 and all(
            v1[v][0].equals(v2[v][0]) and v1[v][1].equals(v2[v][1]) and v1[v][2].equals(v2[v][2])
            for v in surrogates.VERSIONES)
        print(f"  {familia}: celdas corregidas por el piso {corr1} y {corr2}; "
              f"serie, 4 versiones y controles idénticos: {marca(iguales)}")


if __name__ == "__main__":
    main()
