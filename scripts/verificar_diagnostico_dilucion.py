"""
verificar_diagnostico_dilucion.py

Semana 7: verificaciones previas al diagnóstico de dilución
(Bitacora_Semana7.md, sección 8). No escribe archivos.

  a) V0 reproduce, para efecto 0.0 y 1.6 e índices 0 a 199, el efecto
     observado del par plantado, la marca no_estimable y las celdas
     corregidas por el piso de data/control_positivo/control_positivo.csv
     (igualdad exacta; no dependen del remuestreo).
  b) La serie sin piso de V2, con el piso aplicado, es exactamente la serie de
     V0; coincide con ella donde es >= 0 y difiere solo donde el piso la
     corrigió. El control propio de V2 es el de V0, alineado con la serie, y
     diagnostico.series_versiones aplicada a la serie con piso reproduce
     exactamente las series de surrogates.construir_versiones.
  c) Una misma réplica y variante da exactamente el mismo resultado dos veces.
  d) Unicidad de todas las claves de semilla, incluidas las nuevas
     ("diag_remuestreo", variante, efecto, índice, versión, origen, destino).
  e) En V1 y V3 el diseño tiene una columna menos (parámetros por ecuación =
     K * p + 1) y df_den aumenta en 1 respecto de V0 y V2.
"""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence import diagnostico, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

N_SUSTITUTOS = 1000
N_CONTROL_POSITIVO = 300
SEMILLAS_SESGO = 5
REPLICA_C = ("1.6", 0)

_run = runpy.run_path(str(Path(__file__).resolve().parent / "run_diagnostico_dilucion.py"))
procesar_replica = _run["procesar_replica"]
pares_por_version = _run["pares_por_version"]
leer_csv = _run["leer_csv"]
N_POR_EFECTO = _run["N_POR_EFECTO"]


def marca(ok: bool) -> str:
    return "OK" if ok else "FALLA"


def iguales(a: object, b: object) -> bool:
    """Igualdad exacta, con NaN igual a NaN."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def verificar_claves(pares: dict[str, list[tuple[str, str]]], plantables: list[tuple[str, str]]) -> bool:
    identidades: list[tuple[str | int, ...]] = []
    for version, lista in pares.items():
        identidades += [("remuestreo", version, o, d) for o, d in lista]
    for familia in surrogates.FAMILIAS:
        for i in range(N_SUSTITUTOS):
            identidades.append(("generacion", familia, i))
            for version, lista in pares.items():
                identidades += [("remuestreo", familia, i, version, o, d) for o, d in lista]
    for efecto in planted.TAMANOS:
        if float(efecto) != 0.0:
            identidades += [("plantado_calibracion", efecto, o, d) for o, d in plantables]
            identidades += [("plantado_sesgo", efecto, o, d, s) for o, d in plantables for s in range(SEMILLAS_SESGO)]
        for i in range(N_CONTROL_POSITIVO):
            identidades.append(("plantado_par", efecto, i))
            identidades.append(("plantado_generacion", efecto, i))
            for version, lista in pares.items():
                identidades += [("plantado_remuestreo", efecto, i, version, o, d) for o, d in lista]
    n_previas = len(identidades)
    for variante in diagnostico.VARIANTES:
        for efecto in diagnostico.EFECTOS:
            for i in range(N_POR_EFECTO):
                for version, lista in pares.items():
                    identidades += [("diag_remuestreo", variante, efecto, i, version, o, d) for o, d in lista]
    claves = [granger.clave_semilla(*ident) for ident in identidades]
    distintas = len(set(claves))
    print(f"  identidades: {len(identidades)} ({n_previas} previas + {len(identidades) - n_previas} nuevas), "
          f"claves distintas: {distintas}")
    ok = distintas == len(identidades)
    print(f"  sin claves repetidas: {marca(ok)}")
    return ok


def main() -> None:
    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    columnas = list(real.columns)
    pares = pares_por_version(columnas)
    betas = leer_csv(OUTPUT_DIR / "control_positivo" / "betas.csv")
    cp = leer_csv(OUTPUT_DIR / "control_positivo" / "control_positivo.csv").set_index(["efecto", "indice"])
    resultados: dict[str, bool] = {}

    print("=== d) Unicidad de las claves de semilla ===")
    resultados["d"] = verificar_claves(pares, planted.pares_plantables(columnas))

    print(f"\n=== a) V0 contra data/control_positivo/ (efectos {diagnostico.EFECTOS}, índices 0 a {N_POR_EFECTO - 1}) ===")
    print("=== b) Serie de V2 contra la de V0 ===")
    dif_a: list[str] = []
    dif_b: list[str] = []
    celdas_corregidas = 0
    for efecto in diagnostico.EFECTOS:
        for i in range(N_POR_EFECTO):
            x_con, corregidas, par, beta = planted.generar_replica(efecto, i, real, betas)
            x_sin, par_sin, beta_sin = diagnostico.replica_sin_piso(efecto, i, real, betas)
            guardada = cp.loc[(efecto, i)]
            if corregidas != guardada["celdas_piso"] or par != (guardada["origen"], guardada["destino"]):
                dif_a.append(f"{efecto}/{i} celdas o par")
            v0 = diagnostico.versiones_variante(x_sin, x_con, con_piso=True)
            for version, (serie, propio) in v0.items():
                disenos = [diagnostico.disenos(serie, propio[d], o, d, con_control=True) for o, d in pares[version]]
                no_estimable = int(any(np.linalg.matrix_rank(X) < X.shape[1] for X, _, _ in disenos))
                j = pares[version].index(par)
                efecto_plantado = np.nan if no_estimable else diagnostico.prueba(*disenos[j])["efecto"]
                if not iguales(float(efecto_plantado), float(guardada[f"efecto_plantado_{version}"])):
                    dif_a.append(f"{efecto}/{i} {version} efecto {efecto_plantado!r} != "
                                 f"{guardada[f'efecto_plantado_{version}']!r}")
                if no_estimable != int(guardada[f"no_estimable_{version}"]):
                    dif_a.append(f"{efecto}/{i} {version} no_estimable")
            # b)
            con_piso_aplicado, _ = surrogates.aplicar_piso(x_sin.to_numpy())
            no_negativas = x_sin.to_numpy() >= 0
            corregidas_aqui = ~no_negativas
            celdas_corregidas += int(corregidas_aqui.sum())
            ok_b = (par_sin == par and beta_sin == beta
                    and np.array_equal(con_piso_aplicado, x_con.to_numpy())
                    and np.array_equal(x_sin.to_numpy()[no_negativas], x_con.to_numpy()[no_negativas])
                    and bool(np.all(x_sin.to_numpy()[corregidas_aqui] != x_con.to_numpy()[corregidas_aqui]))
                    and int(corregidas_aqui.sum()) == corregidas)
            v2 = diagnostico.versiones_variante(x_sin, x_con, con_piso=False)
            ok_b &= all(v2[v][1].equals(v0[v][1]) for v in surrogates.VERSIONES)
            # La composición de series_versiones reproduce las series de construir_versiones.
            series_con = diagnostico.series_versiones(x_con)
            ok_b &= all(series_con[v].equals(v0[v][0]) for v in surrogates.VERSIONES)
            ok_b &= all(v2[v][0].index.equals(v2[v][1].index) for v in surrogates.VERSIONES)
            if not ok_b:
                dif_b.append(f"{efecto}/{i}")
    resultados["a"] = not dif_a
    resultados["b"] = not dif_b
    print(f"  a) {2 * N_POR_EFECTO} réplicas: efecto observado del par plantado, no_estimable y celdas "
          f"corregidas idénticos a control_positivo.csv: {marca(resultados['a'])}"
          + (f" (diferencias: {dif_a[:10]})" if dif_a else ""))
    print(f"  b) {2 * N_POR_EFECTO} réplicas ({celdas_corregidas} celdas corregidas en total): serie sin piso igual "
          f"a la de V0 donde es >= 0, distinta solo donde el piso la corrigió, piso aplicado igual a V0, "
          f"control propio de V2 igual al de V0 y alineado, y series_versiones igual a construir_versiones "
          f"sobre la serie con piso: {marca(resultados['b'])}"
          + (f" (réplicas: {dif_b[:10]})" if dif_b else ""))

    print(f"\n=== c) Misma réplica dos veces (efecto {REPLICA_C[0]}, índice {REPLICA_C[1]}) ===")
    r1 = procesar_replica(*REPLICA_C, real, betas, pares)
    r2 = procesar_replica(*REPLICA_C, real, betas, pares)
    ok_c = True
    for v in diagnostico.VARIANTES:
        (f1, a1), (f2, a2) = r1[v], r2[v]
        ok = (f1.keys() == f2.keys() and all(iguales(f1[k], f2[k]) for k in f1)
              and a1.keys() == a2.keys() and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1))
        ok_c &= ok
        print(f"  {v}: fila y arreglos idénticos: {marca(ok)}")
    resultados["c"] = ok_c

    print(f"\n=== e) Parámetros y df_den por variante (réplica {REPLICA_C[0]}/{REPLICA_C[1]}, par plantado; "
          f"y forma de los diseños de todos los pares) ===")
    ok_e = True
    filas_e = []
    x_con, _, _, _ = planted.generar_replica(*REPLICA_C, real, betas)
    x_sin, _, _ = diagnostico.replica_sin_piso(*REPLICA_C, real, betas)
    for version in surrogates.VERSIONES:
        K = len(diagnostico.versiones_variante(x_sin, x_con, con_piso=True)[version][0].columns)
        fila = {"version": version, "K": K}
        for v in diagnostico.VARIANTES:
            fila[f"params_{v}"] = r1[v][0][f"n_params_{version}"]
            fila[f"df_den_{v}"] = r1[v][0][f"df_den_{version}"]
            con_control, con_piso = diagnostico.VARIANTES[v]
            serie, propio = diagnostico.versiones_variante(x_sin, x_con, con_piso)[version]
            formas = {diagnostico.disenos(serie, propio[d], o, d, con_control)[0].shape[1] for o, d in pares[version]}
            ok_e &= formas == {fila[f"params_{v}"]}
        ok_e &= fila["params_V1"] == fila["params_V3"] == K * 1 + 1
        ok_e &= fila["params_V0"] == fila["params_V2"] == K * 1 + 2
        ok_e &= fila["df_den_V1"] == fila["df_den_V0"] + 1 and fila["df_den_V3"] == fila["df_den_V2"] + 1
        filas_e.append(fila)
    print(pd.DataFrame(filas_e).to_string(index=False))
    print(f"  V1 y V3 con K * p + 1 parámetros (todos los pares) y df_den de V0 y V2 + 1: {marca(ok_e)}")
    resultados["e"] = ok_e

    print(f"\nResumen: {', '.join(f'{k}) {marca(v)}' for k, v in sorted(resultados.items()))}")
    if not all(resultados.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")


if __name__ == "__main__":
    main()
