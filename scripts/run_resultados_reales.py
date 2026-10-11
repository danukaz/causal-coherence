"""
run_resultados_reales.py

Semana 7: prueba de Granger sobre los datos reales con el valor p por
remuestreo salvaje, corrección de Benjamini y Hochberg, regla entre
versiones y comprobación de las dos rutas de significancia. El criterio
está fijado en Bitacora_Semana7.md, sección 4, antes de esta corrida.

La serie real pasa por el mismo camino que la verificación 3 de
scripts/verificar_sustitutos.py: surrogates.construir_versiones (n_t =
suma de masas) y granger.estadistico_f sobre los pares de
pares_por_version.

Para cada control (propio y compartido), versión y par ordenado: efecto,
F, df_num, df_den, p_teorico y p_remuestreo (B = 9999, generador
granger.generador("remuestreo", versión, origen, destino), el mismo para
los dos controles). Benjamini y Hochberg a q = 0.05 dentro de cada versión
y control, con m explícito (90, 90, 90 y 72).

Regla entre versiones (propio como principal, compartido como
sensibilidad): un enlace se declara detectado si BH lo rechaza en al menos
2 versiones; los pares con topic_5 solo existen en 3.

Rutas: r = p_remuestreo / max(p_teorico, 1/(B+1)). Pasa en una versión y
control si al menos el 95% de los pares cumple 0.1 <= r <= 10 y la mediana
de r está en [1/3, 3].

Resultados:
  data/resultados_reales.csv
  data/resultados_reales_enlaces.csv              (control propio)
  data/resultados_reales_enlaces_compartido.csv   (control compartido)
  data/rutas.csv
"""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence import granger, surrogates
from causal_coherence.config import OUTPUT_DIR

B = 9999
Q = 0.05
M_FAMILIA = {"original": 90, "diferenciada": 90, "detendenciada": 90, "residualizada": 72}
MIN_VERSIONES = 2
CONTROLES = ("propio", "compartido")
R_MIN, R_MAX = 0.1, 10.0
FRACCION_MIN = 0.95
MEDIANA_MIN, MEDIANA_MAX = 1 / 3, 3.0
TOLERANCIA = 1e-9

SALIDA_RESULTADOS = OUTPUT_DIR / "resultados_reales.csv"
SALIDA_ENLACES = {"propio": OUTPUT_DIR / "resultados_reales_enlaces.csv",
                  "compartido": OUTPUT_DIR / "resultados_reales_enlaces_compartido.csv"}
SALIDA_RUTAS = OUTPUT_DIR / "rutas.csv"
COLUMNAS = ["control_tipo", "version", "origen", "destino", "n_obs", "efecto", "F", "df_num",
            "df_den", "p_teorico", "p_remuestreo", "B", "rechaza_BH"]

# Mismo armado de pares que la verificación 3 de verificar_sustitutos.py.
_verificar = runpy.run_path(str(Path(__file__).resolve().parent / "verificar_sustitutos.py"))
pares_por_version = _verificar["pares_por_version"]


def marca(ok: bool) -> str:
    return "OK" if ok else "FALLA"


def calcular(real: pd.DataFrame) -> pd.DataFrame:
    """Una fila por control, versión y par, con BH dentro de cada versión y control."""
    pares = pares_por_version(real)
    versiones = surrogates.construir_versiones(real)
    filas = []
    for control_tipo in CONTROLES:
        for version, (serie, compartido, propio) in versiones.items():
            for origen, destino in pares[version]:
                control = propio[destino] if control_tipo == "propio" else compartido
                obs = granger.estadistico_f(serie, control, origen, destino)
                X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
                X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
                rng = granger.generador("remuestreo", version, origen, destino)
                rem = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B, rng)
                filas.append({"control_tipo": control_tipo, "version": version,
                              "origen": origen, "destino": destino, "n_obs": obs["n_obs"],
                              "efecto": np.log(obs["rss_r"] / obs["rss_f"]), "F": obs["F"],
                              "df_num": obs["df_num"], "df_den": obs["df_den"],
                              "p_teorico": granger.p_teorico(obs["F"], obs["df_num"], obs["df_den"]),
                              "p_remuestreo": rem["p"], "B": B})
    tabla = pd.DataFrame(filas)
    tabla["rechaza_BH"] = 0
    for (control_tipo, version), g in tabla.groupby(["control_tipo", "version"], sort=False):
        rechazos = granger.benjamini_hochberg(g["p_remuestreo"].to_numpy(), Q, M_FAMILIA[version])
        tabla.loc[g.index, "rechaza_BH"] = rechazos.astype(int)
    return tabla[COLUMNAS]


def enlaces(tabla: pd.DataFrame, control_tipo: str) -> pd.DataFrame:
    """Pares rechazados por BH en al menos MIN_VERSIONES versiones, en formato ancho."""
    t = tabla[tabla["control_tipo"] == control_tipo]
    columnas = ["origen", "destino", "versiones_rechazado", "n_versiones_rechazado", "n_versiones_disponibles"]
    for v in surrogates.VERSIONES:
        columnas += [f"efecto_{v}", f"p_remuestreo_{v}", f"rechaza_{v}"]
    filas = []
    for (origen, destino), g in t.groupby(["origen", "destino"], sort=False):
        g = g.set_index("version")
        rechazadas = [v for v in surrogates.VERSIONES if v in g.index and g.loc[v, "rechaza_BH"] == 1]
        if len(rechazadas) < MIN_VERSIONES:
            continue
        fila: dict[str, object] = {"origen": origen, "destino": destino,
                                   "versiones_rechazado": ";".join(rechazadas),
                                   "n_versiones_rechazado": len(rechazadas),
                                   "n_versiones_disponibles": len(g)}
        for v in surrogates.VERSIONES:
            existe = v in g.index
            fila[f"efecto_{v}"] = g.loc[v, "efecto"] if existe else np.nan
            fila[f"p_remuestreo_{v}"] = g.loc[v, "p_remuestreo"] if existe else np.nan
            fila[f"rechaza_{v}"] = int(g.loc[v, "rechaza_BH"]) if existe else pd.NA
        filas.append(fila)
    resultado = pd.DataFrame(filas, columns=columnas)
    for v in surrogates.VERSIONES:
        resultado[f"rechaza_{v}"] = resultado[f"rechaza_{v}"].astype("Int64")
    return resultado


def rutas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Comprobación de las dos rutas por versión y control (Bitacora_Semana7.md, sección 4)."""
    filas = []
    for (control_tipo, version), g in tabla.groupby(["control_tipo", "version"], sort=False):
        r = g["p_remuestreo"] / np.maximum(g["p_teorico"], 1 / (B + 1))
        fraccion = float(((r >= R_MIN) & (r <= R_MAX)).mean())
        mediana = float(r.median())
        orden = fraccion >= FRACCION_MIN
        sistematica = not (MEDIANA_MIN <= mediana <= MEDIANA_MAX)
        filas.append({"control_tipo": control_tipo, "version": version, "n_pares": len(g),
                      "fraccion_r_en_0.1_10": fraccion, "mediana_r": mediana,
                      "cumple_orden_magnitud": orden, "discrepancia_sistematica": sistematica,
                      "pasa": orden and not sistematica})
    return pd.DataFrame(filas)


def verificar(tabla: pd.DataFrame) -> None:
    print("=== Verificaciones a), b) y d) ===")
    barrido = pd.read_csv(OUTPUT_DIR / "granger_results.csv", float_precision="round_trip")
    unido = tabla.merge(barrido, on=["control_tipo", "version", "origen", "destino"], how="outer",
                        validate="one_to_one", indicator=True, suffixes=("", "_barrido"))
    emparejadas = int((unido["_merge"] == "both").sum())
    print(f"  filas emparejadas con data/granger_results.csv: {emparejadas} de {len(barrido)}")
    for col, col_b in (("F", "F_barrido"), ("efecto", "efecto_barrido"), ("p_teorico", "p_valor_orientativo")):
        dif = (unido[col] - unido[col_b]).abs()
        ok = emparejadas == len(barrido) and np.allclose(unido[col], unido[col_b], rtol=0, atol=TOLERANCIA)
        etiqueta = "a)" if col != "p_teorico" else "b)"
        print(f"  {etiqueta} {col} contra {col_b}: max |dif| = {dif.max():.3e}, "
              f"np.allclose(rtol=0, atol={TOLERANCIA}): {marca(ok)}")
    numericas = tabla[["n_obs", "efecto", "F", "df_num", "df_den", "p_teorico", "p_remuestreo"]].to_numpy(float)
    sin_nan_inf = bool(np.isfinite(numericas).all())
    p = tabla["p_remuestreo"]
    p_en_rango = bool(((p >= 1 / (B + 1)) & (p <= 1)).all())
    pt = tabla["p_teorico"]
    print(f"  d) filas: {len(tabla)} (esperado 684): {marca(len(tabla) == 684)}")
    print(f"  d) sin NaN ni infinitos: {marca(sin_nan_inf)}")
    print(f"  d) p_remuestreo en [1/(B+1), 1]: {marca(p_en_rango)} "
          f"(min {p.min()!r}, max {p.max()!r})")
    print(f"  d) p_teorico en [0, 1]: {marca(bool(((pt >= 0) & (pt <= 1)).all()))} "
          f"(min {pt.min()!r}, max {pt.max()!r})")


def reportar(tabla: pd.DataFrame, por_control: dict[str, pd.DataFrame], tabla_rutas: pd.DataFrame) -> None:
    print("\n=== Por versión y control ===")
    resumen = tabla.groupby(["control_tipo", "version"], sort=False).agg(
        m=("p_remuestreo", "size"),
        rechazos_BH=("rechaza_BH", "sum"),
        p_remuestreo_min=("p_remuestreo", "min"),
        pares_p_remuestreo_le_0001=("p_remuestreo", lambda s: int((s <= 0.001).sum())),
    )
    print(resumen.to_string())

    for control_tipo, t in por_control.items():
        print(f"\n=== Enlaces detectados con la regla entre versiones, control {control_tipo} "
              f"({SALIDA_ENLACES[control_tipo].name}) ===")
        if t.empty:
            print("  ningún enlace cumple la regla; el archivo se guardó solo con los encabezados")
        else:
            with pd.option_context("display.width", 300, "display.max_columns", 50):
                print(t.to_string(index=False))

    claves = {c: set(zip(t["origen"], t["destino"])) for c, t in por_control.items()}
    print(f"\nEnlaces con control propio: {len(claves['propio'])}; con control compartido: "
          f"{len(claves['compartido'])}; en ambos: {len(claves['propio'] & claves['compartido'])}")

    print(f"\n=== Rutas ({SALIDA_RUTAS.name}) ===")
    print(tabla_rutas.to_string(index=False))


def main() -> None:
    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    tabla = calcular(real)
    por_control = {c: enlaces(tabla, c) for c in CONTROLES}
    tabla_rutas = rutas(tabla)

    tabla.to_csv(SALIDA_RESULTADOS, index=False)
    for c, t in por_control.items():
        t.to_csv(SALIDA_ENLACES[c], index=False)
    tabla_rutas.to_csv(SALIDA_RUTAS, index=False)

    verificar(tabla)
    reportar(tabla, por_control, tabla_rutas)
    print("\nGuardados:")
    for ruta in (SALIDA_RESULTADOS, *SALIDA_ENLACES.values(), SALIDA_RUTAS):
        print(f"  {ruta}")


if __name__ == "__main__":
    main()
