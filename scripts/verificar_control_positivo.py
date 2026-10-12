"""
verificar_control_positivo.py

Semana 7: verificaciones previas al control positivo (Bitacora_Semana7.md,
sección 6). Calibra beta (72 pares por 5 tamaños) y guarda
data/control_positivo/betas.csv, que run_control_positivo.py reutiliza.

  d) Unicidad de todas las claves de semilla: las de los datos reales y los
     sustitutos (verificar_sustitutos.py) más las del control positivo
     (plantado_calibracion, plantado_par, plantado_generacion y
     plantado_remuestreo).
  a) Cada beta calibrado, re-simulado con planted.simular_cruzado (T = 20000,
     misma clave de calibración), da el efecto poblacional objetivo con error
     relativo < 1%, y el radio espectral de la matriz de coeficientes es < 1.
     Media de B (c_B centrado en media_A = c_A / (1 - a_A)), en los 288 pares
     con efecto > 0 y T = 20000:
       a1) explicación de la desviación: (media de B con beta - media de B con
           beta = 0, misma clave) - beta * (media simulada de A - media_A) /
           (1 - a_B) tiene valor absoluto <= 0.005;
       a2) ausencia de sesgo: para los 8 pares con mayor desviación relativa,
           media y desviación estándar (ddof=1) de (media simulada de A -
           media_A) sobre 5 semillas con claves ("plantado_sesgo", efecto,
           origen, destino, semilla). Falla si, para algún tópico A, la media
           de las 5 semillas tiene el mismo signo en todos sus pares y
           magnitud > 0.01 en todos ellos.
     La desviación relativa de la media de B respecto de beta = 0 se reporta
     solo como información, sin umbral. Además, para los 3 primeros pares
     plantables: media simulada de B con beta = 0, sin centrar y centrado.
  b) Con beta = 0, planted.simular_cruzado da exactamente la misma serie que
     surrogates.simular_ar_diagonal con el mismo generador (mismos c, a y
     residuos), y una réplica de efecto 0.0 construye las 4 versiones.
  c) Una misma réplica (mismo efecto e índice) da exactamente la misma fila
     dos veces (serie, versiones y p de todos los pares).
"""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence import granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

N_SUSTITUTOS = 1000
N_POR_TAMANO = 300
PARES_MEDIA = 3
RESIDUO_EXPLICACION_MAX = 0.005
PARES_SESGO = 8
SEMILLAS_SESGO = 5
SESGO_MAX = 0.01

_run = runpy.run_path(str(Path(__file__).resolve().parent / "run_control_positivo.py"))
procesar_replica = _run["procesar_replica"]
pares_por_version = _run["pares_por_version"]
RUTA_BETAS = _run["RUTA_BETAS"]


def marca(ok: bool) -> str:
    return "OK" if ok else "FALLA"


def verificar_claves(pares: dict[str, list[tuple[str, str]]], plantables: list[tuple[str, str]]) -> bool:
    identidades: list[tuple[str | int, ...]] = []
    for version, lista in pares.items():
        identidades += [("remuestreo", version, o, d) for o, d in lista]
    for familia in surrogates.FAMILIAS:
        for i in range(N_SUSTITUTOS):
            identidades.append(("generacion", familia, i))
            for version, lista in pares.items():
                identidades += [("remuestreo", familia, i, version, o, d) for o, d in lista]
    n_previas = len(identidades)
    for efecto in planted.TAMANOS:
        if float(efecto) != 0.0:
            identidades += [("plantado_calibracion", efecto, o, d) for o, d in plantables]
            identidades += [("plantado_sesgo", efecto, o, d, s) for o, d in plantables
                            for s in range(SEMILLAS_SESGO)]
        for i in range(N_POR_TAMANO):
            identidades.append(("plantado_par", efecto, i))
            identidades.append(("plantado_generacion", efecto, i))
            for version, lista in pares.items():
                identidades += [("plantado_remuestreo", efecto, i, version, o, d) for o, d in lista]
    claves = [granger.clave_semilla(*ident) for ident in identidades]
    distintas = len(set(claves))
    print(f"  identidades: {len(identidades)} ({n_previas} previas + {len(identidades) - n_previas} "
          f"del control positivo), claves distintas: {distintas}")
    ok = distintas == len(identidades)
    if not ok:
        vistas: dict[tuple[int, ...], tuple[str | int, ...]] = {}
        for ident, clave in zip(identidades, claves):
            if clave in vistas:
                print(f"  repetida: {ident} y {vistas[clave]}")
            vistas.setdefault(clave, ident)
    print(f"  sin claves repetidas: {marca(ok)}")
    return ok


def simular_larga(real: pd.DataFrame, efecto: str, origen: str, destino: str, beta: float,
                  media_A: float, clave: tuple[str | int, ...] | None = None) -> np.ndarray:
    """Simulación larga con la clave de calibración del par, u otra clave si se indica."""
    columnas = list(real.columns)
    c, a, residuos = surrogates.ajustar_ar1(real)
    rng = granger.generador(*(clave or ("plantado_calibracion", efecto, origen, destino)))
    return planted.simular_cruzado(c, a, residuos, real.to_numpy().mean(axis=0), planted.T_LARGA,
                                   columnas.index(origen), columnas.index(destino), beta, media_A, rng)


def main() -> None:
    real = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
    columnas = list(real.columns)
    pares = pares_por_version(columnas)
    plantables = planted.pares_plantables(columnas)
    print(f"Pares plantables: {len(plantables)}")

    print("\n=== d) Unicidad de las claves de semilla ===")
    if not verificar_claves(pares, plantables):
        print("Hay claves repetidas: se detiene aquí.")
        return

    print("\n=== a) Calibración de beta ===")
    betas = planted.calibrar_todos(real)
    RUTA_BETAS.parent.mkdir(parents=True, exist_ok=True)
    betas.to_csv(RUTA_BETAS, index=False)
    c, a, _ = surrogates.ajustar_ar1(real)
    errores = []
    radios = []
    desviaciones = []
    residuos_explicacion = []
    for fila in betas.itertuples(index=False):
        i_A, i_B = columnas.index(fila.origen), columnas.index(fila.destino)
        M = np.diag(a)
        M[i_B, i_A] = fila.beta
        radios.append(float(np.max(np.abs(np.linalg.eigvals(M)))))
        if float(fila.efecto) == 0.0:
            errores.append(np.nan)
            desviaciones.append(np.nan)
            residuos_explicacion.append(np.nan)
            continue
        media_A = planted.media_proceso(c, a, i_A)
        X = simular_larga(real, fila.efecto, fila.origen, fila.destino, fila.beta, media_A)
        X_0 = simular_larga(real, fila.efecto, fila.origen, fila.destino, 0.0, media_A)
        valor = planted.efecto_poblacional(X, i_A, i_B)
        errores.append(abs(valor - float(fila.efecto)) / float(fila.efecto))
        media_0 = X_0[:, i_B].mean()
        diferencia = X[:, i_B].mean() - media_0
        desviaciones.append(abs(diferencia) / abs(media_0))
        prediccion = fila.beta * (X_0[:, i_A].mean() - media_A) / (1 - a[i_B])
        residuos_explicacion.append(abs(diferencia - prediccion))
    betas["error_relativo_resimulado"] = errores
    betas["radio_espectral"] = radios
    betas["desviacion_relativa_media_B"] = desviaciones
    betas["residuo_explicacion"] = residuos_explicacion
    no_nulos = betas[betas["efecto"].astype(float) > 0]
    ok_error = bool((no_nulos["error_relativo_resimulado"] < planted.ERROR_RELATIVO_MAX).all())
    ok_radio = bool((betas["radio_espectral"] < 1).all())
    ok_explicacion = bool((no_nulos["residuo_explicacion"] <= RESIDUO_EXPLICACION_MAX).all())
    resumen = betas.groupby("efecto", sort=False).agg(
        n_pares=("beta", "size"), beta_min=("beta", "min"), beta_mediana=("beta", "median"),
        beta_max=("beta", "max"), iteraciones_max=("iteraciones", "max"),
        error_bisec_max=("error_relativo", "max"), error_resimulado_max=("error_relativo_resimulado", "max"),
        radio_espectral_max=("radio_espectral", "max"),
        residuo_explicacion_max=("residuo_explicacion", "max"))
    with pd.option_context("display.width", 250):
        print(resumen.to_string())
    print(f"  error relativo re-simulado < {planted.ERROR_RELATIVO_MAX} en los "
          f"{len(no_nulos)} pares con efecto > 0: {marca(ok_error)}")
    print(f"  radio espectral < 1 en las {len(betas)} filas: {marca(ok_radio)}")
    print(f"  a1) |(media B con beta - media B con beta = 0) - beta (media A simulada - media_A) / (1 - a_B)|, "
          f"máximo sobre los {len(no_nulos)} pares: {no_nulos['residuo_explicacion'].max():.6f} "
          f"<= {RESIDUO_EXPLICACION_MAX}: {marca(ok_explicacion)}")
    print(f"  betas guardados en {RUTA_BETAS}")

    print("\n  Información, sin umbral: desviación relativa de la media simulada de B respecto de beta = 0 "
          f"(T = {planted.T_LARGA})")
    info = no_nulos.groupby("efecto", sort=False)["desviacion_relativa_media_B"].agg(
        maximo="max", mediana="median", pares_sobre_0_02=lambda s: int((s > 0.02).sum()))
    print(info.to_string())

    print(f"\n  a2) Ausencia de sesgo: {PARES_SESGO} pares con mayor desviación relativa, "
          f"{SEMILLAS_SESGO} semillas (\"plantado_sesgo\", efecto, origen, destino, semilla)")
    filas_sesgo = []
    for fila in no_nulos.nlargest(PARES_SESGO, "desviacion_relativa_media_B").itertuples(index=False):
        i_A = columnas.index(fila.origen)
        media_A = planted.media_proceso(c, a, i_A)
        errores_A = [simular_larga(real, fila.efecto, fila.origen, fila.destino, 0.0, media_A,
                                   clave=("plantado_sesgo", fila.efecto, fila.origen, fila.destino, s))[:, i_A].mean()
                     - media_A for s in range(SEMILLAS_SESGO)]
        filas_sesgo.append({"efecto": fila.efecto, "par": f"{fila.origen}->{fila.destino}", "topico_A": fila.origen,
                            "desv_rel_media_B": fila.desviacion_relativa_media_B, "media_A_proceso": media_A,
                            "media_error_A_5_semillas": float(np.mean(errores_A)),
                            "desv_est_error_A_5_semillas": float(np.std(errores_A, ddof=1))})
    sesgo = pd.DataFrame(filas_sesgo)
    with pd.option_context("display.width", 250):
        print(sesgo.to_string(index=False))
    alertas = []
    for topico, g in sesgo.groupby("topico_A"):
        m = g["media_error_A_5_semillas"]
        if (np.all(m > 0) or np.all(m < 0)) and bool((m.abs() > SESGO_MAX).all()):
            alertas.append(topico)
    ok_sesgo = not alertas
    print(f"  a2) ningún tópico A con la media de las {SEMILLAS_SESGO} semillas del mismo signo en todos sus "
          f"pares y magnitud > {SESGO_MAX}: {marca(ok_sesgo)}" + (f" (tópicos: {alertas})" if alertas else ""))
    if not (ok_error and ok_radio and ok_explicacion and ok_sesgo):
        print("  ALGUNA VERIFICACIÓN DE a) FALLA: no lanzar la corrida.")

    print(f"\n  Media simulada de B (T = {planted.T_LARGA}), primeros {PARES_MEDIA} pares plantables:")
    filas_media = []
    for origen, destino in plantables[:PARES_MEDIA]:
        media_A = planted.media_proceso(c, a, columnas.index(origen))
        base = simular_larga(real, "1.6", origen, destino, 0.0, media_A)[:, columnas.index(destino)].mean()
        for efecto in planted.TAMANOS[1:]:
            beta = float(betas[(betas["efecto"] == efecto) & (betas["origen"] == origen)
                               & (betas["destino"] == destino)]["beta"].iloc[0])
            sin = simular_larga(real, efecto, origen, destino, beta, 0.0)[:, columnas.index(destino)].mean()
            con = simular_larga(real, efecto, origen, destino, beta, media_A)[:, columnas.index(destino)].mean()
            filas_media.append({"par": f"{origen}->{destino}", "efecto": efecto, "beta": beta,
                                "media_real_B": float(real[destino].mean()), "media_A": media_A,
                                "media_B_beta_0": float(base), "media_B_sin_centrar": float(sin),
                                "media_B_centrado": float(con)})
    print(pd.DataFrame(filas_media).to_string(index=False))
    print("  (media_B_beta_0 usa la clave de calibración del tamaño 1.6; con beta = 0 la serie no depende "
          "de media_A)")

    print("\n=== b) Efecto 0.0 contra el sustituto AR diagonal ===")
    c, a, residuos = surrogates.ajustar_ar1(real)
    origen, destino = plantables[0]
    X_cruzado = planted.simular_cruzado(c, a, residuos, real.to_numpy().mean(axis=0), len(real),
                                        columnas.index(origen), columnas.index(destino), 0.0,
                                        planted.media_proceso(c, a, columnas.index(origen)),
                                        granger.generador("generacion", "ar_diagonal", 0))
    X_ar = surrogates.simular_ar_diagonal(real, granger.generador("generacion", "ar_diagonal", 0))
    print(f"  simular_cruzado(beta = 0) idéntica a simular_ar_diagonal con el mismo generador: "
          f"{marca(np.array_equal(X_cruzado, X_ar))}")
    x, corregidas, par, beta = planted.generar_replica("0.0", 0, real, betas)
    versiones = surrogates.construir_versiones(x)
    formas = {v: versiones[v][0].shape for v in surrogates.VERSIONES}
    print(f"  réplica 0.0 índice 0: par {par[0]}->{par[1]}, beta {beta}, celdas corregidas {corregidas}, "
          f"versiones {formas}: OK")

    print("\n=== c) Misma réplica dos veces ===")
    for efecto in ("0.0", "1.6"):
        f1 = procesar_replica(efecto, 0, real, betas, pares)
        f2 = procesar_replica(efecto, 0, real, betas, pares)
        x1, *_ = planted.generar_replica(efecto, 0, real, betas)
        x2, *_ = planted.generar_replica(efecto, 0, real, betas)
        print(f"  efecto {efecto} índice 0: serie idéntica {marca(x1.equals(x2))}; "
              f"fila del CSV idéntica {marca(f1 == f2)}")


if __name__ == "__main__":
    main()
