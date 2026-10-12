"""
calculadora_poder.py

Semana 7, B2 (Bitacora_Semana7.md, sección 12): calculadora de poder sobre
el banco de respuesta conocida de causal_coherence.auditoria (VAR gaussiano
con la dinámica de bpoil semanal, sin piso y sin control; para K < 10, los
primeros K tópicos con su submatriz). Modelo con K rezagos más intercepto.

Rejilla: K en {3, 5, 10}, n_obs en {22, 30, 45, 60, 90} (T = n_obs + 1) y
efecto poblacional en {0.0, 0.4, 0.8} del par plantado topic_0 -> topic_1
(beta por bisección con la clave ("calc_calibracion", K, efecto); con 0.0,
beta = 0). Réplicas: 200 con efecto 0.0 y 100 con 0.4 y 0.8. En cada una,
p de los m = K(K-1) pares por remuestreo (B = 9999, principal) y por la F;
Benjamini y Hochberg a q = 0.05 con cada ruta.

Celda (K, n_obs) APTA: detección del par plantado con efecto 0.8 y
remuestreo >= 0.80, y con efecto 0.0, conteo de réplicas con al menos un
rechazo (remuestreo) <= binom.isf(0.05, 200, 0.05).

Salidas en data/calculadora_poder/: un .npz por celda (punto de control
cada 50 réplicas), betas.csv, celdas.csv y aptitud.csv.

Uso:
  python scripts/calculadora_poder.py --verificar   verificación contra A6 (ruta F)
  python scripts/calculadora_poder.py --medir 3     tiempo; no guarda réplicas
  python scripts/calculadora_poder.py               rejilla completa (reanuda)
  python scripts/calculadora_poder.py --resumen     tablas desde lo guardado
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence import auditoria, granger, planted, surrogates
from causal_coherence.config import OUTPUT_DIR

SALIDA = OUTPUT_DIR / "calculadora_poder"
KS = (3, 5, 10)
N_OBS = (22, 30, 45, 60, 90)
EFECTOS = ("0.0", "0.4", "0.8")
REPLICAS = {"0.0": 200, "0.4": 100, "0.8": 100}
B = 9999
Q = 0.05
CADA = 50
DETECCION_MIN = 0.80
UMBRAL_FP = stats.binom.isf(0.05, 200, 0.05)
# Verificación contra A6 (ruta F, efecto 0.8): (K, n_obs) -> valor de A6.
VERIFICACION = {(10, 22): 0.229, (10, 45): 0.962}
TOL_VERIFICACION = 0.06
PAR_PLANTADO = (0, 1)


def celdas() -> list[tuple[int, int, str]]:
    return [(K, n, e) for K in KS for n in N_OBS for e in EFECTOS]


def leer_real() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)


def betas() -> pd.DataFrame:
    """Beta por K y efecto (0.0: beta = 0), con la verificación re-simulada."""
    ruta = SALIDA / "betas.csv"
    if ruta.exists():
        return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")
    real = leer_real()
    filas = []
    for K in KS:
        c, a, sigma = auditoria.parametros_var(real.iloc[:, :K])
        for e in EFECTOS:
            if float(e) == 0.0:
                filas.append({"K": K, "efecto": e, "beta": 0.0, "efecto_poblacional": np.nan,
                              "error_relativo": np.nan, "iteraciones": 0, "error_relativo_resimulado": np.nan})
                continue
            clave = ("calc_calibracion", K, e)
            cal = auditoria.calibrar_beta(c, a, sigma, e, clave)
            X = auditoria.simular_var(c, a, sigma, auditoria.T_LARGA, granger.generador(*clave), 0, 1, cal["beta"])
            valor = planted.efecto_poblacional(X, 0, 1)
            filas.append({"K": K, "efecto": e, **cal, "error_relativo_resimulado": abs(valor - float(e)) / float(e)})
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return pd.read_csv(ruta, dtype={"efecto": str}, float_precision="round_trip")


def replica(K: int, n_obs: int, e: str, i: int, beta: float, c: np.ndarray, a: np.ndarray, sigma: np.ndarray,
            columnas: list[str], con_remuestreo: bool) -> tuple[np.ndarray, np.ndarray | None]:
    X = auditoria.simular_var(c, a, sigma, n_obs + 1, granger.generador("calc_generacion", K, n_obs, e, i),
                              PAR_PLANTADO[0], PAR_PLANTADO[1], beta)
    pares = list(itertools.permutations(range(K), 2))
    F, p_f = auditoria.p_ruta_f(X, pares)
    if not con_remuestreo:
        return p_f, None
    claves = [("calc_remuestreo", K, n_obs, e, i, columnas[o], columnas[d]) for o, d in pares]
    return p_f, auditoria.p_ruta_remuestreo(X, pares, F, B, claves)


def parametros(real: pd.DataFrame, tabla_betas: pd.DataFrame, K: int, e: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    c, a, sigma = auditoria.parametros_var(real.iloc[:, :K])
    beta = float(tabla_betas[(tabla_betas["K"] == K) & (tabla_betas["efecto"] == e)]["beta"].iloc[0])
    return beta, c, a, sigma


def deteccion(p: np.ndarray, K: int) -> np.ndarray:
    """Por réplica: si BH (m = K(K-1)) rechaza el par plantado."""
    j = list(itertools.permutations(range(K), 2)).index(PAR_PLANTADO)
    return np.array([granger.benjamini_hochberg(fila, Q, K * (K - 1))[j] for fila in p])


def algun_rechazo(p: np.ndarray, K: int) -> np.ndarray:
    """Por réplica: si BH (m = K(K-1)) rechaza algún par."""
    return np.array([granger.benjamini_hochberg(fila, Q, K * (K - 1)).any() for fila in p])


# ---------------------------------------------------------------------------
# Claves de semilla
# ---------------------------------------------------------------------------

def verificar_claves() -> bool:
    """Unicidad de las claves nuevas contra todas las ya usadas (como en auditoria_independiente.py)."""
    columnas = list(leer_real().columns)
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
    n_previas = len(ids)
    for K in KS:
        ids += [("calc_calibracion", K, e) for e in EFECTOS if float(e) != 0.0]
    for K, n, e in celdas():
        pares_K = list(itertools.permutations(columnas[:K], 2))
        for i in range(REPLICAS[e]):
            ids.append(("calc_generacion", K, n, e, i))
            ids += [("calc_remuestreo", K, n, e, i, o, d) for o, d in pares_K]
    distintas = len({granger.clave_semilla(*ident) for ident in ids})
    ok = distintas == len(ids)
    print(f"Claves de semilla: {len(ids)} ({n_previas} previas + {len(ids) - n_previas} nuevas), "
          f"distintas: {distintas}: {'OK' if ok else 'FALLA'}")
    return ok


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------

def verificar() -> bool:
    real, tabla_betas = leer_real(), betas()
    columnas = list(real.columns)
    print("Betas calibrados:")
    print(tabla_betas.to_string(index=False))
    ok = True
    for (K, n), referencia in VERIFICACION.items():
        beta, c, a, sigma = parametros(real, tabla_betas, K, "0.8")
        p_f = np.array([replica(K, n, "0.8", i, beta, c, a, sigma, columnas, False)[0] for i in range(REPLICAS["0.8"])])
        det = float(deteccion(p_f, K).mean())
        cumple = abs(det - referencia) <= TOL_VERIFICACION
        ok &= cumple
        print(f"  K={K} n_obs={n} efecto 0.8: detección ruta F {det:.3f} ({REPLICAS['0.8']} réplicas), A6 {referencia}, "
              f"|dif| {abs(det - referencia):.3f} <= {TOL_VERIFICACION}: {'OK' if cumple else 'FALLA'}")
    print(f"Verificación: {'PASA' if ok else 'FALLA'}")
    return ok


def medir(n_rep: int) -> None:
    real, tabla_betas = leer_real(), betas()
    columnas = list(real.columns)
    total = 0.0
    for K, n, e in celdas():
        beta, c, a, sigma = parametros(real, tabla_betas, K, e)
        t_f, t_r = [], []
        for i in range(n_rep):
            t0 = time.perf_counter()
            replica(K, n, e, i, beta, c, a, sigma, columnas, False)
            t_f.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            replica(K, n, e, i, beta, c, a, sigma, columnas, True)
            t_r.append(time.perf_counter() - t0)
        celda = float(np.mean(t_r)) * REPLICAS[e]
        total += celda
        print(f"  K={K:2d} n_obs={n:2d} efecto {e}: ruta F {np.mean(t_f):.3f} s, F + remuestreo {np.mean(t_r):.3f} s "
              f"por réplica x {REPLICAS[e]} = {celda / 60:.2f} min", flush=True)
    print(f"Total rejilla ({sum(REPLICAS[e] for _, _, e in celdas())} réplicas): {total / 60:.1f} min")


def ruta_celda(K: int, n: int, e: str) -> Path:
    return SALIDA / f"K{K}_n{n}_e{e}.npz"


def correr() -> None:
    real, tabla_betas = leer_real(), betas()
    columnas = list(real.columns)
    for K, n, e in celdas():
        beta, c, a, sigma = parametros(real, tabla_betas, K, e)
        ruta = ruta_celda(K, n, e)
        guardado = {}
        if ruta.exists():
            with np.load(ruta) as z:
                guardado = {k: z[k] for k in z.files}
        p_f, p_rem = list(guardado.get("p_f", [])), list(guardado.get("p_rem", []))
        if len(p_f) < REPLICAS[e]:
            print(f"K={K} n_obs={n} efecto {e}: {len(p_f)} guardadas, faltan {REPLICAS[e] - len(p_f)}", flush=True)
        for i in range(len(p_f), REPLICAS[e]):
            pf, pr = replica(K, n, e, i, beta, c, a, sigma, columnas, True)
            p_f.append(pf)
            p_rem.append(pr)
            if (i + 1) % CADA == 0 or i + 1 == REPLICAS[e]:
                tmp = ruta.with_name(ruta.stem + ".tmp.npz")
                np.savez(tmp, p_f=np.array(p_f), p_rem=np.array(p_rem))
                os.replace(tmp, ruta)
    resumen()


def resumen() -> None:
    filas = []
    for K, n, e in celdas():
        with np.load(ruta_celda(K, n, e)) as z:
            p_f, p_rem = z["p_f"], z["p_rem"]
        m = K * (K - 1)
        assert p_f.shape == (REPLICAS[e], m) and p_rem.shape == (REPLICAS[e], m), f"K={K} n={n} e={e} incompleta"
        filas.append({"K": K, "n_obs": n, "efecto": e, "m": m, "replicas": REPLICAS[e],
                      "deteccion_remuestreo": float(deteccion(p_rem, K).mean()),
                      "deteccion_F": float(deteccion(p_f, K).mean()),
                      "replicas_con_rechazo_remuestreo": int(algun_rechazo(p_rem, K).sum()),
                      "replicas_con_rechazo_F": int(algun_rechazo(p_f, K).sum())})
    tabla = pd.DataFrame(filas)
    aptitud = []
    for (K, n), g in tabla.groupby(["K", "n_obs"], sort=False):
        g = g.set_index("efecto")
        det08 = g.loc["0.8", "deteccion_remuestreo"]
        fp = g.loc["0.0", "replicas_con_rechazo_remuestreo"]
        aptitud.append({"K": K, "n_obs": n, "deteccion_0.8_remuestreo": det08,
                        "deteccion_0.4_remuestreo": g.loc["0.4", "deteccion_remuestreo"],
                        "falsos_positivos_0.0_remuestreo": int(fp), "umbral_falsos_positivos": UMBRAL_FP,
                        "apta": bool(det08 >= DETECCION_MIN and fp <= UMBRAL_FP)})
    tabla_aptitud = pd.DataFrame(aptitud)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "celdas.csv", index=False)
    tabla_aptitud.to_csv(SALIDA / "aptitud.csv", index=False)
    with pd.option_context("display.width", 250):
        print(tabla.to_string(index=False))
        print()
        print(tabla_aptitud.to_string(index=False))
    print(f"\nUmbral de falsos positivos: binom.isf(0.05, 200, 0.05) = {UMBRAL_FP}")
    for K in KS:
        aptas = tabla_aptitud[(tabla_aptitud["K"] == K) & tabla_aptitud["apta"]]
        print(f"  K={K}: menor n_obs APTO de la rejilla: {int(aptas['n_obs'].min()) if len(aptas) else 'ninguno'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true", help="verificación contra A6 (ruta F)")
    grupo.add_argument("--medir", type=int, metavar="N", help="tiempo con N réplicas por celda, sin guardarlas")
    grupo.add_argument("--resumen", action="store_true", help="tablas desde lo guardado")
    args = parser.parse_args()
    if args.resumen:
        resumen()
        return
    if not verificar_claves():
        print("Hay claves repetidas: se detiene aquí.")
        return
    if args.verificar:
        verificar()
    elif args.medir:
        medir(args.medir)
    else:
        correr()


if __name__ == "__main__":
    main()
