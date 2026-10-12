"""
make_informe_viabilidad.py

Semana 7: escribe docs/viabilidad_granger_bpoil.md a partir de los CSV ya
generados (guía, R6: ninguna cifra escrita a mano). Cada cifra del
documento pasa por cifra(), que la registra con su valor de origen y su
archivo; al final se verifica que cada cifra esté en el documento y que su
texto coincida con el valor de origen a la precisión mostrada, y se imprime
la lista.

Antes de escribir, verifica con git que cada commit de origen existe
(git cat-file) y que el script productor de cada CSV figura en ese commit
(git ls-tree). Si algo no coincide, no escribe el documento.

A1 de la auditoría no deja CSV: se vuelve a correr aquí
(scripts/auditoria_independiente.py, prueba_a1) para obtener su resultado.
"""

import contextlib
import hashlib
import io
import math
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence.config import OUTPUT_DIR, PROJECT_ROOT

SALIDA = PROJECT_ROOT / "docs" / "viabilidad_granger_bpoil.md"
Q = 0.05
FECHA_FIJACION = "5 de octubre de 2026"
VERSIONES = ["original", "diferenciada", "detendenciada", "residualizada"]
CONTROLES = ["propio", "compartido"]
FAMILIAS_SUST = ["ar_diagonal", "fase_aleatoria"]

# Archivo de origen: (script productor, commits).
ORIGENES: dict[str, tuple[str, list[str]]] = {
    "topic_series_volumen.csv": ("scripts/build_volumen_series.py", ["2147655", "9b06eba"]),
    "resultados_reales.csv": ("scripts/run_resultados_reales.py", ["e15a9c7"]),
    "rutas.csv": ("scripts/run_resultados_reales.py", ["e15a9c7"]),
    "calibracion/calibracion_ar_diagonal.csv": ("scripts/run_calibracion.py", ["b2a047b"]),
    "calibracion/calibracion_fase_aleatoria.csv": ("scripts/run_calibracion.py", ["b2a047b"]),
    "control_positivo/resumen_por_version.csv": ("scripts/run_control_positivo.py", ["07bccd1", "e5c3e85"]),
    "control_positivo/resumen_por_tamano.csv": ("scripts/run_control_positivo.py", ["07bccd1", "e5c3e85"]),
    "diagnostico_dilucion/resumen_por_version.csv": ("scripts/run_diagnostico_dilucion.py", ["c475fff"]),
    "diagnostico_dilucion/resumen_por_variante.csv": ("scripts/run_diagnostico_dilucion.py", ["c475fff"]),
    "auditoria/A2.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A3.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A4_resumen.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A5_por_par.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A5_fracciones.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A5.npz": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
    "auditoria/A6_celdas.csv": ("scripts/auditoria_independiente.py", ["2ae02d7"]),
}

REGISTRO: list[dict[str, object]] = []


def cifra(nombre: str, valor: float | int, archivo: str, decimales: int | None = None, pct: bool = False) -> str:
    """Formatea un valor de origen, lo registra y devuelve su texto."""
    mostrado = 100 * float(valor) if pct else valor
    if decimales is None:
        texto = str(int(mostrado))
        assert float(texto) == float(mostrado), f"{nombre}: {mostrado} no es entero"
    else:
        texto = f"{mostrado:.{decimales}f}"
    REGISTRO.append({"cifra": nombre, "valor": texto, "origen": float(mostrado), "archivo": archivo,
                     "decimales": decimales})
    return texto


def tabla_md(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "|".join("---" for _ in encabezados) + "|"]
    lineas += ["| " + " | ".join(f) + " |" for f in filas]
    return "\n".join(lineas)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def verificar_origenes() -> bool:
    ok = True
    for archivo, (script, commits) in ORIGENES.items():
        if not (OUTPUT_DIR / archivo).exists():
            print(f"  FALTA el archivo data/{archivo}")
            ok = False
        for commit in commits:
            existe = git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
            en_commit = existe and script in git("ls-tree", "-r", "--name-only", commit).stdout.splitlines()
            if not (existe and en_commit):
                print(f"  NO COINCIDE: data/{archivo}: commit {commit} existe={existe}, {script} en el commit={en_commit}")
                ok = False
    return ok


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def leer(archivo: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / archivo, float_precision="round_trip", dtype={"efecto": str})


def requisito(m: int, df_den: int) -> tuple[float, float, float, float]:
    """q/m, F_req = f.isf(q/m, 1, df_den), efecto_req = ln(1 + F_req / df_den) y 1 - exp(-efecto_req)."""
    umbral = Q / m
    F_req = float(stats.f.isf(umbral, 1, df_den))
    efecto_req = math.log(1 + F_req / df_den)
    return umbral, F_req, efecto_req, 1 - math.exp(-efecto_req)


def resultado_a1() -> bool:
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        ok = runpy.run_path(str(PROJECT_ROOT / "scripts" / "auditoria_independiente.py"))["prueba_a1"]()
    return bool(ok)


def main() -> None:
    print("Orígenes (git cat-file y git ls-tree):")
    if not verificar_origenes():
        print("Algún origen no coincide: no se escribe el documento.")
        return
    print("  todos coinciden")

    rr = pd.read_csv(OUTPUT_DIR / "resultados_reales.csv", float_precision="round_trip")  # efecto numérico
    rutas = pd.read_csv(OUTPUT_DIR / "rutas.csv", float_precision="round_trip")
    cp_ver = leer("control_positivo/resumen_por_version.csv")
    cp_tam = leer("control_positivo/resumen_por_tamano.csv")
    dg_ver = leer("diagnostico_dilucion/resumen_por_version.csv")
    dg_var = leer("diagnostico_dilucion/resumen_por_variante.csv")
    a6 = leer("auditoria/A6_celdas.csv")
    a2 = pd.read_csv(OUTPUT_DIR / "auditoria/A2.csv", float_precision="round_trip")
    a3 = pd.read_csv(OUTPUT_DIR / "auditoria/A3.csv")
    a4 = pd.read_csv(OUTPUT_DIR / "auditoria/A4_resumen.csv")
    a5_par = pd.read_csv(OUTPUT_DIR / "auditoria/A5_por_par.csv", float_precision="round_trip")
    a5_frac = pd.read_csv(OUTPUT_DIR / "auditoria/A5_fracciones.csv", float_precision="round_trip")
    with np.load(OUTPUT_DIR / "auditoria/A5.npz") as z:
        a5_p_f = z["p_f"]

    RR, RU = "data/resultados_reales.csv", "data/rutas.csv"
    CPV, CPT = "data/control_positivo/resumen_por_version.csv", "data/control_positivo/resumen_por_tamano.csv"
    DGV = "data/diagnostico_dilucion/resumen_por_version.csv"
    DGA = "data/diagnostico_dilucion/resumen_por_variante.csv"
    A6 = "data/auditoria/A6_celdas.csv"
    TS = "data/topic_series_volumen.csv"
    T_real = len(pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv"))

    # Configuración, desde resultados_reales.csv.
    orig = rr[(rr["control_tipo"] == "propio") & (rr["version"] == "original")]
    n_obs = int(orig["n_obs"].iloc[0])
    df_den_orig = int(orig["df_den"].iloc[0])
    K = len(set(orig["origen"]) | set(orig["destino"]))
    m_ver = {v: int(len(rr[(rr["control_tipo"] == "propio") & (rr["version"] == v)])) for v in VERSIONES}
    df_den_ver = {v: int(rr[(rr["control_tipo"] == "propio") & (rr["version"] == v)]["df_den"].iloc[0]) for v in VERSIONES}
    assert [df_den_ver[v] for v in VERSIONES] == [10, 9, 10, 11], df_den_ver
    B = int(rr["B"].iloc[0])
    T_txt = cifra("T (ventanas): filas de la serie", T_real, TS)
    n_obs_txt = cifra("observaciones por regresión", n_obs, RR)

    # Conclusión.
    def celda_a6(K_: int, T_: int, e: str) -> pd.Series:
        return a6[(a6["K"] == K_) & (a6["T"] == T_) & (a6["efecto"] == e)].iloc[0]

    propio = rr[rr["control_tipo"] == "propio"]
    fila_min = propio.loc[propio["p_remuestreo"].idxmin()]
    n_familias = rr.groupby(["control_tipo", "version"]).ngroups
    conclusion = (
        f"Con bpoil a resolución semanal ({cifra('T en la conclusión: filas de la serie', T_real, TS)} ventanas, "
        f"{cifra('n_obs en la conclusión', n_obs, RR)} observaciones por regresión) y la configuración fijada, "
        "el procedimiento no tiene poder para detectar enlaces entre tópicos. Un enlace plantado que elimina el "
        "55% del error de predicción (efecto 0.8) se detecta en "
        f"{cifra('det_08: A6 K=10 T=23 e=0.8, remuestreo (%)', celda_a6(10, 23, '0.8')['deteccion_BH_remuestreo'], A6, 0, True)}% "
        "de las réplicas con la ruta de remuestreo en el banco de respuesta conocida, y uno que elimina el 33% "
        f"(efecto 0.4), en {cifra('det_04: A6 K=10 T=23 e=0.4, remuestreo (%)', celda_a6(10, 23, '0.4')['deteccion_BH_remuestreo'], A6, 0, True)}%. "
        f"Sobre los datos reales no se rechazó ninguna de las {cifra('n_familias (control x versión)', n_familias, RR)} "
        f"familias de hipótesis; el p mínimo fue {cifra('pmin: p_remuestreo mínimo, control propio', fila_min['p_remuestreo'], RR, 4)} "
        f"frente a un umbral de {cifra('umbral: q/m de la familia del pmin', Q / m_ver[fila_min['version']], RR, 6)}. "
        "El resultado nulo no distingue entre ausencia de enlaces y falta de poder. El mismo código, con "
        f"{cifra('T92', celda_a6(10, 92, '0.4')['T'], A6)} semanas, detecta el "
        f"{cifra('det_92_04: A6 K=10 T=92 e=0.4, remuestreo (%)', celda_a6(10, 92, '0.4')['deteccion_BH_remuestreo'], A6, 0, True)}% "
        "de los enlaces de efecto 0.4, de modo que la limitación es la cantidad de puntos en el tiempo y no la "
        "implementación. En consecuencia, bpoil a resolución semanal no es apto como corpus principal para estimar "
        "causalidad condicional entre tópicos."
    )
    assert (rr["rechaza_BH"] == 0).all(), "hay rechazos en los datos reales: el texto fijo no aplica"

    partes: list[str] = []
    partes.append("# Viabilidad de la prueba de Granger condicional sobre bpoil\n")
    partes.append("Documento generado por `scripts/make_informe_viabilidad.py` a partir de los CSV de `data/`; "
                  "todas las cifras salen de esos archivos o de fórmulas calculadas en el script.\n")

    partes.append("## 1. Resumen\n")
    partes.append("Este documento reúne los resultados de la prueba de Granger condicional entre tópicos sobre "
                  "bpoil (serie de volumen semanal): la corrida sobre los datos reales, la calibración con "
                  "sustitutos, el control positivo, el diagnóstico de dilución y la auditoría independiente de la "
                  "implementación.\n")
    partes.append(conclusion + "\n")

    partes.append("## 2. Configuración fijada\n")
    partes.append(tabla_md(["Parámetro", "Valor"], [
        ["Ventanas (T)", T_txt],
        ["Observaciones por regresión", n_obs_txt],
        ["Tópicos (K)", cifra("K", K, RR)],
        ["Retardo", cifra("retardo = T - n_obs", T_real - n_obs, f"{TS} y {RR}")],
        ["Parámetros por ecuación (K rezagos, control, intercepto)", cifra("parámetros = n_obs - df_den (original)", n_obs - df_den_orig, RR)],
        ["m (original, diferenciada, detendenciada)", cifra("m original", m_ver["original"], RR)],
        ["m (residualizada)", cifra("m residualizada", m_ver["residualizada"], RR)],
        ["q (Benjamini y Hochberg)", cifra("q", Q, "constante Q de scripts/run_resultados_reales.py", 2)],
        ["B (remuestreo salvaje)", cifra("B", B, RR)],
        ["Fecha de fijación", f"{FECHA_FIJACION} (sección 6 de Bitacora_Semana6)"],
    ]) + "\n")

    partes.append("## 3. Resultado sobre los datos reales\n")
    filas = []
    for c in CONTROLES:
        for v in VERSIONES:
            g = rr[(rr["control_tipo"] == c) & (rr["version"] == v)]
            filas.append([c, v, cifra(f"m {c} {v}", len(g), RR), cifra(f"rechazos BH {c} {v}", int(g["rechaza_BH"].sum()), RR),
                          cifra(f"p_remuestreo mínimo {c} {v}", g["p_remuestreo"].min(), RR, 4),
                          cifra(f"efecto máximo {c} {v}", g["efecto"].max(), RR, 3),
                          cifra(f"F máximo {c} {v}", g["F"].max(), RR, 3)])
    partes.append(tabla_md(["Control", "Versión", "m", "Rechazos BH", "p_remuestreo mínimo", "Mayor efecto", "Mayor F"], filas) + "\n")
    partes.append("Efecto y F necesarios para que el menor p de la familia quede por debajo del primer umbral de Benjamini y "
                  "Hochberg (q/m), con F_req = f.isf(q/m, 1, df_den), efecto_req = ln(1 + F_req/df_den) y reducción "
                  "del error = 1 - exp(-efecto_req), frente al mayor efecto observado:\n")
    filas = []
    for v in VERSIONES:
        umbral, F_req, e_req, red = requisito(m_ver[v], df_den_ver[v])
        filas.append([v, cifra(f"m {v}", m_ver[v], RR), cifra(f"q/m {v}", umbral, RR, 6), cifra(f"df_den {v}", df_den_ver[v], RR),
                      cifra(f"F_req {v}", F_req, RR, 2), cifra(f"efecto_req {v}", e_req, RR, 3),
                      cifra(f"reducción requerida {v} (%)", red, RR, 1, True),
                      cifra(f"mayor efecto propio {v}", rr[(rr['control_tipo'] == 'propio') & (rr['version'] == v)]['efecto'].max(), RR, 3),
                      cifra(f"mayor F propio {v}", rr[(rr['control_tipo'] == 'propio') & (rr['version'] == v)]['F'].max(), RR, 3)])
    partes.append(tabla_md(["Versión", "m", "q/m", "df_den", "F_req", "efecto_req", "Reducción del error requerida (%)",
                            "Mayor efecto observado (propio)", "Mayor F observado (propio)"], filas) + "\n")
    partes.append("Comprobación de las dos rutas de significancia (remuestreo y F asintótica):\n")
    filas = [[r["control_tipo"], r["version"],
              cifra(f"rutas fracción r en [0.1, 10] {r['control_tipo']} {r['version']}", r["fraccion_r_en_0.1_10"], RU, 3),
              cifra(f"rutas mediana r {r['control_tipo']} {r['version']}", r["mediana_r"], RU, 3),
              "sí" if r["pasa"] else "no"]
             for _, r in rutas.iterrows()]
    partes.append(tabla_md(["Control", "Versión", "Fracción de pares con 0.1 <= r <= 10", "Mediana de r", "Pasa"], filas) + "\n")

    partes.append("## 4. Calibración con sustitutos\n")
    umbral_cal = stats.binom.isf(0.05 / 8, 1000, 0.05)
    filas = []
    for fam in FAMILIAS_SUST:
        archivo = f"data/calibracion/calibracion_{fam}.csv"
        t = pd.read_csv(PROJECT_ROOT / archivo)
        for v in VERSIONES:
            n = int((t[f"rechazos_{v}"] >= 1).sum())
            filas.append([fam, v, cifra(f"calibración {fam} {v} sustitutos con rechazo", n, archivo),
                          cifra(f"calibración {fam} {v} fracción", n / len(t), archivo, 3),
                          cifra(f"umbral calibración ({fam} {v})", umbral_cal, "fórmula binom.isf(0.05 / 8, 1000, 0.05)")])
    partes.append(tabla_md(["Familia", "Versión", "Sustitutos con al menos un rechazo (de 1000)", "Fracción", "Umbral"], filas) + "\n")

    partes.append("## 5. Control positivo\n")
    filas = [[r.efecto, cifra(f"control positivo tasa regla {r.efecto}", r.tasa_deteccion_regla_2_versiones, CPT, 3),
              cifra(f"control positivo tasa sin no estimables {r.efecto}", r.tasa_deteccion_regla_sin_no_estimables, CPT, 3),
              cifra(f"control positivo réplicas no estimables {r.efecto}", r.n_replicas_no_estimables, CPT)]
             for r in cp_tam.itertuples(index=False)]
    partes.append("Detección del par plantado con la regla de 2 versiones, por tamaño de efecto:\n")
    partes.append(tabla_md(["Efecto nominal", "Detección (regla)", "Detección sin no estimables", "Réplicas no estimables"], filas) + "\n")
    filas = [[r.efecto, r.version, cifra(f"control positivo tasa BH {r.efecto} {r.version}", r.tasa_deteccion_plantado_BH, CPV, 3),
              cifra(f"control positivo efecto observado {r.efecto} {r.version}", r.efecto_observado_medio, CPV, 3)]
             for r in cp_ver.itertuples(index=False)]
    partes.append("Detección del par plantado por Benjamini y Hochberg y efecto observado medio, por tamaño y versión:\n")
    partes.append(tabla_md(["Efecto nominal", "Versión", "Detección (BH)", "Efecto observado medio"], filas) + "\n")

    partes.append("## 6. Diagnóstico de dilución\n")
    partes.append("V0: análisis original; V1: sin control de volumen; V2: sin piso en 0 en la serie de la regresión; "
                  "V3: V1 y V2. Falsos positivos: máximo, sobre las 4 versiones, de réplicas con al menos un rechazo "
                  "con efecto 0.0 (de 200).\n")
    filas = []
    for r in dg_var[dg_var["efecto"] == "1.6"].itertuples(index=False):
        nulos = dg_ver[(dg_ver["variante"] == r.variante) & (dg_ver["ruta"] == r.ruta) & (dg_ver["efecto"] == "0.0")]
        filas.append([r.variante, r.ruta,
                      cifra(f"dilución detección 1.6 {r.variante} {r.ruta}", r.tasa_deteccion_regla_2_versiones, DGA, 3),
                      cifra(f"dilución falsos positivos máx {r.variante} {r.ruta}", nulos["replicas_con_al_menos_un_rechazo"].max(), DGV),
                      cifra(f"dilución umbral {r.variante} {r.ruta}", nulos["umbral_binomial"].iloc[0], DGV),
                      "sí" if r.recupera_poder_valido else "no"])
    partes.append(tabla_md(["Variante", "Ruta", "Detección a efecto 1.6 (regla)", "Falsos positivos a efecto 0.0 (máximo)",
                            "Umbral", "Recupera poder válido"], filas) + "\n")

    partes.append("## 7. Auditoría independiente\n")
    ok_a1 = resultado_a1()
    ok_a2 = bool((a2[["error_rel_F", "error_rel_p"]].to_numpy().max() <= 1e-8) and (a2["df"] == a2["df_barrido"]).all())
    ok_a3 = bool(a3["pasa"].all())
    ok_a4 = bool(a4["cumple"].dropna().astype(bool).all())
    por_par = a5_par["frac_pF<=0.05"].to_numpy()
    ok_a5a = bool(((por_par >= 0.0325) & (por_par <= 0.0700)).all())
    por_replica = np.mean(a5_p_f <= 0.05, axis=1)
    ok_a5b = abs(por_replica.mean() - 0.05) < 3 * por_replica.std(ddof=1) / np.sqrt(len(por_replica))
    det = {(r.K, r.T, r.efecto): r.deteccion_BH_F for r in a6.itertuples(index=False)}
    ok_a6_i = all(det[(10, T2, e)] >= det[(10, T1, e)] - 0.03 for e in ("0.4", "0.8", "1.6") for T1, T2 in ((23, 46), (46, 92))) \
        and all(det[(K_, T_, e2)] >= det[(K_, T_, e1)] - 0.03 for K_, T_ in ((10, 23), (10, 46), (10, 92), (5, 23))
                for e1, e2 in (("0.4", "0.8"), ("0.8", "1.6")))
    ok_a6_ii = all(det[(10, 92, e)] >= 0.80 for e in ("0.8", "1.6"))
    marca = {True: "PASA", False: "FALLA"}
    fila_f = a5_frac[a5_frac["ruta"] == "F"].iloc[0]
    fila_r = a5_frac[a5_frac["ruta"] == "remuestreo"].iloc[0]
    partes.append(tabla_md(["Prueba", "Qué verifica", "Resultado"], [
        ["A1", "cadena de datos, del corpus T17 a la serie", marca[ok_a1]],
        ["A2", "F y p recalculados con statsmodels y scipy (error relativo máximo "
               f"{cifra('A2 error relativo máximo', a2[['error_rel_F', 'error_rel_p']].to_numpy().max(), 'data/auditoria/A2.csv', 13)})", marca[ok_a2]],
        ["A3", "remuestreo vectorizado contra bucle explícito", marca[ok_a3]],
        ["A4", "realismo de los sustitutos (criterio corregido)", marca[ok_a4]],
        ["A5 (a)", "tamaño del test, ruta F, por par", marca[ok_a5a]],
        ["A5 (b)", "tamaño del test, ruta F, conjunto (fracción con p <= 0.05: "
                   f"{cifra('A5 fracción p_F <= 0.05', fila_f['frac_p<=0.05'], 'data/auditoria/A5_fracciones.csv', 4)}; "
                   f"remuestreo: {cifra('A5 fracción p_rem <= 0.05', fila_r['frac_p<=0.05'], 'data/auditoria/A5_fracciones.csv', 4)})",
         marca[ok_a5b]],
        ["A6 (i)", "la detección no decrece al aumentar T ni el efecto", marca[ok_a6_i]],
        ["A6 (ii)", "a T = 92, detección >= 0.80 para efectos 0.8 y 1.6", marca[ok_a6_ii]],
    ]) + "\n")
    partes.append("Detección del par plantado (topic_0 -> topic_1) por Benjamini y Hochberg con la ruta de remuestreo "
                  "en el banco de respuesta conocida (VAR gaussiano con la dinámica de bpoil, sin piso ni control):\n")
    filas = [[cifra(f"A6 K fila {r.K} {r.T} {r.efecto}", r.K, A6), cifra(f"A6 T fila {r.K} {r.T} {r.efecto}", r.T, A6),
              cifra(f"A6 n_obs {r.K} {r.T} {r.efecto}", r.n_obs, A6), r.efecto,
              cifra(f"A6 detección remuestreo {r.K} {r.T} {r.efecto}", r.deteccion_BH_remuestreo, A6, 2)]
             for r in a6.itertuples(index=False)]
    partes.append(tabla_md(["K", "T", "Observaciones", "Efecto", "Detección (remuestreo)"], filas) + "\n")

    partes.append("## 8. Qué se afirma y qué no\n")
    partes.append(
        "Se afirma: con esta configuración y este corpus la prueba no tiene resolución para efectos moderados o "
        "fuertes; la implementación se verificó con una librería independiente y con un banco de respuesta conocida. "
        "No se afirma: que no existan relaciones entre tópicos en bpoil; que bpoil sea inadecuado para las demás "
        "partes del proyecto (línea base, juicio, E2 a E4); que otro corpus o resolución resuelva el problema, lo "
        "cual no se ha medido; ni que los resultados del banco gaussiano se trasladen sin cambios a otros corpus, "
        "porque usa la dinámica de bpoil. Opciones previstas por la guía ante este caso: corpus de mayor densidad "
        "temporal con comparación entre ambos, reducir la familia de hipótesis, otra familia de referencia, o "
        "tamaños de efecto con intervalos de confianza en lugar de enlaces con umbral.\n")

    partes.append("## 9. Archivos y commits de origen\n")
    partes.append(tabla_md(["Archivo", "Script productor", "Commit", "MD5"],
                           [[f"`data/{a}`", f"`{s}`", ", ".join(f"`{c}`" for c in cs), f"`{md5(OUTPUT_DIR / a)}`"]
                            for a, (s, cs) in ORIGENES.items()]) + "\n")
    partes.append("A1 no deja archivo: su resultado se obtiene al generar este documento, corriendo "
                  "`prueba_a1` de `scripts/auditoria_independiente.py`.\n")

    documento = "\n".join(partes)
    assert "\u2014" not in documento, "el documento contiene guiones largos"

    # Verificación de cada cifra.
    print(f"\nVerificación de cifras ({len(REGISTRO)}):")
    todas = True
    for r in REGISTRO:
        tolerancia = 0.0 if r["decimales"] is None else 0.5 * 10 ** -r["decimales"] + 1e-12
        ok = r["valor"] in documento and abs(float(r["valor"]) - r["origen"]) <= tolerancia
        todas &= ok
        print(f"  {'OK   ' if ok else 'FALLA'} {r['cifra']}: {r['valor']}  (origen {r['origen']!r}, {r['archivo']})")
    if not todas:
        print("Alguna cifra no coincide: no se escribe el documento.")
        return
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(documento, encoding="utf-8")
    print(f"\nTodas las cifras coinciden. Escrito {SALIDA} (MD5 {md5(SALIDA)})")


if __name__ == "__main__":
    main()
