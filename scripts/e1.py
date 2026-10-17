"""
e1.py

Semana 8, paso 5, parte B (Bitacora_Semana8.md, sección 29): E1, ¿la
estructura detectada es dirigida? Serie de 2 días con K = 10.

  B1: prueba por pares sin condicionar sobre los datos reales (ambos
      controles), comparada con la condicional ya estimada
      (data/resultados_reales_2d/resultados.csv, que no se vuelve a estimar).
  B2: calibración de la prueba por pares con los 300 sustitutos de volumen
      exógeno del bloque B (sección 26).
  B3: sustituto con impulsor común latente (kappa 0.5, 1.0, 2.0 y 4.0), pruebas por
      pares y condicional.
  B4: 300 permutaciones de las ventanas reales, pruebas por pares y
      condicional.
  B5: asimetría S de los F, con nulo por la ruta F sobre los sustitutos de B2.
  B6: transferencia de entropía (exploratoria) con sustitutos por bloques.

Las réplicas de B2 a B5 corren en procesos independientes (cada una depende
solo de sus claves); OpenBLAS usa 1 hilo en todos los procesos. Salidas en
data/e1/.

Uso:
  python scripts/e1.py --verificar
  python scripts/e1.py --medir 3            tiempo por bloque; no guarda nada
  python scripts/e1.py                      B1, B5, B6, B2, B4 y B3, en ese orden (reanuda)
  python scripts/e1.py --bloque B1|...|B6   un bloque (reanuda)
  python scripts/e1.py --resumen            tablas desde lo guardado
  --reducido                                200 sustitutos en B2 y B4 y 75 por kappa en B3 (regla B7)
"""

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from causal_coherence import candidatas, e1, granger  # noqa: E402
from causal_coherence.config import OUTPUT_DIR  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
SALIDA = OUTPUT_DIR / "e1"
PRINCIPAL = OUTPUT_DIR / "resultados_reales_2d" / "resultados.csv"
B_SUSTITUTOS = OUTPUT_DIR / "validez" / "B_sustitutos.npz"
VERSIONES = e1.VERSIONES
PRUEBAS = e1.PRUEBAS
CONTROLES = ("propio", "compartido")
B = 9999
Q = 0.05
CADA = 25
TRABAJADORES = 12
KAPPAS = ("0.5", "1.0", "2.0", "4.0")
N_COMPLETO = {"B2": 300, "B3": 100, "B4": 300}  # B3: por kappa
N_REDUCIDO = {"B2": 200, "B3": 75, "B4": 200}  # regla B7 (nota del 17 de octubre)
N_B5 = 300
N_TE = 1000
BLOQUE_TE = 4
HORAS_MAX = 4.0
CLAVES_PREVIAS = 4359148  # sección 27, verificación e)
T_VERIFICACION_C = 20000
N_TE_INDEPENDIENTES = 200
PARES_FIJOS_TOPIC_5 = [("topic_5", "topic_0"), ("topic_3", "topic_5")]

_SCRIPTS: dict[str, dict[str, object]] = {}


def script(nombre: str) -> dict[str, object]:
    """Carga un script existente con runpy (sin modificarlo), una sola vez."""
    if nombre not in _SCRIPTS:
        _SCRIPTS[nombre] = runpy.run_path(str(SCRIPTS / f"{nombre}.py"))
    return _SCRIPTS[nombre]


def contexto() -> dict[str, object]:
    """Serie real de 2 días con K = 10 y el ajuste AR(1) del clr del bloque B (paquete_validez.preparar_b)."""
    pv = script("paquete_validez")
    prep = pv["preparar_b"]()
    real = prep["real"]
    descartada = pv["DESCARTE"][10]
    return {"real": real, "n_t": prep["n_t"], "c": prep["c"], "a": prep["a"], "residuos": prep["residuos"],
            "x0": prep["clr"].to_numpy().mean(axis=0), "calent": pv["CALENT_B"], "descartada": descartada,
            "pares": e1.pares_por_version(list(real.columns), descartada), "B": B,
            "sd_e": prep["residuos"].std(axis=0, ddof=1)}


def guardar_tablas(tablas: dict[str, pd.DataFrame]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400):
        for nombre, tabla in tablas.items():
            tabla.to_csv(SALIDA / f"{nombre}.csv", index=False)
            print(f"\n=== {nombre} (data/e1/{nombre}.csv) ===")
            print(tabla.to_string(index=False) if len(tabla) else "  sin filas")


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(SALIDA / nombre, float_precision="round_trip", dtype={"kappa": str})


def principal() -> pd.DataFrame:
    return pd.read_csv(PRINCIPAL, float_precision="round_trip")


def guardar_avance(nombre: str, filas: list[dict[str, object]], arreglos: dict[str, list[np.ndarray]]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    tmp = SALIDA / f"{nombre}.tmp.npz"
    np.savez(tmp, **{k: np.array(v) for k, v in arreglos.items()})
    os.replace(tmp, SALIDA / f"{nombre}.npz")
    tmp = SALIDA / f"{nombre}.tmp.csv"
    pd.DataFrame(filas).to_csv(tmp, index=False)
    os.replace(tmp, SALIDA / f"{nombre}.csv")


def correr(ctx: dict[str, object], nombre: str, tareas: list[tuple[str | int, ...]]) -> None:
    """Réplicas en procesos independientes, con punto de control cada CADA réplicas; reanuda desde lo guardado."""
    ruta_csv, ruta_npz = SALIDA / f"{nombre}.csv", SALIDA / f"{nombre}.npz"
    filas: list[dict[str, object]] = []
    arreglos: dict[str, list[np.ndarray]] = {}
    if ruta_csv.exists():
        tabla = leer(f"{nombre}.csv")
        assert tabla["indice"].tolist() == [int(t[-1]) for t in tareas[:len(tabla)]], f"{nombre}: índices guardados no calzan"
        filas = tabla.to_dict("records")
        with np.load(ruta_npz) as z:
            arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
    pendientes = tareas[len(filas):]
    print(f"{nombre}: {len(filas)} réplicas guardadas, faltan {len(pendientes)}", flush=True)
    if not pendientes:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=TRABAJADORES, initializer=e1.iniciar_trabajador, initargs=(ctx,)) as ex:
        for inicio in range(0, len(pendientes), CADA):
            for fila, arr in ex.map(e1.trabajo, pendientes[inicio:inicio + CADA]):
                filas.append(fila)
                for k, a in arr.items():
                    arreglos.setdefault(k, []).append(a)
            guardar_avance(nombre, filas, arreglos)
            print(f"  {nombre}: {len(filas)}/{len(tareas)} ({time.perf_counter() - t0:.0f} s en esta sesión)", flush=True)


def tareas(bloque: str, N: dict[str, int], kappa: str | None = None) -> list[tuple[str | int, ...]]:
    if bloque == "B3":
        return [("B3", kappa, i) for i in range(N["B3"])]
    n = N_B5 if bloque == "B5" else N[bloque]
    return [(bloque, i) for i in range(n)]


def umbral(n: int) -> float:
    return float(stats.binom.isf(0.05 / 4, n, 0.05))


def por_par(tabla: pd.DataFrame, control: str, version: str, pares: list[tuple[str, str]], columna: str) -> np.ndarray:
    g = tabla[(tabla["control_tipo"] == control) & (tabla["version"] == version)].set_index(["origen", "destino"])
    return g.loc[pares, columna].to_numpy()


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------

def bloque_b1(ctx: dict[str, object]) -> None:
    real, descartada, pares = ctx["real"], ctx["descartada"], ctx["pares"]
    filas, arreglos = [], {}
    for control in CONTROLES:
        fila, arr = e1.analizar(real, None, descartada, pares, ("pares",), B,
                                lambda prueba, v, o, d: ("e1_b1_remuestreo", v, o, d), control=control)
        assert not any(fila[f"no_estimable_pares_{v}"] for v in VERSIONES), "B1: diseño sin rango completo"
        for v in VERSIONES:
            for j, (o, d) in enumerate(pares[v]):
                filas.append({"control_tipo": control, "version": v, "origen": o, "destino": d,
                              "n_obs": int(arr[f"pares_n_obs_{v}"][j]), "efecto": arr[f"pares_efecto_{v}"][j],
                              "F": arr[f"pares_F_{v}"][j], "df_num": 1, "df_den": int(arr[f"pares_df_den_{v}"][j]),
                              "p_teorico": arr[f"pares_p_teo_{v}"][j], "p_remuestreo": arr[f"pares_p_rem_{v}"][j], "B": B,
                              "rechaza_BH": int(arr[f"pares_rechaza_{v}"][j])})
        arreglos.update({f"{control}_{k}": a for k, a in arr.items()})
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA / "B1_pares_real.csv", index=False)
    np.savez(SALIDA / "B1_pares_real.npz", **arreglos,
             **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in VERSIONES})
    resumen_b1()


def resumen_b1() -> None:
    pares_t = leer("B1_pares_real.csv")
    cond = principal()
    claves = ["control_tipo", "version", "origen", "destino"]
    comp = pares_t.merge(cond[claves + ["efecto", "F", "p_remuestreo", "rechaza_BH"]], on=claves,
                         suffixes=("_pares", "_condicional"), validate="one_to_one")
    assert len(comp) == len(pares_t) == len(cond)
    resumen, rechazados, enlaces = [], [], []
    for (control, v), g in comp.groupby(["control_tipo", "version"], sort=False):
        resumen.append({"control_tipo": control, "version": v, "m": len(g),
                        "rechazos_pares": int(g["rechaza_BH_pares"].sum()),
                        "rechazos_condicional": int(g["rechaza_BH_condicional"].sum()),
                        "rechazados_pares_no_condicional": int(((g["rechaza_BH_pares"] == 1) & (g["rechaza_BH_condicional"] == 0)).sum()),
                        "p_remuestreo_min_pares": float(g["p_remuestreo_pares"].min()),
                        "efecto_max_pares": float(g["efecto_pares"].max()), "F_max_pares": float(g["F_pares"].max()),
                        "df_den_pares": int(g["df_den"].iloc[0])})
    for fila in comp[comp["rechaza_BH_pares"] == 1].itertuples(index=False):
        rechazados.append({"control_tipo": fila.control_tipo, "version": fila.version, "origen": fila.origen,
                           "destino": fila.destino, "efecto_pares": fila.efecto_pares, "p_remuestreo_pares": fila.p_remuestreo_pares,
                           "efecto_condicional": fila.efecto_condicional, "p_remuestreo_condicional": fila.p_remuestreo_condicional,
                           "rechaza_condicional": int(fila.rechaza_BH_condicional),
                           "impulsor_comun_candidato": int(fila.rechaza_BH_condicional == 0)})
    regla = []
    for control, g in comp.groupby("control_tipo", sort=False):
        veces_p = g.groupby(["origen", "destino"])["rechaza_BH_pares"].sum()
        veces_c = g.groupby(["origen", "destino"])["rechaza_BH_condicional"].sum()
        declarados = veces_p[veces_p >= e1.MIN_VERSIONES].index
        regla.append({"control_tipo": control, "enlaces_regla_pares": len(declarados),
                      "enlaces_regla_condicional": int((veces_c >= e1.MIN_VERSIONES).sum()),
                      "enlaces_pares_no_condicional": int(sum(veces_c[par] < e1.MIN_VERSIONES for par in declarados))})
        for o, d in declarados:
            fila = {"control_tipo": control, "origen": o, "destino": d, "n_versiones_rechazado_pares": int(veces_p[(o, d)]),
                    "n_versiones_rechazado_condicional": int(veces_c[(o, d)]),
                    "impulsor_comun_candidato": int(veces_c[(o, d)] < e1.MIN_VERSIONES)}
            for v in VERSIONES:
                h = g[(g["version"] == v) & (g["origen"] == o) & (g["destino"] == d)]
                for col in ("efecto_pares", "p_remuestreo_pares", "rechaza_BH_pares", "efecto_condicional",
                            "p_remuestreo_condicional", "rechaza_BH_condicional"):
                    fila[f"{col}_{v}"] = h[col].iloc[0] if len(h) else np.nan
            enlaces.append(fila)
    guardar_tablas({"B1_resumen": pd.DataFrame(resumen), "B1_regla": pd.DataFrame(regla),
                    "B1_rechazados": pd.DataFrame(rechazados), "B1_enlaces_pares": pd.DataFrame(enlaces)})


# ---------------------------------------------------------------------------
# B2, B3 y B4
# ---------------------------------------------------------------------------

def conteos(t: pd.DataFrame, prueba: str) -> list[dict[str, object]]:
    filas = []
    for v in VERSIONES:
        col = t[f"rechazos_{prueba}_{v}"]
        filas.append({"prueba": prueba, "version": v, "replicas": len(t), "replicas_con_rechazo": int((col >= 1).sum()),
                      "fraccion": float((col >= 1).mean()), "rechazos_medios": float(col.mean()),
                      "no_estimables": int(t[f"no_estimable_{prueba}_{v}"].sum())})
    col = t[f"enlaces_regla_{prueba}"]
    filas.append({"prueba": prueba, "version": "regla_2_versiones", "replicas": len(t),
                  "replicas_con_rechazo": int((col >= 1).sum()), "fraccion": float((col >= 1).mean()),
                  "rechazos_medios": float(col.mean()), "no_estimables": np.nan})
    return filas


def resumen_b2(N: dict[str, int]) -> None:
    t = leer("B2_sustitutos.csv")
    assert len(t) == N["B2"], f"B2 incompleto: {len(t)}"
    cal = pd.DataFrame(conteos(t, "pares"))
    cuentas = {v: int(cal[cal["version"] == v]["replicas_con_rechazo"].iloc[0]) for v in ("original", "detendenciada")}
    u = umbral(len(t))
    criterio = pd.DataFrame([{"prueba": "pares", "original": cuentas["original"], "detendenciada": cuentas["detendenciada"],
                              "umbral": u, "pasa_informativo": all(c <= u for c in cuentas.values())}])
    guardar_tablas({"B2_calibracion": cal, "B2_criterio": criterio})


def resumen_b3(N: dict[str, int]) -> None:
    filas, criterio = [], []
    for kappa in KAPPAS:
        t = leer(f"B3_kappa_{kappa}.csv")
        assert len(t) == N["B3"], f"B3 kappa {kappa} incompleto: {len(t)}"
        u = umbral(len(t))
        for prueba in PRUEBAS:
            cuenta = conteos(t, prueba)
            filas += [{"kappa": kappa, **c} for c in cuenta]
            c = {f["version"]: f["replicas_con_rechazo"] for f in cuenta}
            criterio.append({"kappa": kappa, "prueba": prueba, "original": c["original"], "detendenciada": c["detendenciada"],
                             "umbral": u, "ambas_en_o_bajo_umbral": c["original"] <= u and c["detendenciada"] <= u,
                             "ambas_sobre_umbral": c["original"] > u and c["detendenciada"] > u})
    criterio_t = pd.DataFrame(criterio)
    # Lectura (nota del 17 de octubre, sección 29): la de pares se deja engañar en el menor kappa en que supera el umbral
    # en original y detendenciada; la condicional controla si en ese kappa y en los mayores no lo supera en ninguna de
    # las dos; si la de pares no se deja engañar en ningún kappa, B3 no discrimina.
    par = criterio_t[criterio_t["prueba"] == "pares"].set_index("kappa")
    cond = criterio_t[criterio_t["prueba"] == "condicional"].set_index("kappa")
    enganada = [k for k in KAPPAS if bool(par.loc[k, "ambas_sobre_umbral"])]
    if enganada:
        k_min = enganada[0]
        desde = KAPPAS[KAPPAS.index(k_min):]
        controla = all(cond.loc[k, "original"] <= cond.loc[k, "umbral"] and cond.loc[k, "detendenciada"] <= cond.loc[k, "umbral"]
                       for k in desde)
        lectura = pd.DataFrame([{"kappa_minimo_pares_se_deja_enganar": k_min, "kappas_evaluados_condicional": ", ".join(desde),
                                 "condicional_controla_el_impulsor_comun": controla, "resultado": "discrimina"}])
    else:
        lectura = pd.DataFrame([{"kappa_minimo_pares_se_deja_enganar": "ninguno", "kappas_evaluados_condicional": "",
                                 "condicional_controla_el_impulsor_comun": np.nan, "resultado": "no discrimina"}])
    guardar_tablas({"B3_resumen": pd.DataFrame(filas), "B3_criterio": criterio_t, "B3_lectura": lectura})


def resumen_b4(N: dict[str, int]) -> None:
    t = leer("B4_permutaciones.csv")
    assert len(t) == N["B4"], f"B4 incompleto: {len(t)}"
    assert t["max_dif_masa_n_t"].max() <= 1e-9, "B4: la suma de masas no coincide con n_t"
    guardar_tablas({"B4_resumen": pd.DataFrame(conteos(t, "condicional") + conteos(t, "pares"))})


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------

def bloque_b5(ctx: dict[str, object]) -> None:
    assert (SALIDA / "B1_pares_real.csv").exists(), "B5 necesita B1"
    correr(ctx, "B5_nulo", tareas("B5", N_COMPLETO))
    with np.load(SALIDA / "B5_nulo.npz") as w, np.load(B_SUSTITUTOS) as z:
        iguales = all(np.array_equal(w[f"condicional_F_{v}"], z[f"propio_F_{v}"], equal_nan=True) for v in VERSIONES)
    assert iguales, "B5: el F condicional del nulo no coincide con data/validez/B_sustitutos.npz"
    print("B5: el F condicional de los 300 sustitutos coincide con el guardado del bloque B (control propio)")
    resumen_b5(ctx["pares"])


def resumen_b5(pares: dict[str, list[tuple[str, str]]]) -> None:
    nulo = leer("B5_nulo.csv")
    fuentes = {"condicional": principal(), "pares": leer("B1_pares_real.csv")}
    filas, recip = [], []
    for prueba, fuente in fuentes.items():
        for control in CONTROLES:
            for v in VERSIONES:
                S_obs = e1.asimetria(por_par(fuente, control, v, pares[v], "F"), pares[v])
                n = nulo[f"S_{prueba}_{v}"].dropna().to_numpy()
                filas.append({"prueba": prueba, "control_tipo": control, "version": v, "pares_no_ordenados": len(pares[v]) // 2,
                              "S_observado": S_obs, "nulo_media": float(n.mean()), "nulo_p2.5": float(np.percentile(n, 2.5)),
                              "nulo_p97.5": float(np.percentile(n, 97.5)), "p_empirico": float((n >= S_obs).mean()),
                              "n_nulo": len(n), "nota": "" if control == "propio" else "dato: nulo con control propio"})
            g = fuente[fuente["control_tipo"] == control]
            veces = g.groupby(["origen", "destino"])["rechaza_BH"].sum()
            enlaces = set(veces[veces >= e1.MIN_VERSIONES].index)
            medida = e1.reciprocos(enlaces)
            recip.append({"prueba": prueba, "control_tipo": control, "enlaces_declarados": len(enlaces),
                          "fraccion_reciprocos": "no aplica" if medida is None else f"{medida:.4f}"})
    guardar_tablas({"B5_asimetria": pd.DataFrame(filas), "B5_reciprocos": pd.DataFrame(recip)})


# ---------------------------------------------------------------------------
# B6
# ---------------------------------------------------------------------------

def bloque_b6(ctx: dict[str, object]) -> None:
    assert (SALIDA / "B1_pares_real.csv").exists(), "B6 necesita B1"
    real = ctx["real"]
    simbolos = {c: e1.simbolizar(real[c].to_numpy()) for c in real.columns}
    pares = ctx["pares"]["original"]
    filas, sust = [], []
    for o, d in pares:
        r = e1.te_con_sustitutos(simbolos[o], simbolos[d], N_TE, BLOQUE_TE, granger.generador("e1_b6_te", o, d))
        filas.append({"origen": o, "destino": d, "te_observada": r["te_obs"], "te_media_sustitutos": r["te_media_sust"],
                      "te_corregida": r["te_corregida"], "p": r["p"]})
        sust.append(r["te_sust"])
    tabla = pd.DataFrame(filas)
    tabla["rechaza_BH"] = granger.benjamini_hochberg(tabla["p"].to_numpy(), Q, len(pares)).astype(int)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "B6_te.csv", index=False)
    np.savez(SALIDA / "B6_te.npz", te_observada=tabla["te_observada"].to_numpy(), te_corregida=tabla["te_corregida"].to_numpy(),
             p=tabla["p"].to_numpy(), te_sustitutos=np.array(sust), pares=np.array([f"{o}->{d}" for o, d in pares]))
    resumen_b6()


def resumen_b6() -> None:
    te = leer("B6_te.csv")
    resumen = pd.DataFrame([{"m": len(te), "rechazos_BH": int(te["rechaza_BH"].sum()), "p_min": float(te["p"].min()),
                             "te_corregida_media": float(te["te_corregida"].mean()),
                             "te_corregida_min": float(te["te_corregida"].min()),
                             "te_corregida_max": float(te["te_corregida"].max()),
                             "te_observada_media": float(te["te_observada"].mean())}])
    filas = []
    for prueba, fuente in (("condicional", principal()), ("pares", leer("B1_pares_real.csv"))):
        for control in CONTROLES:
            for v in VERSIONES:
                g = fuente[(fuente["control_tipo"] == control) & (fuente["version"] == v)]
                m = te.merge(g[["origen", "destino", "efecto"]], on=["origen", "destino"], validate="one_to_one")
                r = stats.spearmanr(m["te_corregida"], m["efecto"])
                filas.append({"prueba": prueba, "control_tipo": control, "version": v, "n_pares": len(m),
                              "spearman": float(r.statistic), "p": float(r.pvalue)})
    guardar_tablas({"B6_resumen": resumen, "B6_spearman": pd.DataFrame(filas)})


# ---------------------------------------------------------------------------
# Bloques y medición
# ---------------------------------------------------------------------------

def bloque(nombre: str, ctx: dict[str, object], N: dict[str, int]) -> None:
    print(f"\n##### {nombre} #####", flush=True)
    t0 = time.perf_counter()
    if nombre == "B1":
        bloque_b1(ctx)
    elif nombre == "B5":
        bloque_b5(ctx)
    elif nombre == "B6":
        bloque_b6(ctx)
    elif nombre == "B2":
        correr(ctx, "B2_sustitutos", tareas("B2", N))
        resumen_b2(N)
    elif nombre == "B4":
        correr(ctx, "B4_permutaciones", tareas("B4", N))
        resumen_b4(N)
    elif nombre == "B3":
        for kappa in KAPPAS:
            correr(ctx, f"B3_kappa_{kappa}", tareas("B3", N, kappa))
        resumen_b3(N)
    print(f"{nombre}: {time.perf_counter() - t0:.0f} s", flush=True)


def medir(ctx: dict[str, object], n: int, N: dict[str, int]) -> None:
    t0 = time.perf_counter()
    for control in CONTROLES:
        e1.analizar(ctx["real"], None, ctx["descartada"], ctx["pares"], ("pares",), B,
                    lambda prueba, v, o, d: ("e1_b1_remuestreo", v, o, d), control=control)
    t_b1 = time.perf_counter() - t0
    por_replica: dict[str, float] = {}
    for nombre, f in (("B2", lambda i: e1.replica_b2(ctx, i)), ("B3", lambda i: e1.replica_b3(ctx, "1.0", i)),
                      ("B4", lambda i: e1.replica_b4(ctx, i)), ("B5", lambda i: e1.replica_b5(ctx, i))):
        tiempos = []
        for i in range(n):
            t0 = time.perf_counter()
            f(i)
            tiempos.append(time.perf_counter() - t0)
        por_replica[nombre] = float(np.mean(tiempos))
    simbolos = {c: e1.simbolizar(ctx["real"][c].to_numpy()) for c in ctx["real"].columns}
    t0 = time.perf_counter()
    for o, d in ctx["pares"]["original"][:n]:
        e1.te_con_sustitutos(simbolos[o], simbolos[d], N_TE, BLOQUE_TE, granger.generador("e1_b6_te", o, d))
    t_b6 = (time.perf_counter() - t0) / n * len(ctx["pares"]["original"])
    for etiqueta, N_ in (("completo", N_COMPLETO), ("reducido", N_REDUCIDO)):
        serie = {"B1": t_b1, "B5": por_replica["B5"] * N_B5, "B6": t_b6, "B2": por_replica["B2"] * N_["B2"],
                 "B4": por_replica["B4"] * N_["B4"], "B3": por_replica["B3"] * N_["B3"] * len(KAPPAS)}
        en_serie = sum(serie.values())
        paralelo = serie["B1"] + serie["B6"] + sum(serie[b] for b in ("B5", "B2", "B4", "B3")) / TRABAJADORES
        print(f"\n{etiqueta} (B2 {N_['B2']}, B4 {N_['B4']}, B3 {N_['B3']} por kappa, B5 {N_B5}):")
        for b, s in serie.items():
            extra = f" ({por_replica[b]:.2f} s por réplica)" if b in por_replica else ""
            print(f"  {b}: {s / 60:.1f} min en serie{extra}")
        print(f"  total en serie {en_serie / 3600:.2f} h; con {TRABAJADORES} procesos, unos {paralelo / 3600:.2f} h; "
              f"límite {HORAS_MAX:.0f} h: {'supera' if en_serie > HORAS_MAX * 3600 else 'no supera'} (en serie)")


# ---------------------------------------------------------------------------
# Verificaciones
# ---------------------------------------------------------------------------

def claves_previas() -> list[tuple[str | int, ...]]:
    """Las claves de la verificación e) de paquete_validez (previas más las del paquete de validez)."""
    pv = script("paquete_validez")
    conf, cand = pv["_conf"], pv["_cand"]
    serie_real, pares_pv = pv["serie_real"], pv["pares_por_version"]
    previas = conf["claves_previas_y_candidatas"]()
    for cand_ in conf["CONFIRMAR"]:
        K_ = cand["CANDIDATAS"][cand_][4]
        pares_c = pares_pv(list(serie_real(cand_)[0].columns), pv["DESCARTE"][K_])
        for e, n in conf["EFECTOS"].items():
            for i in range(n):
                previas.append(("conf_generacion", cand_, e, str(i)))
                if float(e) > 0:
                    previas.append(("conf_par", cand_, e, str(i)))
                for v, lista in pares_c.items():
                    previas += [("conf_remuestreo", cand_, "C0", e, str(i), v, o, d) for o, d in lista]
    real, _ = serie_real("2D_K10")
    pares10 = pares_pv(list(real.columns), pv["DESCARTE"][10])
    previas += [("real2_remuestreo", v, o, d) for v, lista in pares10.items() for o, d in lista]
    previas.append(("val_verificacion", "c"))
    for i in range(pv["N_SUSTITUTOS"]):
        previas.append(("val_generacion", str(i)))
        for control in CONTROLES:
            previas += [("val_remuestreo", control, str(i), v, o, d) for v, lista in pares10.items() for o, d in lista]
    for nombre, spec in pv["sensibilidades"]().items():
        if isinstance(spec, int):
            continue
        serie, descartada, _ = spec
        for v, lista in pares_pv(list(serie.columns), descartada).items():
            previas += [("sens_remuestreo", nombre, v, o, d) for o, d in lista]
    return previas


def claves_nuevas(pares: dict[str, list[tuple[str, str]]]) -> list[tuple[str | int, ...]]:
    todos = [(v, o, d) for v, lista in pares.items() for o, d in lista]
    nuevas: list[tuple[str | int, ...]] = [("e1_verificacion", "c"), ("e1_verificacion", "c0"),
                                           ("e1_verificacion_te", "enlace"), ("e1_verificacion_te", "enlace_ab"),
                                           ("e1_verificacion_te", "enlace_ba")]
    nuevas += [("e1_verificacion_te", "independiente", str(i)) for i in range(N_TE_INDEPENDIENTES)]
    nuevas += [("e1_b1_remuestreo", v, o, d) for v, o, d in todos]
    for i in range(N_COMPLETO["B2"]):
        nuevas += [("e1_b2_remuestreo", str(i), v, o, d) for v, o, d in todos]
    for kappa in KAPPAS:
        for i in range(N_COMPLETO["B3"]):
            nuevas.append(("e1_b3_generacion", kappa, str(i)))
            for prueba in PRUEBAS:
                nuevas += [("e1_b3_remuestreo", prueba, kappa, str(i), v, o, d) for v, o, d in todos]
    for i in range(N_COMPLETO["B4"]):
        nuevas.append(("e1_b4_permutacion", str(i)))
        for prueba in PRUEBAS:
            nuevas += [("e1_b4_remuestreo", prueba, str(i), v, o, d) for v, o, d in todos]
    nuevas += [("e1_b6_te", o, d) for o, d in pares["original"]]
    return nuevas


def iguales(r1: tuple[dict[str, object], dict[str, np.ndarray]], r2: tuple[dict[str, object], dict[str, np.ndarray]]) -> bool:
    (f1, a1), (f2, a2) = r1, r2
    filas = f1.keys() == f2.keys() and all(
        (isinstance(f1[k], float) and math.isnan(f1[k]) and math.isnan(f2[k])) or f1[k] == f2[k] for k in f1)
    return filas and a1.keys() == a2.keys() and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)


def f_statsmodels_pares(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> tuple[float, float, float]:
    import statsmodels.api as sm
    datos = pd.DataFrame({"y": serie[destino].iloc[1:].to_numpy(), "destino_rezago": serie[destino].iloc[:-1].to_numpy(),
                          "origen_rezago": serie[origen].iloc[:-1].to_numpy(), "control": control.iloc[1:].to_numpy()})
    X_f = sm.add_constant(datos[["destino_rezago", "origen_rezago", "control"]])
    completo = sm.OLS(datos["y"], X_f).fit()
    restringido = sm.OLS(datos["y"], X_f.drop(columns="origen_rezago")).fit()
    F, p, _ = completo.compare_f_test(restringido)
    return float(F), float(p), float(completo.df_resid)


def coeficientes_impulsor(x: pd.DataFrame, control: pd.DataFrame | None, origen: str,
                          destino: str) -> tuple[float, float, float, float]:
    """Coeficiente y t de origen_{t-1} por MCO (statsmodels): regresión por pares y condicional (todos los rezagos)."""
    import statsmodels.api as sm
    y = x[destino].iloc[1:].to_numpy()
    rez = x.iloc[:-1].reset_index(drop=True)
    extra = {} if control is None else {"control": control[destino].iloc[1:].to_numpy()}
    X_p = sm.add_constant(pd.DataFrame({"destino_rezago": rez[destino], "origen_rezago": rez[origen], **extra}))
    X_c = sm.add_constant(pd.concat([rez.add_suffix("_rezago"), pd.DataFrame(extra)], axis=1))
    aj_p, aj_c = sm.OLS(y, X_p).fit(), sm.OLS(y, X_c).fit()
    return (float(aj_p.params["origen_rezago"]), float(aj_p.tvalues["origen_rezago"]),
            float(aj_c.params[f"{origen}_rezago"]), float(aj_c.tvalues[f"{origen}_rezago"]))


def verificar(ctx: dict[str, object]) -> bool:
    import statsmodels.api as sm
    res: dict[str, bool] = {}
    real, n_t, descartada, pares = ctx["real"], ctx["n_t"], ctx["descartada"], ctx["pares"]

    print("=== a) La condicional reproduce la estimación principal (data/resultados_reales_2d/resultados.csv) ===")
    prin = principal()
    ok_a, n_filas = True, 0
    for control in CONTROLES:
        _, arr = e1.analizar(real, None, descartada, pares, ("condicional",), B,
                             lambda prueba, v, o, d: ("real2_remuestreo", v, o, d), control=control)
        for v in VERSIONES:
            g = prin[(prin["control_tipo"] == control) & (prin["version"] == v)]
            assert list(zip(g["origen"], g["destino"])) == pares[v], f"orden de pares distinto en {control} {v}"
            for col, k in (("efecto", "efecto"), ("F", "F"), ("p_teorico", "p_teo"), ("p_remuestreo", "p_rem"),
                           ("n_obs", "n_obs"), ("df_den", "df_den"), ("rechaza_BH", "rechaza")):
                ok_a &= np.array_equal(g[col].to_numpy(dtype=float), arr[f"condicional_{k}_{v}"].astype(float))
            n_filas += len(g)
    res["a"] = ok_a and n_filas == len(prin)
    print(f"  {n_filas} filas (F, efecto, p teórico, p de remuestreo, n_obs, df_den y rechazos) idénticas: "
          f"{'OK' if res['a'] else 'FALLA'}")

    print("\n=== b) Prueba por pares contra statsmodels (OLS y compare_f_test) ===")
    versiones = candidatas.construir_versiones_k(real, descartada)
    max_F, max_p, ok_df, n_b = 0.0, 0.0, True, 0
    for control in CONTROLES:
        _, arr = e1.analizar(real, None, descartada, pares, ("pares",), 0, None, control=control)
        for v in VERSIONES:
            F = arr[f"pares_F_{v}"]
            elegidos = [pares[v][j] for j in np.argsort(-F, kind="stable")[:5]]
            elegidos += [par for par in PARES_FIJOS_TOPIC_5 if par in pares[v] and par not in elegidos]
            for o, d in elegidos:
                j = pares[v].index((o, d))
                serie, compartido, propio = versiones[v]
                F_sm, p_sm, df_sm = f_statsmodels_pares(serie, propio[d] if control == "propio" else compartido, o, d)
                max_F = max(max_F, abs(F[j] - F_sm) / abs(F_sm))
                max_p = max(max_p, abs(arr[f"pares_p_teo_{v}"][j] - p_sm) / abs(p_sm))
                ok_df &= arr[f"pares_df_den_{v}"][j] == df_sm == arr[f"pares_n_obs_{v}"][j] - 4
                n_b += 1
    res["b"] = max_F <= 1e-8 and max_p <= 1e-8 and ok_df
    print(f"  {n_b} pares (2 controles; por versión los 5 de mayor F y los fijos con topic_5): error relativo máximo "
          f"F {max_F:.2e}, p {max_p:.2e}; df_den = n_obs - 4: {ok_df}: {'OK' if res['b'] else 'FALLA'}")

    print("\n=== c) Impulsor común, kappa = 1.0, T = 20000, n_t real en ciclo: coeficiente de A_{t-1} ===")
    lam = 1.0 * ctx["sd_e"] * e1.U_IMPULSOR
    z = e1.simular_impulsor(ctx["c"], ctx["a"], ctx["residuos"], ctx["x0"], T_VERIFICACION_C, ctx["calent"], lam,
                            e1.PHI_IMPULSOR, granger.generador("e1_verificacion", "c"))
    n_ciclo = pd.Series(np.resize(n_t.to_numpy(dtype=float), T_VERIFICACION_C), name="n_t")
    x = e1.a_series(z, n_ciclo, real.columns)
    propio_c = np.log1p(np.expm1(x).rsub(n_ciclo, axis=0))
    o, d = "topic_0", "topic_1"
    coef_p, t_p, coef_c, t_c = coeficientes_impulsor(x, propio_c, o, d)
    print(f"  {o} -> {d} (u = +1 y +1): por pares coef = {coef_p:.5f} (t = {t_p:.2f}); condicional coef = {coef_c:.5f} "
          f"(t = {t_c:.2f})")
    columnas = list(real.columns)
    mismos = [(a_, b_) for a_, b_ in itertools.permutations(columnas, 2)
              if e1.U_IMPULSOR[columnas.index(a_)] == e1.U_IMPULSOR[columnas.index(b_)]]
    z_df = pd.DataFrame(z, columns=columnas)
    for etiqueta, serie_i, control_i in (("log1p(masa), control propio", x, propio_c), ("clr, sin control", z_df, None)):
        coefs = [coeficientes_impulsor(serie_i, control_i, a_, b_) for a_, b_ in mismos]
        baja = sum(abs(cc) < abs(cp) for cp, _, cc, _ in coefs)
        print(f"  información: en {etiqueta}, el |coef| baja al condicionar en {baja} de {len(mismos)} pares con u del "
              f"mismo signo (mediana |t| por pares {np.median([abs(f[1]) for f in coefs]):.1f}, condicional "
              f"{np.median([abs(f[3]) for f in coefs]):.1f})")
    z0 = e1.simular_impulsor(ctx["c"], ctx["a"], ctx["residuos"], ctx["x0"], len(real), ctx["calent"], np.zeros(len(lam)),
                             e1.PHI_IMPULSOR, granger.generador("e1_verificacion", "c0"))
    z_ref = candidatas.simular(ctx["c"], ctx["a"], ctx["residuos"], ctx["x0"], len(real), ctx["calent"], 0, 1, 0.0, 0.0,
                               granger.generador("e1_verificacion", "c0"))
    sin_impulsor = np.array_equal(z0, z_ref)
    print(f"  con lambda = 0 el simulador da la misma serie que candidatas.simular: {sin_impulsor}")
    res["c"] = abs(t_p) > 5 and abs(coef_c) < abs(coef_p) and sin_impulsor
    print(f"  |t| por pares > 5: {abs(t_p) > 5}; |coef| condicional < |coef| por pares: {abs(coef_c) < abs(coef_p)}: "
          f"{'OK' if res['c'] else 'FALLA'}")

    print("\n=== d) Permutaciones: suma de masas por ventana igual a n_t ===")
    dif, filas_ok = 0.0, True
    for i in range(N_COMPLETO["B4"]):
        xp, n_perm, orden = e1.permutacion_b4(ctx, i)
        dif = max(dif, float(np.abs(np.expm1(xp).sum(axis=1).to_numpy() - n_perm.to_numpy(dtype=float)).max()))
        filas_ok &= np.array_equal(np.sort(orden), np.arange(len(real))) and np.array_equal(xp.to_numpy(), real.to_numpy()[orden])
    res["d"] = dif <= 1e-9 and filas_ok
    print(f"  {N_COMPLETO['B4']} permutaciones: max |suma de masas - n_t| = {dif:.3e}; filas completas conservadas: "
          f"{filas_ok}: {'OK' if res['d'] else 'FALLA'}")

    print("\n=== e) Misma réplica dos veces, proceso principal contra proceso del grupo, y unicidad de claves ===")
    ok_e = True
    referencia = {}
    for nombre, args in (("B2", (0,)), ("B3", ("1.0", 0)), ("B4", (0,)), ("B5", (0,))):
        r1, r2 = e1.REPLICAS[nombre](ctx, *args), e1.REPLICAS[nombre](ctx, *args)
        referencia[nombre] = r1
        print(f"  {nombre} {args}: {'iguales' if iguales(r1, r2) else 'DISTINTAS'}")
        ok_e &= iguales(r1, r2)
    with ProcessPoolExecutor(max_workers=2, initializer=e1.iniciar_trabajador, initargs=(ctx,)) as ex:
        r_grupo = ex.submit(e1.trabajo, ("B2", 0)).result()
    print(f"  B2 (0) en un proceso del grupo: {'igual' if iguales(r_grupo, referencia['B2']) else 'DISTINTA'}")
    ok_e &= iguales(r_grupo, referencia["B2"])
    previas = claves_previas()
    nuevas = claves_nuevas(pares)
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    ok_claves = len(previas) == CLAVES_PREVIAS and distintas == len(todas)
    print(f"  claves: {len(todas)} ({len(previas)} previas, esperadas {CLAVES_PREVIAS}, + {len(nuevas)} nuevas), "
          f"distintas: {distintas}")
    res["e"] = ok_e and ok_claves
    print(f"  {'OK' if res['e'] else 'FALLA'}")

    print("\n=== f) B2 usa los mismos sustitutos del bloque B ===")
    pv = script("paquete_validez")
    prep = pv["preparar_b"]()
    ok_f = True
    with np.load(B_SUSTITUTOS) as zb:
        for i in range(3):
            x_e1 = e1.sustituto_b2(ctx, i)
            x_pv, _ = pv["sustituto_b"](prep, i)
            ok_f &= x_e1.equals(x_pv)
            _, arr = e1.analizar(x_e1, n_t, descartada, pares, ("condicional",), 0, None)
            ok_f &= all(np.array_equal(arr[f"condicional_F_{v}"], zb[f"propio_F_{v}"][i]) for v in VERSIONES)
    res["f"] = ok_f
    print(f"  sustitutos 0 a 2 idénticos a paquete_validez.sustituto_b y F condicional igual al guardado: "
          f"{'OK' if ok_f else 'FALLA'}")

    print("\n=== g) Transferencia de entropía (verificación previa de B6) ===")
    rng = granger.generador("e1_verificacion_te", "enlace")
    dif_te = 0.0
    for _ in range(20):
        fx, fy = rng.integers(0, 2, size=53), rng.integers(0, 2, size=53)
        dif_te = max(dif_te, abs(float(e1.te_conteo(fx, fy)[0]) - e1.te_lazo(fx, fy)))
    A = rng.standard_normal(500)
    ruido = rng.standard_normal(500)
    Bs = np.concatenate([[0.5 * ruido[0]], A[:-1] + 0.5 * ruido[1:]])
    sA, sB = e1.simbolizar(A), e1.simbolizar(Bs)
    r_ab = e1.te_con_sustitutos(sA, sB, N_TE, BLOQUE_TE, granger.generador("e1_verificacion_te", "enlace_ab"))
    r_ba = e1.te_con_sustitutos(sB, sA, N_TE, BLOQUE_TE, granger.generador("e1_verificacion_te", "enlace_ba"))
    ok_enlace = r_ab["te_obs"] > r_ba["te_obs"] and r_ab["p"] < 0.05
    print(f"  te_conteo contra el bucle explícito (20 pares): diferencia máxima {dif_te:.2e}")
    print(f"  enlace A -> B (n = 500): TE(A->B) = {r_ab['te_obs']:.4f} bits (p = {r_ab['p']:.4f}), "
          f"TE(B->A) = {r_ba['te_obs']:.4f} bits (p = {r_ba['p']:.4f}): {'OK' if ok_enlace else 'FALLA'}")
    corregidas, observadas = [], []
    for i in range(N_TE_INDEPENDIENTES):
        rng_i = granger.generador("e1_verificacion_te", "independiente", str(i))
        fx, fy = e1.simbolizar(rng_i.standard_normal(53)), e1.simbolizar(rng_i.standard_normal(53))
        r = e1.te_con_sustitutos(fx, fy, N_TE, BLOQUE_TE, rng_i)
        corregidas.append(r["te_corregida"])
        observadas.append(r["te_obs"])
    media, ee = float(np.mean(corregidas)), float(np.std(corregidas, ddof=1) / math.sqrt(len(corregidas)))
    ok_indep = abs(media) <= 3 * ee
    print(f"  {N_TE_INDEPENDIENTES} pares independientes (n = 53): TE observada media {np.mean(observadas):.4f} bits; "
          f"TE corregida media {media:.5f} (error estándar {ee:.5f}); |media| <= 3 EE: {'OK' if ok_indep else 'FALLA'}")
    res["g"] = dif_te <= 1e-12 and ok_enlace and ok_indep

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


def pares_guardados() -> dict[str, list[tuple[str, str]]]:
    with np.load(SALIDA / "B1_pares_real.npz") as z:
        return {v: [tuple(str(s).split("->")) for s in z[f"pares_{v}"]] for v in VERSIONES}


def resumen(N: dict[str, int]) -> None:
    if (SALIDA / "B1_pares_real.csv").exists():
        resumen_b1()
    if (SALIDA / "B5_nulo.csv").exists():
        resumen_b5(pares_guardados())
    if (SALIDA / "B6_te.csv").exists():
        resumen_b6()
    if (SALIDA / "B2_sustitutos.csv").exists():
        resumen_b2(N)
    if (SALIDA / "B4_permutaciones.csv").exists():
        resumen_b4(N)
    if all((SALIDA / f"B3_kappa_{k}.csv").exists() for k in KAPPAS):
        resumen_b3(N)


def main() -> None:
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--verificar", action="store_true")
    grupo.add_argument("--medir", type=int, metavar="N")
    grupo.add_argument("--resumen", action="store_true")
    grupo.add_argument("--bloque", choices=["B1", "B2", "B3", "B4", "B5", "B6"])
    parser.add_argument("--reducido", action="store_true", help="200 sustitutos en B2 y B4 y 75 por kappa en B3 (regla B7)")
    args = parser.parse_args()
    N = N_REDUCIDO if args.reducido else N_COMPLETO
    if args.resumen:
        resumen(N)
        return
    ctx = contexto()
    if args.verificar:
        verificar(ctx)
    elif args.medir:
        medir(ctx, args.medir, N)
    elif args.bloque:
        bloque(args.bloque, ctx, N)
    else:
        for nombre in ("B1", "B5", "B6", "B2", "B4", "B3"):
            bloque(nombre, ctx, N)


if __name__ == "__main__":
    main()
