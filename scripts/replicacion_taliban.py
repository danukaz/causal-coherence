"""
replicacion_taliban.py

Semana 8, paso 6 (Bitacora_Semana8.md, sección 32): réplica preregistrada
del proceso de Granger condicional sobre el subconjunto NEWS-TLS Entities
(Taliban) del corpus de Afganistán. Salidas en data/taliban/.

Etapas (en este orden):
  auditoria    auditoría previa (solo reporte), exclusiones (fecha corrupta o
               contenido vacío; para si pasan del 5%) y preprocesamiento igual
               al de bpoil.
  modelo       RollingLDA con chunks anuales para un K (--K), semilla 42
               salvo --semilla.
  series       grillas de 14 y 30 días, recorte, mayor racha, series de cada
               candidata, tópico descartado y chequeos de R1.
  verificar    verificaciones (a) a (e) de la sección 32.
  medir        tiempo de la evaluación (--n réplicas por familia); no guarda.
  evaluar      betas, sustituto realista, fase aleatorizada y control positivo
               por candidata (reanuda), aptitud y regla de elección.
  real         estimación sobre datos reales, una sola vez, con la elegida.
  e1           batería de E1 con la elegida (reanuda).
  estabilidad  solo si la estimación declaró enlaces; si no, "no aplica".
  resumen      vuelve a imprimir las tablas de la evaluación.

Uso:
  python scripts/replicacion_taliban.py --etapa <etapa> [--K 10] [--semilla 42] [--n 3] [--reducido]
"""

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from causal_coherence import candidatas, e1, granger, surrogates, taliban  # noqa: E402
from causal_coherence.config import OUTPUT_DIR  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
SALIDA = OUTPUT_DIR / "taliban"
RUTA_DOCUMENTOS = SALIDA / "documentos.csv"
RUTA_TOKENS = SALIDA / "tokens.pkl"


_SCRIPTS_CARGADOS: dict[str, dict[str, object]] = {}


def cargar_script(nombre: str) -> dict[str, object]:
    """Un script existente con runpy (sin modificarlo), una sola vez."""
    if nombre not in _SCRIPTS_CARGADOS:
        _SCRIPTS_CARGADOS[nombre] = runpy.run_path(str(SCRIPTS / f"{nombre}.py"))
    return _SCRIPTS_CARGADOS[nombre]


def guardar_tablas(tablas: dict[str, pd.DataFrame], cabeza: int = 40) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.max_colwidth", 70):
        for nombre, tabla in tablas.items():
            tabla.to_csv(SALIDA / f"{nombre}.csv", index=False)
            print(f"\n=== {nombre} (data/taliban/{nombre}.csv, {len(tabla)} filas) ===")
            print(tabla.head(cabeza).to_string(index=False) if len(tabla) else "  sin filas")


def ruta_modelo(K: int, semilla: int = 42) -> Path:
    sufijo = "" if semilla == 42 else f"_semilla{semilla}"
    return SALIDA / f"roll_lda_taliban_K{K}{sufijo}.pickle"


def etapa_auditoria() -> bool:
    from causal_coherence.preprocessing import build_english_pipeline, get_english_stopwords, preprocess_batch

    df = taliban.documentos_taliban()
    tablas = taliban.auditoria(df)
    guardar_tablas(tablas)
    resumen = tablas["A_resumen"].iloc[0]
    if resumen["fraccion_excluida"] > taliban.MAX_EXCLUIDOS:
        print(f"\nLa auditoría excluye {resumen['fraccion_excluida']:.1%} (> 5%): se detiene.")
        return False
    excluir = set(tablas["A_excluidos"]["id"])
    datos = df[~df["id"].isin(excluir)].sort_values("date", kind="stable").reset_index(drop=True)
    print(f"\nPreprocesando {len(datos)} documentos (spaCy en_core_web_sm y stopwords de NLTK, como en bpoil)...", flush=True)
    t0 = time.perf_counter()
    tokens = preprocess_batch(datos["content"].tolist(), build_english_pipeline(), get_english_stopwords())
    print(f"  {time.perf_counter() - t0:.0f} s; documentos sin tokens: {sum(len(t) == 0 for t in tokens)}")
    datos[["id", "date", "title"]].to_csv(RUTA_DOCUMENTOS, index=False)
    with open(RUTA_TOKENS, "wb") as f:
        pickle.dump(tokens, f)
    return True


def etapa_modelo(K: int, semilla: int) -> None:
    from causal_coherence.topic_model import check_hash_seed, chunk_summary
    from ttta.methods.rolling_lda import RollingLDA

    check_hash_seed()
    ruta = ruta_modelo(K, semilla)
    if ruta.exists():
        print(f"Ya existe {ruta}: no se vuelve a ajustar.")
        return
    datos = pd.read_csv(RUTA_DOCUMENTOS, parse_dates=["date"])
    with open(RUTA_TOKENS, "rb") as f:
        datos["preprocessed_text"] = pickle.load(f)
    config = taliban.configuracion_lda(K, semilla)
    print(f"RollingLDA K = {K}, semilla {semilla}, gamma = {config['gamma']}, chunks: "
          f"{[d.date().isoformat() for d in config['how']]}", flush=True)
    t0 = time.perf_counter()
    roll = RollingLDA(**config)
    roll.fit(datos.copy(), text_column="preprocessed_text", date_column="date")
    minutos = (time.perf_counter() - t0) / 60
    roll.save(str(ruta))
    chunks = chunk_summary(roll, datos).reset_index()
    assert int(chunks["documentos"].sum()) == len(datos), "los chunks no cubren todos los documentos"
    sufijo = f"K{K}" if semilla == 42 else f"K{K}_semilla{semilla}"
    palabras = roll.top_words(number=10)
    palabras = palabras if isinstance(palabras, pd.DataFrame) else pd.DataFrame(palabras)
    guardar_tablas({f"M_chunks_{sufijo}": chunks, f"M_palabras_{sufijo}": palabras.reset_index(),
                    f"M_tiempo_{sufijo}": pd.DataFrame([{"K": K, "semilla": semilla, "minutos": minutos}])})
    print(f"\nModelo guardado en {ruta} ({minutos:.1f} min)")




# ---------------------------------------------------------------------------
# Series, R1 y candidatas (secciones 2 a 4)
# ---------------------------------------------------------------------------

SERIES = SALIDA / "series"
EVAL = SALIDA / "evaluacion"
REAL = SALIDA / "real"
E1_DIR = SALIDA / "e1"
B = 9999
CADA = 25
TRABAJADORES = 12
KS = (10, 5)
N_COMPLETO = {"realista": 300, "fase": 300, "0.4": 100, "0.8": 100}
N_REDUCIDO = {"realista": 200, "fase": 200, "0.4": 60, "0.8": 60}
N_E1 = {"B2": 300, "B3": 100, "B4": 300}
N_B5 = 300
KAPPAS = ("0.5", "1.0", "2.0", "4.0")
DETECCION_MIN = 0.80
HORAS_MAX = 12.0
CLAVES_PREVIAS = 4941885  # sección 30, verificación e)
VARIACIONES_ESTABILIDAD = ("otra_grilla", "sin_5_primeras", "sin_5_ultimas", "retardo_2", "semilla_43", "semilla_44")


def umbral_aptitud(n: int) -> float:
    return float(stats.binom.isf(0.05 / 8, n, 0.05))


def fechas_documentos() -> pd.Series:
    return pd.read_csv(RUTA_DOCUMENTOS, parse_dates=["date"])["date"].dt.normalize()


@lru_cache(maxsize=None)
def theta_modelo(K: int, semilla: int = 42) -> np.ndarray:
    from ttta.methods.rolling_lda import RollingLDA
    roll = RollingLDA(**taliban.configuracion_lda(K, semilla))
    roll.load(str(ruta_modelo(K, semilla)))
    theta = roll.get_document_topic_matrix()
    assert theta.shape == (len(fechas_documentos()), K), f"theta de K = {K} no calza con los documentos"
    return theta / theta.sum(axis=1, keepdims=True)


def palabras_topicos(K: int, semilla: int = 42) -> dict[str, list[str]]:
    """Las 10 palabras de cada tópico de M_palabras (roll.top_words(number = 10)), como topic_k -> lista."""
    sufijo = f"K{K}" if semilla == 42 else f"K{K}_semilla{semilla}"
    tabla = pd.read_csv(SALIDA / f"M_palabras_{sufijo}.csv")
    columnas = [c for c in tabla.columns if c != "index"]
    assert len(columnas) == K, f"M_palabras_{sufijo}: {len(columnas)} columnas"
    return {f"topic_{k}": tabla[col].astype(str).tolist() for k, col in enumerate(columnas)}


def candidatas_definidas() -> dict[str, tuple[str, int]]:
    """Nombre -> (grilla, K) para las grillas que alcanzan 45 observaciones (lee S_grillas.csv)."""
    grillas = pd.read_csv(SALIDA / "S_grillas.csv")
    aptas = grillas[grillas["apta_por_observaciones"]]["grilla"].tolist()
    return {f"{taliban.GRILLAS[g][1]}D_K{K}": (g, K) for g in aptas for K in KS}


@lru_cache(maxsize=None)
def serie(nombre: str) -> tuple[pd.DataFrame, pd.Series]:
    t = pd.read_csv(SERIES / f"{nombre}.csv", parse_dates=["inicio"], index_col="inicio", float_precision="round_trip")
    return t.drop(columns="n_t"), t["n_t"]


def descartada(nombre: str) -> str:
    c = pd.read_csv(SALIDA / "S_candidatas.csv").set_index("candidata")
    return str(c.loc[nombre, "descartada"])


def contexto(nombre: str, con_betas: bool = True) -> dict[str, object]:
    real, n_t = serie(nombre)
    grilla, K = candidatas_definidas()[nombre]
    desc = descartada(nombre)
    ctx: dict[str, object] = {"nombre": nombre, "real": real, "n_t": n_t, "descartada": desc, "K": K, "grilla": grilla,
                              "pares": taliban.pares_por_version(list(real.columns), desc), "B": B,
                              "calent": candidatas.calentamiento(taliban.GRILLAS[grilla][1]),
                              "clr": taliban.preparar_clr(real, n_t), "ar": surrogates.ajustar_ar1(real)}
    if con_betas and (EVAL / "betas.csv").exists():
        b = pd.read_csv(EVAL / "betas.csv", dtype={"efecto": str}, float_precision="round_trip")
        b = b[b["candidata"] == nombre]
        ctx["betas"] = {(f.efecto, f.origen, f.destino): f.beta for f in b.itertuples(index=False)}
    return ctx


def etapa_series() -> None:
    test_column = cargar_script("check_stationarity")["test_column"]
    fechas = fechas_documentos()
    resumen, candidatas_t = [], []
    for grilla in taliban.GRILLAS:
        r = taliban.recorte(fechas, grilla)
        resumen.append(r["resumen"])
        guardar_tablas({f"S_recorte_{taliban.GRILLAS[grilla][1]}D": r["detalle"]}, cabeza=200)
    guardar_tablas({"S_grillas": pd.DataFrame(resumen)})
    SERIES.mkdir(parents=True, exist_ok=True)
    palabras_t = []
    for nombre, (grilla, K) in candidatas_definidas().items():
        real, n_t = taliban.serie_candidata(theta_modelo(K), fechas, grilla)
        palabras = palabras_topicos(K)
        masa = np.expm1(real).sum(axis=0)
        desc, criterio, cuenta = taliban.topico_plantilla(palabras, masa)
        for t in real.columns:
            palabras_t.append({"candidata": nombre, "topico": t, "masa_total": float(masa[t]),
                               "palabras_plantilla": cuenta[t], "palabras": ", ".join(palabras[t]),
                               "descartado": t == desc})
        candidatas_t.append({"candidata": nombre, "grilla": grilla, "K": K, "ventanas": len(real), "observaciones": len(real) - 1,
                             "descartada": desc, "criterio": criterio})
        real.assign(n_t=n_t).rename_axis("inicio").to_csv(SERIES / f"{nombre}.csv")
        # R1 (bloque A de paquete_validez).
        versiones = candidatas.construir_versiones_k(real, desc)
        a1 = pd.DataFrame([{"serie": v, "topico": t, **test_column(versiones[v][0][t].dropna())}
                           for v in taliban.VERSIONES for t in versiones[v][0].columns])
        pca = PCA().fit(StandardScaler().fit_transform(real.to_numpy()))
        a2 = pd.DataFrame({"componente": np.arange(1, real.shape[1] + 1), "varianza_explicada": pca.explained_variance_ratio_,
                           "acumulada": np.cumsum(pca.explained_variance_ratio_)})
        cargas = pd.DataFrame({"topico": real.columns, "carga_componente_1": pca.components_[0]})
        _, _, residuos = surrogates.ajustar_ar1(real)
        n_res = n_t.to_numpy()[1:].astype(float)
        a3 = pd.DataFrame([{"topico": t, "spearman_abs_residuo_n_t": float(stats.spearmanr(np.abs(residuos[:, k]), n_res).statistic),
                            "p_n_t": float(stats.spearmanr(np.abs(residuos[:, k]), n_res).pvalue)}
                           for k, t in enumerate(real.columns)])
        guardar_tablas({f"R1_estacionariedad_{nombre}": a1, f"R1_pca_varianza_{nombre}": a2, f"R1_pca_cargas_{nombre}": cargas,
                        f"R1_spearman_{nombre}": a3})
        resumen_r1 = []
        for v in taliban.VERSIONES:
            g = a1[a1["serie"] == v]
            resumen_r1.append({"candidata": nombre, "version": v, "topicos": len(g), "adf_rechaza": int((g["adf_pvalue"] < 0.05).sum()),
                               "kpss_rechaza": int((g["kpss_pvalue"] < 0.05).sum())})
        guardar_tablas({f"R1_resumen_{nombre}": pd.DataFrame(resumen_r1)})
    guardar_tablas({"S_candidatas": pd.DataFrame(candidatas_t), "S_palabras": pd.DataFrame(palabras_t)}, cabeza=200)


# ---------------------------------------------------------------------------
# Verificaciones (sección 9)
# ---------------------------------------------------------------------------

def f_statsmodels(serie_v: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> tuple[float, float]:
    import statsmodels.api as sm
    rezagos = serie_v.shift(1).add_suffix("_rezago")
    datos = pd.concat([serie_v[destino].rename("y"), rezagos, control.rename("control")], axis=1).dropna()
    X_f = sm.add_constant(datos.drop(columns="y"))
    completo = sm.OLS(datos["y"], X_f).fit()
    restringido = sm.OLS(datos["y"], X_f.drop(columns=f"{origen}_rezago")).fit()
    F, p, _ = completo.compare_f_test(restringido)
    return float(F), float(p)


def iguales(r1: tuple[dict[str, object], dict[str, np.ndarray]], r2: tuple[dict[str, object], dict[str, np.ndarray]]) -> bool:
    (f1, a1), (f2, a2) = r1, r2
    filas = f1.keys() == f2.keys() and all(
        (isinstance(f1[k], float) and math.isnan(f1[k]) and math.isnan(f2[k])) or f1[k] == f2[k] for k in f1)
    return filas and a1.keys() == a2.keys() and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)


def claves_nuevas() -> list[tuple[str | int, ...]]:
    """Todas las claves nuevas de la réplica. Las de la estimación real, E1 y estabilidad no llevan la candidata,
    así que se cuentan sobre la unión de los pares de todas las candidatas (un superconjunto de la elegida)."""
    nuevas: list[tuple[str | int, ...]] = []
    union: set[tuple[str, str, str]] = set()
    columnas: set[str] = set()
    for nombre in candidatas_definidas():
        real, _ = serie(nombre)
        desc = descartada(nombre)
        pares = taliban.pares_por_version(list(real.columns), desc)
        todos = [(v, o, d) for v, lista in pares.items() for o, d in lista]
        union |= set(todos)
        columnas |= set(real.columns)
        plantables = list(itertools.permutations([c for c in real.columns if c != desc], 2))
        for e in ("0.4", "0.8"):
            nuevas += [("tal_calibracion", nombre, e, o, d) for o, d in plantables]
            for i in range(N_COMPLETO[e]):
                nuevas += [("tal_generacion", nombre, e, str(i)), ("tal_par", nombre, e, str(i))]
                nuevas += [("tal_cp_remuestreo", nombre, c, e, str(i), v, o, d) for c in taliban.CONTROLES for v, o, d in todos]
        for familia, n in (("val", N_COMPLETO["realista"]), ("fase", N_COMPLETO["fase"])):
            for i in range(n):
                nuevas.append((f"tal_{familia}_generacion", nombre, str(i)))
                nuevas += [(f"tal_{familia}_remuestreo", nombre, c, str(i), v, o, d) for c in taliban.CONTROLES for v, o, d in todos]
    todos = sorted(union)
    nuevas += [("tal_real_remuestreo", v, o, d) for v, o, d in todos]
    nuevas += [("tal_e1_b1_remuestreo", v, o, d) for v, o, d in todos]
    for i in range(N_E1["B2"]):
        nuevas += [("tal_e1_b2_remuestreo", str(i), v, o, d) for v, o, d in todos]
    for kappa in KAPPAS:
        for i in range(N_E1["B3"]):
            nuevas.append(("tal_e1_b3_generacion", kappa, str(i)))
            nuevas += [("tal_e1_b3_remuestreo", prueba, kappa, str(i), v, o, d) for prueba in e1.PRUEBAS for v, o, d in todos]
    for i in range(N_E1["B4"]):
        nuevas.append(("tal_e1_b4_permutacion", str(i)))
        nuevas += [("tal_e1_b4_remuestreo", prueba, str(i), v, o, d) for prueba in e1.PRUEBAS for v, o, d in todos]
    nuevas += [("tal_e1_b6_te", o, d) for o, d in itertools.permutations(sorted(columnas), 2)]
    for var in VARIACIONES_ESTABILIDAD:
        nuevas += [("tal_estab_remuestreo", var, v, o, d) for v, o, d in todos]
    return nuevas


def etapa_verificar() -> bool:
    res: dict[str, bool] = {}
    nombres = list(candidatas_definidas())
    print(f"Candidatas: {nombres}")
    print("\n=== a) Series: suma de masas igual a n_t, racha consecutiva y sin ceros artificiales ===")
    ok_a = True
    fechas = fechas_documentos()
    for nombre in nombres:
        real, n_t = serie(nombre)
        grilla, K = candidatas_definidas()[nombre]
        dias = taliban.GRILLAS[grilla][1]
        dif = float(np.abs(np.expm1(real).sum(axis=1).to_numpy() - n_t.to_numpy(dtype=float)).max())
        pasos = np.diff(real.index.to_numpy()).astype("timedelta64[D]").astype(int)
        consecutiva = bool((pasos == dias).all())
        tramo = taliban.recorte(fechas, grilla)["tramo"]
        sin_ceros = bool((n_t >= taliban.MIN_DOCS).all()) and bool((np.expm1(real).sum(axis=1) > 0).all())
        coincide = bool((tramo["inicio"].to_numpy() == real.index.to_numpy()).all() and (tramo["documentos"].to_numpy() == n_t.to_numpy()).all())
        ok = dif <= 1e-9 and consecutiva and sin_ceros and coincide
        ok_a &= ok
        print(f"  {nombre}: {len(real)} ventanas, max |suma - n_t| = {dif:.2e}, pasos de {dias} días: {consecutiva}, "
              f"n_t >= 5 y sin filas en cero: {sin_ceros}, igual a la racha: {coincide}: {'OK' if ok else 'FALLA'}")
    res["a"] = ok_a

    print("\n=== b) F y p teórico contra statsmodels (5 pares de mayor F por versión, sustituto realista 0) ===")
    max_F, max_p, n_b = 0.0, 0.0, 0
    for nombre in nombres:
        ctx = contexto(nombre, con_betas=False)
        x = taliban.sustituto_realista(ctx, 0)
        versiones = candidatas.construir_versiones_k(x, ctx["descartada"])
        for control in taliban.CONTROLES:
            control_de = taliban.control_propio_interno(versiones) if control == "propio" else taliban.control_compartido_fijo(ctx["n_t"])
            for v in taliban.VERSIONES:
                val, _ = e1.estimar_version(versiones[v][0], lambda d, v=v: control_de(v, d), ctx["pares"][v], "condicional", 0, None)
                for j in np.argsort(-val["F"], kind="stable")[:5]:
                    o, d = ctx["pares"][v][j]
                    F_sm, p_sm = f_statsmodels(versiones[v][0], control_de(v, d), o, d)
                    max_F = max(max_F, abs(val["F"][j] - F_sm) / abs(F_sm))
                    max_p = max(max_p, abs(val["p_teo"][j] - p_sm) / abs(p_sm))
                    n_b += 1
    res["b"] = max_F <= 1e-8 and max_p <= 1e-8
    print(f"  {n_b} pares: error relativo máximo F {max_F:.2e}, p {max_p:.2e}: {'OK' if res['b'] else 'FALLA'}")

    print("\n=== c) Sustituto realista: n_t exacto y proporciones que suman 1 ===")
    dif_n, dif_s = 0.0, 0.0
    for nombre in nombres:
        ctx = contexto(nombre, con_betas=False)
        print(f"  {nombre}: ceros reemplazados para el clr {ctx['clr']['ceros_reemplazados']}, delta = {ctx['clr']['delta']:.6g}")
        for i in range(5):
            x = taliban.sustituto_realista(ctx, i)
            masa = np.expm1(x)
            dif_n = max(dif_n, float(np.abs(masa.sum(axis=1).to_numpy() - ctx["n_t"].to_numpy(dtype=float)).max()))
            dif_s = max(dif_s, float(np.abs((masa.div(ctx["n_t"].astype(float), axis=0)).sum(axis=1) - 1).max()))
    res["c"] = dif_n <= 1e-9 and dif_s <= 1e-12
    print(f"  max |suma de masas - n_t| = {dif_n:.2e}; max |suma de proporciones - 1| = {dif_s:.2e}: {'OK' if res['c'] else 'FALLA'}")

    print("\n=== d) Misma réplica dos veces (y en un proceso del grupo) ===")
    nombre = nombres[0]
    ctx = contexto(nombre, con_betas=False)
    plantables = list(itertools.permutations([c for c in ctx["real"].columns if c != ctx["descartada"]], 2))
    ctx["betas"] = {("0.8", o, d): 0.3 for o, d in plantables}  # beta fijo solo para esta comprobación
    ok_d = True
    referencia = {}
    for familia, efecto in (("realista", "0.0"), ("fase", "0.0"), ("cp", "0.8")):
        r1, r2 = (taliban.replica_evaluacion(ctx, familia, 0, efecto) for _ in range(2))
        referencia[familia] = r1
        ok_d &= iguales(r1, r2)
        print(f"  {nombre} {familia}: {'iguales' if iguales(r1, r2) else 'DISTINTAS'}")
    with ProcessPoolExecutor(max_workers=2, initializer=taliban.iniciar_trabajador, initargs=({nombre: ctx},)) as ex:
        r_grupo = ex.submit(taliban.trabajo, ("eval", nombre, "fase", "0.0", 0)).result()
    ok_d &= iguales(r_grupo, referencia["fase"])
    print(f"  fase en un proceso del grupo: {'igual' if iguales(r_grupo, referencia['fase']) else 'DISTINTA'}")
    res["d"] = ok_d

    print("\n=== e) Claves de semilla únicas, con prefijo tal_ ===")
    e1s = cargar_script("e1")
    previas = e1s["claves_previas"]() + e1s["claves_nuevas"](e1s["contexto"]()["pares"])
    nuevas = claves_nuevas()
    prefijo = all(str(n[0]).startswith("tal_") for n in nuevas)
    todas = previas + nuevas
    distintas = len({granger.clave_semilla(*ident) for ident in todas})
    res["e"] = len(previas) == CLAVES_PREVIAS and distintas == len(todas) and prefijo
    print(f"  {len(todas)} ({len(previas)} previas, esperadas {CLAVES_PREVIAS}, + {len(nuevas)} nuevas), distintas: {distintas}; "
          f"todas las nuevas con prefijo tal_: {prefijo}: {'OK' if res['e'] else 'FALLA'}")

    print("\nResumen: " + ", ".join(f"{k}) {'OK' if v else 'FALLA'}" for k, v in sorted(res.items())))
    if not all(res.values()):
        print("ALGUNA VERIFICACIÓN FALLA: no seguir.")
    return all(res.values())


# ---------------------------------------------------------------------------
# Evaluación (sección 5)
# ---------------------------------------------------------------------------

def guardar_avance(carpeta: Path, nombre: str, filas: list[dict[str, object]], arreglos: dict[str, list[np.ndarray]]) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    tmp = carpeta / f"{nombre}.tmp.npz"
    np.savez(tmp, **{k: np.array(v) for k, v in arreglos.items()})
    os.replace(tmp, carpeta / f"{nombre}.npz")
    tmp = carpeta / f"{nombre}.tmp.csv"
    pd.DataFrame(filas).to_csv(tmp, index=False)
    os.replace(tmp, carpeta / f"{nombre}.csv")


def correr(contextos: dict[str, dict[str, object]], carpeta: Path, nombre: str, tareas: list[tuple[str | int, ...]]) -> None:
    """Réplicas en procesos independientes con punto de control cada CADA; reanuda desde lo guardado."""
    ruta_csv = carpeta / f"{nombre}.csv"
    filas: list[dict[str, object]] = []
    arreglos: dict[str, list[np.ndarray]] = {}
    if ruta_csv.exists():
        tabla = pd.read_csv(ruta_csv, dtype={"efecto": str, "kappa": str}, float_precision="round_trip")
        assert tabla["indice"].tolist() == [int(t[-1]) for t in tareas[:len(tabla)]], f"{nombre}: índices guardados no calzan"
        filas = tabla.to_dict("records")
        with np.load(carpeta / f"{nombre}.npz") as z:
            arreglos = {k: list(z[k][:len(filas)]) for k in z.files}
    pendientes = tareas[len(filas):]
    print(f"{nombre}: {len(filas)} guardadas, faltan {len(pendientes)}", flush=True)
    if not pendientes:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=TRABAJADORES, initializer=taliban.iniciar_trabajador, initargs=(contextos,)) as ex:
        for inicio in range(0, len(pendientes), CADA):
            for fila, arr in ex.map(taliban.trabajo, pendientes[inicio:inicio + CADA]):
                filas.append(fila)
                for k, a in arr.items():
                    arreglos.setdefault(k, []).append(a)
            guardar_avance(carpeta, nombre, filas, arreglos)
            print(f"  {nombre}: {len(filas)}/{len(tareas)} ({time.perf_counter() - t0:.0f} s)", flush=True)


def betas_evaluacion() -> None:
    ruta = EVAL / "betas.csv"
    if ruta.exists():
        return
    contextos = {n: contexto(n, con_betas=False) for n in candidatas_definidas()}
    tareas = [("beta", n, e, o, d) for n, ctx in contextos.items() for e in ("0.4", "0.8")
              for o, d in itertools.permutations([c for c in ctx["real"].columns if c != ctx["descartada"]], 2)]
    print(f"Betas: {len(tareas)} bisecciones", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=TRABAJADORES, initializer=taliban.iniciar_trabajador, initargs=(contextos,)) as ex:
        filas = list(ex.map(taliban.trabajo, tareas, chunksize=4))
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)
    print(f"  {time.perf_counter() - t0:.0f} s; radio espectral máximo {max(f['radio_espectral'] for f in filas):.3f}")


def unidades_evaluacion(N: dict[str, int]) -> list[tuple[str, str, str, int]]:
    """(candidata, familia, efecto, réplicas)."""
    salida = []
    for nombre in candidatas_definidas():
        salida += [(nombre, "realista", "0.0", N["realista"]), (nombre, "fase", "0.0", N["fase"]),
                   (nombre, "cp", "0.4", N["0.4"]), (nombre, "cp", "0.8", N["0.8"])]
    return salida


def etapa_evaluar(N: dict[str, int]) -> None:
    betas_evaluacion()
    contextos = {n: contexto(n) for n in candidatas_definidas()}
    for nombre, familia, efecto, n in unidades_evaluacion(N):
        unidad = f"{nombre}_{familia}" + (f"_e{efecto}" if familia == "cp" else "")
        correr(contextos, EVAL, unidad, [("eval", nombre, familia, efecto, i) for i in range(n)])
    resumen_evaluacion(N)


def resumen_evaluacion(N: dict[str, int]) -> None:
    umbral = umbral_aptitud(N["realista"])
    assert N["realista"] == N["fase"]
    filas, apt = [], []
    for nombre in candidatas_definidas():
        datos = {(fam, e): pd.read_csv(EVAL / (f"{nombre}_{fam}" + (f"_e{e}" if fam == "cp" else "") + ".csv"), dtype={"efecto": str})
                 for fam, e in (("realista", "0.0"), ("fase", "0.0"), ("cp", "0.4"), ("cp", "0.8"))}
        for (fam, e), t in datos.items():
            n_esperado = N["realista" if fam == "realista" else "fase" if fam == "fase" else e]
            assert len(t) == n_esperado, f"{nombre} {fam} {e}: {len(t)} réplicas"
        grilla, K = candidatas_definidas()[nombre]
        real, _ = serie(nombre)
        for control in taliban.CONTROLES:
            fila_apt: dict[str, object] = {"candidata": nombre, "grilla": grilla, "K": K, "observaciones": len(real) - 1,
                                           "control": control}
            for fam in taliban.FAMILIAS:
                t = datos[(fam, "0.0")]
                for v in taliban.VERSIONES:
                    filas.append({"candidata": nombre, "control": control, "familia": fam, "version": v, "replicas": len(t),
                                  "replicas_con_rechazo": int((t[f"rechazos_{control}_{v}"] >= 1).sum()),
                                  "fraccion": float((t[f"rechazos_{control}_{v}"] >= 1).mean()),
                                  "no_estimables": int(t[f"no_estimable_{control}_{v}"].sum())})
                filas.append({"candidata": nombre, "control": control, "familia": fam, "version": "regla_2_versiones",
                              "replicas": len(t), "replicas_con_rechazo": int((t[f"enlaces_regla_{control}"] >= 1).sum()),
                              "fraccion": float((t[f"enlaces_regla_{control}"] >= 1).mean()), "no_estimables": np.nan})
                fila_apt[f"fp_{fam}_original"] = int((t[f"rechazos_{control}_original"] >= 1).sum())
                fila_apt[f"fp_{fam}_detendenciada"] = int((t[f"rechazos_{control}_detendenciada"] >= 1).sum())
            for e in ("0.4", "0.8"):
                t = datos[("cp", e)]
                fila_apt[f"deteccion_regla_{e}"] = float(t[f"detectado_regla_{control}"].mean())
                for v in taliban.VERSIONES:
                    filas.append({"candidata": nombre, "control": control, "familia": f"control_positivo_{e}", "version": v,
                                  "replicas": len(t), "replicas_con_rechazo": int(t[f"rechaza_plantado_{control}_{v}"].sum()),
                                  "fraccion": float(t[f"rechaza_plantado_{control}_{v}"].mean()),
                                  "no_estimables": int(t[f"no_estimable_{control}_{v}"].sum())})
            fila_apt["umbral_fp"] = umbral
            fila_apt["apta"] = bool(fila_apt["deteccion_regla_0.8"] >= DETECCION_MIN and all(
                fila_apt[f"fp_{fam}_{v}"] <= umbral for fam in taliban.FAMILIAS for v in ("original", "detendenciada")))
            apt.append(fila_apt)
    apt_t = pd.DataFrame(apt)
    propio = apt_t[(apt_t["control"] == "propio") & apt_t["apta"]]
    if len(propio):
        orden = propio.sort_values(["K", "deteccion_regla_0.4", "observaciones"], ascending=False, kind="stable")
        elegida = orden.iloc[0]
        eleccion = pd.DataFrame([{"elegida": elegida["candidata"], "grilla": elegida["grilla"], "K": int(elegida["K"]),
                                  "observaciones": int(elegida["observaciones"]),
                                  "aptas_con_control_propio": ", ".join(propio["candidata"]), "se_estima": True}])
    else:
        eleccion = pd.DataFrame([{"elegida": "ninguna", "grilla": "", "K": np.nan, "observaciones": np.nan,
                                  "aptas_con_control_propio": "", "se_estima": False}])
    guardar_tablas({"E_calibracion_y_deteccion": pd.DataFrame(filas), "E_aptitud": apt_t, "E_eleccion": eleccion}, cabeza=200)


def elegida() -> str | None:
    e = pd.read_csv(SALIDA / "E_eleccion.csv").iloc[0]
    return str(e["elegida"]) if bool(e["se_estima"]) else None


def etapa_medir(n: int, N: dict[str, int]) -> None:
    nombres = list(candidatas_definidas())
    total = 0.0
    filas = []
    for nombre in nombres:
        ctx = contexto(nombre, con_betas=False)
        plantables = list(itertools.permutations([c for c in ctx["real"].columns if c != ctx["descartada"]], 2))
        t0 = time.perf_counter()
        b = taliban.tarea_beta(ctx, "0.8", *plantables[0])
        t_beta = time.perf_counter() - t0
        ctx["betas"] = {(e, o, d): b["beta"] for e in ("0.4", "0.8") for o, d in plantables}
        tiempos = {}
        for familia, efecto in (("realista", "0.0"), ("fase", "0.0"), ("cp", "0.8")):
            t0 = time.perf_counter()
            for i in range(n):
                taliban.replica_evaluacion(ctx, familia, i, efecto)
            tiempos[familia] = (time.perf_counter() - t0) / n
        serie_s = (t_beta * 2 * len(plantables) + tiempos["realista"] * N["realista"] + tiempos["fase"] * N["fase"]
                   + tiempos["cp"] * (N["0.4"] + N["0.8"]))
        total += serie_s
        filas.append({"candidata": nombre, "T": len(ctx["real"]), "pares": sum(len(l) for l in ctx["pares"].values()),
                      "s_beta": t_beta, **{f"s_{k}": v for k, v in tiempos.items()}, "horas_en_serie": serie_s / 3600})
    # E1 (sobre la candidata más cara, como cota) y estimación real.
    peor = max(filas, key=lambda f: f["s_realista"])
    e1_s = peor["s_realista"] / 2 * (N_E1["B2"] + 2 * N_E1["B3"] * len(KAPPAS) + 2 * N_E1["B4"])
    total += e1_s
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nE1 (cota con {peor['candidata']}): {e1_s / 3600:.2f} h en serie")
    print(f"Total en serie {total / 3600:.2f} h; con {TRABAJADORES} procesos, unas {total / 3600 / TRABAJADORES * 1.5:.2f} h "
          f"(factor 1.5 por la eficiencia observada en E1); límite {HORAS_MAX:.0f} h: "
          f"{'supera' if total / TRABAJADORES * 1.5 > HORAS_MAX * 3600 else 'no supera'}")


# ---------------------------------------------------------------------------
# Estimación real (sección 6)
# ---------------------------------------------------------------------------

def requisito(m: int, df_den: int) -> dict[str, float]:
    """confirmacion_y_real.requisito."""
    F_req = float(stats.f.isf(taliban.Q / m, 1, df_den))
    efecto_req = math.log(1 + F_req / df_den)
    return {"q_sobre_m": taliban.Q / m, "F_req": F_req, "efecto_req": efecto_req, "reduccion_req": 1 - math.exp(-efecto_req)}


def etapa_real() -> None:
    ruta = REAL / "resultados.csv"
    if ruta.exists():
        print(f"Ya existe {ruta}: la estimación real se ejecuta una sola vez. No se repite.")
        return
    nombre = elegida()
    if nombre is None:
        print("Ninguna combinación es APTA con el control propio: no se estima sobre datos reales (sección 5).")
        return
    conf = cargar_script("confirmacion_y_real")
    ctx = contexto(nombre, con_betas=False)
    real, desc, pares = ctx["real"], ctx["descartada"], ctx["pares"]
    filas, arreglos = [], {}
    for control in taliban.CONTROLES:
        fila, arr = e1.analizar(real, None, desc, pares, ("condicional",), B,
                                lambda prueba, v, o, d: ("tal_real_remuestreo", v, o, d), control=control)
        assert not any(fila[f"no_estimable_condicional_{v}"] for v in taliban.VERSIONES), "diseño sin rango completo"
        for v in taliban.VERSIONES:
            for j, (o, d) in enumerate(pares[v]):
                filas.append({"configuracion": nombre, "control_tipo": control, "version": v, "origen": o, "destino": d,
                              "n_obs": int(arr[f"condicional_n_obs_{v}"][j]), "efecto": arr[f"condicional_efecto_{v}"][j],
                              "F": arr[f"condicional_F_{v}"][j], "df_num": 1, "df_den": int(arr[f"condicional_df_den_{v}"][j]),
                              "p_teorico": arr[f"condicional_p_teo_{v}"][j], "p_remuestreo": arr[f"condicional_p_rem_{v}"][j],
                              "B": B, "rechaza_BH": int(arr[f"condicional_rechaza_{v}"][j])})
        arreglos.update({f"{control}_{k}": a for k, a in arr.items()})
    tabla = pd.DataFrame(filas)
    REAL.mkdir(parents=True, exist_ok=True)
    np.savez(REAL / "p_todos_los_pares.npz", **arreglos,
             **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in taliban.VERSIONES})
    tabla.to_csv(ruta, index=False)
    por_version, rutas_r = [], []
    for (control, v), g in tabla.groupby(["control_tipo", "version"], sort=False):
        por_version.append({"control_tipo": control, "version": v, "m": len(g), "rechazos_BH": int(g["rechaza_BH"].sum()),
                            "p_remuestreo_min": float(g["p_remuestreo"].min()), "efecto_max": float(g["efecto"].max()),
                            "F_max": float(g["F"].max()), "df_den": int(g["df_den"].iloc[0]),
                            **requisito(len(g), int(g["df_den"].iloc[0]))})
        r = g["p_remuestreo"] / np.maximum(g["p_teorico"], 1 / (B + 1))
        frac, med = float(((r >= 0.1) & (r <= 10)).mean()), float(r.median())
        rutas_r.append({"control_tipo": control, "version": v, "n_pares": len(g), "fraccion_r_en_0.1_10": frac, "mediana_r": med,
                        "cumple_orden_magnitud": frac >= 0.95, "discrepancia_sistematica": not (1 / 3 <= med <= 3),
                        "pasa": frac >= 0.95 and 1 / 3 <= med <= 3})
    enlaces = conf["enlaces"]
    guardar_tablas({"real/por_version": pd.DataFrame(por_version), "real/rutas": pd.DataFrame(rutas_r),
                    "real/enlaces_propio": enlaces(tabla, "propio"), "real/enlaces_compartido": enlaces(tabla, "compartido")})


# ---------------------------------------------------------------------------
# E1 (sección 7)
# ---------------------------------------------------------------------------

def modulo_e1() -> ModuleType:
    """scripts/e1.py como módulo, con sus rutas apuntando a data/taliban/e1 y a la estimación real de la réplica."""
    spec = importlib.util.spec_from_file_location("e1_script_taliban", SCRIPTS / "e1.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.SALIDA = E1_DIR
    modulo.PRINCIPAL = REAL / "resultados.csv"
    assert tuple(modulo.KAPPAS) == KAPPAS and modulo.N_COMPLETO["B3"] == N_E1["B3"]
    return modulo


def etapa_e1() -> None:
    nombre = elegida()
    if nombre is None or not (REAL / "resultados.csv").exists():
        print("E1 no aplica: no hubo estimación real.")
        return
    e1s = modulo_e1()
    ctx = contexto(nombre, con_betas=False)
    contextos = {nombre: ctx}
    real, desc, pares = ctx["real"], ctx["descartada"], ctx["pares"]
    E1_DIR.mkdir(parents=True, exist_ok=True)
    # B1.
    if not (E1_DIR / "B1_pares_real.csv").exists():
        filas, arreglos = [], {}
        for control in taliban.CONTROLES:
            fila, arr = e1.analizar(real, None, desc, pares, ("pares",), B,
                                    lambda prueba, v, o, d: ("tal_e1_b1_remuestreo", v, o, d), control=control)
            for v in taliban.VERSIONES:
                for j, (o, d) in enumerate(pares[v]):
                    filas.append({"control_tipo": control, "version": v, "origen": o, "destino": d,
                                  "n_obs": int(arr[f"pares_n_obs_{v}"][j]), "efecto": arr[f"pares_efecto_{v}"][j],
                                  "F": arr[f"pares_F_{v}"][j], "df_num": 1, "df_den": int(arr[f"pares_df_den_{v}"][j]),
                                  "p_teorico": arr[f"pares_p_teo_{v}"][j], "p_remuestreo": arr[f"pares_p_rem_{v}"][j], "B": B,
                                  "rechaza_BH": int(arr[f"pares_rechaza_{v}"][j])})
            arreglos.update({f"{control}_{k}": a for k, a in arr.items()})
        pd.DataFrame(filas).to_csv(E1_DIR / "B1_pares_real.csv", index=False)
        np.savez(E1_DIR / "B1_pares_real.npz", **arreglos,
                 **{f"pares_{v}": np.array([f"{o}->{d}" for o, d in pares[v]]) for v in taliban.VERSIONES})
    print("\n##### B1 #####")
    e1s.resumen_b1()
    # B5: nulo por la ruta F sobre los sustitutos realistas de la evaluación.
    print("\n##### B5 #####")
    correr(contextos, E1_DIR, "B5_nulo", [("e1", nombre, "B5", i) for i in range(N_B5)])
    with np.load(E1_DIR / "B5_nulo.npz") as w, np.load(EVAL / f"{nombre}_realista.npz") as z:
        coincide = all(np.array_equal(w[f"condicional_F_{v}"], z[f"propio_F_{v}"], equal_nan=True) for v in taliban.VERSIONES)
    assert coincide, "B5: el F condicional del nulo no coincide con el de la evaluación (sustituto realista, control propio)"
    print("B5: el F condicional de los 300 sustitutos coincide con el de la evaluación (control propio)")
    e1s.resumen_b5(pares)
    # B6.
    print("\n##### B6 #####")
    if not (E1_DIR / "B6_te.csv").exists():
        simbolos = {c: e1.simbolizar(real[c].to_numpy()) for c in real.columns}
        filas, sust = [], []
        pares_te = pares["original"]
        for o, d in pares_te:
            r = e1.te_con_sustitutos(simbolos[o], simbolos[d], e1s.N_TE, e1s.BLOQUE_TE, granger.generador("tal_e1_b6_te", o, d))
            filas.append({"origen": o, "destino": d, "te_observada": r["te_obs"], "te_media_sustitutos": r["te_media_sust"],
                          "te_corregida": r["te_corregida"], "p": r["p"]})
            sust.append(r["te_sust"])
        tabla = pd.DataFrame(filas)
        tabla["rechaza_BH"] = granger.benjamini_hochberg(tabla["p"].to_numpy(), taliban.Q, len(pares_te)).astype(int)
        tabla.to_csv(E1_DIR / "B6_te.csv", index=False)
        np.savez(E1_DIR / "B6_te.npz", te_observada=tabla["te_observada"].to_numpy(), te_corregida=tabla["te_corregida"].to_numpy(),
                 p=tabla["p"].to_numpy(), te_sustitutos=np.array(sust), pares=np.array([f"{o}->{d}" for o, d in pares_te]))
    e1s.resumen_b6()
    # B2, B4 y B3.
    print("\n##### B2 #####")
    correr(contextos, E1_DIR, "B2_sustitutos", [("e1", nombre, "B2", i) for i in range(N_E1["B2"])])
    e1s.resumen_b2({"B2": N_E1["B2"]})
    print("\n##### B4 #####")
    correr(contextos, E1_DIR, "B4_permutaciones", [("e1", nombre, "B4", i) for i in range(N_E1["B4"])])
    e1s.resumen_b4({"B4": N_E1["B4"]})
    print("\n##### B3 #####")
    for kappa in KAPPAS:
        correr(contextos, E1_DIR, f"B3_kappa_{kappa}", [("e1", nombre, "B3", kappa, i) for i in range(N_E1["B3"])])
    e1s.resumen_b3({"B3": N_E1["B3"]})


def etapa_estabilidad() -> None:
    nombre = elegida()
    ruta = REAL / "enlaces_propio.csv"
    hay = nombre is not None and ruta.exists() and (len(pd.read_csv(ruta)) or len(pd.read_csv(REAL / "enlaces_compartido.csv")))
    if not hay:
        guardar_tablas({"estabilidad": pd.DataFrame([{"estabilidad": "no aplica",
                                                      "motivo": "la estimación de la sección 6 no declaró enlaces"
                                                      if nombre is not None else "no hubo estimación real"}])})
        return
    raise NotImplementedError("hay enlaces declarados: la estabilidad (sección 8) se implementa como etapa aparte")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etapa", required=True, choices=["auditoria", "modelo", "series", "verificar", "medir", "evaluar",
                                                           "real", "e1", "estabilidad", "resumen"])
    parser.add_argument("--K", type=int, choices=[5, 10])
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--reducido", action="store_true", help="200 réplicas en 5a y 5b y 60 por efecto en 5c (regla 9)")
    args = parser.parse_args()
    N = N_REDUCIDO if args.reducido else N_COMPLETO
    if args.etapa == "auditoria":
        etapa_auditoria()
    elif args.etapa == "modelo":
        etapa_modelo(args.K, args.semilla)
    elif args.etapa == "series":
        etapa_series()
    elif args.etapa == "verificar":
        etapa_verificar()
    elif args.etapa == "medir":
        etapa_medir(args.n, N)
    elif args.etapa == "evaluar":
        etapa_evaluar(N)
    elif args.etapa == "resumen":
        resumen_evaluacion(N)
    elif args.etapa == "real":
        etapa_real()
    elif args.etapa == "e1":
        etapa_e1()
    else:
        etapa_estabilidad()


if __name__ == "__main__":
    main()
