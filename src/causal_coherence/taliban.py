"""
taliban.py

Semana 8, paso 6 (Bitacora_Semana8.md, sección 32): piezas de la réplica
preregistrada del proceso de Granger condicional sobre el subconjunto
NEWS-TLS Entities (Taliban) del corpus de Afganistán.

  - documentos_taliban, auditoria: carga del subconjunto y auditoría previa
    (solo reporte; se excluyen solo fechas que no se pueden interpretar y
    contenido vacío, como en bpoil).
  - CHUNKS_ANUALES, configuracion_lda: RollingLDA con la LDA_CONFIG de bpoil,
    chunks por año calendario, gamma = 1/K y semilla explícitos.
  - topico_plantilla: la regla fijada para el tópico descartado en la
    residualizada.
  - ventanas, recorte, serie_candidata: grillas de 14 y 30 días, recorte de
    bordes, mayor racha y registro de R1.
  - analizar_controles: la condicional por versión con controles explícitos,
    BH por versión, regla de 2 versiones y par plantado.
  - sustituto_realista, fase_aleatoria, sustituto_plantado: las familias de
    la evaluación (sección 5).
  - replica_evaluacion, tarea_beta, replica_e1, trabajo: réplicas para
    procesos independientes, con las claves "tal_".
"""

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd

from causal_coherence import candidatas, e1, granger, planted, surrogates, validez
from causal_coherence.data_loading import AFGHANISTAN_DIR, load_full_corpus
from causal_coherence.topic_model import LDA_CONFIG

SOURCE_TALIBAN = "NEWS-TLS Entities (Taliban)"
RANGO = (pd.Timestamp("2006-07-31"), pd.Timestamp("2017-09-19"))
DIAS_AISLADO = 60
PALABRAS_MINIMAS = 30
FECHA_MASIVA = 10
MAX_EXCLUIDOS = 0.05
CHUNKS_ANUALES = [pd.Timestamp(f"{anio}-01-01") for anio in range(2006, 2018)]
PALABRAS_PLANTILLA = frozenset({
    "reuters", "percent", "comment", "welcome", "latency", "copyright", "rights", "reserved", "subscribe", "newsletter",
    "click", "email", "photo", "image", "caption", "advertisement", "cookie", "browser", "javascript", "download",
    "twitter", "facebook", "share", "print", "page", "link", "website", "editor", "reporting", "editing", "writing",
    "contributed"})
MINIMO_PLANTILLA = 3


def documentos_taliban() -> pd.DataFrame:
    """Los documentos del subconjunto, en el orden del corpus, con la fecha interpretada (NaT si no se puede)."""
    docs = [d for d in load_full_corpus(AFGHANISTAN_DIR) if d["source"] == SOURCE_TALIBAN]
    df = pd.DataFrame(docs)
    df["fecha_texto"] = df["date"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def auditoria(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tablas de la auditoría (solo reporte) y la lista de exclusiones (fecha corrupta o contenido vacío)."""
    contenido = df["content"].fillna("").astype(str)
    palabras = contenido.str.split().str.len()
    fecha_corrupta = df["date"].isna()
    vacio = contenido.str.strip() == ""
    excluidos = df.loc[fecha_corrupta | vacio, ["id", "fecha_texto", "title"]].copy()
    excluidos["motivo"] = np.where(fecha_corrupta[excluidos.index], "fecha corrupta", "contenido vacío")

    validas = df.loc[~fecha_corrupta].copy()
    fechas = validas["date"].sort_values()
    valores = fechas.to_numpy()
    distancia_prev = np.r_[np.inf, np.diff(valores).astype("timedelta64[D]").astype(float)]
    distancia_sig = np.r_[np.diff(valores).astype("timedelta64[D]").astype(float), np.inf]
    cercano = pd.Series(np.minimum(distancia_prev, distancia_sig), index=fechas.index)
    aislados = validas.loc[cercano[cercano > DIAS_AISLADO].index, ["id", "date", "title"]].copy()
    aislados["dias_al_mas_cercano"] = cercano[cercano > DIAS_AISLADO].to_numpy()
    fuera = validas.loc[(validas["date"] < RANGO[0]) | (validas["date"] > RANGO[1]), ["id", "date", "title"]]

    por_anio = validas.groupby(validas["date"].dt.year).size().rename("documentos").rename_axis("anio").reset_index()
    por_mes = validas.groupby(validas["date"].dt.to_period("M")).size().rename("documentos").rename_axis("mes").reset_index()
    por_mes["mes"] = por_mes["mes"].astype(str)
    por_fecha = validas.groupby(validas["date"].dt.date).size().rename("documentos").rename_axis("fecha").reset_index()
    masivas = por_fecha[por_fecha["documentos"] >= FECHA_MASIVA].sort_values("documentos", ascending=False, kind="stable")
    top_fechas = por_fecha.sort_values("documentos", ascending=False, kind="stable").head(10)
    cortos = df.loc[palabras < PALABRAS_MINIMAS, ["id", "fecha_texto", "title"]].assign(palabras=palabras[palabras < PALABRAS_MINIMAS])
    titulos = df["title"].fillna("").astype(str)
    repetidos = titulos[titulos.duplicated(keep=False)]
    dup_titulos = (df.loc[repetidos.index, ["id", "fecha_texto", "title"]].sort_values(["title", "fecha_texto"], kind="stable"))
    resumen = pd.DataFrame([{
        "documentos": len(df), "fecha_corrupta": int(fecha_corrupta.sum()), "contenido_vacio": int(vacio.sum()),
        "excluidos": len(excluidos), "fraccion_excluida": len(excluidos) / len(df),
        "primer_documento": fechas.iloc[0].date(), "ultimo_documento": fechas.iloc[-1].date(),
        "fuera_de_rango": len(fuera), "fechas_aisladas": len(aislados),
        "fechas_con_10_o_mas": len(masivas), "menos_de_30_palabras": len(cortos),
        "titulos_duplicados_documentos": len(dup_titulos), "titulos_duplicados_grupos": int(repetidos.nunique()),
        "palabras_mediana": float(palabras.median())}])
    return {"A_resumen": resumen, "A_por_anio": por_anio, "A_por_mes": por_mes, "A_fechas_masivas": masivas,
            "A_top10_fechas": top_fechas, "A_aislados": aislados, "A_fuera_de_rango": fuera, "A_cortos": cortos,
            "A_titulos_duplicados": dup_titulos, "A_excluidos": excluidos}


def configuracion_lda(K: int, semilla: int = 42) -> dict[str, object]:
    """LDA_CONFIG de bpoil con chunks anuales, K, gamma = 1/K y la semilla explícitos."""
    return {**LDA_CONFIG, "K": K, "gamma": 1 / K, "how": list(CHUNKS_ANUALES), "seed": semilla}


def topico_plantilla(palabras: dict[str, list[str]], masa: pd.Series) -> tuple[str, str, dict[str, int]]:
    """
    Tópico descartado en la residualizada (sección 32): de plantilla si al
    menos 3 de sus 10 palabras están en PALABRAS_PLANTILLA (si hay varios, el
    de más palabras y luego el de menor masa); si ninguno, el de menor masa.
    Devuelve (tópico, criterio, palabras de plantilla por tópico).
    """
    cuenta = {t: sum(w in PALABRAS_PLANTILLA for w in ws) for t, ws in palabras.items()}
    candidatos = [t for t, n in cuenta.items() if n >= MINIMO_PLANTILLA]
    if candidatos:
        elegido = min(candidatos, key=lambda t: (-cuenta[t], float(masa[t])))
        return elegido, "plantilla", cuenta
    return str(masa.idxmin()), "menor masa total", cuenta


# ---------------------------------------------------------------------------
# Grillas, recorte y series (secciones 3 y 4)
# ---------------------------------------------------------------------------

GRILLAS: dict[str, tuple[str, int]] = {"14 días": ("14D", 14), "30 días": ("30D", 30)}
MIN_DOCS = 5
MIN_OBS = 45
VERSIONES = list(surrogates.VERSIONES)
Q = 0.05
MIN_VERSIONES = 2
FAMILIAS = ("realista", "fase")
CONTROLES = ("propio", "compartido")


def ventanas(fechas: pd.Series, freq: str, dias: int) -> pd.DataFrame:
    """Ventanas del resample (origen en el día del primer documento): inicio, fin, documentos, completa y válida."""
    conteo = pd.Series(1, index=pd.DatetimeIndex(fechas)).resample(freq).size()
    desde = conteo.index
    hasta = desde + pd.Timedelta(days=dias - 1)
    completa = np.asarray((desde >= fechas.min()) & (hasta <= fechas.max()))
    docs = conteo.to_numpy()
    return pd.DataFrame({"inicio": desde, "fin": hasta, "documentos": docs, "completa": completa,
                         "valida": completa & (docs >= MIN_DOCS)})


def rachas(validas: np.ndarray) -> list[tuple[int, int]]:
    """(primera posición, largo) de cada racha de ventanas válidas consecutivas (como factibilidad_grillas.rachas)."""
    salida, inicio = [], None
    for i, v in enumerate(list(validas) + [False]):
        if v and inicio is None:
            inicio = i
        elif not v and inicio is not None:
            salida.append((inicio, i - inicio))
            inicio = None
    return salida


def recorte(fechas: pd.Series, grilla: str) -> dict[str, object]:
    """La mayor racha (en empate, la primera) y el registro de R1 de la grilla."""
    freq, dias = GRILLAS[grilla]
    v = ventanas(fechas, freq, dias)
    todas = rachas(v["valida"].to_numpy())
    pos, largo = max(todas, key=lambda r: r[1]) if todas else (0, 0)
    tramo = v.iloc[pos:pos + largo].reset_index(drop=True)
    en_racha = np.zeros(len(v), dtype=bool)
    en_racha[pos:pos + largo] = True
    bordes = v[~v["completa"]]
    pocas = v[v["completa"] & ~v["valida"]]
    fuera = v[v["valida"] & ~en_racha]
    resumen = {"grilla": grilla, "frecuencia": freq, "retardo_dias": dias, "ventanas_totales": len(v),
               "excluidas_por_bordes": len(bordes), "excluidas_menos_de_5": len(pocas),
               "validas_fuera_de_la_racha": len(fuera), "racha": largo, "observaciones": max(largo - 1, 0),
               "racha_inicio": tramo["inicio"].iloc[0].date() if largo else None,
               "racha_fin": tramo["fin"].iloc[-1].date() if largo else None,
               "docs_min": int(tramo["documentos"].min()) if largo else 0,
               "docs_mediana": float(tramo["documentos"].median()) if largo else np.nan,
               "docs_max": int(tramo["documentos"].max()) if largo else 0,
               "rachas_de_10_o_mas": sum(1 for _, n in todas if n >= 10), "apta_por_observaciones": largo - 1 >= MIN_OBS}
    detalle = pd.concat([bordes.assign(motivo="borde"), pocas.assign(motivo="menos de 5 documentos"),
                         fuera.assign(motivo="válida fuera de la racha")]).sort_values("inicio", kind="stable")
    return {"ventanas": v, "tramo": tramo, "resumen": resumen,
            "detalle": detalle[["inicio", "fin", "documentos", "motivo"]].reset_index(drop=True)}


def serie_candidata(theta: np.ndarray, fechas: pd.Series, grilla: str) -> tuple[pd.DataFrame, pd.Series]:
    """log1p de la suma de theta por ventana de la racha y n_t (conteo real de documentos)."""
    freq, _ = GRILLAS[grilla]
    masa, conteo = candidatas.masas_por_ventana(theta, fechas, freq)
    tramo = recorte(fechas, grilla)["tramo"]
    masa, conteo = masa.loc[tramo["inicio"]], conteo.loc[tramo["inicio"]]
    assert (conteo.to_numpy() == tramo["documentos"].to_numpy()).all(), f"{grilla}: conteos distintos"
    return np.log1p(masa), conteo.rename("n_t")


def pares_por_version(columnas: list[str], descartada: str) -> dict[str, list[tuple[str, str]]]:
    return e1.pares_por_version(columnas, descartada)


def u_impulsor(K: int) -> np.ndarray:
    """+1 para los primeros floor(K / 2) tópicos y -1 para el resto (con K = 10 es e1.U_IMPULSOR)."""
    return np.array([1.0 if k < K // 2 else -1.0 for k in range(K)])


# ---------------------------------------------------------------------------
# Análisis de una réplica con controles explícitos
# ---------------------------------------------------------------------------

ControlDe = Callable[[str, str], pd.Series]


def control_propio_externo(x: pd.DataFrame, n_t: pd.Series) -> ControlDe:
    """log1p(n_t real - masa), transformado por versión (bloque B de la sección 26)."""
    versiones = e1.versiones_propio(np.log1p(np.expm1(x).rsub(n_t.astype(float), axis=0)))
    return lambda v, d: versiones[v][d]


def control_propio_interno(versiones: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]) -> ControlDe:
    """log1p(suma de masas - masa) de candidatas.construir_versiones_k."""
    return lambda v, d: versiones[v][2][d]


def control_compartido_fijo(n_t: pd.Series) -> ControlDe:
    """log1p(n_t real), por los mismos constructores que la serie (el C2 de la sección 21)."""
    versiones = candidatas.versiones_control_fijo(np.log1p(n_t.astype(float)).rename("control_compartido"))
    return lambda v, d: versiones[v]


def analizar_controles(x: pd.DataFrame, descartada: str, pares: dict[str, list[tuple[str, str]]],
                       controles: dict[str, ControlDe], B: int, clave: Callable[[str, str, str, str], tuple],
                       plantado: tuple[str, str] | None = None) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Condicional por versión con cada control, BH por versión, regla de 2 versiones y, si hay, el par plantado."""
    versiones = candidatas.construir_versiones_k(x, descartada)
    fila: dict[str, object] = {}
    arreglos: dict[str, np.ndarray] = {}
    for control, control_de in controles.items():
        veces: dict[tuple[str, str], int] = {}
        n_plantado = 0
        for v in VERSIONES:
            val, no_estimable = e1.estimar_version(versiones[v][0], lambda d, v=v: control_de(v, d), pares[v], "condicional",
                                                   B, lambda o, d, v=v, c=control: clave(c, v, o, d))
            m = len(pares[v])
            rechaza = granger.benjamini_hochberg(val["p_rem"], Q, m) if B and not no_estimable else np.zeros(m, dtype=bool)
            for par, r in zip(pares[v], rechaza):
                veces[par] = veces.get(par, 0) + int(r)
            fila[f"rechazos_{control}_{v}"] = int(rechaza.sum())
            fila[f"no_estimable_{control}_{v}"] = int(no_estimable)
            if plantado is not None:
                j = pares[v].index(plantado)
                fila[f"rechaza_plantado_{control}_{v}"] = int(rechaza[j])
                fila[f"efecto_plantado_{control}_{v}"] = float(val["efecto"][j])
                n_plantado += int(rechaza[j])
            arreglos[f"{control}_rechaza_{v}"] = rechaza.astype(np.int8)
            for k in ("efecto", "F", "p_teo", "p_rem"):
                arreglos[f"{control}_{k}_{v}"] = val[k]
        fila[f"enlaces_regla_{control}"] = int(sum(n >= MIN_VERSIONES for n in veces.values()))
        if plantado is not None:
            fila[f"detectado_regla_{control}"] = int(n_plantado >= MIN_VERSIONES)
    return fila, arreglos


# ---------------------------------------------------------------------------
# Sustitutos de la evaluación (sección 5)
# ---------------------------------------------------------------------------

def preparar_clr(real: pd.DataFrame, n_t: pd.Series) -> dict[str, object]:
    """AR(1) diagonal del clr de las proporciones reales (paquete_validez.preparar_b)."""
    proporciones = np.expm1(real).div(n_t.astype(float), axis=0)
    sin_ceros, n_ceros, delta = validez.reemplazo_multiplicativo(proporciones)
    z = validez.clr(sin_ceros)
    c, a, residuos = surrogates.ajustar_ar1(z)
    return {"c": c, "a": a, "residuos": residuos, "x0": z.to_numpy().mean(axis=0), "ceros_reemplazados": n_ceros,
            "delta": delta, "proporciones": proporciones, "sd_e": residuos.std(axis=0, ddof=1)}


def sustituto_realista(ctx: dict[str, object], indice: int) -> pd.DataFrame:
    real, n_t, clr = ctx["real"], ctx["n_t"], ctx["clr"]
    z = candidatas.simular(clr["c"], clr["a"], clr["residuos"], clr["x0"], len(real), ctx["calent"], 0, 1, 0.0, 0.0,
                           granger.generador("tal_val_generacion", ctx["nombre"], str(indice)))
    return e1.a_series(z, n_t, real.columns)


def fase_aleatoria(series: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """surrogates.simular_fase_aleatoria con T impar; con T par la frecuencia de Nyquist queda igual (es real)."""
    if len(series) % 2 == 1:
        return surrogates.simular_fase_aleatoria(series, rng)
    X = series.to_numpy()
    T = X.shape[0]
    coef = np.fft.rfft(X, axis=0)
    fases = rng.uniform(0.0, 2 * np.pi, size=(coef.shape[0] - 2, X.shape[1]))
    nuevo = coef.copy()
    nuevo[1:-1] = np.abs(coef[1:-1]) * np.exp(1j * fases)
    return np.fft.irfft(nuevo, n=T, axis=0)


def sustituto_fase(ctx: dict[str, object], indice: int) -> tuple[pd.DataFrame, int]:
    real = ctx["real"]
    X = fase_aleatoria(real, granger.generador("tal_fase_generacion", ctx["nombre"], str(indice)))
    X, corregidas = surrogates.aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas


def sustituto_plantado(ctx: dict[str, object], efecto: str, indice: int) -> tuple[pd.DataFrame, int, tuple[str, str]]:
    """candidatas_grillas.generar con las claves de la réplica."""
    real = ctx["real"]
    columnas = list(real.columns)
    c, a, residuos = ctx["ar"]
    plantables = list(itertools.permutations([col for col in columnas if col != ctx["descartada"]], 2))
    par = plantables[int(granger.generador("tal_par", ctx["nombre"], efecto, str(indice)).integers(0, len(plantables)))]
    beta = float(ctx["betas"][(efecto, par[0], par[1])])
    i_A, i_B = columnas.index(par[0]), columnas.index(par[1])
    X = candidatas.simular(c, a, residuos, real.to_numpy().mean(axis=0), len(real), ctx["calent"], i_A, i_B, beta,
                           planted.media_proceso(c, a, i_A), granger.generador("tal_generacion", ctx["nombre"], efecto, str(indice)))
    X, corregidas = surrogates.aplicar_piso(X)
    return pd.DataFrame(X, index=real.index, columns=real.columns), corregidas, par


def replica_evaluacion(ctx: dict[str, object], familia: str, indice: int,
                       efecto: str = "0.0") -> tuple[dict[str, object], dict[str, np.ndarray]]:
    nombre, n_t = ctx["nombre"], ctx["n_t"]
    comp = control_compartido_fijo(n_t)
    plantado = None
    if familia == "realista":
        x = sustituto_realista(ctx, indice)
        controles = {"propio": control_propio_externo(x, n_t), "compartido": comp}
        clave = lambda c, v, o, d: ("tal_val_remuestreo", nombre, c, str(indice), v, o, d)
        extra: dict[str, object] = {}
    else:
        if familia == "fase":
            x, corregidas = sustituto_fase(ctx, indice)
            clave = lambda c, v, o, d: ("tal_fase_remuestreo", nombre, c, str(indice), v, o, d)
            extra = {"celdas_piso": corregidas}
        else:
            x, corregidas, plantado = sustituto_plantado(ctx, efecto, indice)
            clave = lambda c, v, o, d: ("tal_cp_remuestreo", nombre, c, efecto, str(indice), v, o, d)
            extra = {"celdas_piso": corregidas, "origen": plantado[0], "destino": plantado[1]}
        controles = {"propio": control_propio_interno(candidatas.construir_versiones_k(x, ctx["descartada"])),
                     "compartido": comp}
    fila, arr = analizar_controles(x, ctx["descartada"], ctx["pares"], controles, ctx["B"], clave, plantado)
    return {"candidata": nombre, "familia": familia, "efecto": efecto, "indice": indice, **extra, **fila}, arr


def tarea_beta(ctx: dict[str, object], efecto: str, origen: str, destino: str) -> dict[str, object]:
    real = ctx["real"]
    columnas = list(real.columns)
    i_A, i_B = columnas.index(origen), columnas.index(destino)
    cal = candidatas.calibrar_beta(real, efecto, i_A, i_B, ctx["calent"], ("tal_calibracion", ctx["nombre"], efecto, origen, destino))
    _, a, _ = ctx["ar"]
    M = np.diag(a)
    M[i_B, i_A] = cal["beta"]
    return {"candidata": ctx["nombre"], "efecto": efecto, "origen": origen, "destino": destino, **cal,
            "radio_espectral": float(np.max(np.abs(np.linalg.eigvals(M))))}


# ---------------------------------------------------------------------------
# E1 (sección 7) con las claves de la réplica
# ---------------------------------------------------------------------------

def replica_e1(ctx: dict[str, object], bloque: str, *args: str | int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    real, n_t, desc, pares, B = ctx["real"], ctx["n_t"], ctx["descartada"], ctx["pares"], ctx["B"]
    if bloque == "B2":
        (indice,) = args
        x = sustituto_realista(ctx, int(indice))
        fila, arr = e1.analizar(x, n_t, desc, pares, ("pares",), B,
                                lambda prueba, v, o, d: ("tal_e1_b2_remuestreo", str(indice), v, o, d))
        return {"indice": indice, **fila}, arr
    if bloque == "B3":
        kappa, indice = args
        clr = ctx["clr"]
        lam = float(kappa) * clr["sd_e"] * u_impulsor(real.shape[1])
        z = e1.simular_impulsor(clr["c"], clr["a"], clr["residuos"], clr["x0"], len(real), ctx["calent"], lam,
                                e1.PHI_IMPULSOR, granger.generador("tal_e1_b3_generacion", str(kappa), str(indice)))
        x = e1.a_series(z, n_t, real.columns)
        fila, arr = e1.analizar(x, n_t, desc, pares, e1.PRUEBAS, B,
                                lambda prueba, v, o, d: ("tal_e1_b3_remuestreo", prueba, str(kappa), str(indice), v, o, d))
        return {"kappa": kappa, "indice": indice, **fila}, arr
    if bloque == "B4":
        (indice,) = args
        orden = granger.generador("tal_e1_b4_permutacion", str(indice)).permutation(len(real))
        x = pd.DataFrame(real.to_numpy()[orden], index=real.index, columns=real.columns)
        n_perm = n_t.to_numpy(dtype=float)[orden]
        fila, arr = e1.analizar(x, None, desc, pares, e1.PRUEBAS, B,
                                lambda prueba, v, o, d: ("tal_e1_b4_remuestreo", prueba, str(indice), v, o, d))
        dif = float(np.abs(np.expm1(x).sum(axis=1).to_numpy() - n_perm).max())
        return {"indice": indice, "max_dif_masa_n_t": dif, **fila}, arr
    if bloque == "B5":
        (indice,) = args
        x = sustituto_realista(ctx, int(indice))
        _, arr = e1.analizar(x, n_t, desc, pares, e1.PRUEBAS, 0, None)
        fila: dict[str, object] = {"indice": indice}
        for prueba in e1.PRUEBAS:
            for v in VERSIONES:
                F = arr[f"{prueba}_F_{v}"]
                fila[f"S_{prueba}_{v}"] = float("nan") if np.isnan(F).any() else e1.asimetria(F, pares[v])
        return fila, {k: a for k, a in arr.items() if "_F_" in k}
    raise ValueError(bloque)


# ---------------------------------------------------------------------------
# Procesos independientes
# ---------------------------------------------------------------------------

_CONTEXTOS: dict[str, dict[str, object]] = {}


def iniciar_trabajador(contextos: dict[str, dict[str, object]]) -> None:
    _CONTEXTOS.clear()
    _CONTEXTOS.update(contextos)


def trabajo(tarea: tuple[str | int, ...]) -> tuple[dict[str, object], dict[str, np.ndarray]] | dict[str, object]:
    """("eval", candidata, familia, efecto, índice), ("beta", candidata, efecto, origen, destino) o ("e1", candidata, bloque, ...)."""
    tipo, nombre, *args = tarea
    ctx = _CONTEXTOS[nombre]
    if tipo == "eval":
        familia, efecto, indice = args
        return replica_evaluacion(ctx, familia, int(indice), efecto)
    if tipo == "beta":
        return tarea_beta(ctx, *args)
    if tipo == "e1":
        return replica_e1(ctx, *args)
    raise ValueError(tipo)
