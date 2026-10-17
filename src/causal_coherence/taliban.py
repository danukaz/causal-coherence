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
"""

import numpy as np
import pandas as pd

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
