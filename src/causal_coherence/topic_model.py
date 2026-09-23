"""
topic_model.py

Configuración y ajuste de RollingLDA sobre bpoil.
"""

import os
import warnings

import pandas as pd
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import SOURCE_LABEL, build_dataframe, load_filtered

MODEL_PATH = OUTPUT_DIR / "roll_lda_bpoil.pickle"

# El modelo de referencia (data/roll_lda_bpoil.pickle, θ con hash
# 2d6dc4b1469dbacf) se ajustó con PYTHONHASHSEED=0.
REFERENCE_HASH_SEED = "0"

# EXPERIMENTO DE CONTROL: la primera corrida (chunks mensuales
# automáticos, prototype=100) NO mostró "broken models" en agosto ni
# septiembre -- solo desde octubre en adelante. Toda la fusión de
# chunks que hicimos después la probamos con prototype=10 (bajado por
# velocidad), lo cual pudo introducir fallas artificiales que no son
# reales: con solo 10 candidatos por chunk, es más fácil que ninguno
# alcance el umbral por mala suerte, no por falta real de vocabulario.
# Se vuelve a chunks mensuales simples y prototype=100 para verificar
# si la fusión agresiva era realmente necesaria.
TRUNCATE_AFTER = pd.Timestamp("2010-09-30")

CUSTOM_CHUNKS = pd.to_datetime([
    "2010-04-30",
    "2010-05-31",
    "2010-06-30",
    "2010-07-31",
    "2010-08-31",
    "2010-10-01",  # +1 día del límite de truncamiento (30-sep), no la
                   # fecha límite misma -- para no dejar huérfanos los
                   # documentos fechados exactamente el 30 de septiembre
]).tolist()

# Hiperparámetros de RollingLDA. K=10 y alpha=0.1 coinciden con la
# configuración de referencia encontrada en el HTML que entregó el
# laboratorio. warmup=2 (los primeros 2 chunks de CUSTOM_CHUNKS, es
# decir abril y mayo) es una elección propia, razonable para un corpus
# que solo cubre 10 meses -- hay que documentarla y estar dispuesto a
# ablacionarla, no es un valor fijo de la especificación.
LDA_CONFIG = {
    "K": 10,
    "alpha": 0.1,
    "how": CUSTOM_CHUNKS,
    "warmup": 2,       # abril+mayo como warmup, chunks mensuales normales
    "memory": 3,
    "prototype": 100,  # 100 candidatos por chunk, no 10 como en la fusión agresiva
    "initial_epochs": 500,
    "subsequent_epochs": 500,
    "min_docs_per_chunk": 1,
    # Por defecto es [5, 0.002] -- una palabra necesita 5 ocurrencias Y
    # 0.2% de proporción dentro de un tópico para que ese tópico cuente
    # como válido. Con el chunk final (~125 documentos) eso resultaba
    # en "all models are broken". Bajado a [2, 0.001] para probar si el
    # problema era solo el umbral o si el vocabulario de esos meses es
    # genuinamente demasiado disperso incluso con un criterio más laxo.
    "topic_threshold": [2, 0.001],
    "seed": 42,
    "verbose": 1,
}


def truncate_to_coverage_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorta el corpus a la ventana de cobertura sostenida (hasta
    TRUNCATE_AFTER). Ver el comentario junto a esa constante para la
    justificación completa -- esto no es un descarte silencioso, quien
    llama debe reportar cuántos documentos se dejan fuera.
    """
    return df[df["date"] <= TRUNCATE_AFTER].reset_index(drop=True)


def check_hash_seed() -> None:
    """
    seed=42 no alcanza para repetir el modelo. ttta arma el vocabulario
    recorriendo un set de strings (ttta.preprocessing.preprocess.create_dtm),
    así que el id de cada palabra depende de PYTHONHASHSEED, y con otros ids
    el mismo muestreo da otros tópicos (ver docs/reproducibilidad.md). Sin
    PYTHONHASHSEED fijo el ajuste no es repetible, así que se corta antes de
    gastar los ~20 minutos; con un valor distinto al de referencia solo se
    avisa, porque puede ser a propósito.
    """
    value = os.environ.get("PYTHONHASHSEED")
    if value is None or value == "random":
        raise RuntimeError(
            "PYTHONHASHSEED no está fijo, así que el orden del vocabulario de ttta (y los "
            "tópicos) cambiaría entre corridas. Activar el ambiente (conda activate "
            f"causal-coherence) o definir PYTHONHASHSEED={REFERENCE_HASH_SEED}."
        )
    if value != REFERENCE_HASH_SEED:
        warnings.warn(
            f"PYTHONHASHSEED={value}: el modelo va a ser distinto al de referencia, "
            f"que se ajustó con PYTHONHASHSEED={REFERENCE_HASH_SEED}.",
            stacklevel=2,
        )


def fit_topic_model(df: pd.DataFrame) -> RollingLDA:
    roll = RollingLDA(**LDA_CONFIG)
    roll.fit(df, text_column="preprocessed_text", date_column="date")
    return roll


def chunk_summary(roll: RollingLDA, df: pd.DataFrame) -> pd.DataFrame:
    """
    Documentos y rango de fechas real de cada chunk, calculado sobre el
    DataFrame ordenado por fecha -- no depende de la etiqueta que muestra
    la consola durante el ajuste, que viene retrasada un chunk.
    """
    df_sorted = df.sort_values("date", kind="stable").reset_index(drop=True)
    starts = roll.chunk_indices["chunk_start"].tolist() + [len(df_sorted)]
    rows = []
    for i in range(len(roll.chunk_indices)):
        lo, hi = starts[i], starts[i + 1]
        rows.append({
            "documentos": hi - lo,
            "desde": df_sorted["date"].iloc[lo].date(),
            "hasta": df_sorted["date"].iloc[hi - 1].date(),
        })
    return pd.DataFrame(rows).rename_axis("chunk")


def load_topic_model(path=MODEL_PATH) -> RollingLDA:
    """Carga un modelo ya ajustado (por defecto el que guarda fit_rolling_lda.py)."""
    roll = RollingLDA(**LDA_CONFIG)
    roll.load(str(path))
    return roll


def training_documents() -> pd.DataFrame:
    """
    Documentos con que se ajusta el modelo, en el mismo orden que las filas
    de su matriz documento-tópico: bpoil truncado a TRUNCATE_AFTER y
    ordenado por fecha con orden estable, que es lo que hace
    RollingLDA.fit internamente. El modelo guardado no incluye las fechas.
    """
    docs, _ = load_filtered(SOURCE_LABEL)
    df = truncate_to_coverage_window(build_dataframe(docs))
    return df.sort_values("date", kind="stable").reset_index(drop=True)
