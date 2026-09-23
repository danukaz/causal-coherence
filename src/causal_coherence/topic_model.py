"""
topic_model.py

Configuración y ajuste de RollingLDA sobre bpoil.
"""

import pandas as pd
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence.config import OUTPUT_DIR

MODEL_PATH = OUTPUT_DIR / "roll_lda_bpoil.pickle"

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


def fit_topic_model(df: pd.DataFrame) -> RollingLDA:
    roll = RollingLDA(**LDA_CONFIG)
    roll.fit(df, text_column="preprocessed_text", date_column="date")
    return roll
