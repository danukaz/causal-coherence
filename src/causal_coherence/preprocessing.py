"""
preprocessing.py

Preprocesamiento de texto en inglés -- el corpus bpoil es en inglés, a
diferencia del corpus en español del repo t2s2026 de referencia.
"""

import nltk
import spacy
from nltk.corpus import stopwords


def build_english_pipeline() -> spacy.Language:
    """
    Carga el modelo de spaCy en inglés. Si no lo tienes descargado,
    corre esto una vez en la terminal (con el ambiente activado):
        python -m spacy download en_core_web_sm
    """
    return spacy.load("en_core_web_sm")


def get_english_stopwords() -> set:
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


def preprocess_batch(texts, pipeline, extra_stopwords) -> list[list[str]]:
    """
    Tokeniza, lematiza y limpia una lista de textos.
    Se descartan signos de puntuación, números, y stopwords.
    Nota: esta función es propia del proyecto, no una reutilización de
    cet.preprocessing del repo t2s2026 -- ese módulo asume un config.json
    en el directorio de trabajo y un modelo en español por defecto, así
    que replicar la misma lógica aquí, de forma aislada, es más limpio
    que intentar reusarlo directamente.
    """
    preprocessed = []
    for doc in pipeline.pipe(texts, batch_size=200):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and not token.is_punct
            and not token.is_digit
        ]
        tokens = [t for t in tokens if t not in extra_stopwords]
        preprocessed.append(tokens)
    return preprocessed
