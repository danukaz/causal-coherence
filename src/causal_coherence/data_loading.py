"""
data_loading.py

Carga y filtrado de los corpus del laboratorio (T17 y Afganistán). Sin
dependencias de nltk/spacy/ttta a propósito -- este módulo debe poder
importarse desde cualquier ambiente (capstone o rollinglda) sin
arrastrar paquetes pesados que no se necesitan solo para cargar datos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence.config import DATASETS_DIR

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

T17_DIR = DATASETS_DIR / "t17"
AFGHANISTAN_DIR = DATASETS_DIR / "afghanistan"

TOPIC = "bpoil"
SOURCE_LABEL = f"NEWS-TLS T17 ({TOPIC})"


def corpus_path(dataset_dir: Path) -> Path:
    return dataset_dir / "corpus.jsonl"


def embeddings_path(dataset_dir: Path) -> Path:
    return dataset_dir / "emb=mpnet" / "embeddings_mpnet.npy"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_full_corpus(dataset_dir: Path = T17_DIR) -> list[dict]:
    docs = []
    with open(corpus_path(dataset_dir), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def load_embeddings(dataset_dir: Path = T17_DIR) -> np.ndarray:
    return np.load(embeddings_path(dataset_dir))


# ---------------------------------------------------------------------------
# Filtrado y alineación con los embeddings
# ---------------------------------------------------------------------------

def filter_topic(docs: list[dict], embeddings: np.ndarray, source_label: str):
    assert embeddings.shape[0] == len(docs), (
        f"Los embeddings tienen {embeddings.shape[0]} filas pero el corpus "
        f"completo tiene {len(docs)} documentos -- no se puede asumir que "
        f"están alineados por posición."
    )
    mask = np.array([doc["source"] == source_label for doc in docs])
    filtered_docs = [doc for doc, keep in zip(docs, mask) if keep]
    filtered_embeddings = embeddings[mask]
    assert len(filtered_docs) == filtered_embeddings.shape[0]
    return filtered_docs, filtered_embeddings


def date_range(docs: list[dict]):
    dates = sorted(doc["date"] for doc in docs)
    return dates[0], dates[-1]


def build_dataframe(docs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    return df


def sort_by_date(df: pd.DataFrame, embeddings: np.ndarray):
    """Ordena documentos y embeddings juntos por fecha, preservando la alineación."""
    order = df["date"].argsort(kind="stable").to_numpy()
    return df.iloc[order].reset_index(drop=True), embeddings[order]


# ---------------------------------------------------------------------------
# Atajos de alto nivel
# ---------------------------------------------------------------------------

def load_filtered(source_label: str = SOURCE_LABEL, dataset_dir: Path = T17_DIR):
    """Documentos (en el orden original del corpus) y embeddings de un subset."""
    docs = load_full_corpus(dataset_dir)
    embeddings = load_embeddings(dataset_dir)
    return filter_topic(docs, embeddings, source_label)


def load_subset(source_label: str, dataset_dir: Path):
    """Carga un subset completo (sin truncar) como DataFrame, ordenado cronológicamente."""
    docs, embeddings = load_filtered(source_label, dataset_dir)
    df = build_dataframe(docs)
    return sort_by_date(df, embeddings)


def load_bpoil_full():
    """Carga bpoil completo (sin truncar), ordenado cronológicamente."""
    return load_subset(SOURCE_LABEL, T17_DIR)
