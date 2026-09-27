"""
data_loading.py

Carga y filtrado de los corpus del laboratorio (T17 y Afganistán). Sin
dependencias de nltk/spacy/ttta/umap a propósito -- este módulo debe
poder importarse (desde un notebook, por ejemplo) sin arrastrar
paquetes pesados que no se necesitan solo para cargar datos.
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


def documents_between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fecha y título de los documentos en una ventana de fechas (ambos extremos incluidos)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, ["date", "title"]]


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


# ---------------------------------------------------------------------------
# Ventana de análisis
# ---------------------------------------------------------------------------

# Decisión tomada tras inspeccionar semana a semana: antes del
# 19-04-2010 el corpus tiene solo 2 documentos en 3 semanas (la explosión de
# la plataforma fue el 20-04-2010, así que casi no hay cobertura real antes),
# incluyendo una semana totalmente vacía. RollingLDA debe reajustarse sobre
# esta misma ventana para que la serie de tópicos sea coherente con el
# modelo que la genera.
CORPUS_START = pd.Timestamp("2010-04-19")
CORPUS_END = pd.Timestamp("2010-09-30")

# Mínimo de documentos por ventana de la serie temporal (R1 de la guía: las
# ventanas que no alcanzan un mínimo declarado se excluyen, además de las de
# borde incompletas). Con la grilla semanal actual la ventana más chica tiene
# 7 documentos, así que este mínimo no excluye ninguna.
MIN_DOCS_PER_WINDOW = 5


def truncate_to_analysis_window(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["date"] >= CORPUS_START) & (df["date"] <= CORPUS_END)
    descartados = len(df) - mask.sum()
    print(f"Truncamiento a [{CORPUS_START.date()}, {CORPUS_END.date()}]: "
          f"se dejan fuera {descartados} de {len(df)} documentos.")
    return df.loc[mask].reset_index(drop=True)


def drop_windows_below_min_docs(windows: pd.DataFrame, n_docs: pd.Series):
    """
    Descarta las ventanas con menos de MIN_DOCS_PER_WINDOW documentos.
    windows y n_docs están indexados por la misma fecha de ventana.
    Devuelve (windows, n_docs) filtrados.
    """
    keep = n_docs >= MIN_DOCS_PER_WINDOW
    for fecha in n_docs.index[~keep]:
        print(f"  Se descarta la ventana que cierra {fecha.date()} ({n_docs[fecha]} documentos, "
              f"mínimo {MIN_DOCS_PER_WINDOW}).")
    print(f"{int((~keep).sum())} ventanas excluidas por mínimo de documentos "
          f"(MIN_DOCS_PER_WINDOW = {MIN_DOCS_PER_WINDOW}).")
    return windows.loc[keep], n_docs.loc[keep]