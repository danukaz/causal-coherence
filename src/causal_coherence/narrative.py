"""
narrative.py

Envoltorio mínimo sobre el método original de Narrative Trails
(UMAP + HDBSCAN + grafo de coherencia). El código de referencia vive en
el repo narrative-trails (ver config.NARRATIVE_TRAILS_DIR) y se importa
desde ahí, sin copiarlo.
"""

import hashlib
import sys
from dataclasses import dataclass

import numpy as np

from causal_coherence.config import NARRATIVE_TRAILS_DIR

sys.path.append(str(NARRATIVE_TRAILS_DIR))

from Library.narrative_landscape import NarrativeLandscape  # type: ignore  # noqa: E402
from Library.storyline import Storyline  # type: ignore  # noqa: E402

__all__ = [
    "CONFIG",
    "NarrativeLandscape",
    "Storyline",
    "build_landscape",
    "count_topics",
    "matrix_hash",
]


@dataclass
class Config:
    """Nombres de columnas que espera Storyline.print_narrative_path."""
    data_column = "content"
    date_column = "date"
    summary_column = "title"


CONFIG = Config()


def build_landscape(embeddings: np.ndarray, dates) -> NarrativeLandscape:
    landscape = NarrativeLandscape(
        impose_date_constraint=True,
        n_neighbors=16,
        min_cluster_size=4,
        verbose=True,
    )
    landscape.fit(embeddings, dates=dates)
    return landscape


def count_topics(cluster_labels) -> int:
    """Número de tópicos descubiertos por HDBSCAN, sin contar el ruido (-1)."""
    labels = set(cluster_labels)
    return len(labels) - (1 if -1 in labels else 0)


def matrix_hash(matrix: np.ndarray) -> str:
    """Identificador derivado del contenido de la matriz, no de un nombre elegido a mano (R6)."""
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()[:16]
