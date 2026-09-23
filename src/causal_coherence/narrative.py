"""
narrative.py

Envoltorio mínimo sobre el método original de Narrative Trails
(UMAP + HDBSCAN + grafo de coherencia). El código de referencia vive en
el repo narrative-trails (ver config.NARRATIVE_TRAILS_DIR) y se importa
desde ahí, sin copiarlo.
"""

import hashlib
import pickle
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence.config import NARRATIVE_TRAILS_DIR, OUTPUT_DIR

sys.path.append(str(NARRATIVE_TRAILS_DIR))

from Library.narrative_landscape import NarrativeLandscape  # type: ignore  # noqa: E402
from Library.storyline import Storyline  # type: ignore  # noqa: E402

__all__ = [
    "CONFIG",
    "LANDSCAPE_PARAMS",
    "NarrativeLandscape",
    "Storyline",
    "alternatives_summary",
    "build_landscape",
    "count_topics",
    "extract_alternatives",
    "format_narrative",
    "load_or_build_landscape",
    "matrix_hash",
    "narrative_table",
    "node_degree_report",
    "print_alternatives",
    "random_pair_paths",
]


@dataclass
class Config:
    """Nombres de columnas que espera Storyline.print_narrative_path."""
    data_column = "content"
    date_column = "date"
    summary_column = "title"


CONFIG = Config()

LANDSCAPE_PARAMS = {
    "impose_date_constraint": True,
    "n_neighbors": 16,
    "min_cluster_size": 4,
}

# Paquetes cuya versión cambia el landscape (ver docs/reproducibilidad.md).
# Forman parte de la clave de caché: si cambia una versión, se reajusta.
_VERSIONED_PACKAGES = [
    "umap-learn", "pynndescent", "numba", "hdbscan", "scikit-learn",
    "scipy", "numpy", "networkx",
]

# Atributos que el landscape no necesita para extraer narrativas ni
# evaluarlas; se descartan antes de guardar en caché para no duplicar
# matrices NxN en disco.
_UNCACHED_ATTRS = ("ang_sim", "topic_sim", "mst")


# ---------------------------------------------------------------------------
# Construcción y caché del landscape
# ---------------------------------------------------------------------------

def build_landscape(embeddings: np.ndarray, dates) -> NarrativeLandscape:
    landscape = NarrativeLandscape(**LANDSCAPE_PARAMS, verbose=True)
    landscape.fit(embeddings, dates=dates)
    return landscape


def count_topics(cluster_labels) -> int:
    """Número de tópicos descubiertos por HDBSCAN, sin contar el ruido (-1)."""
    labels = set(cluster_labels)
    return len(labels) - (1 if -1 in labels else 0)


def matrix_hash(matrix: np.ndarray) -> str:
    """Identificador derivado del contenido de la matriz, no de un nombre elegido a mano (R6)."""
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()[:16]


def landscape_cache_key(embeddings: np.ndarray, dates) -> str:
    """
    Clave derivada de todo lo que determina el landscape: embeddings,
    fechas, parámetros y versiones de las librerías que lo afectan.
    Mismo criterio que matrix_hash: la identidad sale del contenido.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(embeddings).tobytes())
    h.update(np.asarray(pd.to_datetime(dates), dtype="datetime64[ns]").tobytes())
    h.update(repr(sorted(LANDSCAPE_PARAMS.items())).encode())
    h.update(repr([(p, version(p)) for p in _VERSIONED_PACKAGES]).encode())
    return h.hexdigest()[:16]


def load_or_build_landscape(embeddings: np.ndarray, dates, cache_dir: Path = None):
    """
    Devuelve (landscape, coherence_hash). Si ya existe un landscape
    ajustado con las mismas entradas, lo carga desde cache_dir en vez de
    volver a correr UMAP + HDBSCAN + grafo de coherencia.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else OUTPUT_DIR / "cache"
    path = cache_dir / f"landscape_{landscape_cache_key(embeddings, dates)}.pkl"

    if path.exists():
        with open(path, "rb") as f:
            cached = pickle.load(f)
        print(f"Landscape cargado desde caché: {path.name}")
        return cached["landscape"], cached["coherence_hash"]

    landscape = build_landscape(embeddings, np.asarray(dates))
    coherence_hash = matrix_hash(landscape.sparse_coherence)
    for attr in _UNCACHED_ATTRS:
        setattr(landscape, attr, None)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"landscape": landscape, "coherence_hash": coherence_hash}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Landscape guardado en caché: {path.name}")
    return landscape, coherence_hash


# ---------------------------------------------------------------------------
# Extracción de narrativas
# ---------------------------------------------------------------------------

def extract_alternatives(landscape: NarrativeLandscape, src: int, tgt: int, n_paths: int = 3):
    """
    Hasta n_paths narrativas entre el mismo par de extremos, excluyendo en
    cada vuelta los nodos intermedios ya usados por las anteriores (el
    mecanismo de "k storylines distintas" del notebook de referencia).
    Devuelve menos de n_paths si se agotan los caminos.
    """
    hidden_nodes = []
    storylines = []
    for _ in range(n_paths):
        path, _ = landscape.extract_narrative(src, tgt, hidden_nodes=hidden_nodes)
        if not path:
            break
        storyline = Storyline(landscape, path)
        storylines.append(storyline)
        hidden_nodes.extend(storyline.chain[1:-1])
    return storylines


def narrative_table(df: pd.DataFrame, landscape: NarrativeLandscape, chain, lead_chars: int = 0) -> pd.DataFrame:
    """
    Un documento por fila, en el orden del camino: fecha, tópico, coherencia
    base con el documento anterior y título completo. Con lead_chars > 0
    agrega el comienzo del contenido (útil cuando el título es solo el
    comienzo del texto, como en bpoil).
    """
    coherence = [np.nan] + [landscape.base_coherence[a][b] for a, b in zip(chain[:-1], chain[1:])]
    table = pd.DataFrame({
        "fecha": df.loc[chain, "date"].dt.date.values,
        "tópico": landscape.cluster_labels[chain],
        "coherencia": coherence,
        "título": df.loc[chain, "title"].values,
    }, index=pd.Index(chain, name="doc"))
    if lead_chars:
        table["inicio"] = [text[:lead_chars].rstrip() + "…" for text in df.loc[chain, "content"]]
    return table


def alternatives_summary(storylines) -> pd.DataFrame:
    """Largo, bottleneck y reliability de cada narrativa alternativa."""
    return pd.DataFrame([
        {"largo": len(s.chain), "bottleneck": s.bottleneck_weight(), "reliability": s.reliability()}
        for s in storylines
    ]).rename_axis("alternativa")


def print_alternatives(df: pd.DataFrame, landscape: NarrativeLandscape, storylines, lead_chars: int = 0) -> None:
    """Imprime cada narrativa alternativa documento por documento (ver narrative_table)."""
    for i, storyline in enumerate(storylines):
        print(f"--- Alternativa {i}: {len(storyline.chain)} documentos · "
              f"bottleneck {storyline.bottleneck_weight():.3f} · "
              f"reliability {storyline.reliability():.3f}")
        print(format_narrative(narrative_table(df, landscape, storyline.chain, lead_chars)))
        print()


def format_narrative(table: pd.DataFrame) -> str:
    """Texto legible de narrative_table: una entrada por documento, sin truncar."""
    lines = []
    for doc, row in table.iterrows():
        coh = "" if pd.isna(row["coherencia"]) else f" · coherencia {row['coherencia']:.3f}"
        lines.append(f"[{doc}] {row['fecha']} · tópico {row['tópico']}{coh}")
        lines.append(f"    {row['título']}")
        if "inicio" in table.columns:
            lines.append(f"    > {row['inicio']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnósticos
# ---------------------------------------------------------------------------

def node_degree_report(landscape: NarrativeLandscape, node: int) -> dict:
    """
    Grado de un nodo (entrante + saliente) comparado con el resto del grafo.
    percentile es la fracción de nodos con grado estrictamente menor; se
    marca como periférico si su grado es menor que la mitad del promedio.

    El diagnóstico vale solo para el landscape que se le pasa: un mismo
    índice es otro documento en otro corpus.
    """
    degrees = dict(landscape.nx_graph.degree())
    values = np.fromiter(degrees.values(), dtype=float)
    degree = degrees[node]
    mean_degree = float(values.mean())
    return {
        "node": node,
        "degree": degree,
        "mean_degree": mean_degree,
        "median_degree": float(np.median(values)),
        "percentile": float((values < degree).mean()),
        "peripheral": degree < mean_degree * 0.5,
    }


def random_pair_paths(landscape: NarrativeLandscape, dates: pd.Series, n_pairs: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    Extrae la narrativa entre n_pairs pares de documentos al azar (el de
    índice menor como origen) y registra su separación en días, el largo
    del camino, su bottleneck y el percentil de grado de cada extremo.
    Sirve para saber qué largo de camino es típico en un grafo,
    independiente de un par de extremos en particular.
    """
    degrees = dict(landscape.nx_graph.degree())
    values = np.fromiter(degrees.values(), dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_pairs):
        src, tgt = sorted(int(i) for i in rng.choice(len(dates), size=2, replace=False))
        path, _ = landscape.extract_narrative(src, tgt, hidden_nodes=[])
        rows.append({
            "src": src,
            "tgt": tgt,
            "días": (dates.iloc[tgt] - dates.iloc[src]).days,
            "largo": len(path) if path else np.nan,
            "bottleneck": Storyline(landscape, path).bottleneck_weight() if path else np.nan,
            "pct_src": float((values < degrees[src]).mean()),
            "pct_tgt": float((values < degrees[tgt]).mean()),
        })
    return pd.DataFrame(rows)
