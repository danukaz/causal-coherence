"""
baseline_narrative.py

Semana 4: extracción de la narrativa de línea base de extremo a extremo,
usando el método original de Narrative Trails (UMAP + HDBSCAN) sobre el
corpus principal (bpoil), con reproducibilidad resuelta.

A diferencia de fit_rolling_lda.py (semana 3), aquí NO se trunca el
corpus ni se usa RollingLDA -- se usa bpoil completo (1133 documentos)
y el pipeline original del repo de referencia.
"""

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import load_bpoil_full
from causal_coherence.narrative import (
    CONFIG,
    Storyline,
    build_landscape,
    count_topics,
    matrix_hash,
)

# Origen (3): reporta el hundimiento de la plataforma y el riesgo de derrame,
# marcando el inicio real de la crisis (no solo la explosión del día
# anterior, que aún no se sabía que derivaría en un derrame de esta escala).
# Destino (975): titular directo sobre el sellado del pozo, marcando el cierre
# narrativo de la historia principal del derrame.
SRC_NODE = 15
TGT_NODE = 572

# Narrativas alternativas a extraer entre el mismo par de extremos.
N_PATHS = 3


def show_candidates(df: pd.DataFrame, start: str, end: str, n: int = 10):
    """Muestra documentos en una ventana de fechas, para elegir origen/destino con criterio."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    print(df.loc[mask, ["date", "title"]].head(n).to_string())


def main():
    df, embeddings = load_bpoil_full()

    print("Candidatos cerca de la explosión (20-23 abril 2010):")
    show_candidates(df, "2010-04-20", "2010-04-23")

    print("\nCandidatos cerca del sellado del pozo (15-20 sept 2010):")
    show_candidates(df, "2010-09-15", "2010-09-20")

    if SRC_NODE is None or TGT_NODE is None:
        print(
            "\nRevisa los candidatos de arriba, fija SRC_NODE/TGT_NODE al "
            "principio del archivo (con un comentario de por qué), y vuelve a correr."
        )
        return

    print(f"\nOrigen  ({SRC_NODE}): {df.loc[SRC_NODE, 'title']} -- {df.loc[SRC_NODE, 'date'].date()}")
    print(f"Destino ({TGT_NODE}): {df.loc[TGT_NODE, 'title']} -- {df.loc[TGT_NODE, 'date'].date()}")

    print("\nAjustando NarrativeLandscape (UMAP + HDBSCAN)...")
    landscape = build_landscape(embeddings, df["date"].values)
    print(f"Coherencia base promedio: {np.mean(landscape.base_coherence):.4f}")
    print(f"Tópicos descubiertos: {count_topics(landscape.cluster_labels)}")

    coherence_hash = matrix_hash(landscape.sparse_coherence)
    print(f"Hash de la matriz de coherencia: {coherence_hash}")

    # Varias narrativas alternativas entre el mismo par de extremos,
    # excluyendo en cada vuelta los nodos intermedios ya usados por las
    # anteriores -- el mismo mecanismo de "k storylines distintas" que
    # vimos en el notebook de referencia.
    hidden_nodes = []
    resultados = []

    for i in range(N_PATHS):
        narrative_path, _ = landscape.extract_narrative(
            SRC_NODE, TGT_NODE, hidden_nodes=hidden_nodes
        )

        if not narrative_path:
            print(f"\nAlternativa {i}: no se encontró camino (nodos disponibles agotados).")
            break

        storyline = Storyline(landscape, narrative_path)
        print(f"\n{'=' * 13} Alternativa {i} {'=' * 13}")
        print("Path (índices):", storyline.chain)
        print("Bottleneck:", storyline.bottleneck_weight())
        print("Reliability:", storyline.reliability())
        print("Length of Path:", len(storyline.chain))

        Storyline.print_narrative_path(df, landscape.cluster_labels, storyline.chain, CONFIG)

        hidden_nodes.extend(storyline.chain[1:-1])
        resultados.append({
            "alternativa": i,
            "chain": storyline.chain,
            "bottleneck": storyline.bottleneck_weight(),
            "reliability": storyline.reliability(),
        })

    if not resultados:
        print("No se encontró ningún camino entre esos dos documentos.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "alternativas": resultados,
        "src_node": SRC_NODE,
        "tgt_node": TGT_NODE,
        "coherence_hash": coherence_hash,
    }
    out_path = OUTPUT_DIR / f"baseline_{coherence_hash}.pkl"
    pd.to_pickle(result, out_path)
    print(f"\nResultado guardado en {out_path}")


if __name__ == "__main__":
    main()
