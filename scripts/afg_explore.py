"""
afg_explore.py

Exploración rápida, NO parte formal del proyecto: correr el método
original de Narrative Trails sobre el subset "Taliban" del corpus de
Afganistán, para comparar contra los resultados ya obtenidos con bpoil.
Ambiente: capstone.
"""

import numpy as np

from causal_coherence.data_loading import AFGHANISTAN_DIR, load_subset
from causal_coherence.narrative import CONFIG, Storyline, build_landscape, count_topics

SOURCE_LABEL = "NEWS-TLS Entities (Taliban)"

# Origen/destino fijados a mano -- rápido para una exploración, no una
# elección justificada como la de bpoil.
SRC_NODE = 15
TGT_NODE = 727


def main():
    df, embeddings = load_subset(SOURCE_LABEL, AFGHANISTAN_DIR)
    print(f"Documentos ({SOURCE_LABEL}): {len(df)}")
    print(f"Rango de fechas: {df['date'].min().date()} a {df['date'].max().date()}")

    print(f"\nOrigen  ({SRC_NODE}): {df.loc[SRC_NODE, 'title']} -- {df.loc[SRC_NODE, 'date'].date()}")
    print(f"Destino ({TGT_NODE}): {df.loc[TGT_NODE, 'title']} -- {df.loc[TGT_NODE, 'date'].date()}")

    print("\nAjustando NarrativeLandscape (UMAP + HDBSCAN)...")
    landscape = build_landscape(embeddings, df["date"].values)
    print(f"Coherencia base promedio: {np.mean(landscape.base_coherence):.4f}")
    print(f"Tópicos descubiertos: {count_topics(landscape.cluster_labels)}")

    narrative_path, _ = landscape.extract_narrative(
        SRC_NODE, TGT_NODE, hidden_nodes=[]
    )

    if not narrative_path:
        print("No se encontró camino entre esos dos documentos.")
        return

    storyline = Storyline(landscape, narrative_path)
    print("-" * 13)
    print("Path (índices):", storyline.chain)
    print("Bottleneck:", storyline.bottleneck_weight())
    print("Reliability:", storyline.reliability())
    print("Length of Path:", len(storyline.chain))

    Storyline.print_narrative_path(df, landscape.cluster_labels, storyline.chain, CONFIG)

    # --- Diagnóstico: ¿el destino es un nodo periférico en el grafo? ---
    grados = dict(landscape.nx_graph.degree())
    grado_destino = grados[TGT_NODE]
    grado_promedio = sum(grados.values()) / len(grados)
    print(f"\nGrado del nodo destino ({TGT_NODE}): {grado_destino}")
    print(f"Grado promedio de todo el grafo: {grado_promedio:.1f}")
    if grado_destino < grado_promedio * 0.5:
        print("-> El destino está notablemente por debajo del promedio: nodo periférico.")


if __name__ == "__main__":
    main()
