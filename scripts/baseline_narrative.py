"""
baseline_narrative.py

Semana 4: extracción de la narrativa de línea base de extremo a extremo,
usando el método original de Narrative Trails (UMAP + HDBSCAN) sobre el
corpus principal (bpoil), con reproducibilidad resuelta.

A diferencia de fit_rolling_lda.py (semana 3), aquí NO se trunca el
corpus ni se usa RollingLDA -- se usa bpoil completo (1133 documentos)
y el pipeline original del repo de referencia.

Uso:
    python scripts/baseline_narrative.py            # par de referencia 15 -> 572
    python scripts/baseline_narrative.py 3 975      # cualquier otro par
"""

import sys

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR, PROJECT_ROOT
from causal_coherence.data_loading import documents_between, load_bpoil_full
from causal_coherence.narrative import (
    CONFIG,
    Storyline,
    build_landscape,
    count_topics,
    extract_alternatives,
    matrix_hash,
    node_degree_report,
)

# Par de referencia: 15 -> 572. Hay dos pares con corridas de referencia en
# results/, y los dos se mantienen porque cada uno muestra algo distinto.
#
# 3 -> 975 fue el primer par, elegido por su lectura narrativa: 3
# (22-04-2010) reporta el hundimiento de la plataforma y el riesgo de
# derrame; 975 (16-09-2010) anuncia el sellado del pozo. Dio caminos cortos
# (3 y 4 documentos) y débiles (bottleneck cercano a 0,5). Son
# results/corrida_1.txt y corrida_2.txt.
#
# 15 -> 572 se eligió después, de forma ad hoc, para ver si otro par daba
# algo distinto. Dio caminos de 6 a 8 documentos con bottleneck cercano a
# 0,82: la elección del par cambia mucho el resultado.
#
# Verificación de grado antes de fijar el par (node_degree_report sobre el
# grafo de bpoil, hash 63d0faab065a6f4c; grado = aristas entrantes +
# salientes; periférico = menos de la mitad del grado promedio, 678):
#
#   nodo   fecha        grado   percentil   periférico
#   3      2010-04-22      18      0,13      sí
#   975    2010-09-16     875      0,43      no
#   15     2010-04-27     878      0,57      no
#   572    2010-06-20     887      0,90      no
#
# El nodo atípico de 3 -> 975 es el origen: 3 tiene 18 aristas salientes,
# cuando los documentos a 15 días o menos tienen una mediana de 826. 975
# es típico para su fecha. 15 y 572 quedan a menos de 2 % de la mediana
# del grafo (876). En bpoil el grado está muy concentrado arriba (el 76 %
# de los nodos tiene entre 860 y 901, que es el máximo), así que el
# percentil 0,90 de 572 equivale a 11 aristas sobre la mediana: en este
# grafo no hay hubs, y el riesgo está en la cola de nodos con pocas
# aristas (264 de 1133 son periféricos). Por eso el chequeo se repite en
# cada corrida (print_degree_check).
SRC_NODE = 15
TGT_NODE = 572

# Narrativas alternativas a extraer entre el mismo par de extremos.
N_PATHS = 3


def show_candidates(df: pd.DataFrame, start: str, end: str, n: int = 10):
    """Muestra documentos en una ventana de fechas, para elegir origen/destino con criterio."""
    print(documents_between(df, start, end).head(n).to_string())


def print_degree_check(landscape, src: int, tgt: int) -> None:
    """Grado total de los extremos frente al resto del grafo; avisa si alguno es periférico."""
    print("\nGrado de los extremos (aristas entrantes + salientes):")
    for role, node in (("Origen ", src), ("Destino", tgt)):
        r = node_degree_report(landscape, node)
        note = "  <- PERIFÉRICO: el camino tiene pocas aristas para entrar o salir" if r["peripheral"] else ""
        print(f"  {role} {node}: grado {r['degree']} (mediana {r['median_degree']:.0f}, "
              f"promedio {r['mean_degree']:.0f}, percentil {r['percentile']:.2f}){note}")


def parse_pair(argv) -> tuple[int, int]:
    if len(argv) == 1:
        return SRC_NODE, TGT_NODE
    if len(argv) == 3:
        return int(argv[1]), int(argv[2])
    raise SystemExit("uso: python scripts/baseline_narrative.py [ORIGEN DESTINO]")


def main():
    src, tgt = parse_pair(sys.argv)
    df, embeddings = load_bpoil_full()

    print("Candidatos cerca de la explosión (20-23 abril 2010):")
    show_candidates(df, "2010-04-20", "2010-04-23")

    print("\nCandidatos cerca del sellado del pozo (15-20 sept 2010):")
    show_candidates(df, "2010-09-15", "2010-09-20")

    print(f"\nOrigen  ({src}): {df.loc[src, 'title']} -- {df.loc[src, 'date'].date()}")
    print(f"Destino ({tgt}): {df.loc[tgt, 'title']} -- {df.loc[tgt, 'date'].date()}")

    print("\nAjustando NarrativeLandscape (UMAP + HDBSCAN)...")
    landscape = build_landscape(embeddings, df["date"].values)
    print(f"Coherencia base promedio: {np.mean(landscape.base_coherence):.4f}")
    print(f"Tópicos descubiertos: {count_topics(landscape.cluster_labels)}")

    coherence_hash = matrix_hash(landscape.sparse_coherence)
    print(f"Hash de la matriz de coherencia: {coherence_hash}")

    print_degree_check(landscape, src, tgt)

    storylines = extract_alternatives(landscape, src, tgt, N_PATHS)
    resultados = []

    for i, storyline in enumerate(storylines):
        print(f"\n{'=' * 13} Alternativa {i} {'=' * 13}")
        print("Path (índices):", storyline.chain)
        print("Bottleneck:", storyline.bottleneck_weight())
        print("Reliability:", storyline.reliability())
        print("Length of Path:", len(storyline.chain))

        Storyline.print_narrative_path(df, landscape.cluster_labels, storyline.chain, CONFIG)

        resultados.append({
            "alternativa": i,
            "chain": storyline.chain,
            "bottleneck": storyline.bottleneck_weight(),
            "reliability": storyline.reliability(),
        })

    if len(storylines) < N_PATHS:
        print(f"\nAlternativa {len(storylines)}: no se encontró camino (nodos disponibles agotados).")

    if not resultados:
        print("No se encontró ningún camino entre esos dos documentos.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "alternativas": resultados,
        "src_node": src,
        "tgt_node": tgt,
        "coherence_hash": coherence_hash,
    }
    out_path = OUTPUT_DIR / f"baseline_{coherence_hash}_{src}-{tgt}.pkl"
    pd.to_pickle(result, out_path)
    shown = out_path.relative_to(PROJECT_ROOT).as_posix() if out_path.is_relative_to(PROJECT_ROOT) else out_path
    print(f"\nResultado guardado en {shown}")


if __name__ == "__main__":
    main()
