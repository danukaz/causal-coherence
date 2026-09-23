"""
find_chunks.py

Explora distintos umbrales de razón tipo-token acumulada para decidir
los cortes de chunk de RollingLDA de forma reproducible, en vez de ir
fusionando meses a mano por prueba y error.

Reutiliza las funciones del paquete causal_coherence -- no las reescribe.
"""

import pandas as pd

from causal_coherence.data_loading import SOURCE_LABEL, build_dataframe, load_filtered
from causal_coherence.preprocessing import (
    build_english_pipeline,
    get_english_stopwords,
    preprocess_batch,
)


def cumulative_ttr_chunks(df, ttr_threshold: float):
    """
    Acumula documentos mes a mes y cierra un chunk apenas la razón
    tipo-token de la ventana acumulada (desde el último corte) cae por
    debajo de ttr_threshold. Imprime cada ventana evaluada, para que
    la decisión sea auditable y no una caja negra.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    months = sorted(df_sorted["date"].dt.to_period("M").unique())

    cuts = []
    window_tokens: list[str] = []
    window_start = None
    ratio = None

    for month in months:
        month_docs = df_sorted[df_sorted["date"].dt.to_period("M") == month]
        if window_start is None:
            window_start = month
        window_tokens.extend(
            t for tokens in month_docs["preprocessed_text"] for t in tokens
        )
        ratio = len(set(window_tokens)) / len(window_tokens)
        print(
            f"    ventana {window_start}-{month}: "
            f"{len(window_tokens)} tokens, razón={ratio:.3f}"
        )

        if ratio <= ttr_threshold:
            # +1 día: el corte marca "el día después del último
            # documento observado", no la fecha del documento mismo.
            # Usar la fecha exacta deja ese día huérfano como su propio
            # chunk de un solo día -- es justo lo que le pasó a Daniel.
            cuts.append(month_docs["date"].max() + pd.Timedelta(days=1))
            window_tokens = []
            window_start = None

    if window_tokens:
        print(
            f"    [cola sin cerrar: {window_start}-{months[-1]}, "
            f"razón final={ratio:.3f} -- candidata a truncar]"
        )

    return cuts


def main():
    docs_bpoil, _ = load_filtered(SOURCE_LABEL)
    df = build_dataframe(docs_bpoil)

    print("Preprocesando texto...")
    nlp = build_english_pipeline()
    extra_stopwords = get_english_stopwords()
    df["preprocessed_text"] = preprocess_batch(
        df["content"].tolist(), nlp, extra_stopwords
    )

    for threshold in (0.10, 0.12, 0.15, 0.18):
        print(f"\n=== Umbral razón tipo-token <= {threshold} ===")
        cuts = cumulative_ttr_chunks(df, threshold)
        print("Cortes propuestos:", [c.date() for c in cuts])


if __name__ == "__main__":
    main()
