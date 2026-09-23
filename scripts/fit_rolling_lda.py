"""
fit_rolling_lda.py  (antes causal_coherence.py)

Semana 3: carga el corpus T17, lo filtra al subset "bpoil" (el caso de
estudio principal), lo trunca a la ventana de cobertura sostenida y
ajusta RollingLDA.
"""

from causal_coherence.data_loading import (
    SOURCE_LABEL,
    TOPIC,
    build_dataframe,
    date_range,
    load_filtered,
)
from causal_coherence.preprocessing import (
    build_english_pipeline,
    get_english_stopwords,
    preprocess_batch,
)
from causal_coherence.topic_model import (
    MODEL_PATH,
    TRUNCATE_AFTER,
    chunk_summary,
    fit_topic_model,
    truncate_to_coverage_window,
)


def print_chunk_summary(roll, df) -> None:
    """Imprime cuántos documentos tiene cada chunk y su rango de fechas real."""
    print("\nResumen real de chunks:")
    for i, row in chunk_summary(roll, df).iterrows():
        print(f"  Chunk {i}: {row['documentos']} documentos, de {row['desde']} a {row['hasta']}")


def main():
    docs_bpoil, _ = load_filtered(SOURCE_LABEL)
    start, end = date_range(docs_bpoil)
    print(f"Documentos en '{TOPIC}': {len(docs_bpoil)} ({start} a {end})")

    df = build_dataframe(docs_bpoil)
    before = len(df)
    df = truncate_to_coverage_window(df)
    dropped = before - len(df)
    print(
        f"Truncamiento a {TRUNCATE_AFTER.date()}: se dejan fuera "
        f"{dropped} de {before} documentos ({dropped / before:.1%})."
    )

    print("Preprocesando texto (puede demorar un minuto)...")
    nlp = build_english_pipeline()
    extra_stopwords = get_english_stopwords()
    df["preprocessed_text"] = preprocess_batch(
        df["content"].tolist(), nlp, extra_stopwords
    )

    print("\nAjustando RollingLDA (esto puede tardar varios minutos)...")
    roll = fit_topic_model(df)
    print_chunk_summary(roll, df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    roll.save(str(MODEL_PATH))
    print(f"Modelo guardado en {MODEL_PATH}")

    print("\nPalabras principales por tópico (interpretabilidad):")
    print(roll.top_words(number=10))


if __name__ == "__main__":
    main()
