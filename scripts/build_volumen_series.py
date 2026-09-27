"""
build_volumen_series.py

Semana 5 (corrección): construye la serie "original" con la
representación de volumen (ecuación 4 del formulario), que es la
decisión de diseño ya fijada en la sección 7 de la guía -- no la razón
logarítmica que se había usado antes.

x_{t,k} = log(1 + masa acumulada del tópico k en la ventana), para los
10 tópicos completos (no hace falta descartar ninguno como
referencia, a diferencia de la razón logarítmica).

Además construye el control de volumen de publicación,
log(1 + n_t), como serie aparte -- nunca como una serie más del
sistema, según exige la guía.
"""

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import CORPUS_END, CORPUS_START, drop_windows_below_min_docs
from causal_coherence.topic_model import load_topic_model, training_documents

TOPIC_SERIES_PATH = OUTPUT_DIR / "topic_series_volumen.csv"
CONTROL_SERIES_PATH = OUTPUT_DIR / "control_volumen.csv"


def get_theta(roll) -> np.ndarray:
    """theta normalizada por fila (cada documento suma 1)."""
    theta_raw = roll.get_document_topic_matrix()
    row_sums = theta_raw.sum(axis=1, keepdims=True)
    return theta_raw / row_sums


def drop_incomplete_edge_windows(weekly: pd.DataFrame) -> pd.DataFrame:
    """Misma regla que en la razón logarítmica: descarta ventanas de borde incompletas."""
    window_start = weekly.index - pd.Timedelta(days=6)
    window_end = weekly.index
    completa = (window_start >= CORPUS_START) & (window_end <= CORPUS_END)
    descartadas = weekly.index[~completa]
    for fecha in descartadas:
        print(f"  Se descarta la ventana que cierra {fecha.date()} (borde incompleto).")
    return weekly.loc[completa]


def main():
    roll = load_topic_model()
    df = training_documents()
    theta = get_theta(roll)

    K = theta.shape[1]
    theta_df = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(K)])
    theta_df["date"] = df["date"].values

    # Masa acumulada por tópico y por semana (suma, no promedio -- a
    # diferencia de la razón logarítmica, aquí sí importa el volumen).
    masa_semanal = theta_df.set_index("date").resample("W").sum()
    n_docs = theta_df.set_index("date").resample("W").size()

    print(f"Ventanas semanales antes de recortar bordes: {len(masa_semanal)}")
    masa_semanal = drop_incomplete_edge_windows(masa_semanal)
    n_docs = n_docs.loc[masa_semanal.index]
    print(f"Ventanas semanales tras recortar bordes: {len(masa_semanal)}")
    masa_semanal, n_docs = drop_windows_below_min_docs(masa_semanal, n_docs)
    print(f"Ventanas semanales tras el mínimo de documentos: {len(masa_semanal)} (T)")

    serie = np.log1p(masa_semanal)
    control = np.log1p(n_docs).rename("control_volumen")

    print(f"\nSerie 'original' (volumen, ecuación 4), los 10 tópicos:")
    print(f"  Forma: {serie.shape[0]} ventanas x {serie.shape[1]} series")
    print(serie.round(3).to_string())

    print(f"\nControl de volumen (log(1+n_t)):")
    print(control.round(3).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serie.to_csv(TOPIC_SERIES_PATH)
    control.to_csv(CONTROL_SERIES_PATH)
    print(f"\nGuardadas en:\n  {TOPIC_SERIES_PATH}\n  {CONTROL_SERIES_PATH}")


if __name__ == "__main__":
    main()
