"""
build_log_ratio_series.py

Semana 5: construye la serie "original" (ecuación 5 del formulario,
razón logarítmica) sobre la ventana de análisis, con:
  - grilla semanal, T=23 (se excluye la última ventana, que cierra el
    2010-10-03 y por lo tanto solo cubre 4 días reales del corpus,
    no los 7 completos -- ver bitácora),
  - topic_3 como tópico de referencia (mínimo de s_{t,k} más alto de
    los 10, 0.068, y temáticamente estable: "rig, safety, transocean,
    bp, drilling, offshore, hearing" -- causas del desastre, presente
    de fondo durante toda la cobertura).

Todavía es la serie "original" nada más -- las otras 3 versiones
(detendenciada, diferenciada, residualizada de la componente común)
son un paso aparte.
"""

import numpy as np
import pandas as pd
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import CORPUS_END, CORPUS_START
from causal_coherence.topic_model import load_topic_model, training_documents

REFERENCE_TOPIC = 3
OUTPUT_PATH = OUTPUT_DIR / "topic_series_log_ratio.csv"


def get_theta(roll: RollingLDA) -> np.ndarray:
    """theta normalizada por fila (cada documento suma 1)."""
    theta_raw = roll.get_document_topic_matrix()
    row_sums = theta_raw.sum(axis=1, keepdims=True)
    return theta_raw / row_sums


def weekly_topic_shares(theta: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """s_{t,k}: promedio de theta por tópico, agregado por semana."""
    K = theta.shape[1]
    theta_df = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(K)])
    theta_df["date"] = dates.values
    return theta_df.set_index("date").resample("W").mean()


def drop_incomplete_edge_windows(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Descarta ventanas cuyo rango de 7 días (label - 6 días, label) se
    sale de [CORPUS_START, CORPUS_END] -- son ventanas de borde que no
    cubren una semana completa de corpus real, no dato faltante.
    """
    window_start = weekly.index - pd.Timedelta(days=6)
    window_end = weekly.index
    completa = (window_start >= CORPUS_START) & (window_end <= CORPUS_END)
    descartadas = weekly.index[~completa]
    for fecha in descartadas:
        print(f"  Se descarta la ventana que cierra {fecha.date()} (borde incompleto).")
    return weekly.loc[completa]


def log_ratio(weekly: pd.DataFrame, reference: int) -> pd.DataFrame:
    ref_col = f"topic_{reference}"
    ratios = {}
    for col in weekly.columns:
        if col == ref_col:
            continue
        ratios[col] = np.log(weekly[col] / weekly[ref_col])
    return pd.DataFrame(ratios, index=weekly.index)


def main() -> None:
    roll = load_topic_model()
    df = training_documents()
    theta = get_theta(roll)

    weekly = weekly_topic_shares(theta, df["date"])
    print(f"Ventanas semanales antes de recortar bordes: {len(weekly)}")

    weekly = drop_incomplete_edge_windows(weekly)
    print(f"Ventanas semanales tras recortar bordes: {len(weekly)} (T)")

    series = log_ratio(weekly, REFERENCE_TOPIC)
    print(f"\nSerie 'original' (razón logarítmica, referencia=topic_{REFERENCE_TOPIC}):")
    print(f"  Forma: {series.shape[0]} ventanas x {series.shape[1]} series")
    print(series.round(3).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(OUTPUT_PATH)
    print(f"\nGuardada en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
