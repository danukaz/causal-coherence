"""
build_series_variants.py

Semana 5: construye las otras 3 versiones de la serie (además de la
"original" ya guardada en data/topic_series_log_ratio.csv), tal como
exige la guía para un corpus organizado alrededor de un único evento:
  - diferenciada: x_t - x_{t-1}
  - detendenciada: se le resta una tendencia lineal en el tiempo a cada
    serie por separado
  - residualizada: se le resta la componente principal común (primer
    componente de un PCA sobre las 9 series), la aproximación a "el
    ciclo de atención común" del evento

La residualizada es la que requiere una revisión conjunta: el script
imprime cuánta varianza explica esa primera componente antes de restar
nada, para decidir si el supuesto de "hay una sola componente común
dominante" se sostiene con los datos reales, no solo se asume.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from causal_coherence.config import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "topic_series_log_ratio.csv"


def load_original_series() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    return df


def build_differenced(series: pd.DataFrame) -> pd.DataFrame:
    return series.diff().dropna()


def build_detrended(series: pd.DataFrame) -> pd.DataFrame:
    """Resta una tendencia lineal en el tiempo a cada serie por separado."""
    t = np.arange(len(series))
    detrended = pd.DataFrame(index=series.index, columns=series.columns, dtype=float)
    for col in series.columns:
        y = series[col].to_numpy()
        b, a = np.polyfit(t, y, deg=1)  # y = a + b*t
        fitted = a + b * t
        detrended[col] = y - fitted
    return detrended


def build_residualized(series: pd.DataFrame):
    """
    PCA sobre las series estandarizadas. Devuelve (residualizada,
    varianza_explicada) para poder revisar el supuesto antes de usarla.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(series.to_numpy())  # (T, 9), cada columna media 0 var 1

    pca = PCA()
    scores = pca.fit_transform(X)  # (T, 9), scores[:, 0] = componente 1
    varianza_explicada = pca.explained_variance_ratio_

    pc1 = scores[:, [0]]  # (T, 1)
    loadings1 = pca.components_[[0], :]  # (1, 9)
    X_reconstruido_pc1 = pc1 @ loadings1  # lo que "explica" la componente 1
    X_residual = X - X_reconstruido_pc1

    # Devolver en la escala original de cada serie (deshacer el estandarizado),
    # para que las unidades sigan siendo log-razón, no z-scores.
    residual = X_residual * scaler.scale_
    residualizada = pd.DataFrame(residual, index=series.index, columns=series.columns)
    return residualizada, varianza_explicada


def main():
    original = load_original_series()
    print(f"Serie original cargada: {original.shape[0]} ventanas x {original.shape[1]} series")

    diferenciada = build_differenced(original)
    diferenciada.to_csv(OUTPUT_DIR / "topic_series_diferenciada.csv")
    print(f"\nDiferenciada: {diferenciada.shape[0]} ventanas (se pierde 1 por la resta)")

    detendenciada = build_detrended(original)
    detendenciada.to_csv(OUTPUT_DIR / "topic_series_detendenciada.csv")
    print(f"Detendenciada: {detendenciada.shape[0]} ventanas (misma cantidad, se resta la tendencia)")

    residualizada, varianza_explicada = build_residualized(original)
    print("\nPCA sobre las 9 series (estandarizadas) -- varianza explicada por componente:")
    for i, v in enumerate(varianza_explicada):
        marca = "  <-- componente 1 (la que se resta)" if i == 0 else ""
        print(f"  Componente {i + 1}: {v:.1%}{marca}")

    residualizada.to_csv(OUTPUT_DIR / "topic_series_residualizada.csv")
    print(f"\nResidualizada: {residualizada.shape[0]} ventanas, guardada igual, "
          f"pendiente de confirmar si tiene sentido restar solo 1 componente.")

    print("\nGuardadas en:")
    print(f"  {OUTPUT_DIR / 'topic_series_diferenciada.csv'}")
    print(f"  {OUTPUT_DIR / 'topic_series_detendenciada.csv'}")
    print(f"  {OUTPUT_DIR / 'topic_series_residualizada.csv'}")


if __name__ == "__main__":
    main()
