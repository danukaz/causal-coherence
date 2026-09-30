"""
build_series_variants.py

Semana 5: construye las otras 3 versiones de la serie de volumen (además
de la "original" guardada en data/topic_series_volumen.csv), tal como
exige la guía para un corpus organizado alrededor de un único evento:
  - diferenciada: x_t - x_{t-1}
  - detendenciada: se le resta una tendencia lineal en el tiempo a cada
    serie por separado
  - residualizada: se le resta la componente principal común (primer
    componente de un PCA sobre las series de tópicos), la aproximación a
    "el ciclo de atención común" del evento

El control de volumen (data/control_volumen.csv, log(1+n_t)) se empareja
con cada versión así:
  - diferenciada: el control también se diferencia.
  - detendenciada: el control también se detendencia, con el mismo
    procedimiento que las series de tópicos.
  - residualizada: el control NO se residualiza; se usa el original. Es
    una variable exógena, no uno de los tópicos que comparte la
    componente principal.

La residualizada es la que requiere una revisión conjunta: el script
imprime cuánta varianza explica esa primera componente antes de restar
nada, para decidir si el supuesto de "hay una sola componente común
dominante" se sostiene con los datos reales, no solo se asume.

Las versiones de la razón logarítmica (topic_series_diferenciada.csv,
etc.) quedan obsoletas y este script ya no las regenera.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from causal_coherence.config import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "topic_series_volumen.csv"
CONTROL_PATH = OUTPUT_DIR / "control_volumen.csv"

OUTPUTS = {
    "diferenciada": OUTPUT_DIR / "topic_series_volumen_diferenciada.csv",
    "detendenciada": OUTPUT_DIR / "topic_series_volumen_detendenciada.csv",
    "residualizada": OUTPUT_DIR / "topic_series_volumen_residualizada.csv",
}
# Columna que se descarta al guardar la residualizada. Restarle a las K series
# su primera componente principal deja residuos en un subespacio de K-1
# dimensiones (el residuo es ortogonal a esa componente), así que las K
# columnas son exactamente colineales y la autorregresión sería singular
# (guía, ecuación 6). El PCA se calcula igual sobre las K columnas; solo se
# descarta una al escribir el CSV. Cuál no importa matemáticamente; se elige
# topic_5 (percent, reuters, comment, ultra, compliance...), que es texto de
# plantilla sin contenido narrativo, para no perder en la residualizada los
# pares de un tópico sustantivo. Los 18 pares ordenados que involucran a
# topic_5 solo se evalúan en las otras 3 versiones.
RESIDUALIZED_DROP_COLUMN = "topic_5"

CONTROL_OUTPUTS = {
    "diferenciado": OUTPUT_DIR / "control_volumen_diferenciado.csv",
    "detendenciado": OUTPUT_DIR / "control_volumen_detendenciado.csv",
}


def load_original_series() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    return df


def load_control() -> pd.DataFrame:
    return pd.read_csv(CONTROL_PATH, index_col=0, parse_dates=True)


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
    X = scaler.fit_transform(series.to_numpy())  # (T, K), cada columna media 0 var 1

    pca = PCA()
    scores = pca.fit_transform(X)  # (T, K), scores[:, 0] = componente 1
    varianza_explicada = pca.explained_variance_ratio_

    pc1 = scores[:, [0]]  # (T, 1)
    loadings1 = pca.components_[[0], :]  # (1, K)
    X_reconstruido_pc1 = pc1 @ loadings1  # lo que "explica" la componente 1
    X_residual = X - X_reconstruido_pc1

    # Devolver en la escala original de cada serie (deshacer el estandarizado),
    # para que las unidades sigan siendo las de la serie, no z-scores.
    residual = X_residual * scaler.scale_
    residualizada = pd.DataFrame(residual, index=series.index, columns=series.columns)
    return residualizada, varianza_explicada


def main():
    original = load_original_series()
    control = load_control()
    print(f"Serie original cargada: {original.shape[0]} ventanas x {original.shape[1]} series")
    print(f"Control de volumen cargado: {control.shape[0]} ventanas")
    assert control.index.equals(original.index), "el control y la serie no tienen las mismas ventanas"

    diferenciada = build_differenced(original)
    diferenciada.to_csv(OUTPUTS["diferenciada"])
    control_dif = build_differenced(control)
    control_dif.to_csv(CONTROL_OUTPUTS["diferenciado"])
    print(f"\nDiferenciada: {diferenciada.shape[0]} ventanas (se pierde 1 por la resta); "
          f"control diferenciado en paralelo: {control_dif.shape[0]} ventanas")

    detendenciada = build_detrended(original)
    detendenciada.to_csv(OUTPUTS["detendenciada"])
    control_det = build_detrended(control)
    control_det.to_csv(CONTROL_OUTPUTS["detendenciado"])
    print(f"Detendenciada: {detendenciada.shape[0]} ventanas (misma cantidad, se resta la tendencia); "
          f"control detendenciado en paralelo: {control_det.shape[0]} ventanas")

    residualizada, varianza_explicada = build_residualized(original)
    print(f"\nPCA sobre las {original.shape[1]} series (estandarizadas) -- varianza explicada por componente:")
    for i, v in enumerate(varianza_explicada):
        marca = "  <-- componente 1 (la que se resta)" if i == 0 else ""
        print(f"  Componente {i + 1}: {v:.1%}{marca}")

    residualizada = residualizada.drop(columns=RESIDUALIZED_DROP_COLUMN)
    residualizada.to_csv(OUTPUTS["residualizada"])
    print(f"\nResidualizada: {residualizada.shape[0]} ventanas x {residualizada.shape[1]} series "
          f"(PCA sobre las {original.shape[1]}; se descarta {RESIDUALIZED_DROP_COLUMN} al guardar), "
          f"pendiente de confirmar si tiene sentido restar solo 1 componente. "
          f"Control: el original, sin residualizar ({CONTROL_PATH.name}).")

    print("\nGuardadas en:")
    for path in (*OUTPUTS.values(), *CONTROL_OUTPUTS.values()):
        print(f"  {path}")
    print(f"Control sin cambios para la original y la residualizada: {CONTROL_PATH}")


if __name__ == "__main__":
    main()
