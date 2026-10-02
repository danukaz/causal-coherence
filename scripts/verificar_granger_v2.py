"""
verificar_granger_v2.py

Extiende verificar_granger.py: prueba el mismo par (topic_3 -> topic_8)
en las 4 versiones (original, diferenciada, detendenciada y
residualizada; la columna que se descarta en la residualizada es
topic_5, no topic_3, así que este par sí se evalúa ahí), y compara las dos
formas del control de volumen que el profesor guía autorizó explorar:
  - compartido: log(1+n_t), igual para las 10 ecuaciones (lo que ya
    tenías)
  - propio: log(1 + n_t - masa_cruda_del_tópico_objetivo), un control
    distinto según cuál tópico se esté prediciendo
"""

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR

ORIGEN = "topic_3"
DESTINO = "topic_8"
P = 1

VERSIONES = {
    "original": ("topic_series_volumen.csv", "control_volumen.csv"),
    "diferenciada": ("topic_series_volumen_diferenciada.csv", "control_volumen_diferenciado.csv"),
    "detendenciada": ("topic_series_volumen_detendenciada.csv", "control_volumen_detendenciado.csv"),
    "residualizada": ("topic_series_volumen_residualizada.csv", "control_volumen.csv"),
}


def cargar(nombre_serie: str, nombre_control: str) -> tuple[pd.DataFrame, pd.Series]:
    series = pd.read_csv(OUTPUT_DIR / nombre_serie, index_col=0, parse_dates=True)
    control = pd.read_csv(OUTPUT_DIR / nombre_control, index_col=0, parse_dates=True).iloc[:, 0]
    assert (series.index == control.index).all()
    return series, control


def control_propio(nombre_serie_original: str = "topic_series_volumen.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    Reconstruye, para el control 'propio', la masa cruda de cada
    tópico (deshaciendo el log1p ya aplicado) y el n_t total, para
    poder calcular log(1 + n_t - masa_del_topico_objetivo) por tópico.
    """
    serie_log = pd.read_csv(OUTPUT_DIR / nombre_serie_original, index_col=0, parse_dates=True)
    masa_cruda = np.expm1(serie_log)  # deshace el log1p
    n_t = np.expm1(pd.read_csv(OUTPUT_DIR / "control_volumen.csv", index_col=0, parse_dates=True).iloc[:, 0])
    return masa_cruda, n_t


def armar_diseno(series: pd.DataFrame, control_series: pd.Series, excluir: str | None) -> tuple[np.ndarray, np.ndarray]:
    topicos = [c for c in series.columns if c != excluir]
    n = len(series)
    X_rezagado = series[topicos].iloc[:-1].to_numpy()
    control_contemporaneo = control_series.iloc[1:].to_numpy().reshape(-1, 1)
    intercepto = np.ones((n - 1, 1))
    X = np.hstack([X_rezagado, control_contemporaneo, intercepto])
    y = series[DESTINO].iloc[1:].to_numpy()
    return X, y


def ajustar(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = np.sum((y - X @ coef) ** 2)
    return rss, X.shape[1]


def probar(series: pd.DataFrame, control_series: pd.Series, etiqueta: str) -> None:
    X_f, y_f = armar_diseno(series, control_series, excluir=None)
    rss_f, params_f = ajustar(X_f, y_f)
    X_r, y_r = armar_diseno(series, control_series, excluir=ORIGEN)
    rss_r, params_r = ajustar(X_r, y_r)

    n_obs = X_f.shape[0]
    tamano_efecto = np.log(rss_r / rss_f)
    df_num = params_f - params_r
    df_den = n_obs - params_f
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)

    print(f"  [{etiqueta}] n={n_obs}  efecto={tamano_efecto:.4f}  F={F:.4f}  "
          f"gl=({df_num},{df_den})")


def main() -> None:
    print(f"Par verificado: {ORIGEN} -> {DESTINO}, p={P}\n")

    print("=== Control COMPARTIDO (log(1+n_t), igual para las 10 ecuaciones) ===")
    for nombre, (archivo_serie, archivo_control) in VERSIONES.items():
        series, control = cargar(archivo_serie, archivo_control)
        probar(series, control, nombre)

    print("\n=== Control PROPIO (excluye la masa del tópico objetivo) ===")
    masa_cruda, n_t = control_propio()
    for nombre, (archivo_serie, _) in VERSIONES.items():
        series = pd.read_csv(OUTPUT_DIR / archivo_serie, index_col=0, parse_dates=True)
        # El control "propio" se arma sobre la serie ORIGINAL de masa cruda,
        # sin importar qué versión (diferenciada/detendenciada) se esté
        # probando -- es una simplificación a discutir, ver nota más abajo.
        control_propio_destino = np.log1p(n_t - masa_cruda[DESTINO])
        control_propio_destino = control_propio_destino.loc[series.index]
        probar(series, control_propio_destino, nombre)

    print("\nNota: para diferenciada/detendenciada, el control 'propio' de arriba")
    print("NO se transformó en paralelo (a diferencia del control compartido, que")
    print("sí tiene sus versiones _diferenciado/_detendenciado ya construidas).")
    print("Esto es una simplificación para esta verificación rápida, no la versión")
    print("final -- revisar si hace falta construir también las versiones")
    print("diferenciada/detendenciada del control propio antes de escalar.")


if __name__ == "__main__":
    main()