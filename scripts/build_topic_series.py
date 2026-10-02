"""
build_topic_series.py

Semana 5, pasos 1-2: extraer theta del modelo ya reajustado, alinearla
con las fechas, y agregarla a nivel semanal (s_{t,k} = promedio de
theta del tópico k entre los documentos de esa semana).

Todavía NO calcula la razón logarítmica -- antes hay que elegir el
tópico de referencia mirando estos números reales (paso 3), y
verificar que las semanas de los bordes no queden incompletas otra vez
(paso 5). Este script es el diagnóstico para tomar esas dos decisiones,
no el cálculo final.
"""

import numpy as np
import pandas as pd
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence.topic_model import load_topic_model, training_documents


def get_theta(roll: RollingLDA) -> np.ndarray:
    """theta normalizada por fila (cada documento suma 1)."""
    theta_raw = roll.get_document_topic_matrix()
    row_sums = theta_raw.sum(axis=1, keepdims=True)
    return theta_raw / row_sums


def main() -> None:
    roll = load_topic_model()
    df = training_documents()

    theta = get_theta(roll)
    assert theta.shape[0] == len(df), (
        f"theta tiene {theta.shape[0]} filas pero training_documents() "
        f"tiene {len(df)} -- no calzan, algo cambió entre el ajuste "
        f"guardado y el corpus actual."
    )

    K = theta.shape[1]
    theta_df = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(K)])
    theta_df["date"] = df["date"].values

    # s_{t,k}: promedio de theta por tópico, agregado por semana
    weekly = theta_df.set_index("date").resample("W").mean()
    n_docs = theta_df.set_index("date").resample("W").size()

    print(f"Ventanas semanales: {len(weekly)}")
    print(f"Rango de fechas de los documentos: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Primera ventana: {weekly.index[0].date()} (día de cierre semanal)")
    print(f"Última ventana:  {weekly.index[-1].date()} (día de cierre semanal)")

    print("\nDocumentos por ventana (para revisar si los bordes quedan incompletos):")
    for fecha, n in n_docs.items():
        print(f"  semana que cierra {fecha.date()}: {n} documentos")

    print("\nMínimo y promedio de s_{t,k} por tópico, a lo largo de las semanas")
    print("(el tópico de referencia debería tener el mínimo más alto -- el que")
    print("nunca casi desaparece en ninguna semana):")
    resumen = pd.DataFrame({
        "min": weekly.min(),
        "promedio": weekly.mean(),
        "semanas_bajo_0.01": (weekly < 0.01).sum(),
    })
    print(resumen.sort_values("min", ascending=False).to_string())


if __name__ == "__main__":
    main()
