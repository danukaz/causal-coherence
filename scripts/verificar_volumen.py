"""
verificar_volumen.py

Chequeo independiente, a mano, de UNA celda de topic_series_volumen.csv
y UNA celda de control_volumen.csv -- calculado de una forma distinta
a build_volumen_series.py (filtrando por fecha directamente, sin usar
resample), para confirmar que no hay un error que se repita igual por
las dos vías.

Elige tú mismo la semana y el tópico a verificar cambiando las dos
variables de abajo.
"""

import numpy as np
import pandas as pd

from causal_coherence.topic_model import load_topic_model, training_documents

# --- Elige qué celda verificar ---
SEMANA_INICIO = "2010-05-02"  # incluida
SEMANA_FIN = "2010-05-02"     # la misma fecha, porque resample("W")
                              # etiqueta la semana por su ÚLTIMO día
TOPICO = 5


def main():
    roll = load_topic_model()
    df = training_documents()

    theta_raw = roll.get_document_topic_matrix()
    theta = theta_raw / theta_raw.sum(axis=1, keepdims=True)

    # Filtro manual por fecha, siete días hacia atrás desde SEMANA_FIN
    fin = pd.Timestamp(SEMANA_FIN)
    inicio = fin - pd.Timedelta(days=6)
    mask = (df["date"] >= inicio) & (df["date"] <= fin)
    indices = df.index[mask].to_numpy()

    print(f"Ventana verificada a mano: {inicio.date()} a {fin.date()}")
    print(f"Documentos encontrados en esa ventana: {len(indices)}")

    masa_topico = theta[indices, TOPICO].sum()
    valor_esperado = np.log1p(masa_topico)
    print(f"\nMasa acumulada del tópico {TOPICO} (suma de theta): {masa_topico:.4f}")
    print(f"log(1 + masa) esperado para topic_{TOPICO} esa semana: {valor_esperado:.4f}")
    print("-> Compara esto con la celda real de topic_series_volumen.csv,")
    print(f"   fila {SEMANA_FIN}, columna topic_{TOPICO}.")

    n_docs = len(indices)
    control_esperado = np.log1p(n_docs)
    print(f"\nDocumentos en la ventana (n_t): {n_docs}")
    print(f"log(1 + n_t) esperado: {control_esperado:.4f}")
    print("-> Compara esto con la celda real de control_volumen.csv,")
    print(f"   fila {SEMANA_FIN}.")


if __name__ == "__main__":
    main()