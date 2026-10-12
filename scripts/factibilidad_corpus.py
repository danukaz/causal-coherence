"""
factibilidad_corpus.py

Semana 7, B1 (Bitacora_Semana7.md, sección 12): factibilidad de la serie
temporal desde las fechas de los documentos, sin ajustar RollingLDA.

Candidatos: los 9 subconjuntos de T17 (bpoil y los 8 restantes) y los 3 del
corpus de Afganistán. Grillas con el resample de pandas de
check_grid_feasibility.py: semanal ("W", cierra en domingo), cada 3 días
("3D", anclada en el primer documento), diaria ("D") y quincenal ("2W",
solo como referencia). Rango: del primer al último documento de cada
subconjunto; para bpoil, además, la ventana de análisis fijada
(CORPUS_START a CORPUS_END).

Por combinación: documentos, días entre el primer y el último documento,
ventanas, vacías y válidas (>= MIN_DOCS_PER_WINDOW documentos) antes de
recortar bordes; tras excluir las ventanas de los extremos que no cubren el
intervalo completo (R1): ventanas, válidas, mayor racha de válidas
consecutivas, observaciones utilizables con retardo 1 (t y t-1 válidas) y
mediana de documentos por ventana válida.

Verificación: bpoil en la ventana de análisis, antes de recortar, da 24
ventanas semanales con 0 vacías, 55 de 3 días con 2 vacías y 164 diarias
con 16 vacías (check_grid_feasibility.py).

Resultado: data/factibilidad_corpus.csv.
"""

import numpy as np
import pandas as pd

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import (
    AFGHANISTAN_DIR,
    CORPUS_END,
    CORPUS_START,
    MIN_DOCS_PER_WINDOW,
    SOURCE_LABEL,
    T17_DIR,
    load_full_corpus,
)

OUTPUT_PATH = OUTPUT_DIR / "factibilidad_corpus.csv"
# grilla: (frecuencia de pandas, días por ventana, la etiqueta es el extremo derecho)
GRILLAS = {"semanal": ("W", 7, True), "cada 3 días": ("3D", 3, False), "diaria": ("D", 1, False),
           "quincenal (referencia)": ("2W", 14, True)}
VERIFICACION = {"semanal": (24, 0), "cada 3 días": (55, 2), "diaria": (164, 16)}


def fechas_por_subconjunto() -> dict[tuple[str, str], pd.Series]:
    """Fechas (normalizadas al día) por (corpus, etiqueta de source)."""
    salida = {}
    for corpus, carpeta in (("T17", T17_DIR), ("Afganistán", AFGHANISTAN_DIR)):
        docs = load_full_corpus(carpeta)
        tabla = pd.DataFrame({"source": [d["source"] for d in docs],
                              "date": pd.to_datetime([d["date"] for d in docs]).normalize()})
        for fuente, g in tabla.groupby("source"):
            salida[(corpus, fuente)] = g["date"].sort_values().reset_index(drop=True)
    return salida


def mayor_racha(validas: np.ndarray) -> int:
    racha = mejor = 0
    for v in validas:
        racha = racha + 1 if v else 0
        mejor = max(mejor, racha)
    return mejor


def medir(fechas: pd.Series, inicio: pd.Timestamp, fin: pd.Timestamp, grilla: str) -> dict[str, object]:
    freq, dias, etiqueta_derecha = GRILLAS[grilla]
    conteo = pd.Series(1, index=fechas).resample(freq).size()
    validas = (conteo >= MIN_DOCS_PER_WINDOW).to_numpy()
    if etiqueta_derecha:
        desde, hasta = conteo.index - pd.Timedelta(days=dias - 1), conteo.index
    else:
        desde, hasta = conteo.index, conteo.index + pd.Timedelta(days=dias - 1)
    completa = np.asarray((desde >= inicio) & (hasta <= fin))
    c_rec = conteo[completa]
    v_rec = validas[completa]
    return {
        "documentos": len(fechas), "primer_documento": fechas.min().date(), "ultimo_documento": fechas.max().date(),
        "dias": (fechas.max() - fechas.min()).days,
        "ventanas_antes": len(conteo), "vacias_antes": int((conteo == 0).sum()), "validas_antes": int(validas.sum()),
        "ventanas_despues": len(c_rec), "validas_despues": int(v_rec.sum()),
        "racha_max_validas": mayor_racha(v_rec),
        "obs_retardo_1": int(np.sum(v_rec[1:] & v_rec[:-1])),
        "mediana_docs_validas": float(c_rec[v_rec].median()) if v_rec.any() else np.nan,
    }


def main() -> None:
    filas = []
    for (corpus, fuente), fechas in fechas_por_subconjunto().items():
        rangos = [("completo", fechas.min(), fechas.max(), fechas)]
        if fuente == SOURCE_LABEL:
            en_ventana = fechas[(fechas >= CORPUS_START) & (fechas <= CORPUS_END)].reset_index(drop=True)
            rangos.append(("ventana de análisis", CORPUS_START, CORPUS_END, en_ventana))
        for rango, inicio, fin, f in rangos:
            for grilla in GRILLAS:
                filas.append({"corpus": corpus, "subconjunto": fuente, "rango": rango, "grilla": grilla,
                              **medir(f, inicio, fin, grilla)})
    tabla = pd.DataFrame(filas)

    print("Verificación (bpoil, ventana de análisis, antes de recortar bordes):")
    ok = True
    for grilla, (n_esp, vacias_esp) in VERIFICACION.items():
        fila = tabla[(tabla["subconjunto"] == SOURCE_LABEL) & (tabla["rango"] == "ventana de análisis")
                     & (tabla["grilla"] == grilla)].iloc[0]
        cumple = fila["ventanas_antes"] == n_esp and fila["vacias_antes"] == vacias_esp
        ok &= cumple
        print(f"  {grilla}: {fila['ventanas_antes']} ventanas, {fila['vacias_antes']} vacías "
              f"(esperado {n_esp} y {vacias_esp}): {'OK' if cumple else 'FALLA'}")
    print(f"Verificación: {'PASA' if ok else 'FALLA'}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUT_PATH, index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.max_rows", 200):
        print(tabla.to_string(index=False))
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
