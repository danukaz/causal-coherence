"""
factibilidad_grillas.py

Semana 8, paso 1 (Bitacora_Semana8.md, sección 19): factibilidad de grillas
finas para bpoil en la ventana de análisis, con el corpus viejo
(Datasets/t17) y el nuevo (Datasets/t17_timeline17/t17). Solo fechas, sin
ajustar modelos.

Reutiliza medir y GRILLAS de scripts/factibilidad_corpus.py vía runpy, sin
modificarlo: las grillas 2D, 4D y 5D y el mínimo de documentos (5 principal,
3 sensibilidad) se fijan en el espacio de nombres de medir, en memoria. Las
métricas nuevas (racha y sus fechas, rachas de al menos 10 ventanas,
mediana dentro de la racha) se calculan sobre las mismas ventanas (mismo
resample y mismo recorte de bordes), y se comprueba que los conteos
coincidan con los de medir.

Regla de huecos: ventana con menos de MIN_DOCS documentos = dato faltante;
la serie de análisis es la mayor racha de ventanas válidas consecutivas,
tras recortar bordes; observaciones utilizables = longitud de la racha - 1.

Resultado: data/factibilidad_grillas.csv.
"""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from causal_coherence.config import DATASETS_DIR, OUTPUT_DIR
from causal_coherence.data_loading import CORPUS_END, CORPUS_START, SOURCE_LABEL, T17_DIR, load_full_corpus

OUTPUT_PATH = OUTPUT_DIR / "factibilidad_grillas.csv"
CORPUS = {"viejo": T17_DIR, "nuevo": DATASETS_DIR / "t17_timeline17" / "t17"}
# grilla: (frecuencia de pandas, días por ventana, la etiqueta es el extremo derecho)
GRILLAS_NUEVAS = {"cada 2 días": ("2D", 2, False), "cada 4 días": ("4D", 4, False), "cada 5 días": ("5D", 5, False)}
ORDEN = ["semanal", "cada 2 días", "cada 3 días", "cada 4 días", "cada 5 días"]
MIN_DOCS = (5, 3)
RACHA_LARGA = 10
MINIMO_CALCULADORA = {3: 30, 5: 45, 10: 90}

_fact = runpy.run_path(str(Path(__file__).resolve().parent / "factibilidad_corpus.py"))
medir = _fact["medir"]
GRILLAS: dict[str, tuple[str, int, bool]] = medir.__globals__["GRILLAS"]
GRILLAS.update(GRILLAS_NUEVAS)


def fechas_bpoil(carpeta: Path) -> pd.Series:
    docs = load_full_corpus(carpeta)
    fechas = pd.to_datetime([d["date"] for d in docs if d["source"] == SOURCE_LABEL]).normalize()
    fechas = pd.Series(fechas).sort_values()
    return fechas[(fechas >= CORPUS_START) & (fechas <= CORPUS_END)].reset_index(drop=True)


def ventanas_recortadas(fechas: pd.Series, grilla: str, min_docs: int) -> pd.DataFrame:
    """Ventanas completas (mismo resample y recorte que medir): inicio, fin, documentos y si es válida."""
    freq, dias, etiqueta_derecha = GRILLAS[grilla]
    conteo = pd.Series(1, index=fechas).resample(freq).size()
    if etiqueta_derecha:
        desde, hasta = conteo.index - pd.Timedelta(days=dias - 1), conteo.index
    else:
        desde, hasta = conteo.index, conteo.index + pd.Timedelta(days=dias - 1)
    completa = np.asarray((desde >= CORPUS_START) & (hasta <= CORPUS_END))
    return pd.DataFrame({"inicio": desde[completa], "fin": hasta[completa], "documentos": conteo.to_numpy()[completa],
                         "valida": conteo.to_numpy()[completa] >= min_docs})


def rachas(validas: np.ndarray) -> list[tuple[int, int]]:
    """(primera posición, largo) de cada racha de ventanas válidas consecutivas."""
    salida, inicio = [], None
    for i, v in enumerate(list(validas) + [False]):
        if v and inicio is None:
            inicio = i
        elif not v and inicio is not None:
            salida.append((inicio, i - inicio))
            inicio = None
    return salida


def fila(corpus: str, fechas: pd.Series, grilla: str, min_docs: int) -> dict[str, object]:
    medir.__globals__["MIN_DOCS_PER_WINDOW"] = min_docs
    base = medir(fechas, CORPUS_START, CORPUS_END, grilla)
    v = ventanas_recortadas(fechas, grilla, min_docs)
    todas = rachas(v["valida"].to_numpy())
    pos, largo = max(todas, key=lambda r: r[1]) if todas else (0, 0)  # empate: la primera
    tramo = v.iloc[pos:pos + largo]
    assert len(v) == base["ventanas_despues"] and int(v["valida"].sum()) == base["validas_despues"] \
        and largo == base["racha_max_validas"], f"{corpus} {grilla} {min_docs}: no coincide con medir"
    utilizables = max(largo - 1, 0)
    return {
        "corpus": corpus, "grilla": grilla, "min_docs": min_docs,
        "documentos": base["documentos"], "ventanas_antes": base["ventanas_antes"], "vacias_antes": base["vacias_antes"],
        "validas_antes": base["validas_antes"], "ventanas_despues": base["ventanas_despues"],
        "validas_despues": base["validas_despues"], "racha_max": largo,
        "racha_inicio": tramo["inicio"].iloc[0].date() if largo else None,
        "racha_fin": tramo["fin"].iloc[-1].date() if largo else None,
        "racha_dias": int((tramo["fin"].iloc[-1] - tramo["inicio"].iloc[0]).days) + 1 if largo else 0,
        "obs_utilizables": utilizables, "pares_consecutivos_validos": base["obs_retardo_1"],
        "rachas_de_10_o_mas": sum(1 for _, n in todas if n >= RACHA_LARGA),
        "mediana_docs_racha": float(tramo["documentos"].median()) if largo else np.nan,
        "K_que_alcanzan": ",".join(str(K) for K, minimo in MINIMO_CALCULADORA.items() if utilizables >= minimo) or "ninguno",
    }


def verificar(tabla: pd.DataFrame) -> bool:
    def f(corpus: str, grilla: str) -> pd.Series:
        return tabla[(tabla["corpus"] == corpus) & (tabla["grilla"] == grilla) & (tabla["min_docs"] == 5)].iloc[0]

    chequeos = [
        ("viejo semanal: 24 ventanas, 0 vacías, 23 válidas tras recortar",
         (f("viejo", "semanal")[["ventanas_antes", "vacias_antes", "validas_despues"]] == [24, 0, 23]).all()),
        ("viejo 3D: 55 ventanas, 2 vacías, 46 válidas antes, racha 36, 39 pares",
         (f("viejo", "cada 3 días")[["ventanas_antes", "vacias_antes", "validas_antes", "racha_max",
                                     "pares_consecutivos_validos"]] == [55, 2, 46, 36, 39]).all()),
        ("nuevo 3D: 55 ventanas, 1 vacía, 48 válidas, racha 41, 42 pares",
         (f("nuevo", "cada 3 días")[["ventanas_antes", "vacias_antes", "validas_antes", "racha_max",
                                     "pares_consecutivos_validos"]] == [55, 1, 48, 41, 42]).all()),
    ]
    print("Verificación:")
    for nombre, ok in chequeos:
        print(f"  {nombre}: {'OK' if ok else 'FALLA'}")
    return all(ok for _, ok in chequeos)


def main() -> None:
    filas = []
    for corpus, carpeta in CORPUS.items():
        fechas = fechas_bpoil(carpeta)
        for min_docs in MIN_DOCS:
            for grilla in ORDEN:
                filas.append(fila(corpus, fechas, grilla, min_docs))
    tabla = pd.DataFrame(filas)
    if not verificar(tabla):
        print("La verificación falla: no se guarda nada.")
        return
    tabla.to_csv(OUTPUT_PATH, index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 30):
        print()
        print(tabla.to_string(index=False))
        corta = tabla[tabla["min_docs"] == 5][["corpus", "grilla", "racha_max", "obs_utilizables", "K_que_alcanzan"]]
        print("\nMIN_DOCS = 5:")
        print(corta.to_string(index=False))
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
