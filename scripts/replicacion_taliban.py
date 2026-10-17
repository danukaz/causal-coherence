"""
replicacion_taliban.py

Semana 8, paso 6 (Bitacora_Semana8.md, sección 32): réplica preregistrada
del proceso de Granger condicional sobre el subconjunto NEWS-TLS Entities
(Taliban) del corpus de Afganistán. Salidas en data/taliban/.

Etapas:
  auditoria   auditoría previa (solo reporte), exclusiones (fecha corrupta o
              contenido vacío; para si pasan del 5%) y preprocesamiento igual
              al de bpoil; guarda los documentos ordenados y sus tokens.
  modelo      ajusta RollingLDA con chunks anuales para un K (--K), semilla
              42 salvo --semilla; guarda el modelo, el resumen de chunks y las
              10 palabras de cada tópico.

Uso:
  python scripts/replicacion_taliban.py --etapa auditoria
  python scripts/replicacion_taliban.py --etapa modelo --K 10 [--semilla 42]
"""

import argparse
import pickle
import time

import pandas as pd

from causal_coherence import taliban
from causal_coherence.config import OUTPUT_DIR

SALIDA = OUTPUT_DIR / "taliban"
RUTA_DOCUMENTOS = SALIDA / "documentos.csv"
RUTA_TOKENS = SALIDA / "tokens.pkl"


def guardar_tablas(tablas: dict[str, pd.DataFrame]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.max_colwidth", 70):
        for nombre, tabla in tablas.items():
            tabla.to_csv(SALIDA / f"{nombre}.csv", index=False)
            print(f"\n=== {nombre} (data/taliban/{nombre}.csv, {len(tabla)} filas) ===")
            print(tabla.head(40).to_string(index=False) if len(tabla) else "  sin filas")


def ruta_modelo(K: int, semilla: int = 42) -> object:
    sufijo = "" if semilla == 42 else f"_semilla{semilla}"
    return SALIDA / f"roll_lda_taliban_K{K}{sufijo}.pickle"


def etapa_auditoria() -> bool:
    from causal_coherence.preprocessing import build_english_pipeline, get_english_stopwords, preprocess_batch

    df = taliban.documentos_taliban()
    tablas = taliban.auditoria(df)
    guardar_tablas(tablas)
    resumen = tablas["A_resumen"].iloc[0]
    if resumen["fraccion_excluida"] > taliban.MAX_EXCLUIDOS:
        print(f"\nLa auditoría excluye {resumen['fraccion_excluida']:.1%} (> 5%): se detiene.")
        return False
    excluir = set(tablas["A_excluidos"]["id"])
    datos = df[~df["id"].isin(excluir)].sort_values("date", kind="stable").reset_index(drop=True)
    print(f"\nPreprocesando {len(datos)} documentos (spaCy en_core_web_sm y stopwords de NLTK, como en bpoil)...", flush=True)
    t0 = time.perf_counter()
    tokens = preprocess_batch(datos["content"].tolist(), build_english_pipeline(), get_english_stopwords())
    print(f"  {time.perf_counter() - t0:.0f} s; documentos sin tokens: {sum(len(t) == 0 for t in tokens)}")
    datos[["id", "date", "title"]].to_csv(RUTA_DOCUMENTOS, index=False)
    with open(RUTA_TOKENS, "wb") as f:
        pickle.dump(tokens, f)
    return True


def etapa_modelo(K: int, semilla: int) -> None:
    from causal_coherence.topic_model import check_hash_seed, chunk_summary
    from ttta.methods.rolling_lda import RollingLDA

    check_hash_seed()
    ruta = ruta_modelo(K, semilla)
    if ruta.exists():
        print(f"Ya existe {ruta}: no se vuelve a ajustar.")
        return
    datos = pd.read_csv(RUTA_DOCUMENTOS, parse_dates=["date"])
    with open(RUTA_TOKENS, "rb") as f:
        datos["preprocessed_text"] = pickle.load(f)
    config = taliban.configuracion_lda(K, semilla)
    print(f"RollingLDA K = {K}, semilla {semilla}, gamma = {config['gamma']}, chunks: "
          f"{[d.date().isoformat() for d in config['how']]}", flush=True)
    t0 = time.perf_counter()
    roll = RollingLDA(**config)
    roll.fit(datos.copy(), text_column="preprocessed_text", date_column="date")
    minutos = (time.perf_counter() - t0) / 60
    roll.save(str(ruta))
    chunks = chunk_summary(roll, datos).reset_index()
    assert int(chunks["documentos"].sum()) == len(datos), "los chunks no cubren todos los documentos"
    sufijo = f"K{K}" if semilla == 42 else f"K{K}_semilla{semilla}"
    palabras = roll.top_words(number=10)
    palabras = palabras if isinstance(palabras, pd.DataFrame) else pd.DataFrame(palabras)
    guardar_tablas({f"M_chunks_{sufijo}": chunks, f"M_palabras_{sufijo}": palabras.reset_index(),
                    f"M_tiempo_{sufijo}": pd.DataFrame([{"K": K, "semilla": semilla, "minutos": minutos}])})
    print(f"\nModelo guardado en {ruta} ({minutos:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etapa", choices=["auditoria", "modelo"], required=True)
    parser.add_argument("--K", type=int, choices=[5, 10])
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    if args.etapa == "auditoria":
        etapa_auditoria()
    else:
        etapa_modelo(args.K, args.semilla)


if __name__ == "__main__":
    main()
