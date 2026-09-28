"""
sweep_topic_k.py  (exploratorio)

Barrido de la cantidad de tópicos K, como exige la guía (sección 7: "no hay
un valor correcto: hay que barrerlo y reportar el barrido"). Para cada K en
K_VALUES:

  - el modelo RollingLDA: K=10 reutiliza data/roll_lda_bpoil.pickle; los
    demás se ajustan con LDA_CONFIG cambiando solo K (alpha queda fijo en
    el valor de LDA_CONFIG, sin escalarlo con K);
  - la serie de volumen, con la misma lógica de build_volumen_series.py
    (mismo corpus, grilla semanal, recorte de bordes y mínimo de documentos);
  - la factibilidad T contra K*p+1 con p=1, con los mismos umbrales de
    check_grid_feasibility.py;
  - la varianza explicada por la primera componente, con el mismo PCA de
    build_series_variants.py;
  - las palabras principales por tópico, en un archivo de texto por K.

Uso:
    python scripts/sweep_topic_k.py fit 5      # ajusta y guarda data/roll_lda_bpoil_K5.pickle
    python scripts/sweep_topic_k.py analyze    # los tres K juntos
"""

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ttta.methods.rolling_lda import RollingLDA

from causal_coherence.config import OUTPUT_DIR
from causal_coherence.data_loading import (
    SOURCE_LABEL,
    build_dataframe,
    drop_windows_below_min_docs,
    load_filtered,
    truncate_to_analysis_window,
)
from causal_coherence.preprocessing import build_english_pipeline, get_english_stopwords, preprocess_batch
from causal_coherence.topic_model import LDA_CONFIG, MODEL_PATH, check_hash_seed, load_topic_model, training_documents

K_VALUES = (5, 8, 10, 12, 15)
P = 1
SWEEP_DIR = OUTPUT_DIR / "barrido_K"

# Misma lógica que los scripts de la semana 5, importada tal cual.
_SCRIPTS = Path(__file__).resolve().parent
_volumen = runpy.run_path(str(_SCRIPTS / "build_volumen_series.py"))
_variants = runpy.run_path(str(_SCRIPTS / "build_series_variants.py"))
get_theta = _volumen["get_theta"]
drop_incomplete_edge_windows = _volumen["drop_incomplete_edge_windows"]
build_residualized = _variants["build_residualized"]


def model_path(k: int) -> Path:
    return MODEL_PATH if k == LDA_CONFIG["K"] else OUTPUT_DIR / f"roll_lda_bpoil_K{k}.pickle"


def fit(k: int) -> None:
    if k == LDA_CONFIG["K"]:
        raise SystemExit(f"K={k} ya está ajustado en {MODEL_PATH.name}; no se reajusta.")
    check_hash_seed()
    config = {**LDA_CONFIG, "K": k}
    assert config["alpha"] == LDA_CONFIG["alpha"]

    docs, _ = load_filtered(SOURCE_LABEL)
    df = truncate_to_analysis_window(build_dataframe(docs))
    print("Preprocesando texto...")
    df["preprocessed_text"] = preprocess_batch(df["content"].tolist(), build_english_pipeline(), get_english_stopwords())

    print(f"\nAjustando RollingLDA con K={k} (resto de LDA_CONFIG sin cambios, alpha={config['alpha']})...")
    t0 = time.perf_counter()
    roll = RollingLDA(**config)
    roll.fit(df, text_column="preprocessed_text", date_column="date")
    seconds = time.perf_counter() - t0

    path = model_path(k)
    roll.save(str(path))
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / f"fit_K{k}.json").write_text(json.dumps({"K": k, "segundos_ajuste": seconds}), encoding="utf-8")
    print(f"K={k}: ajuste en {seconds / 60:.1f} min, modelo guardado en {path.name}")


def volume_series(roll, df):
    """Mismos pasos que build_volumen_series.main."""
    theta = get_theta(roll)
    theta_df = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(theta.shape[1])])
    theta_df["date"] = df["date"].values
    masa = theta_df.set_index("date").resample("W").sum()
    n_docs = theta_df.set_index("date").resample("W").size()
    masa = drop_incomplete_edge_windows(masa)
    n_docs = n_docs.loc[masa.index]
    masa, n_docs = drop_windows_below_min_docs(masa, n_docs)
    return np.log1p(masa), np.log1p(n_docs).rename("control_volumen")


def feasibility_flag(ratio: float) -> str:
    """Mismos umbrales que check_grid_feasibility.py."""
    return "OK" if ratio >= 3 else ("AJUSTADO" if ratio >= 1.5 else "INSUFICIENTE")


def analyze() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df = training_documents()
    rows = []
    for k in K_VALUES:
        print(f"\n===== K={k} ({model_path(k).name}) =====")
        roll = load_topic_model(model_path(k))
        assert roll._K == k, f"el modelo {model_path(k).name} tiene K={roll._K}"

        serie, control = volume_series(roll, df)
        serie.to_csv(SWEEP_DIR / f"topic_series_volumen_K{k}.csv")
        control.to_csv(SWEEP_DIR / f"control_volumen_K{k}.csv")
        if k == LDA_CONFIG["K"]:
            ref = pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv", index_col=0, parse_dates=True)
            print("serie K=10 idéntica a data/topic_series_volumen.csv:",
                  bool(np.allclose(ref.to_numpy(), serie.to_numpy()) and ref.index.equals(serie.index)))

        T = len(serie)
        params = k * P + 1
        _, varianza = build_residualized(serie)

        words = roll.top_words(number=10, return_as_data_frame=False)
        (SWEEP_DIR / f"top_words_K{k}.txt").write_text(
            "\n".join(f"topic_{i}: {', '.join(w)}" for i, w in enumerate(words)) + "\n", encoding="utf-8")

        fit_info = SWEEP_DIR / f"fit_K{k}.json"
        minutes = json.loads(fit_info.read_text(encoding="utf-8"))["segundos_ajuste"] / 60 if fit_info.exists() else np.nan
        rows.append({
            "K": k,
            "ajuste_min": minutes,
            "T": T,
            "params_por_ecuacion": params,
            "T_sobre_params": T / params,
            "factibilidad": feasibility_flag(T / params),
            "varianza_PC1": varianza[0],
        })

    resumen = pd.DataFrame(rows).set_index("K")
    resumen.to_csv(SWEEP_DIR / "resumen.csv")
    print("\nResumen del barrido (p=1; ajuste_min = NaN para K=10, modelo existente):")
    print(resumen.round(3).to_string())
    print(f"\nArchivos en {SWEEP_DIR}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fit":
        fit(int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "analyze":
        analyze()
    else:
        raise SystemExit("uso: python scripts/sweep_topic_k.py fit K | analyze")
