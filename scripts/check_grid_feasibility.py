"""
check_grid_feasibility.py

Semana 5, primer paso: antes de construir ninguna serie, comparar
distintas grillas de muestreo candidatas (independientes de los chunks
de RollingLDA) y verificar si el corpus da suficientes ventanas (T)
para el orden de retardo que se quiera usar en la autorregresión.

No necesita theta ni RollingLDA -- solo las fechas del corpus.
"""

from causal_coherence.data_loading import load_bpoil_full, truncate_to_analysis_window


def main() -> None:
    df, _ = load_bpoil_full()
    df = truncate_to_analysis_window(df)
    print(f"Corpus: {len(df)} documentos, {df['date'].min().date()} a {df['date'].max().date()}")

    grids = [
        ("D", "diaria"),
        ("3D", "cada 3 dias"),
        ("W", "semanal"),
        ("2W", "quincenal"),
    ]

    resultados = []
    for freq, label in grids:
        counts = df.set_index("date").resample(freq).size()
        T = len(counts)
        n_vacias = int((counts == 0).sum())
        n_min = int(counts[counts > 0].min()) if (counts > 0).any() else 0
        n_mediana = counts.median()
        resultados.append((label, freq, T, n_vacias, n_min, n_mediana))
        print(f"\nGrilla {label} ({freq}): T={T} ventanas, {n_vacias} vacias, "
              f"minimo={n_min}, mediana={n_mediana:.1f} docs/ventana")

    print("\n" + "=" * 70)
    print("Factibilidad: T vs. parametros por ecuacion (K*p + 1, con el +1")
    print("del control de volumen)")
    print("=" * 70)

    for label, freq, T, n_vacias, n_min, n_mediana in resultados:
        for K in (9, 10):  # 9 = razon logaritmica (K-1); 10 = volumen/factorial
            for p in (1, 2):
                params = K * p + 1
                ratio = T / params if params else float("inf")
                flag = "OK" if ratio >= 3 else ("AJUSTADO" if ratio >= 1.5 else "INSUFICIENTE")
                print(f"  {label:14s} K={K:2d} p={p}  T={T:3d}  params={params:3d}  "
                      f"T/params={ratio:4.1f}  [{flag}]")


if __name__ == "__main__":
    main()