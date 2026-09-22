# Causal Coherence

Proyecto Capstone: extracción de narrativas sobre corpus de noticias,
partiendo del método original de **Narrative Trails** (UMAP + HDBSCAN +
grafo de coherencia) y de un modelo de tópicos dinámico (**RollingLDA**)
como base para incorporar coherencia causal.

El caso de estudio principal es el subset **bpoil** (derrame de
Deepwater Horizon, 2010) del corpus NEWS-TLS T17. El subset *Taliban*
del corpus de Afganistán se usa solo como exploración comparativa.

## Estructura

```
causal-coherence/
├── src/causal_coherence/     # Paquete importable
│   ├── config.py             # Rutas (datasets, narrative-trails, salidas)
│   ├── data_loading.py       # Carga, filtrado y alineación corpus ↔ embeddings
│   ├── preprocessing.py      # Tokenización/lematización en inglés (spaCy + NLTK)
│   ├── topic_model.py        # Configuración y ajuste de RollingLDA
│   └── narrative.py          # Envoltorio sobre Narrative Trails
├── scripts/                  # Puntos de entrada
│   ├── find_chunks.py        # Busca cortes de chunk por razón tipo-token
│   ├── fit_rolling_lda.py    # Ajusta RollingLDA sobre bpoil (semana 3)
│   ├── baseline_narrative.py # Narrativa de línea base con Narrative Trails (semana 4)
│   └── afg_explore.py        # Exploración rápida sobre el subset Taliban
├── results/                  # Salidas de consola de corridas de referencia
├── data/                     # Modelos y resultados generados (ignorado por git)
└── pyproject.toml
```

## Datos y dependencias externas

Los datos **no** viven en este repo ni se copian aquí. Por defecto se
asume la estructura de la carpeta Capstone:

```
Capstone/
├── causal-coherence/   # este repo
├── Datasets/
│   ├── t17/            # corpus.jsonl, dataset.json, emb=mpnet/embeddings_mpnet.npy
│   └── afghanistan/
└── narrative-trails/   # repo de referencia (se importa su carpeta Library/)
```

Si están en otro lugar, define estas variables de entorno:

| Variable                  | Por defecto              |
|---------------------------|--------------------------|
| `CC_DATASETS_DIR`         | `../Datasets`            |
| `CC_NARRATIVE_TRAILS_DIR` | `../narrative-trails`    |

## Ambientes

El proyecto usa dos ambientes conda, porque ttta y Narrative Trails
tienen dependencias que no conviven bien:

| Ambiente     | Python | Para qué                                   | Scripts                                   |
|--------------|--------|--------------------------------------------|-------------------------------------------|
| `rollinglda` | 3.12   | spaCy, NLTK, ttta (RollingLDA)             | `find_chunks.py`, `fit_rolling_lda.py`    |
| `capstone`   | 3.11   | UMAP, HDBSCAN, networkx (Narrative Trails) | `baseline_narrative.py`, `afg_explore.py` |

`data_loading` y `config` solo dependen de numpy y pandas, así que
funcionan en ambos.

## Instalación

En **cada** ambiente, desde la raíz del repo:

```bash
pip install -e . --no-deps
```

`--no-deps` evita que pip toque los paquetes que ya instaló conda. Si
armas un ambiente desde cero, puedes usar los extras
`pip install -e ".[lda]"` o `pip install -e ".[narrative]"`.

En el ambiente `rollinglda` hace falta además el modelo de spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Uso

```bash
conda activate rollinglda
python scripts/find_chunks.py        # explora umbrales de razón tipo-token
python scripts/fit_rolling_lda.py    # guarda data/roll_lda_bpoil.pickle

conda activate capstone
python scripts/baseline_narrative.py # guarda data/baseline_<hash>.pkl
python scripts/afg_explore.py
```

Los scripts se pueden correr desde cualquier directorio: las salidas
siempre van a `data/` en la raíz del repo.

## Reproducibilidad

- RollingLDA usa `seed=42`; los hiperparámetros y su justificación
  están documentados en `topic_model.py`.
- `baseline_narrative.py` nombra el resultado con un hash del contenido
  de la matriz de coherencia, así dos corridas con el mismo grafo
  producen el mismo archivo. `results/corrida_1.txt` y
  `results/corrida_2.txt` son dos corridas de referencia de ese script.
