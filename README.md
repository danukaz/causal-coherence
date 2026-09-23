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
├── environment.yml           # Ambiente conda único (Python, modelo de spaCy, PYTHONHASHSEED)
└── pyproject.toml            # Dependencias de Python
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

`Library/` de narrative-trails no es un paquete instalable (el repo no
tiene `setup.py` ni `pyproject.toml`), así que `narrative.py` lo agrega
a `sys.path` desde `CC_NARRATIVE_TRAILS_DIR`. No hace falta hacer nada a
mano: basta con que la carpeta exista.

## Instalación

Un solo ambiente conda (`causal-coherence`, Python 3.12) para todo el
pipeline. Desde la raíz del repo:

```bash
conda env create -f environment.yml
```

Eso instala el paquete en modo editable con todas sus dependencias, el
modelo `en_core_web_sm` de spaCy y deja fijo `PYTHONHASHSEED=0` como
variable **del propio ambiente**: se aplica sola cada vez que haces
`conda activate causal-coherence`, sin pasos manuales. Para comprobarlo:

```bash
conda env config vars list -n causal-coherence
```

Si cambias `pyproject.toml`, actualiza el ambiente con
`conda env update -f environment.yml`.

> La variable solo se aplica al **activar** el ambiente. Si llamas
> directamente a `...\envs\causal-coherence\python.exe` sin activarlo
> (por ejemplo desde algunas configuraciones de IDE), no se fija.

## Uso

```bash
conda activate causal-coherence
python scripts/find_chunks.py        # explora umbrales de razón tipo-token
python scripts/fit_rolling_lda.py    # guarda data/roll_lda_bpoil.pickle
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
  `results/corrida_2.txt` son dos corridas de referencia de ese script
  (hash `63d0faab065a6f4c`).
- **Versiones fijadas en `pyproject.toml`.** Con versiones más nuevas
  de scipy, umap-learn/pynndescent/numba y hdbscan/scikit-learn todo
  instala sin conflictos, pero los resultados de Narrative Trails
  cambian: la proyección UMAP depende de scipy (inicialización
  espectral) y el clustering depende de hdbscan + scikit-learn. Con
  las versiones fijadas, el ambiente único reproduce bit a bit los
  cuatro scripts de los dos ambientes anteriores (`rollinglda`, Python
  3.12, y `capstone`, Python 3.11). Antes de actualizar cualquiera de
  esos pines, vuelve a correr `baseline_narrative.py` y compara el hash.
- Ni ttta ni sus dependencias exigen Python 3.12: ese requisito venía
  de `cet` (repo t2s2026), que el proyecto ya no usa. El techo `<3.13`
  lo pone numba 0.60.
