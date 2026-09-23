# Causal Coherence

Proyecto Capstone: extracción de narrativas sobre corpus de noticias,
partiendo del método original de **Narrative Trails** (UMAP + HDBSCAN +
grafo de coherencia) y de un modelo de tópicos dinámico (**RollingLDA**)
como base para incorporar coherencia causal. Los papers y repositorios
de ambos métodos están en [Referencias](#referencias).

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
│   ├── afg_explore.py        # Exploración rápida sobre el subset Taliban
│   └── run_notebooks.py      # Ejecuta los notebooks de Examples/
├── Examples/                 # Notebooks de demostración, uno por pipeline
├── docs/                     # Notas técnicas (reproducibilidad.md)
├── results/                  # Salidas de consola de corridas de referencia
├── data/                     # Modelos y resultados generados (ignorado por git)
├── environment.yml           # Ambiente conda único (Python, modelo de spaCy, variables)
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

`Library/` de narrative-trails ([repo de referencia](#narrative-trails))
no es un paquete instalable (el repo no tiene `setup.py` ni
`pyproject.toml`), así que `narrative.py` lo agrega
a `sys.path` desde `CC_NARRATIVE_TRAILS_DIR`. No hace falta hacer nada a
mano: basta con que la carpeta exista.

## Instalación

Un solo ambiente conda (`causal-coherence`, Python 3.12) para todo el
pipeline. Desde la raíz del repo:

```bash
conda env create -f environment.yml
```

Eso instala el paquete en modo editable con todas sus dependencias y el
extra `[notebooks]` (kernel y ejecutor de Jupyter), el modelo
`en_core_web_sm` de spaCy y deja fijas dos variables **del propio
ambiente**: `PYTHONHASHSEED=0` y `PYTHONUTF8=1`. La segunda hace falta en
Windows porque `jupyter execute` abre los notebooks con la codificación
del sistema y corrompe los acentos. Las dos se aplican solas cada vez que
haces `conda activate causal-coherence`, sin pasos manuales. Para
comprobarlo:

```bash
conda env config vars list -n causal-coherence
```

Si cambias `pyproject.toml`, actualiza el ambiente con
`conda env update -f environment.yml`.

> Las variables solo se aplican al **activar** el ambiente. Si llamas
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

## Notebooks

`Examples/` tiene un notebook por pipeline. Usan las funciones de
`src/causal_coherence/`, las mismas que los scripts.

| Notebook                             | Qué muestra                                                                      |
|--------------------------------------|----------------------------------------------------------------------------------|
| `rolling_lda_bpoil.ipynb`            | Palabras por tópico, matriz theta y prevalencia semanal de tres tópicos           |
| `narrative_trails_bpoil.ipynb`       | Narrativas alternativas entre pares de documentos de bpoil, con sus títulos       |
| `narrative_trails_afghanistan.ipynb` | Lo mismo sobre el subset Taliban, más el diagnóstico de grado y el experimento de pares al azar |

`rolling_lda_bpoil.ipynb` carga `data/roll_lda_bpoil.pickle`, así que
antes hay que correr `scripts/fit_rolling_lda.py`. Los notebooks de
Narrative Trails guardan el landscape ajustado en `data/cache/`: la
primera vez tardan unos 20 segundos (bpoil) o 2 minutos (Taliban), y los
archivos pesan cerca de 35 MB y 170 MB. Después cargan en menos de un
segundo.

Para ejecutarlos sin abrir Jupyter y guardar las salidas en el mismo
archivo:

```bash
python scripts/run_notebooks.py
```

Acepta nombres de notebook como argumento para correr solo algunos. A
diferencia de `jupyter execute --inplace`, no guarda marcas de tiempo por
celda: si las salidas no cambian, volver a correrlos deja los archivos
idénticos y git no muestra diferencias.

## Reproducibilidad

- RollingLDA usa `seed=42`; los hiperparámetros y su justificación
  están documentados en `topic_model.py`.
- `baseline_narrative.py` nombra el resultado con un hash del contenido
  de la matriz de coherencia, así dos corridas con el mismo grafo
  producen el mismo archivo. `results/corrida_1.txt` y
  `results/corrida_2.txt` son dos corridas de referencia de ese script
  (hash `63d0faab065a6f4c`).

`pyproject.toml` fija las versiones de scipy, UMAP y HDBSCAN porque con
versiones más nuevas cambia ese hash. Antes de actualizar una de ellas,
vuelve a correr `baseline_narrative.py` y compara el hash. El detalle
está en [docs/reproducibilidad.md](docs/reproducibilidad.md).

## Referencias

### Narrative Trails

Método base de la línea narrativa: un camino de máxima capacidad sobre
un grafo de coherencia por similitud. `narrative.py` importa su
`Library/` directamente desde el repositorio.

German, F., Keith, B. y North, C. (2025). Narrative Trails: A Method for
Coherent Storyline Extraction via Maximum Capacity Path Optimization.
*Text2Story 2025*, CEUR-WS Vol. 3964, pp. 15-27.
[Paper](https://ceur-ws.org/Vol-3964/paper2.pdf) ·
[Repositorio](https://github.com/faustogerman/narrative-trails/)

### RollingLDA

El paper que define el algoritmo (chunks, `memory`, `warmup`) no es el
mismo que lo aplica al caso de Chile del repo t2s2026. Este proyecto usa
el algoritmo, a través del paquete ttta (`topic_model.py`). Del caso de
Chile solo se usa el repositorio.

#### Algoritmo

Rieger, J., Jentsch, C. y Rahnenführer, J. (2021). RollingLDA: An Update
Algorithm of Latent Dirichlet Allocation to Construct Consistent Time
Series from Textual Data. *Findings of the Association for Computational
Linguistics: EMNLP 2021*, pp. 2337-2347.
[DOI](https://doi.org/10.18653/v1/2021.findings-emnlp.201) ·
[Paquete ttta](https://pypi.org/project/ttta/)

#### Aplicación a Chile (t2s2026)

Análisis espacio-temporal de conflictos socioambientales ligados a la
transición energética en Chile. Su repositorio sirvió como guía de la
API de ttta. Su corpus, su preprocesamiento en español (paquete `cet`) y
su caso de aplicación no forman parte de este proyecto.

Rieger, J., Muñoz, F., Grönberg, L., et al. (2026). Following
Socio-Environmental Conflict Narratives About Energy Transition in
Chile: A Spatio-Temporal Analysis Using Dynamic Topic Modeling.
*Text2Story 2026*, CEUR-WS Vol. 4202.
[Paper](https://ceur-ws.org/Vol-4202/paper5.pdf) ·
[Repositorio](https://github.com/JonasRieger/t2s2026)
