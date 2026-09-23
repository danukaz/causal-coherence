# Reproducibilidad

Por qué `pyproject.toml` fija ocho versiones exactas, de dónde sale
`requires-python = ">=3.11,<3.13"`, por qué RollingLDA necesita
`PYTHONHASHSEED` fijo además de `seed=42`, y qué campo de los notebooks se
puede ignorar en un diff.

## Resultado de referencia

`baseline_narrative.py` nombra su salida con un hash de la matriz de
coherencia dispersa. El valor de referencia es `63d0faab065a6f4c`, con
54 tópicos en bpoil, y corresponde a `results/corrida_1.txt` y
`results/corrida_2.txt`. Se generó en el ambiente `capstone` (Python
3.11), antes de unificar los dos ambientes del proyecto. Las corridas de
referencia actuales están descritas en `results/README.md`.

## Qué pasa sin versiones fijadas

Un ambiente con Python 3.12 y la última versión de cada dependencia
instala sin conflictos y `pip check` no reporta nada. `find_chunks.py` y
`fit_rolling_lda.py` dan exactamente la misma salida que en el ambiente
`rollinglda`, aunque ttta pasa de 0.9.6 a 0.9.9 y pandas de 2.3 a 3.0.
Los dos scripts de Narrative Trails sí cambian:

|                        | Referencia         | Versiones nuevas   |
|------------------------|--------------------|--------------------|
| Hash de coherencia     | `63d0faab065a6f4c` | `438557565f1918b7` |
| Tópicos, bpoil         | 54                 | 55                 |
| Tópicos, Taliban       | 134                | 140                |

## Dónde se origina la diferencia

Para separar UMAP de HDBSCAN se guardó la proyección UMAP de cada
ambiente y se agrupó cada una con el HDBSCAN del otro. Número de
tópicos en bpoil:

| HDBSCAN \ proyección                   | UMAP 0.5.7 | UMAP 0.5.12 |
|----------------------------------------|------------|-------------|
| hdbscan 0.8.40, scikit-learn 1.6.1     | 54         | 56          |
| hdbscan 0.8.44, scikit-learn 1.9.1     | 51         | 55          |

Cambian las dos etapas, así que hay que fijar ambas.

UMAP. Con umap-learn, pynndescent, numba, numpy, hdbscan y scikit-learn
en las versiones de referencia, y scipy todavía en 1.18.1, la proyección
seguía distinta y el resultado bajaba a 46 tópicos. Al bajar scipy a
1.13.1 la proyección quedó idéntica bit a bit a la de referencia. El uso más directo de scipy en UMAP es la
inicialización espectral (`init="spectral"`, el valor por defecto), que
resuelve un problema de valores propios con `scipy.sparse.linalg.eigsh`
o `lobpcg`. No se aisló cuál de las dos llamadas es la que cambia.

HDBSCAN. hdbscan y scikit-learn se cambiaron juntos, así que no se sabe
cuál de los dos altera el agrupamiento. Se fijan ambos.

## Versiones fijadas

| Paquete       | Versión | Motivo                                                        |
|---------------|---------|---------------------------------------------------------------|
| scipy         | 1.13.1  | Cambia la proyección UMAP (inicialización espectral)          |
| umap-learn    | 0.5.7   | Proyección UMAP                                               |
| pynndescent   | 0.5.13  | Grafo de vecinos aproximados que usa UMAP                     |
| numba         | 0.60.0  | Compila las rutinas de UMAP y pynndescent                     |
| llvmlite      | 0.43.0  | Backend de numba 0.60                                         |
| numpy         | 2.0.2   | numba 0.60 exige `numpy<2.1`; es la versión de referencia     |
| hdbscan       | 0.8.40  | Agrupamiento en tópicos                                       |
| scikit-learn  | 1.6.1   | Usado por hdbscan; cambia el agrupamiento junto con él        |

Las versiones de umap-learn, pynndescent, numba y llvmlite se fijaron
juntas a las de referencia. No se probó por separado si alguna de ellas,
con scipy ya fijado, altera el resultado por sí sola.

El resto de las dependencias queda sin fijar. networkx (3.2.1 contra
3.7), spaCy (3.8.11 contra 3.8.16), ttta (0.9.6 contra 0.9.9) y pandas
(2.x contra 3.0.6) cambian de versión respecto de los ambientes
anteriores sin efecto en ninguna salida.

## Versión de Python

Ni ttta ni sus dependencias declaran un mínimo de Python 3.12. Ese
requisito venía de `cet`, el paquete del repo t2s2026
([referencia](../README.md#referencias)), que el proyecto
dejó de usar al escribir su propio preprocesamiento.

El techo `<3.13` lo pone numba 0.60, que no tiene soporte para Python
3.13. El piso 3.11 viene del ambiente `capstone`, donde los scripts de
Narrative Trails generaron la referencia. `fit_rolling_lda.py` y
`find_chunks.py` solo se han corrido en 3.12, así que el pipeline
completo está validado en 3.12 y no en 3.11.

## Validación

Con las versiones fijadas, un ambiente creado desde cero con
`conda env create -f environment.yml` y activado en una terminal nueva
reproduce los cuatro scripts:

| Script                  | Comparado con | Resultado                               |
|-------------------------|---------------|-----------------------------------------|
| `baseline_narrative.py` | `capstone`    | Hash `63d0faab065a6f4c`, salida idéntica |
| `afg_explore.py`        | `capstone`    | Salida idéntica                          |
| `find_chunks.py`        | `rollinglda`  | Salida idéntica                          |
| `fit_rolling_lda.py`    | `rollinglda`  | Matrices θ y φ idénticas (con `PYTHONHASHSEED=0` en los dos) |

## Cómo actualizar una versión fijada

Cambiar el pin en `pyproject.toml`, actualizar el ambiente con
`conda env update -f environment.yml`, correr `baseline_narrative.py` y
comparar el hash con `63d0faab065a6f4c`. Si cambia, los resultados de
Narrative Trails dejan de ser comparables con los de `results/` y hay
que regenerar la referencia a propósito, guardando la corrida nueva en
`results/` junto a la anterior.

## RollingLDA y PYTHONHASHSEED

`seed=42` no alcanza para repetir un modelo de RollingLDA. ttta arma el
vocabulario en `ttta.preprocessing.preprocess.create_dtm` como
`vocab + list(set(...))`. El orden en que se recorre un `set` de strings
depende del hash de cada string, y ese hash cambia de una ejecución de
Python a otra salvo que `PYTHONHASHSEED` esté fijo. El orden del
vocabulario define el id de cada palabra, y con otros ids el mismo
muestreo de Gibbs, sembrado con 42, llega a otros tópicos.

El estado del generador aleatorio antes del ajuste no influye. Con
`workers=1`, ttta vuelve a sembrar `np.random` con la semilla de cada
prototipo (42, 43, ...) justo antes de sortear sus asignaciones iniciales,
y el sampler en C recibe su propia semilla. `RollingLDA.fit` usa del
DataFrame solo la columna de texto, la de fecha y el índice.

Tres comparaciones, todas sobre el modelo de referencia actual
(`data/roll_lda_bpoil.pickle`, ajustado con `PYTHONHASHSEED=0`):

| Modelo comparado                                   | Qué cambia                          | θ y φ     | Tópicos                         |
|----------------------------------------------------|-------------------------------------|-----------|---------------------------------|
| Código original, anterior a la reorganización      | Organización del código             | Idénticas | Idénticos                       |
| Script actual con `PYTHONHASHSEED=1`               | Solo `PYTHONHASHSEED`               | Distintas | 2 a 8 palabras de 10 en común   |
| Modelo del 12/09 (bitácora de esa fecha)           | Semilla de hash desconocida         | Distintas | 1 a 8 palabras de 10 en común   |

El código original no está en git (el repo se creó después de
reorganizarlo); se reconstruyó tal como estaba antes de moverlo a `src/`.
En los dos modelos distintos, la configuración (los 15 parámetros de
RollingLDA y LDAPrototype, incluidas las semillas por prototipo) y las
entradas son las mismas: los 1011 documentos tienen exactamente las mismas
palabras y el vocabulario tiene las mismas 9437. Lo único distinto es el
orden del vocabulario, desde la primera posición. El modelo del 12/09 no se
puede repetir porque no quedó registrado con qué semilla de hash corrió.

El ambiente fija `PYTHONHASHSEED=0` al activarse, y el kernel registrado
para los notebooks lo lleva en su definición. `fit_rolling_lda.py` llama a
`check_hash_seed()` al comienzo: si la variable no está fija, corta antes
de gastar los ~20 minutos del ajuste; si tiene otro valor, avisa que el
modelo no va a coincidir con la referencia. `find_chunks.py` no depende de
la semilla de hash, porque solo cuenta tipos y tokens.

## Metadata de los notebooks

El editor escribe `metadata.kernelspec` (`name` y `display_name`) al
guardar un notebook, según cómo identifica el kernel en esa sesión. VS Code
puede alternar entre "Python 3" y "Python 3 (causal-coherence)" aunque el
intérprete sea el mismo, y eso aparece como un diff sin ningún cambio de
fondo. Para reducirlo, el kernel del ambiente se registra con un nombre
propio (comando en el README) y los notebooks lo nombran así:
`"name": "causal-coherence"`.

Si el campo vuelve a cambiar en un diff, se puede ignorar. No afecta qué
intérprete ejecuta las celdas ni ninguna salida. Lo que sí identifica el
intérprete es `metadata.language_info`, que escribe el kernel al ejecutar,
y las salidas de las celdas. `scripts/run_notebooks.py` usa el kernel
registrado si existe y no modifica `kernelspec`.
