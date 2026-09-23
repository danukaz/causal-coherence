# results/

Salidas de consola de corridas de referencia de `scripts/baseline_narrative.py`.
Son registros tal como salieron del script y no se editan.

| Archivo                        | Par      | Versión del script                                  |
|--------------------------------|----------|-----------------------------------------------------|
| `corrida_1.txt`, `corrida_2.txt` | 3 → 975  | Anterior a la reorganización, sin chequeo de grado  |
| `baseline_3-975.txt`           | 3 → 975  | Actual (`python scripts/baseline_narrative.py 3 975`) |
| `baseline_15-572.txt`          | 15 → 572 | Actual, par de referencia (`python scripts/baseline_narrative.py`) |

Las cuatro dan el mismo hash de la matriz de coherencia, `63d0faab065a6f4c`,
porque el landscape no depende del par elegido. `corrida_1.txt` y
`corrida_2.txt` son idénticas entre sí y se guardaron con la codificación de la
consola de Windows (cp1252), así que los acentos se ven mal al abrirlas como
UTF-8.

## Por qué hay dos pares

3 → 975 fue el primer par, elegido por su lectura narrativa: el documento 3
reporta el hundimiento de la plataforma y el 975 anuncia el sellado del pozo.
Dio caminos cortos (3 y 4 documentos) y débiles (bottleneck cercano a 0,5).
15 → 572 se eligió después, de forma ad hoc, para probar si otro par daba algo
distinto, y dio caminos de 6 a 8 documentos con bottleneck cercano a 0,82.

El diagnóstico de grado explica la diferencia: el origen 3 es un nodo
periférico (grado 18, contra una mediana de 876 en el grafo) cuyas aristas
hacia el grafo principal pesan entre 0,46 y 0,52. 975, 15 y 572 tienen un grado
típico. Los dos pares se mantienen como referencia porque cada uno muestra algo
distinto: 3 → 975, que un extremo atípico no se nota leyendo el titular; 15 →
572, el resultado con extremos verificados. El análisis completo está en
`Examples/narrative_trails_bpoil.ipynb` y la verificación del par de referencia
en el comentario de `scripts/baseline_narrative.py`.
