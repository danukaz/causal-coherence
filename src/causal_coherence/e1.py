"""
e1.py

Semana 8, paso 5 (Bitacora_Semana8.md, sección 29): piezas de E1, ¿la
estructura detectada es dirigida?

  - diseno_pares: prueba por pares sin condicionar, B_t ~ 1 + B_{t-1} +
    control_t (restringido) y + A_{t-1} (completo); df_num = 1,
    df_den = n_obs - 4.
  - diseno_condicional: la condicional de siempre (granger.armar_diseno).
  - analizar: las dos pruebas sobre las 4 versiones de una serie, con F,
    efecto, p teórico y, si B > 0, p por remuestreo salvaje
    (granger.remuestreo_salvaje), BH por versión y regla de 2 versiones.
  - simular_impulsor: el AR(1) diagonal del clr del bloque B con un factor
    latente común g_t ~ AR(1) que entra con rezago 1.
  - asimetria, reciprocos: el estadístico S de B5 y la medida de enlaces
    recíprocos.
  - simbolizar, te_conteo, permutar_bloques: transferencia de entropía con
    b = 2, k = l = 1, estimador de conteo en log2 (B6).
  - iniciar_trabajador, trabajo y replica_*: una réplica de B2 a B5 a partir
    de un contexto fijo, para correr en procesos independientes.
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from causal_coherence import candidatas, granger, surrogates, validez

VERSIONES = list(surrogates.VERSIONES)
PRUEBAS = ("condicional", "pares")
Q = 0.05
MIN_VERSIONES = 2
U_IMPULSOR = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
PHI_IMPULSOR = 0.7

Clave = Callable[[str, str, str, str], tuple[str | int, ...]]
Diseno = tuple[np.ndarray, np.ndarray, np.ndarray]


def pares_por_version(columnas: list[str], descartada: str) -> dict[str, list[tuple[str, str]]]:
    """Pares ordenados por versión (la residualizada sin la columna descartada), como paquete_validez."""
    sin = [c for c in columnas if c != descartada]
    return {v: list(itertools.permutations(sin if v == "residualizada" else columnas, 2)) for v in VERSIONES}


def diseno_pares(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> Diseno:
    """X_f = [B_{t-1}, A_{t-1}, control_t, 1], X_r = [B_{t-1}, control_t, 1] e y = B_t."""
    b = serie[destino].to_numpy()
    a = serie[origen].to_numpy()
    c = control.to_numpy()
    uno = np.ones(len(b) - 1)
    X_r = np.column_stack([b[:-1], c[1:], uno])
    X_f = np.column_stack([b[:-1], a[:-1], c[1:], uno])
    return X_f, X_r, b[1:]


def diseno_condicional(serie: pd.DataFrame, control: pd.Series, origen: str, destino: str) -> Diseno:
    X_f, y = granger.armar_diseno(serie, control, destino, excluir=None)
    X_r, _ = granger.armar_diseno(serie, control, destino, excluir=origen)
    return X_f, X_r, y


DISENOS: dict[str, Callable[[pd.DataFrame, pd.Series, str, str], Diseno]] = {
    "condicional": diseno_condicional, "pares": diseno_pares}


def estimar_version(serie: pd.DataFrame, control_de: Callable[[str], pd.Series], pares: list[tuple[str, str]], prueba: str,
                    B: int, clave: Callable[[str, str], tuple[str | int, ...]] | None) -> tuple[dict[str, np.ndarray], bool]:
    """
    efecto, F, p_teo, p_rem (NaN si B = 0), n_obs y df_den de cada par. Si
    algún diseño completo no tiene rango completo, la versión es no estimable
    y todo queda en NaN (regla de paquete_validez.estimar).
    """
    m = len(pares)
    val = {k: np.full(m, np.nan) for k in ("efecto", "F", "p_teo", "p_rem", "n_obs", "df_den")}
    disenos = [DISENOS[prueba](serie, control_de(d), o, d) for o, d in pares]
    if any(np.linalg.matrix_rank(X_f) < X_f.shape[1] for X_f, _, _ in disenos):
        return val, True
    for j, ((o, d), (X_f, X_r, y)) in enumerate(zip(pares, disenos)):
        obs = validez.prueba_p(X_f, X_r, y)
        val["efecto"][j], val["F"][j], val["p_teo"][j] = obs["efecto"], obs["F"], obs["p_teo"]
        val["n_obs"][j], val["df_den"][j] = X_f.shape[0], obs["df_den"]
        if B:
            val["p_rem"][j] = granger.remuestreo_salvaje(X_f, X_r, y, obs["F"], obs["df_num"], obs["df_den"], B,
                                                         granger.generador(*clave(o, d)))["p"]
    return val, False


def versiones_propio(propio: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """paquete_validez.versiones_propio."""
    return {"original": propio, "diferenciada": surrogates.diferenciar(propio),
            "detendenciada": surrogates.detendenciar(propio), "residualizada": propio}


def analizar(x: pd.DataFrame, n_t: pd.Series | None, descartada: str, pares: dict[str, list[tuple[str, str]]],
             pruebas: tuple[str, ...], B: int, clave: Clave | None,
             control: str = "propio") -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """
    Las pruebas pedidas sobre las 4 versiones de x (log1p de las masas).
    Con n_t, el control propio es log1p(n_t - masa) (como el bloque B); sin
    n_t, los controles salen de candidatas.construir_versiones_k (como la
    estimación real). Devuelve rechazos por prueba y versión, enlaces por la
    regla de 2 versiones, y los arreglos por par.
    """
    versiones = candidatas.construir_versiones_k(x, descartada)
    if n_t is not None:
        assert control == "propio", "con n_t externo solo se usa el control propio"
        propio_v = versiones_propio(np.log1p(np.expm1(x).rsub(n_t.astype(float), axis=0)))
        control_de = {v: (lambda d, v=v: propio_v[v][d]) for v in VERSIONES}
    elif control == "propio":
        control_de = {v: (lambda d, v=v: versiones[v][2][d]) for v in VERSIONES}
    else:
        control_de = {v: (lambda d, v=v: versiones[v][1]) for v in VERSIONES}
    fila: dict[str, object] = {}
    arreglos: dict[str, np.ndarray] = {}
    for prueba in pruebas:
        veces: dict[tuple[str, str], int] = {}
        for v in VERSIONES:
            clave_v = None if clave is None else (lambda o, d, v=v, prueba=prueba: clave(prueba, v, o, d))
            val, no_estimable = estimar_version(versiones[v][0], control_de[v], pares[v], prueba, B, clave_v)
            m = len(pares[v])
            rechaza = granger.benjamini_hochberg(val["p_rem"], Q, m) if B and not no_estimable else np.zeros(m, dtype=bool)
            for par, r in zip(pares[v], rechaza):
                veces[par] = veces.get(par, 0) + int(r)
            fila[f"rechazos_{prueba}_{v}"] = int(rechaza.sum())
            fila[f"no_estimable_{prueba}_{v}"] = int(no_estimable)
            arreglos[f"{prueba}_rechaza_{v}"] = rechaza.astype(np.int8)
            for k, arr in val.items():
                arreglos[f"{prueba}_{k}_{v}"] = arr
        fila[f"enlaces_regla_{prueba}"] = int(sum(n >= MIN_VERSIONES for n in veces.values()))
    return fila, arreglos


def enlaces_declarados(arreglos: dict[str, np.ndarray], prueba: str,
                       pares: dict[str, list[tuple[str, str]]]) -> set[tuple[str, str]]:
    veces: dict[tuple[str, str], int] = {}
    for v in VERSIONES:
        for par, r in zip(pares[v], arreglos[f"{prueba}_rechaza_{v}"]):
            veces[par] = veces.get(par, 0) + int(r)
    return {par for par, n in veces.items() if n >= MIN_VERSIONES}


# ---------------------------------------------------------------------------
# Sustitutos
# ---------------------------------------------------------------------------

def simular_impulsor(c: np.ndarray, a: np.ndarray, residuos: np.ndarray, x0: np.ndarray, T: int, calent: int,
                     lam: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """
    z_t = c + a z_{t-1} + lam g_{t-1} + e_t, con e_t una fila completa de los
    residuos reales y g_t = phi g_{t-1} + nu_t, nu_t ~ N(0, 1 - phi^2),
    g inicial ~ N(0, 1) (varianza estacionaria 1). Las filas se sortean
    primero, como en candidatas.simular, así que con lam = 0 da la misma serie.
    """
    filas = rng.integers(0, residuos.shape[0], size=calent + T)
    nu = rng.normal(0.0, math.sqrt(1 - phi ** 2), size=calent + T)
    g = rng.normal()
    z = x0
    simulada = np.empty((calent + T, len(c)))
    for t, fila in enumerate(filas):
        z = c + a * z + lam * g + residuos[fila]
        g = phi * g + nu[t]
        simulada[t] = z
    return simulada[-T:]


def a_series(z: np.ndarray, n_t: pd.Series, columnas: pd.Index) -> pd.DataFrame:
    """clr simulado -> proporciones (softmax) -> masas n_t * s -> log1p, como paquete_validez.sustituto_b."""
    s = validez.softmax_filas(z)
    masa = s * n_t.to_numpy(dtype=float)[:, None]
    return pd.DataFrame(np.log1p(masa), index=n_t.index, columns=columnas)


def sustituto_b2(ctx: dict[str, object], indice: int) -> pd.DataFrame:
    """El sustituto del bloque B de la sección 26, con su misma clave de generación."""
    real, n_t = ctx["real"], ctx["n_t"]
    z = candidatas.simular(ctx["c"], ctx["a"], ctx["residuos"], ctx["x0"], len(real), ctx["calent"], 0, 1, 0.0, 0.0,
                           granger.generador("val_generacion", str(indice)))
    return a_series(z, n_t, real.columns)


def sustituto_b3(ctx: dict[str, object], kappa: str, indice: int) -> pd.DataFrame:
    real, n_t = ctx["real"], ctx["n_t"]
    lam = float(kappa) * ctx["sd_e"] * U_IMPULSOR
    z = simular_impulsor(ctx["c"], ctx["a"], ctx["residuos"], ctx["x0"], len(real), ctx["calent"], lam, PHI_IMPULSOR,
                         granger.generador("e1_b3_generacion", kappa, str(indice)))
    return a_series(z, n_t, real.columns)


def permutacion_b4(ctx: dict[str, object], indice: int) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Filas completas (masas y n_t) reordenadas; el índice de fechas queda el original."""
    real, n_t = ctx["real"], ctx["n_t"]
    orden = granger.generador("e1_b4_permutacion", str(indice)).permutation(len(real))
    x = pd.DataFrame(real.to_numpy()[orden], index=real.index, columns=real.columns)
    return x, pd.Series(n_t.to_numpy()[orden], index=real.index, name=n_t.name), orden


# ---------------------------------------------------------------------------
# B5: asimetría
# ---------------------------------------------------------------------------

def asimetria(F: np.ndarray, pares: list[tuple[str, str]]) -> float:
    """Media sobre los pares no ordenados de |F_ij - F_ji| / (F_ij + F_ji), 0 si ambos son 0."""
    valor = dict(zip(pares, F))
    columnas = list(dict.fromkeys(o for o, _ in pares))
    terminos = []
    for i, j in itertools.combinations(columnas, 2):
        f_ij, f_ji = valor[(i, j)], valor[(j, i)]
        suma = f_ij + f_ji
        terminos.append(0.0 if suma == 0 else abs(f_ij - f_ji) / suma)
    return float(np.mean(terminos))


def reciprocos(enlaces: set[tuple[str, str]]) -> float | None:
    """Fracción de enlaces declarados cuyo inverso también está declarado; None con menos de 2 enlaces."""
    if len(enlaces) < 2:
        return None
    return sum((d, o) in enlaces for o, d in enlaces) / len(enlaces)


# ---------------------------------------------------------------------------
# B6: transferencia de entropía
# ---------------------------------------------------------------------------

def simbolizar(x: np.ndarray) -> np.ndarray:
    """2 niveles por la mediana de la serie; los valores iguales a la mediana van al nivel bajo."""
    return (x > np.median(x)).astype(np.int64)


def te_conteo(fuente: np.ndarray, destino: np.ndarray) -> np.ndarray:
    """
    TE(fuente -> destino) en bits con k = l = 1 y estimador de conteo, para
    cada fila de `fuente` (R x n, o un vector):
        TE = sum p(y1, y0, x0) log2 [ p(y1 | y0, x0) / p(y1 | y0) ]
           = sum n/N log2 [ n * n(y0) / (n(y0, x0) * n(y1, y0)) ].
    """
    X = np.atleast_2d(fuente)
    R = X.shape[0]
    indice = 4 * destino[1:] + 2 * destino[:-1] + X[:, :-1]
    N = indice.shape[1]
    n = np.stack([(indice == k).sum(axis=1) for k in range(8)], axis=1).astype(float).reshape(R, 2, 2, 2)  # [y1, y0, x0]
    n_y0x0 = n.sum(axis=1)
    n_y1y0 = n.sum(axis=3)
    n_y0 = n.sum(axis=(1, 3))
    num = n * n_y0[:, None, :, None]
    den = n_y0x0[:, None, :, :] * n_y1y0[:, :, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        termino = np.where(n > 0, n / N * np.log2(num / den), 0.0)
    return termino.sum(axis=(1, 2, 3))


def te_lazo(fuente: np.ndarray, destino: np.ndarray) -> float:
    """La misma TE con probabilidades explícitas y un bucle, para verificar te_conteo."""
    trios = list(zip(destino[1:], destino[:-1], fuente[:-1]))
    N = len(trios)
    total = 0.0
    for y1, y0, x0 in set(trios):
        p_trio = trios.count((y1, y0, x0)) / N
        p_y0x0 = sum(1 for t in trios if t[1] == y0 and t[2] == x0) / N
        p_y1y0 = sum(1 for t in trios if t[0] == y1 and t[1] == y0) / N
        p_y0 = sum(1 for t in trios if t[1] == y0) / N
        total += p_trio * math.log2((p_trio / p_y0x0) / (p_y1y0 / p_y0))
    return total


def permutar_bloques(x: np.ndarray, largo: int, rng: np.random.Generator) -> np.ndarray:
    """Bloques consecutivos de `largo` desde el inicio (el último puede ser más corto), en orden permutado."""
    bloques = [x[i:i + largo] for i in range(0, len(x), largo)]
    return np.concatenate([bloques[i] for i in rng.permutation(len(bloques))])


def te_con_sustitutos(fuente: np.ndarray, destino: np.ndarray, n_sust: int, largo: int,
                      rng: np.random.Generator) -> dict[str, object]:
    """TE observada, sustitutos del origen por bloques, TE corregida y p = (1 + #{TE_s >= TE_obs}) / (n_sust + 1)."""
    te_obs = float(te_conteo(fuente, destino)[0])
    te_s = te_conteo(np.stack([permutar_bloques(fuente, largo, rng) for _ in range(n_sust)]), destino)
    return {"te_obs": te_obs, "te_sust": te_s, "te_media_sust": float(te_s.mean()), "te_corregida": te_obs - float(te_s.mean()),
            "p": (1 + int((te_s >= te_obs).sum())) / (n_sust + 1)}


# ---------------------------------------------------------------------------
# Réplicas en procesos independientes
# ---------------------------------------------------------------------------

_CONTEXTO: dict[str, object] = {}


def iniciar_trabajador(contexto: dict[str, object]) -> None:
    _CONTEXTO.clear()
    _CONTEXTO.update(contexto)


def replica_b2(ctx: dict[str, object], indice: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x = sustituto_b2(ctx, indice)
    fila, arr = analizar(x, ctx["n_t"], ctx["descartada"], ctx["pares"], ("pares",), ctx["B"],
                         lambda prueba, v, o, d: ("e1_b2_remuestreo", str(indice), v, o, d))
    return {"indice": indice, **fila}, arr


def replica_b3(ctx: dict[str, object], kappa: str, indice: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x = sustituto_b3(ctx, kappa, indice)
    fila, arr = analizar(x, ctx["n_t"], ctx["descartada"], ctx["pares"], PRUEBAS, ctx["B"],
                         lambda prueba, v, o, d: ("e1_b3_remuestreo", prueba, kappa, str(indice), v, o, d))
    return {"kappa": kappa, "indice": indice, **fila}, arr


def replica_b4(ctx: dict[str, object], indice: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x, n_perm, _ = permutacion_b4(ctx, indice)
    fila, arr = analizar(x, None, ctx["descartada"], ctx["pares"], PRUEBAS, ctx["B"],
                         lambda prueba, v, o, d: ("e1_b4_remuestreo", prueba, str(indice), v, o, d))
    dif = float(np.abs(np.expm1(x).sum(axis=1).to_numpy() - n_perm.to_numpy(dtype=float)).max())
    return {"indice": indice, "max_dif_masa_n_t": dif, **fila}, arr


def replica_b5(ctx: dict[str, object], indice: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Ruta F (sin remuestreo) de las dos pruebas sobre el sustituto de B2, y S por prueba y versión."""
    x = sustituto_b2(ctx, indice)
    _, arr = analizar(x, ctx["n_t"], ctx["descartada"], ctx["pares"], PRUEBAS, 0, None)
    fila: dict[str, object] = {"indice": indice}
    for prueba in PRUEBAS:
        for v in VERSIONES:
            F = arr[f"{prueba}_F_{v}"]
            fila[f"S_{prueba}_{v}"] = float("nan") if np.isnan(F).any() else asimetria(F, ctx["pares"][v])
    return fila, {k: a for k, a in arr.items() if "_F_" in k}


REPLICAS: dict[str, Callable[..., tuple[dict[str, object], dict[str, np.ndarray]]]] = {
    "B2": replica_b2, "B3": replica_b3, "B4": replica_b4, "B5": replica_b5}


def trabajo(tarea: tuple[str | int, ...]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    bloque, *argumentos = tarea
    return REPLICAS[bloque](_CONTEXTO, *argumentos)
