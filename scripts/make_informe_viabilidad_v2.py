"""
make_informe_viabilidad_v2.py

Semana 8, paso 5, parte A: regenera docs/viabilidad_granger_bpoil.md con los
trabajos posteriores al informe v1 (series largas, diagnóstico factorial,
factibilidad de grillas, candidatas, confirmación y estimación con la grilla
de 2 días, paquete de validez).

Las secciones 2 a 7 se conservan del v1: el script v1
(scripts/make_informe_viabilidad.py, sin modificarlo) se carga como módulo y
se ejecuta escribiendo en un archivo temporal; su propia verificación de
cifras debe pasar y su salida debe coincidir con la versión del informe del
commit 6e29f00. De ahí se toman, sin cambios, las secciones 2 a 7.

Cada cifra nueva pasa por cifra() (o fecha()), que la registra con su valor
de origen y su archivo; al final se verifica que cada una esté en el
documento y que su texto coincida con el origen a la precisión mostrada, y se
imprime la lista (cifra, valor, archivo). Antes de escribir se verifica con
git que cada commit de origen existe (git cat-file) y que el script
productor figura en él (git ls-tree). Si algo no coincide, no se escribe.
"""

import contextlib
import hashlib
import importlib.util
import io
import math
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import stats

from causal_coherence.config import OUTPUT_DIR, PROJECT_ROOT

SALIDA = PROJECT_ROOT / "docs" / "viabilidad_granger_bpoil.md"
V1 = PROJECT_ROOT / "scripts" / "make_informe_viabilidad.py"
COMMIT_V1 = "6e29f00"
Q = 0.05
VERSIONES = ["original", "diferenciada", "detendenciada", "residualizada"]
CONTROLES = ["propio", "compartido"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
MINIMOS_K = {3: 30, 5: 45, 10: 90}  # calculadora de poder (sección 13 de Bitacora_Semana7.md)
NOMBRE_CONTROL = {"C0": "C0 (propio)", "C1": "C1 (sin control)", "C2": "C2 (compartido)"}
# MD5 de data/factibilidad_corpus_nuevo.csv registrado en la sección 20 de Bitacora_Semana8.md.
MD5_FACTIBILIDAD_NUEVO = "256b67da229bab3b841ffc5f61552c33"

CPL, DFP, CAN, CON, RR2, VAL = ("control_positivo_largo", "diagnostico_fp", "candidatas", "confirmacion",
                                "resultados_reales_2d", "validez")
# Archivos nuevos respecto del v1: (script productor, commits).
ORIGENES_NUEVOS: dict[str, tuple[str, list[str]]] = {
    **{f"{CPL}/{a}": ("scripts/run_control_positivo_largo.py", ["8744f72"])
       for a in ["resumen_por_celda.csv", "resumen_por_version.csv", "comparacion.csv"]},
    **{f"{DFP}/{a}": ("scripts/diagnostico_falsos_positivos.py", ["0d9f838"])
       for a in ["resumen_etapa1_efectos_principales.csv", "resumen_criterio_a.csv", "resumen_etapa2_por_version.csv",
                 "resumen_etapa2_regla.csv", "resumen_criterio_b.csv"]},
    "factibilidad_corpus.csv": ("scripts/factibilidad_corpus.py", ["be30272"]),
    "factibilidad_corpus_nuevo.csv": ("scripts/factibilidad_corpus.py", ["be30272"]),
    "factibilidad_grillas.csv": ("scripts/factibilidad_grillas.py", ["1282e81"]),
    **{f"{CAN}/{a}": ("scripts/candidatas_grillas.py", ["d3b0c49", "077cc8a"])
       for a in ["resumen_aptitud.csv", "resumen_c_calibracion.csv", "2D_K10_C0_e0.8.csv"]},
    **{f"{CON}/{a}": ("scripts/confirmacion_y_real.py", ["15e5ed7"])
       for a in ["resumen_decision.csv", "resumen_deteccion.csv", "resumen_falsos_positivos.csv",
                 "resumen_combinado_informativo.csv"]},
    **{f"{RR2}/{a}": ("scripts/confirmacion_y_real.py", ["15e5ed7"])
       for a in ["resultados.csv", "por_version.csv", "rutas.csv", "comparacion_resumen.csv", "enlaces_propio.csv",
                 "enlaces_compartido.csv"]},
    **{f"{VAL}/{a}": ("scripts/paquete_validez.py", ["ba24e7a"])
       for a in ["A1_estacionariedad.csv", "A2_pca_varianza.csv", "A3_documentos.csv", "A3_spearman.csv",
                 "B_calibracion.csv", "B_criterio.csv", "B_sustitutos.csv", "C_resumen.csv", "C_traslape.csv",
                 "D_resumen.csv"]},
}

REGISTRO: list[dict[str, object]] = []


def cifra(nombre: str, valor: float | int, archivo: str, decimales: int | None = None, pct: bool = False) -> str:
    """Formatea un valor de origen, lo registra y devuelve su texto."""
    mostrado = 100 * float(valor) if pct else float(valor)
    if decimales is None:
        texto = str(int(round(mostrado)))
        assert float(texto) == mostrado, f"{nombre}: {mostrado} no es entero"
    else:
        texto = f"{mostrado:.{decimales}f}"
    REGISTRO.append({"cifra": nombre, "valor": texto, "origen": mostrado, "archivo": archivo, "decimales": decimales})
    return texto


def fecha_es(iso: str) -> str:
    d = pd.Timestamp(iso)
    return f"{d.day} de {MESES[d.month - 1]} de {d.year}"


def fecha(nombre: str, iso: str, archivo: str) -> str:
    """Fecha ISO de origen escrita en castellano; se registra como cifra."""
    texto = fecha_es(iso)
    REGISTRO.append({"cifra": nombre, "valor": texto, "origen": iso, "archivo": archivo, "decimales": "fecha"})
    return texto


def tabla_md(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "|".join("---" for _ in encabezados) + "|"]
    lineas += ["| " + " | ".join(f) + " |" for f in filas]
    return "\n".join(lineas)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")


def verificar_origenes(origenes: dict[str, tuple[str, list[str]]]) -> bool:
    ok = True
    for archivo, (script, commits) in origenes.items():
        if not (OUTPUT_DIR / archivo).exists():
            print(f"  FALTA el archivo data/{archivo}")
            ok = False
        for commit in commits:
            existe = git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
            en_commit = existe and script in git("ls-tree", "-r", "--name-only", commit).stdout.splitlines()
            if not (existe and en_commit):
                print(f"  NO COINCIDE: data/{archivo}: commit {commit} existe={existe}, {script} en el commit={en_commit}")
                ok = False
    return ok


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def leer(archivo: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / archivo, float_precision="round_trip")


def requisito(m: int, df_den: int) -> tuple[float, float]:
    """F_req = f.isf(q/m, 1, df_den) y efecto_req = ln(1 + F_req / df_den), como en el v1."""
    F_req = float(stats.f.isf(Q / m, 1, df_den))
    return F_req, math.log(1 + F_req / df_den)


def k_que_alcanzan(obs: int) -> str:
    ks = [str(k) for k, minimo in MINIMOS_K.items() if obs >= minimo]
    return ",".join(ks) if ks else "ninguno"


def cargar_v1() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_informe_viabilidad_v1", V1)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def secciones_v1(v1: ModuleType) -> tuple[str, int]:
    """Corre el v1 hacia un archivo temporal y devuelve sus secciones 2 a 7 y la cantidad de cifras que verificó."""
    with tempfile.TemporaryDirectory() as tmp:
        v1.SALIDA = Path(tmp) / "informe_v1.md"
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            v1.main()
        texto_salida = salida.getvalue()
        assert "Todas las cifras coinciden" in texto_salida, "la verificación de cifras del v1 no pasó:\n" + texto_salida
        documento = v1.SALIDA.read_text(encoding="utf-8")
    comprometido = git("show", f"{COMMIT_V1}:docs/viabilidad_granger_bpoil.md").stdout
    assert documento.replace("\r\n", "\n") == comprometido.replace("\r\n", "\n"), \
        f"el v1 regenerado no coincide con docs/viabilidad_granger_bpoil.md del commit {COMMIT_V1}"
    i2, i8 = documento.index("## 2. Configuración fijada"), documento.index("## 8. ")
    n_ok = sum(1 for linea in texto_salida.splitlines() if linea.strip().startswith("OK "))
    return documento[i2:i8].rstrip("\n") + "\n", n_ok


def main() -> None:
    v1 = cargar_v1()
    origenes = {**v1.ORIGENES, **ORIGENES_NUEVOS}
    print("Orígenes (git cat-file y git ls-tree):")
    if not verificar_origenes(origenes):
        print("Algún origen no coincide: no se escribe el documento.")
        return
    print(f"  todos coinciden ({len(origenes)} archivos)")
    assert md5(OUTPUT_DIR / "factibilidad_corpus_nuevo.csv") == MD5_FACTIBILIDAD_NUEVO, \
        "data/factibilidad_corpus_nuevo.csv cambió respecto del MD5 registrado en la sección 20"

    print("Secciones 2 a 7: se regeneran con el v1 y se comparan con el commit", COMMIT_V1)
    bloque_2_7, n_v1 = secciones_v1(v1)
    print(f"  coinciden; el v1 verificó {n_v1} cifras")

    # Lecturas.
    TS = "data/topic_series_volumen.csv"
    CPT = "data/control_positivo/resumen_por_tamano.csv"
    T_sem = len(pd.read_csv(OUTPUT_DIR / "topic_series_volumen.csv"))
    cp_tam = leer("control_positivo/resumen_por_tamano.csv")
    cpl_celda, cpl_ver, cpl_comp = (leer(f"{CPL}/{a}") for a in
                                    ["resumen_por_celda.csv", "resumen_por_version.csv", "comparacion.csv"])
    fp_ef, fp_a, fp_e2, fp_regla, fp_b = (leer(f"{DFP}/{a}") for a in
                                          ["resumen_etapa1_efectos_principales.csv", "resumen_criterio_a.csv",
                                           "resumen_etapa2_por_version.csv", "resumen_etapa2_regla.csv",
                                           "resumen_criterio_b.csv"])
    fc_viejo, fc_nuevo, fg = leer("factibilidad_corpus.csv"), leer("factibilidad_corpus_nuevo.csv"), leer("factibilidad_grillas.csv")
    apt, cal = leer(f"{CAN}/resumen_aptitud.csv"), leer(f"{CAN}/resumen_c_calibracion.csv")
    n_cp_cand = len(leer(f"{CAN}/2D_K10_C0_e0.8.csv"))
    dec, det, fpc, comb = (leer(f"{CON}/{a}") for a in ["resumen_decision.csv", "resumen_deteccion.csv",
                                                          "resumen_falsos_positivos.csv", "resumen_combinado_informativo.csv"])
    rr2, pv, rutas2, comp2 = (leer(f"{RR2}/{a}") for a in ["resultados.csv", "por_version.csv", "rutas.csv",
                                                            "comparacion_resumen.csv"])
    enl = {c: leer(f"{RR2}/enlaces_{c}.csv") for c in CONTROLES}
    a1, a2, a3d, a3s, bcal, bcri, cres, ctra, dres = (leer(f"{VAL}/{a}") for a in
                                                       ["A1_estacionariedad.csv", "A2_pca_varianza.csv", "A3_documentos.csv",
                                                        "A3_spearman.csv", "B_calibracion.csv", "B_criterio.csv",
                                                        "C_resumen.csv", "C_traslape.csv", "D_resumen.csv"])
    F_CPC, F_CPV, F_CPA = (f"data/{CPL}/{a}" for a in ["resumen_por_celda.csv", "resumen_por_version.csv", "comparacion.csv"])
    F_FPE, F_FPA, F_FP2, F_FPR, F_FPB = (f"data/{DFP}/{a}" for a in
                                         ["resumen_etapa1_efectos_principales.csv", "resumen_criterio_a.csv",
                                          "resumen_etapa2_por_version.csv", "resumen_etapa2_regla.csv", "resumen_criterio_b.csv"])
    F_FCV, F_FCN, F_FG = "data/factibilidad_corpus.csv", "data/factibilidad_corpus_nuevo.csv", "data/factibilidad_grillas.csv"
    F_APT, F_CAL, F_CPR = f"data/{CAN}/resumen_aptitud.csv", f"data/{CAN}/resumen_c_calibracion.csv", f"data/{CAN}/2D_K10_C0_e0.8.csv"
    F_DEC, F_DET, F_FPC, F_COMB = (f"data/{CON}/{a}" for a in ["resumen_decision.csv", "resumen_deteccion.csv",
                                                                 "resumen_falsos_positivos.csv", "resumen_combinado_informativo.csv"])
    F_RR2, F_PV, F_RU2, F_CMP = (f"data/{RR2}/{a}" for a in ["resultados.csv", "por_version.csv", "rutas.csv",
                                                              "comparacion_resumen.csv"])
    F_A1, F_A2, F_A3D, F_A3S, F_BCAL, F_BCRI, F_CRES, F_CTRA, F_DRES = (
        f"data/{VAL}/{a}" for a in ["A1_estacionariedad.csv", "A2_pca_varianza.csv", "A3_documentos.csv", "A3_spearman.csv",
                                    "B_calibracion.csv", "B_criterio.csv", "C_resumen.csv", "C_traslape.csv", "D_resumen.csv"])

    # Valores del resumen y comprobaciones de los textos fijos.
    fila_2d = fg[(fg["corpus"] == "viejo") & (fg["grilla"] == "cada 2 días") & (fg["min_docs"] == 5)].iloc[0]
    orig_2d = rr2[rr2["version"] == "original"]
    n_obs_2d = int(orig_2d["n_obs"].iloc[0])  # versión original; la diferenciada pierde una fila
    assert (orig_2d["n_obs"] == n_obs_2d).all() and n_obs_2d == int(fila_2d["obs_utilizables"])
    assert int(rr2["rechaza_BH"].sum()) == 0 and int(pv["rechazos_BH"].sum()) == 0, "hay rechazos en la estimación de 2 días"
    assert all(len(e) == 0 for e in enl.values()), "hay enlaces declarados en la estimación de 2 días"
    dec_f = dec.iloc[0]
    assert dec_f["adoptada"] == "2D_K10"
    regla_k10 = det[(det["candidata"] == "2D_K10") & (det["version"] == "regla_2_versiones")].set_index("efecto")
    assert abs(regla_k10.loc[0.8, "deteccion_BH"] - dec_f["deteccion_regla_0.8_2D_K10"]) < 1e-12
    fp_vals = bcal[bcal["version"] != "regla_2_versiones"]["fraccion"]
    # Efecto necesario por versión (igual en los dos controles) y mayor efecto observado por versión (de los dos controles);
    # la comparación principal es la de la versión con menor efecto necesario.
    req_v = pv.groupby("version")["efecto_req"].agg(["min", "max"])
    assert (req_v["max"] - req_v["min"]).max() < 1e-12
    req_v = req_v["min"].to_dict()
    obs_v = pv.groupby("version")["efecto_max"].max().to_dict()
    v_comp = min(req_v, key=req_v.get)
    assert obs_v[v_comp] == pv["efecto_max"].max(), "la versión de menor efecto necesario no tiene el mayor efecto observado"
    n_fam = int(pv.groupby(["control_tipo", "version"]).ngroups)
    sens_k10 = cres[(cres["estimada"]) & (cres["K"] == 10)]
    assert int(sens_k10["rechazos_BH"].sum()) == 0 and int(sens_k10["enlaces_regla"].sum()) == 0
    con_enlace = cres[cres["enlaces_regla"].fillna(0) > 0][["sensibilidad", "control_tipo"]].drop_duplicates()
    assert con_enlace.values.tolist() == [["C3_3D_K5", "compartido"]], con_enlace
    assert not bool(apt[(apt["candidata"] == "3D_K5") & (apt["control"] == "C2")]["apta"].iloc[0])
    assert len(apt) == 15

    resumen = (
        f"Con bpoil a resolución semanal ({cifra('T_sem: ventanas de la serie semanal', T_sem, TS)} ventanas) el "
        "procedimiento de Granger condicional no tiene poder: un enlace plantado que elimina el "
        f"{cifra('reducción del error con efecto 0.8 (%)', 1 - math.exp(-0.8), 'fórmula 1 - exp(-0.8)', 0, True)}% del error "
        "de predicción se detecta en "
        f"{cifra('det_sem: control positivo semanal, efecto 0.8, regla de 2 versiones (%)', cp_tam[cp_tam['efecto'] == 0.8]['tasa_deteccion_regla_2_versiones'].iloc[0], CPT, 0, True)}% "
        "de las réplicas. Con una grilla de 2 días "
        f"({cifra('n_ventanas: racha de 2 días', fila_2d['racha_max'], F_FG)} ventanas consecutivas, del "
        f"{fecha('fecha_ini: inicio de la racha de 2 días', fila_2d['racha_inicio'], F_FG)} al "
        f"{fecha('fecha_fin: fin de la racha de 2 días', fila_2d['racha_fin'], F_FG)}, "
        f"{cifra('n_obs: observaciones por regresión, 2 días', n_obs_2d, F_RR2)} observaciones) y los "
        f"{cifra('K de la estimación de 2 días', len(set(rr2['origen']) | set(rr2['destino'])), F_RR2)} tópicos, el poder "
        f"confirmado es de {cifra('det_08: confirmación 2D_K10, efecto 0.8, regla (%)', regla_k10.loc[0.8, 'deteccion_BH'], F_DET, 0, True)}% "
        "para un enlace de efecto 0.8 y de "
        f"{cifra('det_04: confirmación 2D_K10, efecto 0.4, regla (%)', regla_k10.loc[0.4, 'deteccion_BH'], F_DET, 0, True)}% "
        "para uno de efecto 0.4, con falsos positivos de "
        f"{cifra('fp_min: menor fracción con rechazo, bloque B, 2 controles y 4 versiones (%)', fp_vals.min(), F_BCAL, 1, True)}% a "
        f"{cifra('fp_max: mayor fracción con rechazo, bloque B, 2 controles y 4 versiones (%)', fp_vals.max(), F_BCAL, 1, True)}% "
        "en sustitutos con volumen exógeno. La estimación real no rechazó ninguna hipótesis en las "
        f"{cifra('n_fam: familias (control x versión), 2 días', n_fam, F_PV)} familias (p mínimo "
        f"{cifra('pmin_2d: menor p_remuestreo, 2 días, ambos controles', pv['p_remuestreo_min'].min(), F_PV, 4)}; efecto necesario "
        f"{cifra(f'ef_nec: efecto_req, {v_comp}', req_v[v_comp], F_PV, 2)} frente a un máximo observado de "
        f"{cifra(f'ef_obs: mayor efecto observado, {v_comp}, ambos controles', obs_v[v_comp], F_PV, 2)} en la {v_comp}; en las "
        "otras tres versiones, "
        + ", ".join(f"{v} {cifra(f'efecto_req, {v}', req_v[v], F_PV, 2)} frente a "
                    f"{cifra(f'mayor efecto observado, {v}, ambos controles', obs_v[v], F_PV, 2)}"
                    for v in VERSIONES if v != v_comp)
        + "). El resultado nulo es "
        "informativo para enlaces que eliminan más del 50% del error y no descarta efectos débiles. Las sensibilidades con K = 10 (retardo 2, grilla de 3 días, recortes) no "
        "rechazan nada; la única configuración con un enlace declarado (3 días, K = 5, control compartido) no fue apta en "
        "la calibración. Se reportan las dos estimaciones sobre bpoil y las "
        f"{cifra('configuraciones evaluadas en candidatas', len(apt), F_APT)} configuraciones evaluadas."
    )

    partes: list[str] = []
    partes.append("# Viabilidad de la prueba de Granger condicional sobre bpoil\n")
    partes.append("Documento generado por `scripts/make_informe_viabilidad_v2.py` a partir de los CSV de `data/`; todas las "
                  "cifras salen de esos archivos o de fórmulas calculadas en el script. Las secciones 2 a 7 corresponden a "
                  "la estimación semanal y se conservan sin cambios de la versión anterior del informe "
                  f"(`scripts/make_informe_viabilidad.py`, commit `{COMMIT_V1}`); las secciones 8 a 13 agregan los trabajos "
                  "posteriores, hasta la estimación con la grilla de 2 días y su paquete de validez.\n")
    partes.append("## 1. Resumen\n")
    partes.append(resumen + "\n")
    partes.append(bloque_2_7)

    # 8. Series largas.
    partes.append("## 8. Control positivo con series largas\n")
    n_rep_cpl = int(cpl_celda["n_replicas"].iloc[0])
    assert (cpl_celda["n_replicas"] == n_rep_cpl).all()
    partes.append("Pipeline completo de la estimación semanal (control propio, piso en 0, remuestreo salvaje, Benjamini y "
                  "Hochberg por versión y regla de 2 versiones) con la dinámica semanal de bpoil y series más largas, con "
                  f"{cifra('réplicas por celda, series largas', n_rep_cpl, F_CPC)} réplicas por celda. Detección del par "
                  "plantado con la regla de 2 versiones:\n")
    filas = [[cifra(f"largo T {r.T} {r.efecto}", r.T, F_CPC), f"{r.efecto:.1f}",
              cifra(f"largo detección regla T {r.T} efecto {r.efecto}", r.tasa_deteccion_regla_2_versiones, F_CPC, 2),
              cifra(f"largo no estimables T {r.T} efecto {r.efecto}", r.n_replicas_no_estimables, F_CPC)]
             for r in cpl_celda.itertuples(index=False)]
    partes.append(tabla_md(["T", "Efecto", "Detección (regla)", "Réplicas no estimables"], filas) + "\n")
    partes.append("Comparación con el pipeline a T = 23 (sección 5, regla de 2 versiones) y con el banco gaussiano de A6 "
                  "(sección 7, remuestreo):\n")
    filas = [[f"{r.efecto:.1f}"] + [cifra(f"comparación {col} efecto {r.efecto}", getattr(r, col), F_CPA, 3)
                                    for col in ["pipeline_T23_regla", "pipeline_T46_regla", "pipeline_T92_regla",
                                                "A6_K10_T23_remuestreo", "A6_K10_T46_remuestreo", "A6_K10_T92_remuestreo"]]
             for r in cpl_comp.itertuples(index=False)]
    partes.append(tabla_md(["Efecto", "Pipeline T = 23", "Pipeline T = 46", "Pipeline T = 92", "A6 T = 23", "A6 T = 46",
                            "A6 T = 92"], filas) + "\n")
    partes.append("Fracción de réplicas con al menos un rechazo sin enlace plantado (efecto 0.0), por versión; esta "
                  "fracción, por encima del 5% nominal, motivó el diagnóstico de la sección 9:\n")
    nulos = cpl_ver[cpl_ver["efecto"] == 0.0]
    filas = [[cifra(f"largo nulo T {T_}", T_, F_CPV)] +
             [cifra(f"largo nulo T {T_} {v}", nulos[(nulos["T"] == T_) & (nulos["version"] == v)]["fraccion_con_al_menos_un_rechazo"].iloc[0], F_CPV, 2)
              for v in VERSIONES]
             for T_ in sorted(nulos["T"].unique())]
    partes.append(tabla_md(["T", *[v.capitalize() for v in VERSIONES]], filas) + "\n")

    # 9. Diagnóstico factorial.
    partes.append("## 9. Diagnóstico factorial de falsos positivos\n")
    partes.append("Sin enlace plantado, T = 92 y K = 10, se cruzaron tres factores: C, control de volumen (C0 propio, el "
                  "actual; C1 sin control; C2 control exógeno fijo); P, piso en 0 (P0 aplicado, P1 sin piso); I, "
                  "innovaciones (I0 filas reales, I1 gaussianas). Etapa 1 (ruta F): fracción media de réplicas con algún "
                  "rechazo en la versión original por nivel de cada factor; un factor está implicado si su nivel limpio "
                  "reduce esa fracción en al menos 8 puntos:\n")
    filas = []
    for r in fp_ef.itertuples(index=False):
        a = fp_a[(fp_a["factor"] == r.factor) & (fp_a["nivel_limpio"] == r.nivel)]
        filas.append([r.factor, r.nivel, cifra(f"factorial fracción {r.nivel}", r.fraccion_media_original, F_FPE, 3),
                      cifra(f"factorial reducción {r.nivel}", a["reduccion"].iloc[0], F_FPA, 3) if len(a) else "nivel actual",
                      ("sí" if bool(a["implicado"].iloc[0]) else "no") if len(a) else ""])
    partes.append(tabla_md(["Factor", "Nivel", "Fracción media (original)", "Reducción respecto del nivel actual", "Implicado"],
                           filas) + "\n")
    umbral_b = stats.binom.isf(0.05 / 8, 100, 0.05)
    assert (fp_b["umbral"] == umbral_b).all()
    partes.append("Etapa 2 (remuestreo, 100 réplicas por variante): réplicas con algún rechazo en las versiones original y "
                  "detendenciada; la variante restaura la calibración si ambas quedan en o bajo el umbral "
                  f"binom.isf(0.05 / 8, 100, 0.05) = {cifra('umbral etapa 2', umbral_b, 'fórmula binom.isf(0.05 / 8, 100, 0.05)')}:\n")
    filas = []
    for r in fp_b.itertuples(index=False):
        g = fp_regla[fp_regla["variante"] == r.variante].iloc[0]
        filas.append([r.variante, cifra(f"etapa 2 original {r.variante}", r.replicas_con_rechazo_original, F_FPB),
                      cifra(f"etapa 2 detendenciada {r.variante}", r.replicas_con_rechazo_detendenciada, F_FPB),
                      cifra(f"etapa 2 regla {r.variante}", g["fraccion_con_enlace_regla_rem"], F_FPR, 2),
                      "sí" if r.restaura_calibracion else "no"])
        e2 = fp_e2[(fp_e2["variante"] == r.variante) & (fp_e2["version"].isin(["original", "detendenciada"]))]
        assert e2["replicas_con_rechazo_rem"].tolist() == [r.replicas_con_rechazo_original, r.replicas_con_rechazo_detendenciada]
    partes.append(tabla_md(["Variante", "Réplicas con rechazo (original)", "Réplicas con rechazo (detendenciada)",
                            "Fracción con algún enlace (regla)", "Restaura la calibración"], filas) + "\n")
    partes.append("El factor implicado es el control de volumen (C): las dos variantes sin control (C1-P0-I0 y C1-P1-I1) "
                  "restauran la calibración y las tres con el control propio no.\n")

    # 10. Factibilidad de grillas.
    partes.append("## 10. Factibilidad de grillas\n")
    g5 = fg[fg["min_docs"] == 5]
    docs = {c: int(g5[g5["corpus"] == c]["documentos"].iloc[0]) for c in ["viejo", "nuevo"]}
    partes.append("Ventana de análisis de bpoil, con al menos 5 documentos por ventana; la serie de análisis es la mayor "
                  "racha de ventanas válidas consecutivas (sin imputar), y las observaciones utilizables con retardo 1 son "
                  f"el largo de la racha menos 1. Corpus viejo ({cifra('documentos corpus viejo', docs['viejo'], F_FG)} "
                  f"documentos) y nuevo ({cifra('documentos corpus nuevo', docs['nuevo'], F_FG)} documentos). Mínimos de la "
                  f"calculadora de poder: {MINIMOS_K[3]} observaciones para K = 3, {MINIMOS_K[5]} para K = 5 y "
                  f"{MINIMOS_K[10]} para K = 10.\n")
    filas = []
    for grilla in ["semanal", "cada 2 días", "cada 3 días", "cada 4 días", "cada 5 días"]:
        v_, n_ = (g5[(g5["corpus"] == c) & (g5["grilla"] == grilla)].iloc[0] for c in ["viejo", "nuevo"])
        for f_ in (v_, n_):
            assert f_["obs_utilizables"] == f_["racha_max"] - 1 and f_["K_que_alcanzan"] == k_que_alcanzan(int(f_["obs_utilizables"]))
        filas.append([grilla, cifra(f"grilla {grilla} ventanas válidas viejo", v_["validas_despues"], F_FG),
                      cifra(f"grilla {grilla} racha viejo", v_["racha_max"], F_FG),
                      cifra(f"grilla {grilla} observaciones viejo", v_["obs_utilizables"], F_FG),
                      f"{fecha(f'grilla {grilla} inicio viejo', v_['racha_inicio'], F_FG)} al "
                      f"{fecha(f'grilla {grilla} fin viejo', v_['racha_fin'], F_FG)}",
                      cifra(f"grilla {grilla} racha nuevo", n_["racha_max"], F_FG),
                      cifra(f"grilla {grilla} observaciones nuevo", n_["obs_utilizables"], F_FG),
                      v_["K_que_alcanzan"].replace(",", ", ") + ("" if v_["K_que_alcanzan"] == n_["K_que_alcanzan"]
                                                                  else f" (nuevo: {n_['K_que_alcanzan']})")])
    dia = {}
    for nombre, t, archivo in [("viejo", fc_viejo, F_FCV), ("nuevo", fc_nuevo, F_FCN)]:
        dia[nombre] = t[(t["subconjunto"] == "NEWS-TLS T17 (bpoil)") & (t["rango"] == "ventana de análisis") &
                        (t["grilla"] == "diaria")].iloc[0]
        assert int(dia[nombre]["documentos"]) == docs[nombre]
        for grilla in ["semanal", "cada 3 días"]:  # mismo cálculo de racha que factibilidad_grillas.csv
            fila_c = t[(t["subconjunto"] == "NEWS-TLS T17 (bpoil)") & (t["rango"] == "ventana de análisis") & (t["grilla"] == grilla)].iloc[0]
            assert fila_c["racha_max_validas"] == g5[(g5["corpus"] == nombre) & (g5["grilla"] == grilla)]["racha_max"].iloc[0]
    k_dia = {c: k_que_alcanzan(int(dia[c]["racha_max_validas"]) - 1) for c in dia}
    filas.append(["diaria", cifra("diaria ventanas válidas viejo", dia["viejo"]["validas_despues"], F_FCV),
                  cifra("diaria racha viejo", dia["viejo"]["racha_max_validas"], F_FCV),
                  cifra("diaria observaciones viejo (racha - 1)", dia["viejo"]["racha_max_validas"] - 1, F_FCV), "",
                  cifra("diaria racha nuevo", dia["nuevo"]["racha_max_validas"], F_FCN),
                  cifra("diaria observaciones nuevo (racha - 1)", dia["nuevo"]["racha_max_validas"] - 1, F_FCN),
                  k_dia["viejo"] + ("" if k_dia["viejo"] == k_dia["nuevo"] else f" (nuevo: {k_dia['nuevo']})")])
    partes.append(tabla_md(["Grilla", "Ventanas válidas (viejo)", "Racha (viejo)", "Observaciones (viejo)",
                            "Período de la racha (viejo)", "Racha (nuevo)", "Observaciones (nuevo)", "K que alcanzan el mínimo"],
                           filas) + "\n")
    partes.append("La fila diaria sale de `data/factibilidad_corpus.csv` y `data/factibilidad_corpus_nuevo.csv`; las demás, "
                  "de `data/factibilidad_grillas.csv`. Ninguna grilla alcanza el mínimo de K = 10; la de 2 días es la única "
                  "que alcanza K = 5. Las rachas de 2 a 5 días terminan en agosto: la cola de agosto y septiembre queda fuera.\n")

    # 11. Candidatas.
    partes.append("## 11. Configuraciones candidatas\n")
    umbral_c = stats.binom.isf(0.05 / 15, 300, 0.05)
    assert (apt["umbral_fp"] == umbral_c).all()
    n_cal = int(cal["n_replicas"].iloc[0])
    assert (cal["n_replicas"] == n_cal).all()
    for r in apt.itertuples(index=False):
        apta = r._3 >= 0.80 and r.fp_original <= umbral_c and r.fp_detendenciada <= umbral_c
        assert apta == bool(r.apta), r
    partes.append(f"{cifra('combinaciones candidatas', len(apt), F_APT)} combinaciones (5 candidatas de grilla y K por 3 "
                  "controles: C0 propio, C1 sin control, C2 compartido con el conteo real de documentos), evaluadas con "
                  "sustitutos AR(1) diagonales con la dinámica de cada serie: "
                  f"{cifra('réplicas de control positivo por efecto, candidatas', n_cp_cand, F_CPR)} réplicas de control "
                  f"positivo por efecto y {cifra('sustitutos sin enlace, candidatas', n_cal, F_CAL)} sustitutos sin enlace. "
                  "Una combinación es APTA si la detección con la regla de 2 versiones a efecto 0.8 es de al menos 0.80 y las "
                  "réplicas sin enlace con algún rechazo en la original y la detendenciada quedan en o bajo "
                  f"binom.isf(0.05 / 15, 300, 0.05) = {cifra('umbral de aptitud', umbral_c, 'fórmula binom.isf(0.05 / 15, 300, 0.05)')}.\n")
    filas = [[r.candidata, cifra(f"candidata K {r.candidata} {r.control}", r.K, F_APT), NOMBRE_CONTROL[r.control],
              cifra(f"candidata det 0.8 {r.candidata} {r.control}", r._3, F_APT, 2),
              cifra(f"candidata det 0.4 {r.candidata} {r.control}", r._4, F_APT, 2),
              cifra(f"candidata FP original {r.candidata} {r.control}", r.fp_original, F_APT),
              cifra(f"candidata FP detendenciada {r.candidata} {r.control}", r.fp_detendenciada, F_APT),
              "sí" if r.apta else "no"]
             for r in apt.itertuples(index=False)]
    partes.append(tabla_md(["Candidata", "K", "Control", "Detección 0.8 (regla)", "Detección 0.4 (regla)",
                            "Réplicas con rechazo (original)", "Réplicas con rechazo (detendenciada)", "APTA"], filas) + "\n")
    aptas = apt[apt["apta"]]
    elegida = aptas[aptas["K"] == aptas["K"].max()].sort_values("deteccion_regla_0.4", ascending=False, kind="stable").iloc[0]
    partes.append(f"Regla de elección fijada antes de correr: entre las APTAS, la de mayor K y, en empate, la de mayor "
                  f"detección con efecto 0.4. Resultado: {elegida['candidata']} con "
                  f"{NOMBRE_CONTROL[elegida['control']]}. La segunda estimación se fijó con la grilla de 2 días, K = 10 y "
                  "el control propio como principal (compartido como sensibilidad); el control C1 quedó excluido porque la "
                  "guía exige controlar el volumen.\n")

    # 12. Confirmación y estimación real.
    partes.append("## 12. Confirmación de poder y estimación real (2 días, K = 10)\n")
    n_conf = int(det[det["candidata"] == "2D_K10"]["replicas"].iloc[0])
    partes.append(f"Confirmación con {cifra('réplicas de confirmación por efecto', n_conf, F_DET)} réplicas nuevas por "
                  "efecto, control propio. Detección del par plantado por Benjamini y Hochberg en cada versión y con la "
                  "regla de 2 versiones, y réplicas sin enlace con algún rechazo:\n")
    filas = []
    for v in VERSIONES + ["regla_2_versiones"]:
        d = det[(det["candidata"] == "2D_K10") & (det["version"] == v)].set_index("efecto")
        f_ = fpc[(fpc["candidata"] == "2D_K10") & (fpc["version"] == v)].iloc[0]
        filas.append([v.replace("_", " "), cifra(f"confirmación det 0.8 {v}", d.loc[0.8, "deteccion_BH"], F_DET, 3),
                      cifra(f"confirmación det 0.4 {v}", d.loc[0.4, "deteccion_BH"], F_DET, 3),
                      cifra(f"confirmación efecto observado 0.8 {v}", d.loc[0.8, "efecto_observado_medio"], F_DET, 3)
                      if v in VERSIONES else "",
                      cifra(f"confirmación FP {v}", f_["replicas_con_rechazo"], F_FPC),
                      cifra(f"confirmación FP fracción {v}", f_["fraccion"], F_FPC, 3)])
    partes.append(tabla_md(["Versión", "Detección 0.8", "Detección 0.4", "Efecto observado medio (0.8)",
                            f"Réplicas sin enlace con rechazo (de {n_conf})", "Fracción"], filas) + "\n")
    fp8 = fpc[fpc["candidata"] == "2D_K8"].set_index("version")
    partes.append("Regla de respaldo fijada: se adopta K = 10 si la detección confirmada con efecto 0.8 (regla) es de al "
                  f"menos {cifra('umbral de respaldo', dec_f['umbral_respaldo'], F_DEC, 2)}. Resultado: "
                  f"{cifra('detección confirmada 0.8 K10', dec_f['deteccion_regla_0.8_2D_K10'], F_DEC, 3)}; se adopta "
                  f"{dec_f['adoptada']}, con {cifra('FP original K10 (decisión)', dec_f['fp_original'], F_DEC)} y "
                  f"{cifra('FP detendenciada K10 (decisión)', dec_f['fp_detendenciada'], F_DEC)} réplicas con rechazo "
                  f"(umbral {cifra('umbral FP (decisión)', dec_f['umbral_fp'], F_DEC)}). La configuración de respaldo "
                  f"(K = 8, control propio) habría tenido {cifra('FP original K8', fp8.loc['original', 'replicas_con_rechazo'], F_FPC)} "
                  f"y {cifra('FP detendenciada K8', fp8.loc['detendenciada', 'replicas_con_rechazo'], F_FPC)}. Combinando "
                  f"con las {cifra('réplicas de la sección 11 en el combinado', comb['replicas_seccion_21'].iloc[0], F_COMB)} "
                  "réplicas de la sección 11 (solo informativo), la detección con efecto 0.8 es "
                  f"{cifra('combinada 0.8 K10', comb[(comb['candidata'] == '2D_K10') & (comb['efecto'] == 0.8)]['deteccion_combinada'].iloc[0], F_COMB, 3)} "
                  f"y con 0.4, {cifra('combinada 0.4 K10', comb[(comb['candidata'] == '2D_K10') & (comb['efecto'] == 0.4)]['deteccion_combinada'].iloc[0], F_COMB, 3)}.\n")
    partes.append(f"Estimación real, ejecutada una sola vez (B = {cifra('B 2 días', rr2['B'].iloc[0], F_RR2)}, q = 0.05 por "
                  "versión). Efecto y F necesarios para que el menor p quede por debajo del primer umbral de Benjamini y "
                  "Hochberg, con las mismas fórmulas de la sección 3:\n")
    filas = []
    for r in pv.itertuples(index=False):
        F_req, e_req = requisito(int(r.m), int(r.df_den))
        assert abs(F_req - r.F_req) <= 1e-9 * F_req and abs(e_req - r.efecto_req) <= 1e-9 * e_req
        g = rr2[(rr2["control_tipo"] == r.control_tipo) & (rr2["version"] == r.version)]
        assert len(g) == r.m and int(g["rechaza_BH"].sum()) == r.rechazos_BH and g["p_remuestreo"].min() == r.p_remuestreo_min
        c_, v_ = r.control_tipo, r.version
        filas.append([c_, v_, cifra(f"2d m {c_} {v_}", r.m, F_PV), cifra(f"2d rechazos {c_} {v_}", r.rechazos_BH, F_PV),
                      cifra(f"2d p mínimo {c_} {v_}", r.p_remuestreo_min, F_PV, 4),
                      cifra(f"2d efecto máximo {c_} {v_}", r.efecto_max, F_PV, 3), cifra(f"2d F máximo {c_} {v_}", r.F_max, F_PV, 2),
                      cifra(f"2d df_den {c_} {v_}", r.df_den, F_PV), cifra(f"2d F_req {c_} {v_}", r.F_req, F_PV, 2),
                      cifra(f"2d efecto_req {c_} {v_}", r.efecto_req, F_PV, 3)])
    partes.append(tabla_md(["Control", "Versión", "m", "Rechazos BH", "p mínimo", "Mayor efecto", "Mayor F", "df_den", "F_req",
                            "efecto_req"], filas) + "\n")
    partes.append(f"Enlaces declarados por la regla de 2 versiones: {cifra('enlaces propio 2d', len(enl['propio']), f'data/{RR2}/enlaces_propio.csv')} "
                  f"con el control propio y {cifra('enlaces compartido 2d', len(enl['compartido']), f'data/{RR2}/enlaces_compartido.csv')} "
                  "con el compartido. Comprobación de las dos rutas (criterio de la sección 3):\n")
    filas = [[r.control_tipo, r.version, cifra(f"2d rutas fracción {r.control_tipo} {r.version}", r._3, F_RU2, 3),
              cifra(f"2d rutas mediana {r.control_tipo} {r.version}", r.mediana_r, F_RU2, 3), "sí" if r.pasa else "no"]
             for r in rutas2.itertuples(index=False)]
    partes.append(tabla_md(["Control", "Versión", "Fracción de pares con 0.1 <= r <= 10", "Mediana de r", "Pasa"], filas) + "\n")
    partes.append("Comparación con la estimación semanal (mismos pares): correlación de Spearman de los efectos y de los p, "
                  "pares en común entre los 10 de menor p de cada estimación, y enlaces declarados en cada una:\n")
    filas = [[r.control_tipo, r.version, cifra(f"comparación spearman efecto {r.control_tipo} {r.version}", r.spearman_efecto, F_CMP, 2),
              cifra(f"comparación spearman p {r.control_tipo} {r.version}", r.spearman_p, F_CMP, 2),
              cifra(f"comparación top10 {r.control_tipo} {r.version}", r.top10_en_comun, F_CMP),
              cifra(f"comparación enlaces semanal {r.control_tipo} {r.version}", r.enlaces_regla_semanal, F_CMP),
              cifra(f"comparación enlaces 2d {r.control_tipo} {r.version}", r.enlaces_regla_2d, F_CMP)]
             for r in comp2.itertuples(index=False)]
    partes.append(tabla_md(["Control", "Versión", "Spearman de efectos", "Spearman de p", "Top 10 en común", "Enlaces (semanal)",
                            "Enlaces (2 días)"], filas) + "\n")
    partes.append("Las dos estimaciones comparten documentos y período, así que no son evidencia independiente; los efectos se "
                  "comparan por orden porque dependen del número de observaciones.\n")

    # 13. Paquete de validez.
    partes.append("## 13. Paquete de validez de la serie de 2 días\n")
    partes.append("Estacionariedad (A1), con nivel 0.05; ambas pruebas sesgan hacia \"no estacionaria\" con 53 puntos:\n")
    filas = []
    for v in VERSIONES:
        g = a1[a1["serie"] == v]
        filas.append([v, cifra(f"A1 tópicos {v}", len(g), F_A1), cifra(f"A1 ADF rechaza raíz unitaria {v}", int((g["adf_pvalue"] < 0.05).sum()), F_A1),
                      cifra(f"A1 KPSS rechaza estacionariedad {v}", int((g["kpss_pvalue"] < 0.05).sum()), F_A1)])
    partes.append(tabla_md(["Versión", "Tópicos", "ADF rechaza la raíz unitaria", "KPSS rechaza la estacionariedad"], filas) + "\n")
    a3_ = a3d.iloc[0]
    partes.append(f"Componente común (A2): la primera componente explica el "
                  f"{cifra('A2 varianza PC1 (%)', a2['varianza_explicada'].iloc[0], F_A2, 1, True)}% de la varianza estandarizada. "
                  f"Documentos por ventana (A3): mínimo {cifra('A3 mínimo', a3_['minimo'], F_A3D)}, mediana "
                  f"{cifra('A3 mediana', a3_['mediana'], F_A3D)} y máximo {cifra('A3 máximo', a3_['maximo'], F_A3D)}; la "
                  "correlación de Spearman entre el valor absoluto del residuo AR(1) y n_t va de "
                  f"{cifra('A3 Spearman mínimo', a3s['spearman_abs_residuo_n_t'].min(), F_A3S, 2)} a "
                  f"{cifra('A3 Spearman máximo', a3s['spearman_abs_residuo_n_t'].max(), F_A3S, 2)} (p mínimo "
                  f"{cifra('A3 p mínimo', a3s['p_n_t'].min(), F_A3S, 2)}).\n")
    umbral_v = stats.binom.isf(0.05 / 4, 300, 0.05)
    assert (bcri["umbral"] == umbral_v).all()
    partes.append("Calibración con un sustituto realista (bloque B): volumen exógeno con los n_t reales de las 53 ventanas "
                  "(incluido su ciclo semanal), proporciones simuladas por un AR(1) del log-ratio centrado, sin enlaces, "
                  f"{cifra('sustitutos bloque B', len(leer(f'{VAL}/B_sustitutos.csv')), f'data/{VAL}/B_sustitutos.csv')} sustitutos. Pasa si las réplicas con algún "
                  f"rechazo en la original y la detendenciada quedan en o bajo binom.isf(0.05 / 4, 300, 0.05) = "
                  f"{cifra('umbral bloque B', umbral_v, 'fórmula binom.isf(0.05 / 4, 300, 0.05)')}:\n")
    filas = []
    for c in CONTROLES:
        g = bcal[bcal["control_tipo"] == c].set_index("version")
        crit = bcri[bcri["control_tipo"] == c].iloc[0]
        assert crit["original"] == g.loc["original", "replicas_con_rechazo"] and crit["detendenciada"] == g.loc["detendenciada", "replicas_con_rechazo"]
        filas.append([c] + [f"{cifra(f'B {c} {v}', g.loc[v, 'replicas_con_rechazo'], F_BCAL)} "
                            f"({cifra(f'B {c} {v} (%)', g.loc[v, 'fraccion'], F_BCAL, 1, True)}%)"
                            for v in VERSIONES + ["regla_2_versiones"]] + ["sí" if crit["pasa"] else "no"])
    partes.append(tabla_md(["Control", *VERSIONES, "Regla de 2 versiones", "Pasa"], filas) + "\n")
    partes.append("Sensibilidades sobre los datos reales (bloque C, sin declarar enlaces): rechazos de Benjamini y Hochberg "
                  "sumados sobre las 4 versiones, menor p, enlaces de la regla y, para K = 10, cuántos de los 10 pares de "
                  "menor p de la estimación principal (propio, original) están entre los 10 de la sensibilidad:\n")
    filas = []
    for sens in cres["sensibilidad"].unique():
        t = ctra[ctra["sensibilidad"] == sens].iloc[0]
        g_all = cres[cres["sensibilidad"] == sens]
        if not bool(g_all["estimada"].iloc[0]):
            filas.append([sens.replace("_", " "), cifra(f"C ventanas {sens}", g_all["ventanas"].iloc[0], F_CRES), "", "", "", "", "",
                          "", "", "", "no se estima (racha menor que 30)"])
            continue
        celdas = []
        for c in CONTROLES:
            g = g_all[g_all["control_tipo"] == c]
            assert g["enlaces_regla"].nunique() == 1
            celdas += [cifra(f"C rechazos {sens} {c}", g["rechazos_BH"].sum(), F_CRES),
                       cifra(f"C p mínimo {sens} {c}", g["p_remuestreo_min"].min(), F_CRES, 4),
                       cifra(f"C enlaces {sens} {c}", g["enlaces_regla"].iloc[0], F_CRES)]
        filas.append([sens.replace("_", " "), cifra(f"C ventanas {sens}", g_all["ventanas"].iloc[0], F_CRES),
                      cifra(f"C retardo {sens}", g_all["retardo"].iloc[0], F_CRES), cifra(f"C K {sens}", g_all["K"].iloc[0], F_CRES),
                      *celdas, cifra(f"C traslape {sens}", t["top10_en_comun"], F_CTRA) if bool(t["comparable"]) else "no comparable"])
    partes.append(tabla_md(["Sensibilidad", "Ventanas", "Retardo", "K", "Rechazos (propio)", "p mínimo (propio)",
                            "Enlaces (propio)", "Rechazos (compartido)", "p mínimo (compartido)", "Enlaces (compartido)",
                            "Top 10 en común"], filas) + "\n")
    partes.append("Orden de retardo (bloque D, control compartido, mismas filas para p = 1 y p = 2): ecuaciones en que gana "
                  "cada orden:\n")
    filas = [[r.version, cifra(f"D ecuaciones {r.version}", r.ecuaciones, F_DRES),
              cifra(f"D AIC p1 {r.version}", r.gana_p1_aic, F_DRES), cifra(f"D AIC p2 {r.version}", r.gana_p2_aic, F_DRES),
              cifra(f"D BIC p1 {r.version}", r.gana_p1_bic, F_DRES), cifra(f"D BIC p2 {r.version}", r.gana_p2_bic, F_DRES)]
             for r in dres.itertuples(index=False)]
    partes.append(tabla_md(["Versión", "Ecuaciones", "AIC: gana p = 1", "AIC: gana p = 2", "BIC: gana p = 1", "BIC: gana p = 2"],
                           filas) + "\n")

    # 14. Qué se afirma y qué no.
    partes.append("## 14. Qué se afirma y qué no\n")
    partes.append(
        "Se afirma: con la grilla de 2 días y 10 tópicos el procedimiento detecta enlaces fuertes "
        f"({cifra('det_08 en la sección 14 (%)', regla_k10.loc[0.8, 'deteccion_BH'], F_DET, 0, True)}%) y moderados "
        f"({cifra('det_04 en la sección 14 (%)', regla_k10.loc[0.4, 'deteccion_BH'], F_DET, 0, True)}%) con falsos positivos "
        "dentro de la tolerancia fijada (hasta "
        f"{cifra('fp_max en la sección 14 (%)', fp_vals.max(), F_BCAL, 1, True)}%), y bpoil no muestra enlaces de ese tamaño en la fase de crisis. No se afirma: que no existan "
        "relaciones débiles; que el resultado valga para agosto y septiembre (la serie cubre solo "
        f"{cifra('días de la racha de 2 días', fila_2d['racha_dias'], F_FG)} días); que el poder medido sea igual en datos "
        "reales (los sustitutos no reproducen relaciones no lineales); ni que otro corpus se comporte igual. La estimación "
        "sobre datos reales se ejecutó dos veces (semanal y 2 días), y ambas se reportan.\n")

    # 15. Archivos y commits de origen.
    partes.append("## 15. Archivos y commits de origen\n")
    partes.append(tabla_md(["Archivo", "Script productor", "Commit", "MD5"],
                           [[f"`data/{a}`", f"`{s}`", ", ".join(f"`{c}`" for c in cs), f"`{md5(OUTPUT_DIR / a)}`"]
                            for a, (s, cs) in origenes.items()]) + "\n")
    partes.append("A1 de la auditoría (sección 7) no deja archivo: su resultado se obtiene al generar este documento, "
                  "corriendo `prueba_a1` de `scripts/auditoria_independiente.py` a través del v1. "
                  "`data/factibilidad_corpus_nuevo.csv` se generó con un script auxiliar no versionado que importa `medir` y "
                  "`GRILLAS` de `scripts/factibilidad_corpus.py` (commit `be30272`, sin modificarlo) y los aplica al corpus "
                  "nuevo (`Datasets/t17_timeline17/t17`); su MD5 coincide con el registrado en la sección 20 de "
                  "Bitacora_Semana8.md.\n")

    documento = "\n".join(partes)
    assert chr(0x2014) not in documento and chr(0x2013) not in documento, "el documento contiene guiones largos"

    print(f"\nVerificación de cifras nuevas ({len(REGISTRO)}): cifra, valor, archivo")
    todas = True
    for r in REGISTRO:
        if r["decimales"] == "fecha":
            ok = r["valor"] in documento and fecha_es(str(r["origen"])) == r["valor"]
        else:
            tolerancia = 0.0 if r["decimales"] is None else 0.5 * 10 ** -r["decimales"] + 1e-12
            ok = r["valor"] in documento and abs(float(r["valor"]) - r["origen"]) <= tolerancia
        todas &= ok
        print(f"  {'OK   ' if ok else 'FALLA'} {r['cifra']}: {r['valor']}  (origen {r['origen']!r}, {r['archivo']})")
    if not todas:
        print("Alguna cifra no coincide: no se escribe el documento.")
        return
    SALIDA.write_text(documento, encoding="utf-8")
    print(f"\nTodas las cifras coinciden ({len(REGISTRO)} nuevas y {n_v1} del v1). Escrito {SALIDA} (MD5 {md5(SALIDA)})")


if __name__ == "__main__":
    main()
