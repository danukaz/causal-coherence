"""
config.py

Rutas del proyecto. Los datos NO viven en este repo -- viven en la
carpeta de datasets entregada por el laboratorio. Por defecto se asume
la estructura de la carpeta Capstone (Datasets/ y narrative-trails/ al
lado de este repo), pero ambas rutas se pueden sobrescribir con
variables de entorno.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_ROOT = PROJECT_ROOT.parent

DATASETS_DIR = Path(os.environ.get("CC_DATASETS_DIR", CAPSTONE_ROOT / "Datasets"))
NARRATIVE_TRAILS_DIR = Path(
    os.environ.get("CC_NARRATIVE_TRAILS_DIR", CAPSTONE_ROOT / "narrative-trails")
)

# Artefactos generados (modelos, resultados en pickle). Ignorado por git.
OUTPUT_DIR = PROJECT_ROOT / "data"
