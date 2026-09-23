"""
run_notebooks.py

Ejecuta los notebooks de Examples/ y guarda las salidas en el mismo
archivo. A diferencia de `jupyter execute`, no registra marcas de tiempo
por celda, así que volver a correrlos no genera diffs si las salidas no
cambian, y lee los notebooks como UTF-8 también en Windows.

Uso:
    python scripts/run_notebooks.py                       # todos
    python scripts/run_notebooks.py narrative_trails_bpoil
"""

import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from nbclient import NotebookClient

from causal_coherence.config import PROJECT_ROOT

EXAMPLES_DIR = PROJECT_ROOT / "Examples"

# Kernel registrado con `python -m ipykernel install --user --name causal-coherence ...`
# (ver README). Si no está, "python3" es el kernel del ambiente activo.
KERNEL_NAME = "causal-coherence"


def kernel_name() -> str:
    try:
        KernelSpecManager().get_kernel_spec(KERNEL_NAME)
        return KERNEL_NAME
    except NoSuchKernel:
        return "python3"


def run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    NotebookClient(
        nb,
        timeout=1200,
        kernel_name=kernel_name(),
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(nb, path)


def main():
    names = sys.argv[1:] or sorted(p.stem for p in EXAMPLES_DIR.glob("*.ipynb"))
    for name in names:
        path = EXAMPLES_DIR / f"{Path(name).stem}.ipynb"
        print(f"Ejecutando {path.relative_to(PROJECT_ROOT).as_posix()}...")
        run(path)


if __name__ == "__main__":
    main()
