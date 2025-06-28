from prefect import flow, task
from nbclient import NotebookClient
from nbformat import read
from pathlib import Path

# Calcular la raíz del proyecto y el path a los notebooks
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" 

@task
def run_notebook(notebook_name: str):
    notebook_path = NOTEBOOK_DIR / notebook_name
    print(f"📓 Ejecutando notebook: {notebook_path}")
    with open(notebook_path) as f:
        nb = read(f, as_version=4)
        client = NotebookClient(nb, timeout=600)
        client.execute()

@flow
def etl_notebooks_flow():
    run_notebook("00_construction_target_variable.ipynb")
    run_notebook("01_exploring_preprocessing.ipynb")

if __name__ == "__main__":
    etl_notebooks_flow()
