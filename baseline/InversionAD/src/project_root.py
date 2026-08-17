import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_project_root_to_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
