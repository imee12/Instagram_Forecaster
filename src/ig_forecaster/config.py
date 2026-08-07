from __future__ import annotations

import os
from pathlib import Path

from .drive import mount_google_drive


PROJECT_ROOT = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-imeetwelve@gmail.com/My Drive/Colab Notebooks/IG_Forecaster"
)
COLAB_PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/IG_Forecaster")
DEFAULT_PROJECT_ROOT = PROJECT_ROOT


def ensure_google_drive_mount() -> None:
    mount_google_drive()


def _candidate_project_roots() -> list[Path]:
    env_root = os.getenv("IG_FORECASTER_PROJECT_ROOT")
    env_dataset = os.getenv("IG_FORECASTER_DATASET_PATH")
    env_media_folder = os.getenv("IG_FORECASTER_MEDIA_FOLDER")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    if env_dataset:
        candidates.append(Path(env_dataset).parent)
    if env_media_folder:
        candidates.append(Path(env_media_folder))

    candidates.extend(
        [
            PROJECT_ROOT,
            COLAB_PROJECT_ROOT,
            Path.cwd(),
        ]
    )

    return candidates


def _find_dataset_path() -> Path | None:
    env_dataset = os.getenv("IG_FORECASTER_DATASET_PATH")
    if env_dataset:
        path = Path(env_dataset)
        if path.exists():
            return path

    for root in _candidate_project_roots():
        if not root.exists():
            continue

        if (root / "IG_Forecaster.csv").exists():
            return root / "IG_Forecaster.csv"

        for candidate in root.rglob("IG_Forecaster.csv"):
            if candidate.is_file():
                return candidate

    return None


def get_project_root(dataset_path: str | Path | None = None) -> Path:
    ensure_google_drive_mount()

    if dataset_path is not None:
        resolved_path = resolve_dataset_path(dataset_path)
        if resolved_path.exists():
            return resolved_path.parent if resolved_path.is_file() else resolved_path

    discovered_path = _find_dataset_path()
    if discovered_path is not None:
        return discovered_path.parent

    return DEFAULT_PROJECT_ROOT


def get_media_folder(dataset_path: str | Path | None = None) -> Path:
    env_media_folder = os.getenv("IG_FORECASTER_MEDIA_FOLDER")
    if env_media_folder:
        media_folder = Path(env_media_folder).expanduser()
        media_folder.mkdir(parents=True, exist_ok=True)
        return media_folder

    project_root = get_project_root(dataset_path=dataset_path)
    media_folder = project_root / "available_media"
    media_folder.mkdir(parents=True, exist_ok=True)
    return media_folder


def get_index_folder(dataset_path: str | Path | None = None) -> Path:
    project_root = get_project_root(dataset_path=dataset_path)
    index_folder = project_root / "indexes"
    index_folder.mkdir(parents=True, exist_ok=True)
    return index_folder


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    if dataset_path is None:
        discovered_path = _find_dataset_path()
        if discovered_path is not None:
            return discovered_path

        project_root = get_project_root()
        return project_root / "IG_Forecaster.csv"

    candidate = Path(dataset_path).expanduser()
    if candidate.is_dir():
        expected_csv = candidate / "IG_Forecaster.csv"
        if expected_csv.exists():
            return expected_csv

        for match in candidate.rglob("IG_Forecaster.csv"):
            if match.is_file():
                return match

        return expected_csv

    if candidate.exists():
        return candidate

    if candidate.suffix != ".csv":
        possible_csv = candidate / "IG_Forecaster.csv"
        if possible_csv.exists():
            return possible_csv

    return candidate


def get_csv_path() -> Path:
    return resolve_dataset_path()
