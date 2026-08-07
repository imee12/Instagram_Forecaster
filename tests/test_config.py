from pathlib import Path

from ig_forecaster.config import PROJECT_ROOT, get_media_folder, get_project_root


def test_project_root_points_to_google_drive_desktop():
    assert PROJECT_ROOT == (
        Path.home()
        / "Library/CloudStorage/GoogleDrive-imeetwelve@gmail.com/My Drive/Colab Notebooks/IG_Forecaster"
    )


def test_get_project_root_uses_workspace_when_expected_files_exist(tmp_path, monkeypatch):
    project_dir = tmp_path / "IG_Forecaster"
    project_dir.mkdir()
    (project_dir / "IG_Forecaster.csv").write_text("description\n", encoding="utf-8")
    (project_dir / "available_media").mkdir()

    monkeypatch.delenv("IG_FORECASTER_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(project_dir)

    assert get_project_root() == project_dir


def test_get_media_folder_uses_explicit_drive_override(tmp_path, monkeypatch):
    dataset_path = tmp_path / "IG_Forecaster.csv"
    dataset_path.write_text("description\n", encoding="utf-8")
    media_folder = tmp_path / "drive_media"

    monkeypatch.setenv("IG_FORECASTER_DATASET_PATH", str(dataset_path))
    monkeypatch.setenv("IG_FORECASTER_MEDIA_FOLDER", str(media_folder))

    assert get_media_folder() == media_folder
