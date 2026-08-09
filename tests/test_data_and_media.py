from pathlib import Path

import pandas as pd

from ig_forecaster.data import build_post_text, load_posts
from ig_forecaster.main import run_pipeline
from ig_forecaster.media import SUPPORTED_EXTENSIONS, find_media_files, find_media_files_in_project


def test_build_post_text_assembles_retrieval_text(tmp_path):
    row = pd.Series(
        {
            "description": "A rehearsal clip",
            "media_type": "video",
            "category": "behind the scenes",
        }
    )

    result = build_post_text(row)

    assert "Description: A rehearsal clip." in result
    assert "Media type: video." in result
    assert "Content category: behind the scenes." in result


def test_load_posts_adds_retrieval_text(tmp_path):
    csv_path = tmp_path / "posts.csv"
    pd.DataFrame(
        [
            {
                "description": "First post",
                "media_type": "image",
                "category": "fashion",
            }
        ]
    ).to_csv(csv_path, index=False)

    posts = load_posts(csv_path)

    assert "retrieval_text" in posts.columns
    assert posts.loc[0, "retrieval_text"].startswith("Description: First post")


def test_find_media_files_filters_supported_extensions(tmp_path):
    (tmp_path / "sub").mkdir()
    supported = [
        tmp_path / "photo.jpg",
        tmp_path / "sub" / "video.mp4",
    ]
    unsupported = [
        tmp_path / "notes.txt",
        tmp_path / "sub" / "archive.mov.bak",
    ]

    for path in supported + unsupported:
        path.write_bytes(b"data")

    results = find_media_files(tmp_path)

    assert [path.name for path in results] == ["photo.jpg", "video.mp4"]
    assert all(path.suffix.lower() in SUPPORTED_EXTENSIONS for path in results)


def test_run_pipeline_uses_explicit_dataset_path(tmp_path, monkeypatch):
    dataset_path = tmp_path / "IG_Forecaster.csv"
    pd.DataFrame(
        [{"description": "A sample post", "media_type": "image", "category": "fashion"}]
    ).to_csv(dataset_path, index=False)

    monkeypatch.setattr("ig_forecaster.main.get_project_root", lambda dataset_path=None: tmp_path)
    monkeypatch.setattr("ig_forecaster.main.get_media_folder", lambda dataset_path=None: tmp_path / "available_media")
    monkeypatch.setattr("ig_forecaster.main.find_media_files_in_project", lambda project_root, media_folder: [])
    monkeypatch.setattr("ig_forecaster.main.build_or_load_index", lambda posts, dataset_path=None: (("index",), {"id": 1}, "model"))
    monkeypatch.setattr("ig_forecaster.main.retrieve_google_trends", lambda: "trend-report")
    monkeypatch.setattr(
        "ig_forecaster.main.save_trend_report",
        lambda report, output_folder: (output_folder / "interest.csv", output_folder / "related.csv"),
    )
    monkeypatch.setattr("ig_forecaster.main.get_output_folder", lambda dataset_path=None: tmp_path / "output")

    media_analyses_df, media_errors_df, index_payload = run_pipeline(dataset_path=dataset_path)

    assert media_analyses_df.empty
    assert media_errors_df.empty
    assert index_payload[0] == ("index",)
    assert "description" in load_posts(dataset_path).columns


def test_run_pipeline_accepts_project_directory_override(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dataset_path = project_dir / "IG_Forecaster.csv"
    pd.DataFrame(
        [{"description": "A sample post", "media_type": "image", "category": "fashion"}]
    ).to_csv(dataset_path, index=False)

    monkeypatch.setattr("ig_forecaster.main.get_project_root", lambda dataset_path=None: project_dir)
    monkeypatch.setattr("ig_forecaster.main.get_media_folder", lambda dataset_path=None: project_dir / "available_media")
    monkeypatch.setattr("ig_forecaster.main.find_media_files_in_project", lambda project_root, media_folder: [])
    monkeypatch.setattr("ig_forecaster.main.build_or_load_index", lambda posts, dataset_path=None: (("index",), {"id": 1}, "model"))
    monkeypatch.setattr("ig_forecaster.main.retrieve_google_trends", lambda: "trend-report")
    monkeypatch.setattr(
        "ig_forecaster.main.save_trend_report",
        lambda report, output_folder: (output_folder / "interest.csv", output_folder / "related.csv"),
    )
    monkeypatch.setattr("ig_forecaster.main.get_output_folder", lambda dataset_path=None: project_dir / "output")

    media_analyses_df, media_errors_df, _ = run_pipeline(dataset_path=project_dir)

    assert media_analyses_df.empty
    assert media_errors_df.empty


def test_find_media_files_in_project_falls_back_to_project_root(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    media_file = project_dir / "photo.jpg"
    media_file.write_bytes(b"data")

    results = find_media_files_in_project(project_root=project_dir, media_folder=project_dir / "available_media")

    assert [path.name for path in results] == ["photo.jpg"]
