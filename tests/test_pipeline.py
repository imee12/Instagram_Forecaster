import json

import pandas as pd

from ig_forecaster.pipeline import PipelineService


def make_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    dataset = project / "IG_Forecaster.csv"
    pd.DataFrame(
        [{"description": "A post", "media_type": "image", "category": "fashion"}]
    ).to_csv(dataset, index=False)
    return project, dataset


def test_pipeline_service_exposes_artifact_paths_without_external_calls(tmp_path):
    project, dataset = make_project(tmp_path)

    service = PipelineService(dataset_path=dataset)
    artifacts = service.load_project()

    assert artifacts.project_root == project
    assert artifacts.dataset_path == dataset
    assert artifacts.output_folder == project / "output"
    assert artifacts.recommendations_json_path.name == "post_recommendations.json"


def test_pipeline_service_loads_saved_artifacts_without_external_calls(tmp_path):
    _, dataset = make_project(tmp_path)
    service = PipelineService(dataset_path=dataset)
    artifacts = service.load_project()

    artifacts.media_analyses_path.write_text(
        json.dumps([{"file_name": "photo.jpg", "themes": ["fashion"]}]),
        encoding="utf-8",
    )
    pd.DataFrame([{"media_file": "photo.jpg", "post_id": 1}]).to_csv(
        artifacts.historical_matches_path,
        index=False,
    )
    artifacts.recommendations_json_path.write_text(
        json.dumps([{"rank": 1, "concept": "Portrait"}]),
        encoding="utf-8",
    )

    assert service.load_saved_media_analyses().iloc[0]["file_name"] == "photo.jpg"
    assert service.load_saved_historical_matches().iloc[0]["post_id"] == 1
    assert service.load_saved_recommendations().iloc[0]["concept"] == "Portrait"


def test_pipeline_stages_can_use_saved_inputs_independently(tmp_path, monkeypatch):
    _, dataset = make_project(tmp_path)
    service = PipelineService(dataset_path=dataset)
    analyses = pd.DataFrame([{"file_name": "photo.jpg"}])
    matches = pd.DataFrame([{"media_file": "photo.jpg", "post_id": 1}])

    monkeypatch.setattr(service, "load_saved_media_analyses", lambda: analyses)
    monkeypatch.setattr(
        service,
        "load_or_build_historical_index",
        lambda: ("index", pd.DataFrame([{"post_id": 1}]), "model"),
    )
    monkeypatch.setattr(
        "ig_forecaster.pipeline.retrieve_historical_matches",
        lambda *args, **kwargs: matches,
    )

    result = service.retrieve_history()

    assert result.equals(matches)
    assert service.load_saved_historical_matches().equals(matches)
