from types import SimpleNamespace

import pandas as pd

from ig_forecaster.agent.workflow import ForecastWorkflow
from ig_forecaster.agent.workflow_routes import choose_next_stage


def test_route_runs_missing_stages_in_dependency_order():
    assert choose_next_stage({}) == "analyze_media"
    assert choose_next_stage({"media_ready": True}) == "retrieve_history"
    assert choose_next_stage({"media_ready": True, "history_ready": True}) == "retrieve_trends"
    assert choose_next_stage(
        {"media_ready": True, "history_ready": True, "trends_ready": True}
    ) == "generate_recommendations"


def test_route_propagates_refreshes_and_errors():
    assert choose_next_stage(
        {"force_media_refresh": True, "media_ready": True, "media_refreshed": False}
    ) == "analyze_media"
    assert choose_next_stage({"errors": ["failed"]}) == "handle_error"
    assert choose_next_stage(
        {
            "media_ready": True,
            "history_ready": True,
            "trends_ready": True,
            "recommendations_ready": True,
            "force_recommendation_refresh": True,
        }
    ) == "generate_recommendations"


class FakeWorkflowService:
    def __init__(self):
        self.calls = []
        self.media = pd.DataFrame()
        self.history = pd.DataFrame()
        self.trends = None
        self.recommendations = pd.DataFrame()
        self.artifacts = SimpleNamespace(
            dataset_path="/project/data.csv",
            media_analyses_path="/project/media.json",
            historical_matches_path="/project/history.csv",
            trend_signals_path="/project/trends.csv",
            recommendations_json_path="/project/recommendations.json",
        )

    def load_project(self):
        return self.artifacts

    def load_saved_media_analyses(self):
        return self.media

    def load_saved_historical_matches(self):
        return self.history

    def load_saved_trends(self):
        return self.trends

    def load_saved_recommendations(self):
        return self.recommendations

    def analyze_media(self, force=False):
        self.calls.append(("media", force))
        self.media = pd.DataFrame([{"file_name": "photo.jpg"}])
        return self.media, pd.DataFrame()

    def retrieve_history(self):
        self.calls.append(("history", False))
        self.history = pd.DataFrame([{"post_id": 1}])
        return self.history

    def retrieve_trends(self, force_refresh=False):
        self.calls.append(("trends", force_refresh))
        self.trends = SimpleNamespace(agent_signals=pd.DataFrame([{"topic": "music"}]))
        return self.trends

    def generate_recommendations(self):
        self.calls.append(("recommendations", False))
        self.recommendations = pd.DataFrame([{"rank": 1}])
        return self.recommendations


def test_workflow_runs_all_stages_and_finishes():
    service = FakeWorkflowService()

    result = ForecastWorkflow(service).invoke(thread_id="test-thread")

    assert result["current_stage"] == "complete"
    assert result["recommendations_ready"] is True
    assert result["current_stage"] == "complete"
    assert result["errors"] == []
    assert service.calls == [
        ("media", False),
        ("history", False),
        ("trends", False),
        ("recommendations", False),
    ]


def test_workflow_reuses_complete_artifacts():
    service = FakeWorkflowService()
    service.media = pd.DataFrame([{"file_name": "photo.jpg"}])
    service.history = pd.DataFrame([{"post_id": 1}])
    service.trends = SimpleNamespace(agent_signals=pd.DataFrame([{"topic": "music"}]))
    service.recommendations = pd.DataFrame([{"rank": 1}])

    result = ForecastWorkflow(service).invoke(thread_id="cached-thread")

    assert result["recommendations_ready"] is True
    assert service.calls == []


def test_workflow_routes_stage_failures_to_error_handler():
    service = FakeWorkflowService()

    def fail_media(force=False):
        raise RuntimeError("service unavailable")

    service.analyze_media = fail_media
    result = ForecastWorkflow(service).invoke(thread_id="failed-thread")

    assert result["current_stage"] == "failed"
    assert result["errors"] == ["media analysis: service unavailable"]
