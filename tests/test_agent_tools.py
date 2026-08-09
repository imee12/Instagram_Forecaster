from types import SimpleNamespace

import pandas as pd

from ig_forecaster.agent.tools import build_agent_tools


class FakeService:
    def __init__(self):
        self.artifacts = SimpleNamespace(
            project_root="/project",
            dataset_path="/project/data.csv",
        )

    def load_project(self):
        return self.artifacts

    def load_saved_media_analyses(self):
        return pd.DataFrame([{"file_name": "photo.jpg"}])

    def load_saved_media_errors(self):
        return pd.DataFrame()

    def load_saved_historical_matches(self):
        return pd.DataFrame([{"media_file": "photo.jpg", "post_id": 1}])

    def load_saved_trends(self):
        return SimpleNamespace(
            agent_signals=pd.DataFrame([{"rank": 1, "topic": "music"}]),
            keyword_momentum=pd.DataFrame([{"keyword": "music"}]),
        )

    def load_saved_recommendations(self):
        return pd.DataFrame([{"rank": 1, "concept": "Music Reel"}])

    def analyze_media(self, force=False):
        return self.load_saved_media_analyses(), pd.DataFrame()

    def retrieve_history(self):
        return self.load_saved_historical_matches()

    def retrieve_trends(self, force_refresh=False):
        return self.load_saved_trends()

    def generate_recommendations(self):
        return self.load_saved_recommendations()


def tools_by_name():
    return {tool.name: tool for tool in build_agent_tools(FakeService())}


def test_agent_tools_expose_pipeline_stages():
    tools = tools_by_name()

    assert set(tools) == {
        "get_project_status",
        "analyze_project_media",
        "retrieve_historical_matches",
        "retrieve_google_trends",
        "generate_post_recommendations",
        "get_saved_recommendations",
    }


def test_project_status_uses_only_saved_artifacts():
    status = tools_by_name()["get_project_status"].invoke({})

    assert status["media_analysis_count"] == 1
    assert status["historical_match_count"] == 1
    assert status["trend_signal_count"] == 1
    assert status["recommendation_count"] == 1


def test_recommendation_tool_returns_serializable_records():
    result = tools_by_name()["generate_post_recommendations"].invoke({})

    assert result["recommendation_count"] == 1
    assert result["recommendations"][0]["concept"] == "Music Reel"
