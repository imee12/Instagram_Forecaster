from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langsmith import traceable

from ..pipeline import PipelineService
from .workflow_state import ForecastWorkflowState


WorkflowNode = Callable[[ForecastWorkflowState], dict[str, Any]]


def _failure(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "current_stage": "failed",
        "errors": [f"{stage}: {exc}"],
    }


def build_workflow_nodes(service: PipelineService) -> dict[str, WorkflowNode]:
    """Create graph nodes bound to a project-specific pipeline service."""

    @traceable(name="Inspect Forecast Artifacts", run_type="chain")
    def inspect_project(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            artifacts = service.load_project()

            def load_cloud_artifact(loader):
                try:
                    return loader()
                except OSError:
                    return None

            def artifact_exists(path) -> bool:
                try:
                    return Path(path).exists()
                except OSError:
                    return False

            # A stage that just completed already put its readiness and counts
            # in state. Reuse those values instead of immediately re-reading
            # cloud-backed artifacts, which can block or time out on hydration.
            analyses = (
                None
                if state.get("media_ready")
                else load_cloud_artifact(service.load_saved_media_analyses)
            )
            history = (
                None
                if state.get("history_ready")
                else load_cloud_artifact(service.load_saved_historical_matches)
            )
            trends = (
                None
                if state.get("trends_ready")
                else load_cloud_artifact(service.load_saved_trends)
            )
            recommendations = (
                None
                if state.get("recommendations_ready")
                else load_cloud_artifact(service.load_saved_recommendations)
            )
            return {
                "dataset_path": str(artifacts.dataset_path),
                "media_ready": state.get("media_ready")
                or (analyses is not None and not analyses.empty)
                or (
                    analyses is None
                    and artifact_exists(artifacts.media_analyses_path)
                ),
                "history_ready": state.get("history_ready")
                or (history is not None and not history.empty)
                or (
                    history is None
                    and artifact_exists(artifacts.historical_matches_path)
                ),
                "trends_ready": state.get("trends_ready")
                or (
                    trends is not None
                    and not trends.agent_signals.empty
                )
                or (
                    trends is None
                    and artifact_exists(artifacts.trend_signals_path)
                ),
                "recommendations_ready": state.get("recommendations_ready")
                or (recommendations is not None and not recommendations.empty)
                or (
                    recommendations is None
                    and artifact_exists(artifacts.recommendations_json_path)
                ),
                "media_analysis_count": state.get("media_analysis_count", 0)
                if analyses is None
                else len(analyses),
                "historical_match_count": state.get("historical_match_count", 0)
                if history is None
                else len(history),
                "trend_signal_count": state.get("trend_signal_count", 0)
                if trends is None
                else len(trends.agent_signals),
                "recommendation_count": state.get("recommendation_count", 0)
                if recommendations is None
                else len(recommendations),
                "media_analysis_path": str(artifacts.media_analyses_path),
                "historical_matches_path": str(artifacts.historical_matches_path),
                "trend_signals_path": str(artifacts.trend_signals_path),
                "recommendations_path": str(artifacts.recommendations_json_path),
                "current_stage": "inspected",
            }
        except Exception as exc:
            return _failure("project inspection", exc)

    @traceable(name="Workflow Analyze Media", run_type="chain")
    def analyze_media(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            analyses, errors = service.analyze_media(
                force=state.get("force_media_refresh", False)
            )
            if analyses.empty:
                raise ValueError("Media analysis produced no usable analyses.")
            return {
                "media_ready": True,
                "media_refreshed": True,
                "media_analysis_count": len(analyses),
                "media_error_count": len(errors),
                "history_ready": False,
                "recommendations_ready": False,
                "current_stage": "media_complete",
            }
        except Exception as exc:
            return _failure("media analysis", exc)

    @traceable(name="Workflow Retrieve History", run_type="chain")
    def retrieve_history(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            matches = service.retrieve_history()
            if matches.empty:
                raise ValueError("Historical retrieval produced no matches.")
            return {
                "history_ready": True,
                "history_refreshed": True,
                "historical_match_count": len(matches),
                "recommendations_ready": False,
                "current_stage": "history_complete",
            }
        except Exception as exc:
            return _failure("historical retrieval", exc)

    @traceable(name="Workflow Retrieve Trends", run_type="chain")
    def retrieve_trends(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            report = service.retrieve_trends(
                force_refresh=state.get("force_trend_refresh", False)
            )
            if report.agent_signals.empty:
                raise ValueError("Google Trends retrieval produced no agent signals.")
            return {
                "trends_ready": True,
                "trends_refreshed": True,
                "trend_signal_count": len(report.agent_signals),
                "recommendations_ready": False,
                "current_stage": "trends_complete",
            }
        except Exception as exc:
            return _failure("Google Trends retrieval", exc)

    @traceable(name="Workflow Generate Recommendations", run_type="chain")
    def generate_recommendations(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            recommendations = service.generate_recommendations()
            if recommendations.empty:
                raise ValueError("Recommendation generation produced no recommendations.")
            return {
                "recommendations_ready": True,
                "recommendations_generated": True,
                "recommendation_count": len(recommendations),
                "current_stage": "recommendations_complete",
            }
        except Exception as exc:
            return _failure("recommendation generation", exc)

    @traceable(name="Handle Forecast Workflow Error", run_type="chain")
    def handle_error(state: ForecastWorkflowState) -> dict[str, Any]:
        return {
            "current_stage": "failed",
            "errors": state.get("errors", ["Unknown workflow error"]),
        }

    @traceable(name="Complete Forecast Workflow", run_type="chain")
    def complete(state: ForecastWorkflowState) -> dict[str, Any]:
        return {"current_stage": "complete"}

    return {
        "inspect_project": inspect_project,
        "analyze_media": analyze_media,
        "retrieve_history": retrieve_history,
        "retrieve_trends": retrieve_trends,
        "generate_recommendations": generate_recommendations,
        "handle_error": handle_error,
        "complete": complete,
    }
