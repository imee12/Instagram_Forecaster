from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langsmith import traceable

from .. import config
from ..gemini_client import get_or_create_client
from ..pipeline import PipelineService
from ..recommendations import (
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_RECOMMENDATION_COUNT,
    DEFAULT_THOUGHT_BRANCH_COUNT,
    RecommendationCandidate,
    RecommendationThoughtBranch,
    _candidate_allocations,
    _expand_thought_branch,
    _plan_thought_branches,
    _prepare_context,
    _rank_thought_candidates,
    save_content_recommendations,
)
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
                or artifact_exists(artifacts.historical_matches_path),
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
                "historical_evidence_mode": state.get("historical_evidence_mode")
                or (
                    str(history.iloc[0].get("historical_evidence_mode", "healthy"))
                    if history is not None and not history.empty
                    else "cold_start"
                ),
                "historical_index_size": state.get("historical_index_size", 0)
                if history is None or history.empty
                else int(history.iloc[0].get("historical_index_size", len(history))),
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
            mode = (
                "cold_start"
                if matches.empty
                else str(matches.iloc[0].get("historical_evidence_mode", "healthy"))
            )
            index_size = (
                0
                if matches.empty
                else int(matches.iloc[0].get("historical_index_size", len(matches)))
            )
            return {
                "history_ready": True,
                "history_refreshed": True,
                "historical_match_count": len(matches),
                "historical_evidence_mode": mode,
                "historical_index_size": index_size,
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

    @traceable(name="Select Recommendation Execution Mode", run_type="chain")
    def generate_recommendations(state: ForecastWorkflowState) -> dict[str, Any]:
        mode = "mock" if config.DEV_MODE else "real_gemini_tot"
        print(f"[IG Forecaster] Recommendation graph mode: {mode}.")
        return {"current_stage": "recommendation_mode_selected"}

    @traceable(name="DEV MODE - Mock ToT Graph Node", run_type="chain")
    def dev_mock_tot(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            recommendations = service.generate_recommendations(
                recommendation_brief=state.get("recommendation_brief") or None
            )
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

    @traceable(name="ToT 1 - Plan Strategy Branches Graph Node", run_type="chain")
    def tot_plan(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            print("[IG Forecaster] DEV_MODE=false: running real Gemini ToT graph nodes.")
            media_context, historical_context, trend_context = _prepare_context(
                service.load_saved_media_analyses(),
                service.load_saved_historical_matches(),
                service.load_saved_trends(),
            )
            branches = _plan_thought_branches(
                get_or_create_client(),
                media_context,
                historical_context,
                trend_context,
                DEFAULT_THOUGHT_BRANCH_COUNT,
                state.get("recommendation_brief"),
            )
            return {
                "tot_media_context": media_context,
                "tot_historical_context": historical_context,
                "tot_trend_context": trend_context,
                "tot_branches": [branch.model_dump() for branch in branches],
                "current_stage": "tot_planned",
            }
        except Exception as exc:
            return _failure("ToT planning", exc)

    @traceable(name="ToT 2 - Expand Thought Branches Graph Node", run_type="chain")
    def tot_expand(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            branches = [
                RecommendationThoughtBranch.model_validate(item)
                for item in state["tot_branches"]
            ]
            allocations = _candidate_allocations(
                DEFAULT_CANDIDATE_COUNT,
                len(branches),
            )
            expanded = []
            client = get_or_create_client()
            for branch, allocation in zip(branches, allocations):
                candidates = _expand_thought_branch(
                    client,
                    branch,
                    allocation,
                    state["tot_media_context"],
                    state["tot_historical_context"],
                    state["tot_trend_context"],
                    state.get("recommendation_brief"),
                )
                expanded.extend(
                    {
                        "branch": branch.model_dump(),
                        "candidate": candidate.model_dump(),
                    }
                    for candidate in candidates
                )
            return {
                "tot_expanded_candidates": expanded,
                "current_stage": "tot_expanded",
            }
        except Exception as exc:
            return _failure("ToT expansion", exc)

    @traceable(name="ToT 3 - Score and Prune Graph Node", run_type="chain")
    def tot_rank(state: ForecastWorkflowState) -> dict[str, Any]:
        try:
            expanded = [
                (
                    RecommendationThoughtBranch.model_validate(item["branch"]),
                    RecommendationCandidate.model_validate(item["candidate"]),
                )
                for item in state["tot_expanded_candidates"]
            ]
            recommendations = _rank_thought_candidates(
                expanded,
                state["tot_media_context"],
                state["tot_historical_context"],
                state["tot_trend_context"],
                DEFAULT_RECOMMENDATION_COUNT,
            )
            save_content_recommendations(
                recommendations,
                service.artifacts.output_folder,
            )
            return {
                "recommendations_ready": True,
                "recommendations_generated": True,
                "recommendation_count": len(recommendations),
                "current_stage": "recommendations_complete",
            }
        except Exception as exc:
            return _failure("ToT scoring and pruning", exc)

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
        "dev_mock_tot": dev_mock_tot,
        "tot_1_plan": tot_plan,
        "tot_2_expand": tot_expand,
        "tot_3_rank": tot_rank,
        "handle_error": handle_error,
        "complete": complete,
    }
