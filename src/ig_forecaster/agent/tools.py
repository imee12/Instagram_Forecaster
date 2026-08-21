from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain.tools import BaseTool, ToolRuntime, tool

from ..pipeline import PipelineService
from .workflow import ForecastWorkflow


def _records(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.head(limit).to_json(orient="records"))


def _artifact_status(service: PipelineService) -> dict[str, Any]:
    artifacts = service.load_project()
    analyses = service.load_saved_media_analyses()
    errors = service.load_saved_media_errors()
    history = service.load_saved_historical_matches()
    trends = service.load_saved_trends()
    recommendations = service.load_saved_recommendations()
    return {
        "project_root": str(artifacts.project_root),
        "dataset_path": str(artifacts.dataset_path),
        "media_analysis_count": len(analyses),
        "media_error_count": len(errors),
        "historical_match_count": len(history),
        "trend_signal_count": len(trends.agent_signals) if trends is not None else 0,
        "recommendation_count": len(recommendations),
        "has_media_analyses": not analyses.empty,
        "has_historical_matches": not history.empty,
        "has_trends": trends is not None and not trends.agent_signals.empty,
        "has_recommendations": not recommendations.empty,
    }


def build_agent_tools(
    service: PipelineService,
    workflow: ForecastWorkflow | None = None,
) -> list[BaseTool]:
    """Bind safe agent tools to one project-specific pipeline service."""

    @tool("get_project_status")
    def get_project_status() -> dict[str, Any]:
        """Inspect saved pipeline artifacts without making external API calls."""
        return _artifact_status(service)

    @tool("analyze_project_media")
    def analyze_project_media(force: bool = False) -> dict[str, Any]:
        """Analyze project media with Gemini, reusing cached analyses by default.

        Set force to true only when the user explicitly requests reanalysis.
        """
        analyses, errors = service.analyze_media(force=force)
        return {
            "analysis_count": len(analyses),
            "error_count": len(errors),
            "analyses": _records(analyses),
            "errors": _records(errors),
        }

    @tool("retrieve_historical_matches")
    def retrieve_history() -> dict[str, Any]:
        """Match saved media analyses to semantically similar historical posts."""
        matches = service.retrieve_history()
        return {"match_count": len(matches), "matches": _records(matches, limit=15)}

    @tool("retrieve_google_trends")
    def retrieve_trends(force_refresh: bool = False) -> dict[str, Any]:
        """Load Google Trends signals, refreshing only when stale by default.

        Set force_refresh to true only when the user explicitly requests it.
        """
        report = service.retrieve_trends(force_refresh=force_refresh)
        return {
            "signal_count": len(report.agent_signals),
            "signals": _records(report.agent_signals, limit=20),
            "keyword_momentum": _records(report.keyword_momentum),
        }

    @tool("generate_post_recommendations")
    def generate_recommendations(
        recommendation_brief: str | None = None,
    ) -> dict[str, Any]:
        """Generate and save three recommendations from existing project artifacts.

        Pass the user's creative direction, constraints, and feedback verbatim in
        recommendation_brief so it can steer the Tree-of-Thoughts search.
        """
        recommendations = service.generate_recommendations(
            recommendation_brief=recommendation_brief
        )
        return {
            "recommendation_count": len(recommendations),
            "recommendations": _records(recommendations),
        }

    @tool("get_saved_recommendations")
    def get_saved_recommendations() -> dict[str, Any]:
        """Load existing recommendations without running models or external APIs."""
        recommendations = service.load_saved_recommendations()
        return {
            "recommendation_count": len(recommendations),
            "recommendations": _records(recommendations),
        }

    tools = [
        get_project_status,
        analyze_project_media,
        retrieve_history,
        retrieve_trends,
        generate_recommendations,
        get_saved_recommendations,
    ]

    if workflow is not None:
        @tool("run_forecast_workflow")
        def run_forecast_workflow(
            runtime: ToolRuntime,
            force_media_refresh: bool = False,
            force_trend_refresh: bool = False,
            force_recommendation_refresh: bool = False,
            recommendation_brief: str | None = None,
        ) -> dict[str, Any]:
            """Run the cache-aware LangGraph forecasting workflow.

            Use this when the user asks to run or update the full forecast.
            Force refreshes only on explicit request. When the user asks for new
            or revised recommendations, pass their creative direction and feedback
            in recommendation_brief and set force_recommendation_refresh to true.
            """
            thread_id = runtime.config.get("configurable", {}).get("thread_id")
            if not thread_id:
                raise ValueError("The LangGraph runtime has no thread_id.")
            result = workflow.invoke(
                thread_id=str(thread_id),
                force_media_refresh=force_media_refresh,
                force_trend_refresh=force_trend_refresh,
                force_recommendation_refresh=force_recommendation_refresh,
                recommendation_brief=recommendation_brief,
            )
            payload = dict(result)
            if payload.get("recommendations_ready") and not payload.get("errors"):
                recommendations = service.load_saved_recommendations()
                payload["recommendations"] = _records(recommendations)
            return payload

        tools.append(run_forecast_workflow)

    return tools
