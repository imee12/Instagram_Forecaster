from __future__ import annotations

from typing import Any, TypedDict


class ForecastWorkflowState(TypedDict, total=False):
    """Serializable state for one forecasting workflow execution."""

    dataset_path: str
    force_media_refresh: bool
    force_trend_refresh: bool
    force_recommendation_refresh: bool

    media_ready: bool
    history_ready: bool
    trends_ready: bool
    recommendations_ready: bool

    media_refreshed: bool
    history_refreshed: bool
    trends_refreshed: bool
    recommendations_generated: bool

    media_analysis_count: int
    media_error_count: int
    historical_match_count: int
    trend_signal_count: int
    recommendation_count: int

    tot_media_context: list[dict[str, Any]]
    tot_historical_context: list[dict[str, Any]]
    tot_trend_context: list[dict[str, Any]]
    tot_branches: list[dict[str, Any]]
    tot_expanded_candidates: list[dict[str, Any]]

    media_analysis_path: str
    historical_matches_path: str
    trend_signals_path: str
    recommendations_path: str

    current_stage: str
    errors: list[str]
