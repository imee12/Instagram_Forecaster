from __future__ import annotations

from .workflow_state import ForecastWorkflowState


def choose_next_stage(state: ForecastWorkflowState) -> str:
    """Choose the next required stage while propagating upstream refreshes."""
    if state.get("errors"):
        return "handle_error"

    if state.get("force_media_refresh") and not state.get("media_refreshed"):
        return "analyze_media"
    if not state.get("media_ready"):
        return "analyze_media"

    if state.get("media_refreshed") and not state.get("history_refreshed"):
        return "retrieve_history"
    if not state.get("history_ready"):
        return "retrieve_history"

    if state.get("force_trend_refresh") and not state.get("trends_refreshed"):
        return "retrieve_trends"
    if not state.get("trends_ready"):
        return "retrieve_trends"

    upstream_changed = any(
        state.get(key)
        for key in ("media_refreshed", "history_refreshed", "trends_refreshed")
    )
    if state.get("force_recommendation_refresh") and not state.get(
        "recommendations_generated"
    ):
        return "generate_recommendations"
    if upstream_changed and not state.get("recommendations_generated"):
        return "generate_recommendations"
    if not state.get("recommendations_ready"):
        return "generate_recommendations"

    return "complete"
