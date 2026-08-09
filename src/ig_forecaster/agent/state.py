from __future__ import annotations

from typing_extensions import NotRequired

from langchain.agents import AgentState


class IGForecasterState(AgentState):
    """Conversation state persisted for each UI or CLI agent thread."""

    user_preferences: NotRequired[dict[str, str]]
    excluded_media: NotRequired[list[str]]
    approved_recommendations: NotRequired[list[dict]]
    project_status: NotRequired[dict]
