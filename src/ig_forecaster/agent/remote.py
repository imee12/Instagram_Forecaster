"""Client used by Streamlit to run graphs through LangGraph Agent Server."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import uuid

from langgraph_sdk import get_sync_client

from ..pipeline import PipelineService
from .workflow import ForecastWorkflow


class RemoteIGForecasterAgent:
    """Server-backed counterpart to ``IGForecasterAgent`` for the web UI."""

    def __init__(
        self,
        dataset_path=None,
        *,
        server_url: str = "http://127.0.0.1:2024",
    ):
        self.service = PipelineService(dataset_path=dataset_path)
        self.client = get_sync_client(url=server_url)

    def _ensure_thread(
        self,
        thread_id: str,
        *,
        graph_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client.threads.create(
            thread_id=thread_id,
            graph_id=graph_id,
            metadata=metadata,
            if_exists="do_nothing",
        )

    @staticmethod
    def workflow_thread_id(thread_id: str) -> str:
        return str(uuid.uuid5(uuid.UUID(thread_id), "ig-forecaster-workflow"))

    def invoke(
        self,
        message: str,
        *,
        thread_id: str,
        user_preferences: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not message.strip():
            raise ValueError("Agent message cannot be empty.")
        self._ensure_thread(thread_id, graph_id="ig_forecaster_agent")
        state: dict[str, Any] = {
            "messages": [{"role": "user", "content": message}]
        }
        if user_preferences:
            state["user_preferences"] = user_preferences
        return self.client.runs.wait(
            thread_id,
            "ig_forecaster_agent",
            input=state,
        )

    def get_state(self, *, thread_id: str):
        self._ensure_thread(thread_id, graph_id="ig_forecaster_agent")
        state = self.client.threads.get_state(thread_id)
        return SimpleNamespace(values=state.get("values") or {})

    def run_workflow(
        self,
        *,
        thread_id: str,
        force_media_refresh: bool = False,
        force_trend_refresh: bool = False,
        force_recommendation_refresh: bool = False,
    ) -> dict[str, Any]:
        workflow_thread_id = self.workflow_thread_id(thread_id)
        self._ensure_thread(
            workflow_thread_id,
            graph_id="ig_forecaster",
            metadata={
                "source": "streamlit",
                "kind": "pipeline",
                "conversation_thread_id": thread_id,
            },
        )
        initial_state = ForecastWorkflow(self.service).initial_state(
            force_media_refresh=force_media_refresh,
            force_trend_refresh=force_trend_refresh,
            force_recommendation_refresh=force_recommendation_refresh,
        )
        return self.client.runs.wait(
            workflow_thread_id,
            "ig_forecaster",
            input=initial_state,
            metadata={"source": "streamlit", "kind": "pipeline"},
        )
