from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..pipeline import PipelineService
from .workflow_nodes import build_workflow_nodes
from .workflow_routes import (
    choose_next_stage,
    choose_recommendation_mode,
    continue_tot_or_error,
)
from .workflow_state import ForecastWorkflowState


WORKFLOW_CHECKPOINT_NAMESPACE = "forecast-workflow"


def build_forecast_workflow(
    service: PipelineService,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
):
    nodes = build_workflow_nodes(service)
    builder = StateGraph(ForecastWorkflowState)
    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "inspect_project")
    builder.add_conditional_edges(
        "inspect_project",
        choose_next_stage,
        {
            "analyze_media": "analyze_media",
            "retrieve_history": "retrieve_history",
            "retrieve_trends": "retrieve_trends",
            "generate_recommendations": "generate_recommendations",
            "handle_error": "handle_error",
            "complete": "complete",
        },
    )
    for stage in (
        "analyze_media",
        "retrieve_history",
        "retrieve_trends",
    ):
        builder.add_edge(stage, "inspect_project")
    builder.add_conditional_edges(
        "generate_recommendations",
        choose_recommendation_mode,
        {
            "dev_mock_tot": "dev_mock_tot",
            "tot_plan": "tot_1_plan",
        },
    )
    builder.add_edge("dev_mock_tot", "inspect_project")
    builder.add_conditional_edges(
        "tot_1_plan",
        lambda state: continue_tot_or_error(state, "tot_expand"),
        {"tot_expand": "tot_2_expand", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "tot_2_expand",
        lambda state: continue_tot_or_error(state, "tot_rank"),
        {"tot_rank": "tot_3_rank", "handle_error": "handle_error"},
    )
    builder.add_edge("tot_3_rank", "inspect_project")
    builder.add_edge("handle_error", END)
    builder.add_edge("complete", END)
    return builder.compile(checkpointer=checkpointer)


class ForecastWorkflow:
    """Checkpointed LangGraph controller for deterministic pipeline stages."""

    def __init__(
        self,
        service: PipelineService,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.service = service
        self.graph = build_forecast_workflow(service, checkpointer=checkpointer)

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": WORKFLOW_CHECKPOINT_NAMESPACE,
            },
            "metadata": {"thread_id": thread_id, "workflow": "forecast"},
        }

    def initial_state(
        self,
        *,
        force_media_refresh: bool = False,
        force_trend_refresh: bool = False,
        force_recommendation_refresh: bool = False,
        recommendation_brief: str | None = None,
    ) -> ForecastWorkflowState:
        return {
            "dataset_path": str(self.service.artifacts.dataset_path),
            "force_media_refresh": force_media_refresh,
            "force_trend_refresh": force_trend_refresh,
            "force_recommendation_refresh": force_recommendation_refresh,
            "recommendation_brief": (recommendation_brief or "").strip(),
            "media_refreshed": False,
            "history_refreshed": False,
            "trends_refreshed": False,
            "recommendations_generated": False,
            "current_stage": "starting",
            "errors": [],
        }

    def invoke(
        self,
        *,
        thread_id: str,
        force_media_refresh: bool = False,
        force_trend_refresh: bool = False,
        force_recommendation_refresh: bool = False,
        recommendation_brief: str | None = None,
    ) -> ForecastWorkflowState:
        return self.graph.invoke(
            self.initial_state(
                force_media_refresh=force_media_refresh,
                force_trend_refresh=force_trend_refresh,
                force_recommendation_refresh=force_recommendation_refresh,
                recommendation_brief=recommendation_brief,
            ),
            config=self._config(thread_id),
        )

    def stream(
        self,
        *,
        thread_id: str,
        force_media_refresh: bool = False,
        force_trend_refresh: bool = False,
        force_recommendation_refresh: bool = False,
        recommendation_brief: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self.graph.stream(
            self.initial_state(
                force_media_refresh=force_media_refresh,
                force_trend_refresh=force_trend_refresh,
                force_recommendation_refresh=force_recommendation_refresh,
                recommendation_brief=recommendation_brief,
            ),
            config=self._config(thread_id),
            stream_mode="updates",
        )

    def get_state(self, *, thread_id: str):
        return self.graph.get_state(self._config(thread_id))
