"""LangGraph Studio entry point for the deterministic forecast workflow."""

from ig_forecaster.agent.workflow import build_forecast_workflow
from ig_forecaster.pipeline import PipelineService


# Agent Server imports this module and discovers this compiled graph through
# ``langgraph.json``. Persistence is supplied by Agent Server in Studio.
service = PipelineService()
graph = build_forecast_workflow(service)
