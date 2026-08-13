"""LangGraph Studio entry point for the conversational forecaster agent."""

from ig_forecaster.agent.graph import build_agent
from ig_forecaster.agent.workflow import ForecastWorkflow
from ig_forecaster.pipeline import PipelineService


service = PipelineService()
workflow = ForecastWorkflow(service)
graph = build_agent(service, workflow=workflow)
