from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import PipelineService, analyze_media_file


def run_pipeline(
    dataset_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """Run every stage while preserving the original public return value."""
    return PipelineService(dataset_path=dataset_path).run_all().as_legacy_tuple()


def run_workflow(
    dataset_path: str | Path | None = None,
    *,
    thread_id: str = "cli",
    force_media_refresh: bool = False,
    force_trend_refresh: bool = False,
    force_recommendation_refresh: bool = False,
    recommendation_brief: str | None = None,
) -> dict:
    """Run the cache-aware pipeline through its LangGraph workflow."""
    # Imported lazily so legacy pipeline users do not need to initialize the
    # agent package merely to call ``run_pipeline``.
    from .agent.workflow import ForecastWorkflow

    service = PipelineService(dataset_path=dataset_path)
    workflow = ForecastWorkflow(service)
    return dict(
        workflow.invoke(
            thread_id=thread_id,
            force_media_refresh=force_media_refresh,
            force_trend_refresh=force_trend_refresh,
            force_recommendation_refresh=force_recommendation_refresh,
            recommendation_brief=recommendation_brief,
        )
    )


if __name__ == "__main__":
    run_workflow()
