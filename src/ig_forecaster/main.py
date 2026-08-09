from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import PipelineService, analyze_media_file


def run_pipeline(
    dataset_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """Run every stage while preserving the original public return value."""
    return PipelineService(dataset_path=dataset_path).run_all().as_legacy_tuple()


if __name__ == "__main__":
    run_pipeline()
