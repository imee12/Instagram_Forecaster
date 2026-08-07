from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_csv_path, resolve_dataset_path


def build_post_text(row: pd.Series) -> str:
    return (
        f"Description: {row['description']}. "
        f"Media type: {row['media_type']}. "
        f"Content category: {row['category']}."
    )


def load_posts(csv_path: Path | str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = get_csv_path()
    else:
        csv_path = resolve_dataset_path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {csv_path}. "
            "Pass either the CSV file path or a project folder containing IG_Forecaster.csv."
        )

    posts = pd.read_csv(csv_path)
    posts["retrieval_text"] = posts.apply(build_post_text, axis=1)
    return posts
