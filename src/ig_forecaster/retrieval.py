from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import faiss
except ImportError:  # pragma: no cover - exercised in lightweight environments
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in lightweight environments
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised in lightweight environments
    SentenceTransformer = None

from .config import get_index_folder
from .data import load_posts


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _require_runtime_dependencies() -> None:
    missing = []
    if faiss is None:
        missing.append("faiss-cpu")
    if np is None:
        missing.append("numpy")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")

    if missing:
        raise ImportError(
            "Missing optional dependencies for retrieval: "
            + ", ".join(missing)
        )


def build_or_load_index(posts: pd.DataFrame | None = None, index_folder: Path | None = None, dataset_path: str | Path | None = None) -> tuple[object, pd.DataFrame, object]:
    index_folder = index_folder or get_index_folder(dataset_path=dataset_path)
    index_path = index_folder / "historical_posts.index"
    metadata_path = index_folder / "historical_posts_metadata.csv"

    posts = posts if posts is not None else load_posts()

    if index_path.exists() and metadata_path.exists():
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        index = faiss.read_index(str(index_path))
        metadata = pd.read_csv(metadata_path)
        return index, metadata, embedding_model

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    post_embeddings = embedding_model.encode(
        posts["retrieval_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    post_embeddings = np.asarray(post_embeddings, dtype="float32")

    dimension = post_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(post_embeddings)

    index_folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    posts.to_csv(metadata_path, index=False)

    return index, posts, embedding_model
