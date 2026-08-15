from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from langsmith import traceable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in lightweight environments
    np = None

from .config import get_index_folder
from .data import load_posts


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
DEFAULT_MIN_SIMILARITY = 0.35
MIN_HEALTHY_INDEX_POSTS = 10
MIN_HEALTHY_MATCHES_PER_MEDIA = 2


class EmptyHistoricalIndex:
    """No-op index used while a project has no historical posts."""

    ntotal = 0
    d = 0

    def search(self, queries: object, top_k: int) -> tuple[object, object]:
        query_count = len(queries)
        return (
            np.empty((query_count, 0), dtype="float32"),
            np.empty((query_count, 0), dtype="int64"),
        )


class NumpyFlatIPIndex:
    """Safe exact cosine search for normalized embeddings."""

    def __init__(self, embeddings: object):
        self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        if self.embeddings.ndim != 2:
            raise ValueError("Historical embeddings must be a two-dimensional matrix.")
        self.ntotal, self.d = self.embeddings.shape

    def search(self, queries: object, top_k: int) -> tuple[object, object]:
        query_matrix = np.ascontiguousarray(queries, dtype="float32")
        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.d:
            raise ValueError(
                f"Query embedding dimension {query_matrix.shape} does not match index dimension {self.d}."
            )

        similarities = query_matrix @ self.embeddings.T
        result_count = min(top_k, self.ntotal)
        positions = np.argsort(-similarities, axis=1)[:, :result_count]
        scores = np.take_along_axis(similarities, positions, axis=1)
        return scores.astype("float32"), positions.astype("int64")


def _index_trace_inputs(inputs: dict) -> dict:
    posts = inputs.get("posts")
    return {
        "post_count": len(posts) if posts is not None else None,
        "index_folder": str(inputs.get("index_folder")) if inputs.get("index_folder") else None,
        "dataset_path": str(inputs.get("dataset_path")) if inputs.get("dataset_path") else None,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


def _index_trace_outputs(output: tuple) -> dict:
    index, metadata, _ = output
    return {
        "indexed_post_count": len(metadata),
        "vector_count": getattr(index, "ntotal", None),
    }


def _search_trace_inputs(inputs: dict) -> dict:
    return {
        "query_text": inputs.get("query_text"),
        "top_k": inputs.get("top_k", DEFAULT_TOP_K),
        "min_similarity": inputs.get("min_similarity", DEFAULT_MIN_SIMILARITY),
    }


def _search_trace_outputs(output: pd.DataFrame) -> dict:
    documents = []
    for row in output.to_dict(orient="records"):
        page_content = row.pop("retrieval_text", row.get("description", ""))
        documents.append({"page_content": page_content, "metadata": row})
    return {"documents": documents}


@lru_cache(maxsize=1)
def _sentence_transformer_class():
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Install compatible sentence-transformers and transformers packages "
            "to use historical retrieval."
        ) from exc
    return SentenceTransformer


def _require_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")

    if missing:
        raise ImportError(
            "Missing optional dependencies for retrieval: "
            + ", ".join(missing)
        )
    _sentence_transformer_class()


@traceable(
    name="Build or Load Historical Index",
    run_type="retriever",
    process_inputs=_index_trace_inputs,
    process_outputs=_index_trace_outputs,
)
def build_or_load_index(posts: pd.DataFrame | None = None, index_folder: Path | None = None, dataset_path: str | Path | None = None) -> tuple[object, pd.DataFrame, object]:
    _require_runtime_dependencies()
    sentence_transformer = _sentence_transformer_class()
    index_folder = index_folder or get_index_folder(dataset_path=dataset_path)
    embeddings_path = index_folder / "historical_posts_embeddings.npy"
    metadata_path = index_folder / "historical_posts_metadata.csv"

    posts = posts if posts is not None else load_posts()

    if posts.empty:
        return EmptyHistoricalIndex(), posts.copy(), None

    if embeddings_path.exists() and metadata_path.exists():
        embedding_model = sentence_transformer(EMBEDDING_MODEL_NAME)
        metadata = pd.read_csv(metadata_path)
        saved_embeddings = np.load(embeddings_path)
        if len(metadata) == len(saved_embeddings) == len(posts):
            return NumpyFlatIPIndex(saved_embeddings), metadata, embedding_model

    embedding_model = sentence_transformer(EMBEDDING_MODEL_NAME)
    post_embeddings = embedding_model.encode(
        posts["retrieval_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    post_embeddings = np.asarray(post_embeddings, dtype="float32")

    safe_index = NumpyFlatIPIndex(post_embeddings)

    index_folder.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, post_embeddings)
    posts.to_csv(metadata_path, index=False)

    return safe_index, posts, embedding_model


def build_media_query_text(analysis: Mapping[str, object]) -> str:
    def join_values(key: str) -> str:
        value = analysis.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return ", ".join(str(item) for item in value if item)
        return str(value) if value else ""

    parts = [
        f"Visual content: {join_values('visual_summary')}",
        f"Media type: {join_values('media_type')}",
        f"Themes: {join_values('themes')}",
        f"Content categories: {join_values('content_categories')}",
        f"Possible uses: {join_values('possible_post_uses')}",
    ]
    return ". ".join(part for part in parts if not part.endswith(": "))


@traceable(
    name="Retrieve Similar Historical Posts",
    run_type="retriever",
    process_inputs=_search_trace_inputs,
    process_outputs=_search_trace_outputs,
)
def retrieve_similar_posts(
    query_text: str,
    index: object,
    metadata: pd.DataFrame,
    embedding_model: object,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> pd.DataFrame:
    _require_runtime_dependencies()
    if not query_text.strip():
        raise ValueError("Historical retrieval query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between -1 and 1.")
    if metadata.empty:
        return pd.DataFrame(columns=["retrieval_rank", "similarity_score", *metadata.columns])

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
    expected_dimension = getattr(index, "d", query_embedding.shape[1])
    if query_embedding.shape != (1, expected_dimension):
        raise ValueError(
            "Query embedding shape does not match the historical index: "
            f"{query_embedding.shape} versus dimension {expected_dimension}."
        )
    result_count = min(top_k, len(metadata))
    scores, positions = index.search(query_embedding, result_count)

    records = []
    for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
        if position < 0 or position >= len(metadata):
            continue
        if float(score) < min_similarity:
            continue
        record = metadata.iloc[int(position)].to_dict()
        record["retrieval_rank"] = rank
        record["similarity_score"] = round(float(score), 6)
        records.append(record)

    leading_columns = ["retrieval_rank", "similarity_score"]
    return pd.DataFrame(records).reindex(columns=[*leading_columns, *metadata.columns])


@traceable(name="Match Media to Historical Posts", run_type="chain")
def retrieve_historical_matches(
    media_analyses: pd.DataFrame,
    index: object,
    metadata: pd.DataFrame,
    embedding_model: object,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> pd.DataFrame:
    match_frames = []
    for analysis in media_analyses.to_dict(orient="records"):
        matches = retrieve_similar_posts(
            build_media_query_text(analysis),
            index,
            metadata,
            embedding_model,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        matches.insert(0, "media_file", analysis.get("file_name"))
        matches.insert(1, "media_path", analysis.get("file_path"))
        match_frames.append(matches)

    if not match_frames:
        results = pd.DataFrame(
            columns=["media_file", "media_path", "retrieval_rank", "similarity_score"]
        )
    else:
        results = pd.concat(match_frames, ignore_index=True)

    index_size = len(metadata)
    if index_size == 0 or results.empty:
        mode = "cold_start"
    else:
        matches_per_media = results.groupby("media_file").size()
        every_media_has_support = (
            len(matches_per_media) == len(media_analyses)
            and matches_per_media.min() >= MIN_HEALTHY_MATCHES_PER_MEDIA
        )
        mode = (
            "healthy"
            if index_size >= MIN_HEALTHY_INDEX_POSTS and every_media_has_support
            else "sparse"
        )

    results["historical_index_size"] = index_size
    results["historical_evidence_mode"] = mode
    return results
