import numpy as np
import pandas as pd
import pytest

from ig_forecaster.retrieval import (
    NumpyFlatIPIndex,
    build_media_query_text,
    retrieve_historical_matches,
    retrieve_similar_posts,
)


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        assert texts
        assert kwargs["normalize_embeddings"] is True
        return np.asarray([[0.25, 0.75]], dtype="float32")


class FakeIndex:
    def search(self, embedding, top_k):
        assert embedding.shape == (1, 2)
        return (
            np.asarray([[0.91, 0.72]], dtype="float32")[:, :top_k],
            np.asarray([[1, 0]], dtype="int64")[:, :top_k],
        )


def historical_posts():
    return pd.DataFrame(
        [
            {"post_id": 1, "description": "Fashion portrait", "retrieval_text": "fashion"},
            {"post_id": 2, "description": "Rehearsal video", "retrieval_text": "rehearsal"},
        ]
    )


def test_numpy_index_returns_highest_inner_product_first():
    index = NumpyFlatIPIndex(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype="float32",
        )
    )

    scores, positions = index.search(np.asarray([[0.0, 1.0]], dtype="float32"), 2)

    assert positions.tolist() == [[1, 2]]
    assert scores.tolist() == [[1.0, 0.5]]


def test_build_media_query_text_combines_structured_analysis():
    query = build_media_query_text(
        {
            "visual_summary": "Singer rehearsing on stage",
            "media_type": "video",
            "themes": ["music", "behind the scenes"],
            "content_categories": ["performance"],
            "possible_post_uses": ["Reel"],
        }
    )

    assert "Singer rehearsing on stage" in query
    assert "music, behind the scenes" in query
    assert "Content categories: performance" in query


def test_retrieve_similar_posts_returns_ranked_metadata(monkeypatch):
    monkeypatch.setattr("ig_forecaster.retrieval._require_runtime_dependencies", lambda: None)

    results = retrieve_similar_posts(
        "stage rehearsal",
        FakeIndex(),
        historical_posts(),
        FakeEmbeddingModel(),
        top_k=2,
    )

    assert results["post_id"].tolist() == [2, 1]
    assert results["retrieval_rank"].tolist() == [1, 2]
    assert results.iloc[0]["similarity_score"] == pytest.approx(0.91)


def test_retrieve_similar_posts_filters_weak_matches(monkeypatch):
    monkeypatch.setattr("ig_forecaster.retrieval._require_runtime_dependencies", lambda: None)

    results = retrieve_similar_posts(
        "stage rehearsal",
        FakeIndex(),
        historical_posts(),
        FakeEmbeddingModel(),
        top_k=2,
        min_similarity=0.8,
    )

    assert results["post_id"].tolist() == [2]


def test_retrieve_historical_matches_identifies_source_media(monkeypatch):
    monkeypatch.setattr("ig_forecaster.retrieval._require_runtime_dependencies", lambda: None)
    analyses = pd.DataFrame(
        [
            {
                "file_name": "rehearsal.mov",
                "file_path": "/media/rehearsal.mov",
                "visual_summary": "Singer rehearsing",
                "media_type": "video",
                "themes": ["music"],
                "content_categories": ["performance"],
                "possible_post_uses": ["Reel"],
            }
        ]
    )

    results = retrieve_historical_matches(
        analyses,
        FakeIndex(),
        historical_posts(),
        FakeEmbeddingModel(),
        top_k=2,
    )

    assert len(results) == 2
    assert results["media_file"].unique().tolist() == ["rehearsal.mov"]
    assert results["historical_evidence_mode"].unique().tolist() == ["sparse"]
    assert results["historical_index_size"].unique().tolist() == [2]
