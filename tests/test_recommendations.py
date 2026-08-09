from pathlib import Path

import pandas as pd
import pytest

from ig_forecaster.recommendations import (
    RecommendationCandidate,
    RecommendationCandidates,
    RecommendationScores,
    generate_content_recommendations,
    save_content_recommendations,
)
from ig_forecaster.trends import TrendReport


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeModels:
    def __init__(self, parsed):
        self.parsed = parsed

    def generate_content(self, **kwargs):
        assert "MEDIA ANALYSES" in kwargs["contents"]
        assert kwargs["config"].response_schema is RecommendationCandidates
        return FakeResponse(self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.models = FakeModels(parsed)


def make_candidate(media_file, concept, scores, confidence=80):
    return RecommendationCandidate(
        media_file=media_file,
        post_format="reel",
        concept=concept,
        hook="Watch the rehearsal come together",
        caption_direction="Invite followers behind the scenes.",
        rationale="The media, trend, and historical post all concern rehearsal.",
        execution_notes=["Open on the strongest moment"],
        supporting_trends=["rehearsal clips"],
        historical_post_ids=["1"],
        scores=RecommendationScores(**scores),
        confidence=confidence,
    )


def recommendation_context():
    media = pd.DataFrame(
        [
            {
                "file_name": "rehearsal.mov",
                "file_path": "/media/rehearsal.mov",
                "media_type": "video",
                "visual_summary": "Singer rehearsing on stage",
                "themes": ["music"],
                "content_categories": ["performance"],
                "possible_post_uses": ["Reel"],
                "quality_notes": "Clear and well lit",
            }
        ]
    )
    historical = pd.DataFrame(
        [
            {
                "media_file": "rehearsal.mov",
                "retrieval_rank": 1,
                "similarity_score": 0.9,
                "post_id": 1,
                "description": "Past rehearsal",
                "views": 1000,
                "likes": 100,
                "saves": 10,
                "shares": 5,
            }
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "rank": 1,
                "signal_type": "rising_related_query",
                "topic": "rehearsal clips",
                "source_keyword": "behind the scenes",
                "trend_direction": "rising",
                "signal_score": 90,
            }
        ]
    )
    report = TrendReport(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), signals)
    return media, historical, report


def test_generate_content_recommendations_scores_and_ranks_candidates():
    media, historical, report = recommendation_context()
    parsed = RecommendationCandidates(
        candidates=[
            make_candidate(
                "rehearsal.mov",
                "Medium candidate",
                {
                    "historical_performance": 60,
                    "trend_alignment": 60,
                    "media_quality": 60,
                    "audience_fit": 60,
                },
            ),
            make_candidate(
                "rehearsal.mov",
                "Best candidate",
                {
                    "historical_performance": 90,
                    "trend_alignment": 90,
                    "media_quality": 80,
                    "audience_fit": 80,
                },
            ),
            make_candidate(
                "rehearsal.mov",
                "Lowest candidate",
                {
                    "historical_performance": 40,
                    "trend_alignment": 40,
                    "media_quality": 50,
                    "audience_fit": 50,
                },
            ),
        ]
    )

    recommendations = generate_content_recommendations(
        media,
        historical,
        report,
        client_instance=FakeClient(parsed),
        candidate_count=3,
        recommendation_count=3,
    )

    assert recommendations.iloc[0]["concept"] == "Best candidate"
    assert recommendations["rank"].tolist() == [1, 2, 3]
    assert recommendations.iloc[0]["overall_score"] == pytest.approx(86)


def test_recommendations_reject_unknown_evidence():
    media, historical, report = recommendation_context()
    candidates = [
        make_candidate(
            "unknown.mov" if number == 0 else "rehearsal.mov",
            f"Candidate {number}",
            {
                "historical_performance": 70,
                "trend_alignment": 70,
                "media_quality": 70,
                "audience_fit": 70,
            },
        )
        for number in range(3)
    ]

    with pytest.raises(ValueError, match="unknown media"):
        generate_content_recommendations(
            media,
            historical,
            report,
            client_instance=FakeClient(RecommendationCandidates(candidates=candidates)),
            candidate_count=3,
        )


def test_save_content_recommendations_writes_json_and_csv(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "rank": 1,
                "media_file": "rehearsal.mov",
                "execution_notes": ["Use captions"],
                "supporting_trends": ["rehearsal clips"],
                "historical_post_ids": ["1"],
            }
        ]
    )

    json_path, csv_path = save_content_recommendations(frame, Path(tmp_path))

    assert json_path.exists()
    assert csv_path.exists()
