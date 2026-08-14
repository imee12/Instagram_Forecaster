from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel, Field

from . import config
from .gemini_client import MODEL_NAME, get_or_create_client
from .trends import TrendReport


MAX_TREND_SIGNALS = 20
MAX_HISTORICAL_MATCHES_PER_MEDIA = 3
DEFAULT_CANDIDATE_COUNT = 6
DEFAULT_RECOMMENDATION_COUNT = 3
DEFAULT_THOUGHT_BRANCH_COUNT = 3
SCORE_WEIGHTS = {
    "historical_performance": 0.30,
    "trend_alignment": 0.30,
    "media_quality": 0.20,
    "audience_fit": 0.20,
}


class RecommendationScores(BaseModel):
    historical_performance: float = Field(ge=0, le=100)
    trend_alignment: float = Field(ge=0, le=100)
    media_quality: float = Field(ge=0, le=100)
    audience_fit: float = Field(ge=0, le=100)


class RecommendationCandidate(BaseModel):
    media_file: str = Field(description="Exact file name from the supplied media analyses.")
    post_format: Literal["reel", "carousel", "static_post", "story"]
    concept: str
    hook: str
    caption_direction: str
    rationale: str
    execution_notes: list[str] = Field(min_length=1)
    supporting_trends: list[str] = Field(
        min_length=1,
        description="Exact trend topics from the supplied trend signals."
    )
    historical_post_ids: list[str] = Field(
        min_length=1,
        description="Exact historical post IDs from the supplied matches."
    )
    scores: RecommendationScores
    confidence: float = Field(ge=0, le=100)


class RecommendationCandidates(BaseModel):
    candidates: list[RecommendationCandidate] = Field(min_length=1, max_length=8)


class RecommendationThoughtBranch(BaseModel):
    name: str = Field(description="Short label for this recommendation strategy.")
    hypothesis: str = Field(
        description="Concise, testable content strategy hypothesis; not hidden reasoning."
    )
    evidence_focus: list[str] = Field(
        min_length=1,
        description="Evidence dimensions this branch should prioritize.",
    )
    target_formats: list[Literal["reel", "carousel", "static_post", "story"]] = (
        Field(min_length=1)
    )


class RecommendationThoughtBranches(BaseModel):
    branches: list[RecommendationThoughtBranch] = Field(min_length=2, max_length=4)


def _recommendation_trace_inputs(inputs: dict) -> dict:
    media = inputs.get("media_analyses")
    historical = inputs.get("historical_matches")
    trend_report = inputs.get("trend_report")
    return {
        "media_count": len(media) if media is not None else 0,
        "historical_match_count": len(historical) if historical is not None else 0,
        "trend_signal_count": (
            len(trend_report.agent_signals) if trend_report is not None else 0
        ),
        "candidate_count": inputs.get("candidate_count", DEFAULT_CANDIDATE_COUNT),
        "recommendation_count": inputs.get(
            "recommendation_count", DEFAULT_RECOMMENDATION_COUNT
        ),
        "thought_branch_count": inputs.get(
            "thought_branch_count", DEFAULT_THOUGHT_BRANCH_COUNT
        ),
    }


def _recommendation_trace_outputs(output: pd.DataFrame | None) -> dict:
    if output is None:
        return {"status": "failed"}
    return {
        "recommendation_count": len(output),
        "recommendations": output[
            ["rank", "media_file", "post_format", "concept", "overall_score"]
        ].to_dict(orient="records"),
    }


def _json_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def _add_historical_performance_metrics(matches: pd.DataFrame) -> pd.DataFrame:
    enriched = matches.copy()
    engagement_columns = [
        column
        for column in ("likes", "saves", "shares", "reshare")
        if column in enriched.columns
    ]
    if engagement_columns:
        numeric_engagement = enriched[engagement_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
        enriched["engagement_total"] = numeric_engagement.sum(axis=1)
    else:
        enriched["engagement_total"] = 0.0

    if "views" in enriched.columns:
        views = pd.to_numeric(enriched["views"], errors="coerce").fillna(0)
        enriched["engagement_rate"] = (
            enriched["engagement_total"].div(views.where(views > 0)) * 100
        ).fillna(0)
    else:
        enriched["engagement_rate"] = 0.0

    enriched["performance_percentile"] = (
        enriched["engagement_rate"].rank(method="average", pct=True) * 100
    ).round(2)
    return enriched


def _prepare_context(
    media_analyses: pd.DataFrame,
    historical_matches: pd.DataFrame,
    trend_report: TrendReport,
) -> tuple[list[dict], list[dict], list[dict]]:
    if media_analyses.empty:
        raise ValueError("Cannot generate recommendations without analyzed media.")
    if historical_matches.empty:
        raise ValueError("Cannot generate recommendations without historical matches.")
    if trend_report.agent_signals.empty:
        raise ValueError("Cannot generate recommendations without trend signals.")

    historical = _add_historical_performance_metrics(historical_matches)
    if "retrieval_rank" in historical.columns:
        historical = historical.sort_values(["media_file", "retrieval_rank"])
    historical = historical.groupby("media_file", as_index=False, group_keys=False).head(
        MAX_HISTORICAL_MATCHES_PER_MEDIA
    )
    trends = trend_report.agent_signals.sort_values("rank").head(MAX_TREND_SIGNALS)

    return (
        _json_records(media_analyses),
        _json_records(historical),
        _json_records(trends),
    )


def _overall_score(scores: RecommendationScores) -> float:
    return round(
        (scores.historical_performance * SCORE_WEIGHTS["historical_performance"])
        + (scores.trend_alignment * SCORE_WEIGHTS["trend_alignment"])
        + (scores.media_quality * SCORE_WEIGHTS["media_quality"])
        + (scores.audience_fit * SCORE_WEIGHTS["audience_fit"]),
        2,
    )


def _validate_evidence(
    candidates: list[RecommendationCandidate],
    media_context: list[dict],
    historical_context: list[dict],
    trend_context: list[dict],
) -> None:
    media_files = {str(item.get("file_name")) for item in media_context}
    trend_topics = {str(item.get("topic")) for item in trend_context}
    historical_ids = {
        str(item.get("post_id"))
        for item in historical_context
        if item.get("post_id") is not None
    }

    for candidate in candidates:
        if candidate.media_file not in media_files:
            raise ValueError(
                f"Recommendation cited unknown media file: {candidate.media_file}"
            )
        unknown_trends = set(candidate.supporting_trends) - trend_topics
        if unknown_trends:
            raise ValueError(
                "Recommendation cited unknown trend topics: "
                + ", ".join(sorted(unknown_trends))
            )
        unknown_posts = set(candidate.historical_post_ids) - historical_ids
        if unknown_posts:
            raise ValueError(
                "Recommendation cited unknown historical post IDs: "
                + ", ".join(sorted(unknown_posts))
            )


def _candidate_allocations(candidate_count: int, branch_count: int) -> list[int]:
    base, remainder = divmod(candidate_count, branch_count)
    return [base + (1 if index < remainder else 0) for index in range(branch_count)]


@traceable(
    name="ToT 1 - Plan Strategy Branches",
    run_type="chain",
    tags=["tree-of-thoughts", "planning"],
)
def _plan_thought_branches(
    client,
    media_context: list[dict],
    historical_context: list[dict],
    trend_context: list[dict],
    branch_count: int,
) -> list[RecommendationThoughtBranch]:
    prompt = f"""
You are planning a bounded Tree-of-Thoughts search for Instagram content
recommendations for a young singer, actor, and fashion performer.

Create exactly {branch_count} distinct strategy branches. Each branch must be a
concise, testable hypothesis with a different evidence emphasis or creative
format. Do not provide hidden chain-of-thought. Do not invent facts. Return only
the structured branch plans.

MEDIA ANALYSES:
{json.dumps(media_context, indent=2)}

SIMILAR HISTORICAL POSTS:
{json.dumps(historical_context, indent=2)}

RANKED GOOGLE TREND SIGNALS:
{json.dumps(trend_context, indent=2)}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendationThoughtBranches,
            temperature=0.45,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        raise ValueError("Gemini returned no recommendation thought branches.")
    if len(parsed.branches) != branch_count:
        raise ValueError(
            f"Gemini returned {len(parsed.branches)} branches; expected {branch_count}."
        )
    return parsed.branches


@traceable(
    name="ToT 2 - Expand Thought Branch",
    run_type="chain",
    tags=["tree-of-thoughts", "expansion"],
)
def _expand_thought_branch(
    client,
    branch: RecommendationThoughtBranch,
    candidate_count: int,
    media_context: list[dict],
    historical_context: list[dict],
    trend_context: list[dict],
) -> list[RecommendationCandidate]:
    prompt = f"""
Expand this strategy branch into exactly {candidate_count} distinct Instagram
post candidates for a young singer, actor, and fashion performer.

STRATEGY BRANCH:
{branch.model_dump_json(indent=2)}

Use only the supplied facts and evidence. Do not invent media, historical
results, trend topics, events, songs, locations, partnerships, or backstory.
Every candidate must select one exact media file, cite at least one exact trend
topic and historical post_id, contain practical execution guidance, and score
the four evidence dimensions from 0 to 100. Return structured candidates only;
do not provide hidden chain-of-thought.

MEDIA ANALYSES:
{json.dumps(media_context, indent=2)}

SIMILAR HISTORICAL POSTS:
{json.dumps(historical_context, indent=2)}

RANKED GOOGLE TREND SIGNALS:
{json.dumps(trend_context, indent=2)}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendationCandidates,
            temperature=0.35,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        raise ValueError(f"Gemini returned no candidates for branch {branch.name!r}.")
    if len(parsed.candidates) != candidate_count:
        raise ValueError(
            f"Gemini returned {len(parsed.candidates)} candidates for branch "
            f"{branch.name!r}; expected {candidate_count}."
        )
    return parsed.candidates


@traceable(
    name="ToT 3 - Validate, Score, and Prune Candidates",
    run_type="chain",
    tags=["tree-of-thoughts", "evaluation", "pruning"],
)
def _rank_thought_candidates(
    expanded: list[tuple[RecommendationThoughtBranch, RecommendationCandidate]],
    media_context: list[dict],
    historical_context: list[dict],
    trend_context: list[dict],
    recommendation_count: int,
) -> pd.DataFrame:
    """Validate grounded evidence, score candidates, and retain the best leaves."""
    _validate_evidence(
        [candidate for _, candidate in expanded],
        media_context,
        historical_context,
        trend_context,
    )
    records = []
    for branch, candidate in expanded:
        record = candidate.model_dump()
        scores = record.pop("scores")
        record.update({f"{key}_score": value for key, value in scores.items()})
        record["overall_score"] = _overall_score(candidate.scores)
        record["thought_branch"] = branch.name
        record["branch_hypothesis"] = branch.hypothesis
        records.append(record)

    recommendations = (
        pd.DataFrame(records)
        .sort_values(["overall_score", "confidence"], ascending=False)
        .head(recommendation_count)
        .reset_index(drop=True)
    )
    recommendations.insert(0, "rank", range(1, len(recommendations) + 1))
    return recommendations


@traceable(
    name="DEV MODE - Mock ToT Recommendations",
    run_type="chain",
    tags=["tree-of-thoughts", "development-mode", "mock"],
)
def _generate_mock_tot_recommendations(
    media_context: list[dict],
    historical_context: list[dict],
    trend_context: list[dict],
    recommendation_count: int,
) -> pd.DataFrame:
    """Build deterministic, evidence-grounded stand-ins for Gemini ToT output."""
    media_files = [str(item["file_name"]) for item in media_context]
    trend_topics = [str(item["topic"]) for item in trend_context]
    historical_ids = [
        str(item["post_id"])
        for item in historical_context
        if item.get("post_id") is not None
    ]
    formats = ["reel", "carousel", "story", "static_post"]
    expanded = []
    for index in range(recommendation_count):
        branch = RecommendationThoughtBranch(
            name=f"Development mock branch {index + 1}",
            hypothesis="Validate the recommendation workflow with saved project evidence.",
            evidence_focus=["workflow integration", "trace visibility"],
            target_formats=[formats[index % len(formats)]],
        )
        candidate = RecommendationCandidate(
            media_file=media_files[index % len(media_files)],
            post_format=formats[index % len(formats)],
            concept=f"Development recommendation {index + 1}",
            hook="A development-mode preview grounded in the current project artifacts.",
            caption_direction="Replace this mock copy when running with DEV_MODE=false.",
            rationale="This deterministic result exercises the pipeline without Gemini ToT calls.",
            execution_notes=["Use this output to validate LangGraph, LangSmith, and the UI."],
            supporting_trends=[trend_topics[index % len(trend_topics)]],
            historical_post_ids=[historical_ids[index % len(historical_ids)]],
            scores=RecommendationScores(
                historical_performance=75 - index,
                trend_alignment=75 - index,
                media_quality=75 - index,
                audience_fit=75 - index,
            ),
            confidence=75 - index,
        )
        expanded.append((branch, candidate))

    return _rank_thought_candidates(
        expanded,
        media_context,
        historical_context,
        trend_context,
        recommendation_count,
    )


@traceable(
    name="Generate Content Recommendations",
    run_type="chain",
    process_inputs=_recommendation_trace_inputs,
    process_outputs=_recommendation_trace_outputs,
)
def generate_content_recommendations(
    media_analyses: pd.DataFrame,
    historical_matches: pd.DataFrame,
    trend_report: TrendReport,
    *,
    client_instance=None,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    thought_branch_count: int = DEFAULT_THOUGHT_BRANCH_COUNT,
) -> pd.DataFrame:
    if candidate_count < recommendation_count:
        raise ValueError("candidate_count cannot be smaller than recommendation_count.")
    if not 2 <= thought_branch_count <= 4:
        raise ValueError("thought_branch_count must be between 2 and 4.")
    if thought_branch_count > candidate_count:
        raise ValueError("thought_branch_count cannot exceed candidate_count.")

    media_context, historical_context, trend_context = _prepare_context(
        media_analyses,
        historical_matches,
        trend_report,
    )
    if config.DEV_MODE:
        print("[IG Forecaster] DEV_MODE=true: using mock ToT recommendation output.")
        return _generate_mock_tot_recommendations(
            media_context,
            historical_context,
            trend_context,
            recommendation_count,
        )

    print("[IG Forecaster] DEV_MODE=false: using real Gemini ToT calls.")
    client = client_instance or get_or_create_client()
    branches = _plan_thought_branches(
        client,
        media_context,
        historical_context,
        trend_context,
        thought_branch_count,
    )
    allocations = _candidate_allocations(candidate_count, thought_branch_count)
    expanded: list[tuple[RecommendationThoughtBranch, RecommendationCandidate]] = []
    for branch, allocation in zip(branches, allocations):
        candidates = _expand_thought_branch(
            client,
            branch,
            allocation,
            media_context,
            historical_context,
            trend_context,
        )
        expanded.extend((branch, candidate) for candidate in candidates)

    return _rank_thought_candidates(
        expanded,
        media_context,
        historical_context,
        trend_context,
        recommendation_count,
    )


def save_content_recommendations(
    recommendations: pd.DataFrame,
    output_folder: Path,
) -> tuple[Path, Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    json_path = output_folder / "post_recommendations.json"
    csv_path = output_folder / "post_recommendations.csv"

    records = _json_records(recommendations)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    csv_frame = recommendations.copy()
    for column in ("execution_notes", "supporting_trends", "historical_post_ids"):
        if column in csv_frame.columns:
            csv_frame[column] = csv_frame[column].apply(json.dumps)
    csv_frame.to_csv(csv_path, index=False)
    return json_path, csv_path
