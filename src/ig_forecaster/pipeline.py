from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import warnings

import pandas as pd
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel

from .config import get_media_folder, get_output_folder, get_project_root, resolve_dataset_path
from .data import load_posts
from .gemini_client import MODEL_NAME, get_or_create_client, wait_until_ready
from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaAnalysis, find_media_files_in_project
from .recommendations import generate_content_recommendations, save_content_recommendations
from .retrieval import build_or_load_index, retrieve_historical_matches
from .trends import (
    TrendReport,
    load_cached_trend_report,
    retrieve_google_trends,
    save_trend_report,
    trend_cache_is_fresh,
    trend_report_paths,
)


class ProjectArtifacts(BaseModel):
    project_root: Path
    dataset_path: Path
    media_folder: Path
    output_folder: Path
    media_cache_path: Path
    media_analyses_path: Path
    media_errors_path: Path
    historical_matches_path: Path
    trend_signals_path: Path
    recommendations_json_path: Path
    recommendations_csv_path: Path
    local_trend_cache_folder: Path


@dataclass
class PipelineRunResult:
    media_analyses: pd.DataFrame
    media_errors: pd.DataFrame
    historical_matches: pd.DataFrame
    trend_report: TrendReport
    recommendations: pd.DataFrame
    retrieval_runtime: tuple[object, pd.DataFrame, object]

    def as_legacy_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
        return self.media_analyses, self.media_errors, self.retrieval_runtime


def _media_trace_inputs(inputs: dict) -> dict:
    return {"file_path": str(inputs.get("path"))}


def _media_trace_outputs(output: MediaAnalysis | None) -> dict:
    return output.model_dump() if output is not None else {"status": "failed"}


@traceable(
    name="Analyze Media Asset",
    run_type="tool",
    process_inputs=_media_trace_inputs,
    process_outputs=_media_trace_outputs,
)
def analyze_media_file(path: Path, client_instance=None) -> MediaAnalysis:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        raise ValueError(f"Unsupported media type: {path}")

    uploaded_file = None
    client = client_instance or get_or_create_client()
    try:
        print(f"Uploading {path.name}...")
        uploaded_file = client.files.upload(file=str(path))
        uploaded_file = wait_until_ready(uploaded_file, client_instance=client)
        prompt = f"""
Analyze this {media_type} as an available asset for an Instagram
content-planning agent.

The account belongs to a young singer, actor, and fashion performer.

Describe only what can reasonably be observed in the media. Do not
invent a song title, campaign, event, date, location, or backstory.

Evaluate:
- visible subject, setting, clothing, objects, and actions
- tone and themes
- possible Instagram content categories
- possible post uses
- lighting, framing, clarity, audio, and usability
- audible speech or singing, when present
- strongest opening or usable moment for video

The exact file name is {path.name}.
The media type is {media_type}.
"""
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MediaAnalysis,
                temperature=0.2,
            ),
        )
        analysis = response.parsed
        if analysis is None:
            raise ValueError(f"Gemini returned no structured analysis for {path.name}")
        analysis.file_name = path.name
        analysis.media_type = media_type
        return analysis
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as exc:
                print(f"Temporary upload cleanup failed for {path.name}: {exc}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        if isinstance(exc, OSError):
            warnings.warn(
                f"Saved artifact could not be read: {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return pd.DataFrame()


def _read_json_records(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        warnings.warn(
            f"Saved artifact could not be read: {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()


class PipelineService:
    """Independent pipeline stages for CLI, UI, and agent callers."""

    def __init__(self, dataset_path: str | Path | None = None):
        resolved_dataset_path = resolve_dataset_path(dataset_path)
        project_root = get_project_root(dataset_path=resolved_dataset_path)
        output_folder = get_output_folder(dataset_path=resolved_dataset_path)
        output_folder.mkdir(parents=True, exist_ok=True)
        local_trend_cache_folder = Path(
            os.getenv(
                "IG_FORECASTER_LOCAL_CACHE",
                str(Path.cwd() / ".ig_forecaster_cache"),
            )
        ).expanduser()
        local_trend_cache_folder.mkdir(parents=True, exist_ok=True)
        self.artifacts = ProjectArtifacts(
            project_root=project_root,
            dataset_path=resolved_dataset_path,
            media_folder=get_media_folder(dataset_path=resolved_dataset_path),
            output_folder=output_folder,
            media_cache_path=project_root / ".media_analysis_cache.json",
            media_analyses_path=output_folder / "media_analyses.json",
            media_errors_path=output_folder / "media_analysis_errors.csv",
            historical_matches_path=output_folder / "historical_media_matches.csv",
            trend_signals_path=output_folder / "google_trends_agent_signals.csv",
            recommendations_json_path=output_folder / "post_recommendations.json",
            recommendations_csv_path=output_folder / "post_recommendations.csv",
            local_trend_cache_folder=local_trend_cache_folder,
        )
        self._retrieval_runtime: tuple[object, pd.DataFrame, object] | None = None

    def load_project(self) -> ProjectArtifacts:
        """Return artifact locations without invoking external services."""
        return self.artifacts

    def load_saved_media_analyses(self) -> pd.DataFrame:
        path = self.artifacts.media_analyses_path
        analyses = _read_json_records(path)
        if not analyses.empty:
            return analyses
        cache = self._load_analysis_cache()
        return pd.DataFrame(cache.values())

    def load_saved_media_errors(self) -> pd.DataFrame:
        return _read_csv(self.artifacts.media_errors_path)

    def load_saved_historical_matches(self) -> pd.DataFrame:
        return _read_csv(self.artifacts.historical_matches_path)

    def load_saved_trends(self) -> TrendReport | None:
        local_report = load_cached_trend_report(
            self.artifacts.local_trend_cache_folder
        )
        if local_report is not None:
            return local_report

        report = load_cached_trend_report(self.artifacts.output_folder)
        if report is not None:
            save_trend_report(report, self.artifacts.local_trend_cache_folder)
        return report

    def load_saved_recommendations(self) -> pd.DataFrame:
        return _read_json_records(self.artifacts.recommendations_json_path)

    def _load_analysis_cache(self) -> dict[str, dict]:
        path = self.artifacts.media_cache_path
        try:
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError) as exc:
            warnings.warn(
                f"Media analysis cache could not be read: {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}

    def _save_media_results(
        self,
        analyses: pd.DataFrame,
        errors: pd.DataFrame,
        cache: dict[str, dict],
    ) -> None:
        self.artifacts.media_cache_path.write_text(
            json.dumps(cache, indent=2), encoding="utf-8"
        )
        analysis_records = json.loads(analyses.to_json(orient="records"))
        self.artifacts.media_analyses_path.write_text(
            json.dumps(analysis_records, indent=2), encoding="utf-8"
        )
        errors.to_csv(self.artifacts.media_errors_path, index=False)

    @traceable(name="Analyze Project Media", run_type="chain")
    def analyze_media(self, *, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        media_files = find_media_files_in_project(
            project_root=self.artifacts.project_root,
            media_folder=self.artifacts.media_folder,
        )
        cache = self._load_analysis_cache()
        analyses: list[dict] = []
        errors: list[dict] = []

        for number, path in enumerate(media_files, start=1):
            print(f"\n[{number}/{len(media_files)}] {path.name}")
            cache_key = str(path.resolve())
            if not force and cache_key in cache:
                analyses.append(cache[cache_key])
                print("Loaded from cache.")
                continue

            try:
                result = analyze_media_file(path)
                record = result.model_dump()
                record["file_path"] = str(path)
                cache[cache_key] = record
                analyses.append(record)
                print("Completed.")
            except Exception as exc:
                errors.append(
                    {"file_name": path.name, "file_path": str(path), "error": str(exc)}
                )
                print(f"Failed: {exc}")

        analysis_frame = pd.DataFrame(analyses)
        error_frame = pd.DataFrame(errors, columns=["file_name", "file_path", "error"])
        self._save_media_results(analysis_frame, error_frame, cache)
        return analysis_frame, error_frame

    def load_or_build_historical_index(self) -> tuple[object, pd.DataFrame, object]:
        if self._retrieval_runtime is None:
            posts = load_posts(self.artifacts.dataset_path)
            self._retrieval_runtime = build_or_load_index(
                posts=posts,
                dataset_path=self.artifacts.dataset_path,
            )
        return self._retrieval_runtime

    @traceable(name="Retrieve Project Historical Matches", run_type="chain")
    def retrieve_history(
        self,
        media_analyses: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        analyses = (
            media_analyses
            if media_analyses is not None
            else self.load_saved_media_analyses()
        )
        if analyses.empty:
            raise ValueError("Analyze media before retrieving historical matches.")
        index, metadata, embedding_model = self.load_or_build_historical_index()
        matches = retrieve_historical_matches(
            analyses,
            index,
            metadata,
            embedding_model,
        )
        matches.to_csv(self.artifacts.historical_matches_path, index=False)
        return matches

    @traceable(name="Retrieve Project Trends", run_type="chain")
    def retrieve_trends(self, *, force_refresh: bool = False) -> TrendReport:
        cached = self.load_saved_trends()
        if (
            not force_refresh
            and cached is not None
            and trend_cache_is_fresh(self.artifacts.output_folder)
        ):
            print("Using Google Trends cache from the last six hours.")
            return cached

        try:
            report = retrieve_google_trends()
            save_trend_report(report, self.artifacts.output_folder)
            save_trend_report(report, self.artifacts.local_trend_cache_folder)
            return report
        except Exception as exc:
            if cached is None:
                raise RuntimeError(
                    "Google Trends retrieval failed and no cached trend report is available."
                ) from exc
            print(f"Google Trends refresh failed; using the last saved report: {exc}")
            return cached

    @traceable(name="Generate Project Recommendations", run_type="chain")
    def generate_recommendations(
        self,
        media_analyses: pd.DataFrame | None = None,
        historical_matches: pd.DataFrame | None = None,
        trend_report: TrendReport | None = None,
    ) -> pd.DataFrame:
        analyses = (
            media_analyses
            if media_analyses is not None
            else self.load_saved_media_analyses()
        )
        matches = (
            historical_matches
            if historical_matches is not None
            else self.load_saved_historical_matches()
        )
        trends = trend_report if trend_report is not None else self.load_saved_trends()
        if trends is None:
            raise ValueError("Retrieve trends before generating recommendations.")

        recommendations = generate_content_recommendations(analyses, matches, trends)
        save_content_recommendations(recommendations, self.artifacts.output_folder)
        return recommendations

    @traceable(name="IG Forecaster Pipeline", run_type="chain")
    def run_all(self) -> PipelineRunResult:
        print(f"Project root: {self.artifacts.project_root}")
        print(f"Dataset path: {self.artifacts.dataset_path}")
        self.load_or_build_historical_index()

        media_analyses, media_errors = self.analyze_media()

        print("\nRetrieving similar historical posts...")
        historical_matches = self.retrieve_history(media_analyses)
        print(f"Saved historical matches: {self.artifacts.historical_matches_path}")

        print("\nRetrieving Google Trends data...")
        trend_report = self.retrieve_trends()
        for label, path in zip(
            (
                "trend interest",
                "related queries",
                "keyword momentum",
                "agent-ready trend signals",
            ),
            trend_report_paths(self.artifacts.output_folder),
        ):
            print(f"Saved {label}: {path}")

        print("\nGenerating content recommendations...")
        recommendations = self.generate_recommendations(
            media_analyses,
            historical_matches,
            trend_report,
        )
        print(f"Saved recommendations JSON: {self.artifacts.recommendations_json_path}")
        print(f"Saved recommendations CSV: {self.artifacts.recommendations_csv_path}")

        return PipelineRunResult(
            media_analyses=media_analyses,
            media_errors=media_errors,
            historical_matches=historical_matches,
            trend_report=trend_report,
            recommendations=recommendations,
            retrieval_runtime=self.load_or_build_historical_index(),
        )
