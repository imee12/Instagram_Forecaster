from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from google.genai import types
from langsmith import traceable

from .config import get_media_folder, get_output_folder, get_project_root, resolve_dataset_path
from .data import load_posts
from .gemini_client import MODEL_NAME, get_or_create_client, wait_until_ready
from .media import MediaAnalysis, find_media_files_in_project
from .retrieval import build_or_load_index, retrieve_historical_matches
from .trends import (
    load_cached_trend_report,
    retrieve_google_trends,
    save_trend_report,
    trend_cache_is_fresh,
    trend_report_paths,
)


def _media_trace_inputs(inputs: dict) -> dict:
    return {"file_path": str(inputs.get("path"))}


def _media_trace_outputs(output: MediaAnalysis) -> dict:
    return output.model_dump()


def _pipeline_trace_inputs(inputs: dict) -> dict:
    dataset_path = inputs.get("dataset_path")
    return {"dataset_path": str(dataset_path) if dataset_path is not None else None}


def _pipeline_trace_outputs(output: tuple | None) -> dict:
    if output is None:
        return {"status": "failed"}
    analyses, errors, retrieval_payload = output
    _, metadata, _ = retrieval_payload
    return {
        "media_analysis_count": len(analyses),
        "media_error_count": len(errors),
        "historical_post_count": len(metadata),
    }


@traceable(
    name="Analyze Media Asset",
    run_type="tool",
    process_inputs=_media_trace_inputs,
    process_outputs=_media_trace_outputs,
)
def analyze_media_file(path: Path, client_instance=None) -> MediaAnalysis:
    from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        media_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        raise ValueError(f"Unsupported media type: {path}")

    uploaded_file = None
    client_used = client_instance or get_or_create_client()

    try:
        print(f"Uploading {path.name}...")
        uploaded_file = client_used.files.upload(file=str(path))
        uploaded_file = wait_until_ready(uploaded_file, client_instance=client_used)

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

        response = client_used.models.generate_content(
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
                client_used.files.delete(name=uploaded_file.name)
            except Exception as exc:
                print(f"Temporary upload cleanup failed for {path.name}: {exc}")


def _load_analysis_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_analysis_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


@traceable(
    name="IG Forecaster Pipeline",
    run_type="chain",
    process_inputs=_pipeline_trace_inputs,
    process_outputs=_pipeline_trace_outputs,
)
def run_pipeline(dataset_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    resolved_dataset_path = resolve_dataset_path(dataset_path)
    project_root = get_project_root(dataset_path=resolved_dataset_path)
    print(f"Project root: {project_root}")
    print(f"Dataset path: {resolved_dataset_path}")

    posts = load_posts(resolved_dataset_path)
    index, metadata, embedding_model = build_or_load_index(posts=posts, dataset_path=resolved_dataset_path)

    media_folder = get_media_folder(dataset_path=resolved_dataset_path)
    media_files = find_media_files_in_project(project_root=project_root, media_folder=media_folder)

    analyses = []
    errors = []
    cache_path = project_root / ".media_analysis_cache.json"
    analysis_cache = _load_analysis_cache(cache_path)

    for number, path in enumerate(media_files, start=1):
        print(f"\n[{number}/{len(media_files)}] {path.name}")
        cache_key = str(path.resolve())
        if cache_key in analysis_cache:
            record = analysis_cache[cache_key]
            analyses.append(record)
            print("Loaded from cache.")
            continue

        try:
            result = analyze_media_file(path)
            record = result.model_dump()
            record["file_path"] = str(path)
            analysis_cache[cache_key] = record
            analyses.append(record)
            print("Completed.")
        except Exception as exc:
            errors.append({"file_name": path.name, "file_path": str(path), "error": str(exc)})
            print(f"Failed: {exc}")

    _save_analysis_cache(cache_path, analysis_cache)

    media_analyses_df = pd.DataFrame(analyses)
    media_errors_df = pd.DataFrame(errors)
    output_folder = get_output_folder(dataset_path=resolved_dataset_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("\nRetrieving similar historical posts...")
    historical_matches = retrieve_historical_matches(
        media_analyses_df,
        index,
        metadata,
        embedding_model,
    )
    historical_matches_path = output_folder / "historical_media_matches.csv"
    historical_matches.to_csv(historical_matches_path, index=False)
    print(f"Saved historical matches: {historical_matches_path}")

    print("\nRetrieving Google Trends data...")
    cached_trend_report = load_cached_trend_report(output_folder)
    if cached_trend_report is not None and trend_cache_is_fresh(output_folder):
        trend_report = cached_trend_report
        trend_paths = trend_report_paths(output_folder)
        print("Using Google Trends cache from the last six hours.")
    else:
        try:
            trend_report = retrieve_google_trends()
            trend_paths = save_trend_report(trend_report, output_folder)
        except Exception as exc:
            if cached_trend_report is None:
                raise RuntimeError(
                    "Google Trends retrieval failed and no cached trend report is available."
                ) from exc
            trend_report = cached_trend_report
            trend_paths = trend_report_paths(output_folder)
            print(f"Google Trends refresh failed; using the last saved report: {exc}")
    print(f"Saved trend interest: {trend_paths[0]}")
    print(f"Saved related queries: {trend_paths[1]}")
    print(f"Saved keyword momentum: {trend_paths[2]}")
    print(f"Saved agent-ready trend signals: {trend_paths[3]}")

    return media_analyses_df, media_errors_df, (index, metadata, embedding_model)


if __name__ == "__main__":
    run_pipeline()
