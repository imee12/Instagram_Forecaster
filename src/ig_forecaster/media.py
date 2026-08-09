from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


class MediaAnalysis(BaseModel):
    file_name: str = Field(description="Exact name of the analyzed file.")
    media_type: Literal["image", "video"] = Field(description="Whether the file is an image or video.")
    visual_summary: str = Field(
        description=(
            "A factual description of the visible content, setting, "
            "subject, clothing, objects, and actions."
        )
    )
    themes: list[str] = Field(
        description=(
            "Relevant themes such as music, fashion, rehearsal, "
            "whimsical, professional, or behind the scenes."
        )
    )
    content_categories: list[str] = Field(
        description=(
            "Possible Instagram categories such as music promotion, "
            "performance, fashion, lifestyle, or behind the scenes."
        )
    )
    possible_post_uses: list[str] = Field(
        description=(
            "Specific ways this asset might be used in an Instagram "
            "post, Reel, Story, or carousel."
        )
    )
    quality_notes: str = Field(
        description=(
            "Notes about framing, lighting, clarity, audio, motion, "
            "and whether the asset appears usable."
        )
    )
    spoken_or_sung_content: str | None = Field(
        default=None,
        description=(
            "A brief summary of audible speech or singing in a video. "
            "Use null for images or when no speech or singing is present."
        ),
    )
    strongest_moment: str | None = Field(
        default=None,
        description=(
            "For videos, describe the strongest usable moment or hook. "
            "Use null for static images."
        ),
    )


def find_media_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def find_media_files_in_project(project_root: Path, media_folder: Path | None = None) -> list[Path]:
    candidates = []
    if media_folder is not None:
        candidates.append(media_folder)

    candidates.extend(
        [
            project_root / "available_media",
            project_root,
        ]
    )

    seen_folders: set[Path] = set()
    seen_files: set[Path] = set()
    discovered: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate in seen_folders:
            continue
        seen_folders.add(candidate)
        for media_file in find_media_files(candidate):
            resolved_file = media_file.resolve()
            if resolved_file in seen_files:
                continue
            seen_files.add(resolved_file)
            discovered.append(media_file)

    return sorted(discovered)
