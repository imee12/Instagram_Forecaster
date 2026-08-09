from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
from langsmith import traceable

try:
    from pytrends.request import TrendReq
except ImportError:  # pragma: no cover - exercised in lightweight environments
    TrendReq = None


DEFAULT_TREND_KEYWORDS = (
    "new music",
    "singer songwriter",
    "fashion trends",
    "acting",
    "behind the scenes",
)


class TrendsClient(Protocol):
    def build_payload(
        self,
        kw_list: list[str],
        cat: int = 0,
        timeframe: str = "today 3-m",
        geo: str = "US",
        gprop: str = "",
    ) -> None: ...

    def interest_over_time(self) -> pd.DataFrame: ...

    def related_queries(self) -> dict: ...


@dataclass(frozen=True)
class TrendReport:
    interest_over_time: pd.DataFrame
    related_queries: pd.DataFrame


def _trend_trace_inputs(inputs: dict) -> dict:
    return {
        "keywords": list(inputs.get("keywords", DEFAULT_TREND_KEYWORDS)),
        "timeframe": inputs.get("timeframe", "today 3-m"),
        "geo": inputs.get("geo", "US"),
    }


def _trend_trace_outputs(output: TrendReport) -> dict:
    return {
        "interest_data_points": output.interest_over_time.size,
        "related_query_count": len(output.related_queries),
    }


def create_trends_client() -> TrendsClient:
    if TrendReq is None:
        raise ImportError("Install pytrends to retrieve Google Trends data.")

    # pytrends is an unofficial Google Trends client. Avoid its legacy retry
    # arguments, which are incompatible with some recent urllib3 releases.
    return TrendReq(hl="en-US", tz=300, timeout=(10, 25))


def _normalize_keywords(keywords: Sequence[str]) -> list[str]:
    normalized = list(dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip()))
    if not normalized:
        raise ValueError("At least one trend keyword is required.")
    if len(normalized) > 5:
        raise ValueError("Google Trends accepts at most five keywords per request.")
    return normalized


def _flatten_related_queries(related: dict, keywords: Sequence[str]) -> pd.DataFrame:
    records: list[dict] = []
    for keyword in keywords:
        groups = related.get(keyword) or {}
        for query_type in ("top", "rising"):
            frame = groups.get(query_type)
            if frame is None or frame.empty:
                continue
            for row in frame.to_dict(orient="records"):
                records.append(
                    {
                        "keyword": keyword,
                        "query_type": query_type,
                        "query": row.get("query"),
                        "value": row.get("value"),
                    }
                )
    return pd.DataFrame(records, columns=["keyword", "query_type", "query", "value"])


@traceable(
    name="Retrieve Google Trends",
    run_type="tool",
    process_inputs=_trend_trace_inputs,
    process_outputs=_trend_trace_outputs,
)
def retrieve_google_trends(
    keywords: Sequence[str] = DEFAULT_TREND_KEYWORDS,
    *,
    timeframe: str = "today 3-m",
    geo: str = "US",
    client: TrendsClient | None = None,
) -> TrendReport:
    normalized_keywords = _normalize_keywords(keywords)
    client = client or create_trends_client()
    client.build_payload(
        kw_list=normalized_keywords,
        cat=0,
        timeframe=timeframe,
        geo=geo,
        gprop="",
    )

    interest = client.interest_over_time().copy()
    if "isPartial" in interest.columns:
        interest = interest.drop(columns="isPartial")
    interest.index.name = "date"

    related = _flatten_related_queries(client.related_queries(), normalized_keywords)
    return TrendReport(interest_over_time=interest, related_queries=related)


def save_trend_report(report: TrendReport, output_folder: Path) -> tuple[Path, Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    interest_path = output_folder / "google_trends_interest.csv"
    related_path = output_folder / "google_trends_related_queries.csv"
    report.interest_over_time.to_csv(interest_path)
    report.related_queries.to_csv(related_path, index=False)
    return interest_path, related_path
