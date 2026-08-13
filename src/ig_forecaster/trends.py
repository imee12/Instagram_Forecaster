from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Protocol, Sequence
import warnings

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
TREND_CACHE_MAX_AGE = timedelta(hours=6)
INTEREST_FILE_NAME = "google_trends_interest.csv"
RELATED_FILE_NAME = "google_trends_related_queries.csv"
MOMENTUM_FILE_NAME = "google_trends_keyword_momentum.csv"
AGENT_SIGNALS_FILE_NAME = "google_trends_agent_signals.csv"


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
    keyword_momentum: pd.DataFrame
    agent_signals: pd.DataFrame


def _trend_trace_inputs(inputs: dict) -> dict:
    return {
        "keywords": list(inputs.get("keywords", DEFAULT_TREND_KEYWORDS)),
        "timeframe": inputs.get("timeframe", "today 3-m"),
        "geo": inputs.get("geo", "US"),
    }


def _trend_trace_outputs(output: TrendReport | None) -> dict:
    if output is None:
        return {"status": "failed"}
    return {
        "interest_data_points": output.interest_over_time.size,
        "related_query_count": len(output.related_queries),
        "agent_signal_count": len(output.agent_signals),
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


def _summarize_keyword_momentum(interest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for keyword in interest.columns:
        values = pd.to_numeric(interest[keyword], errors="coerce").dropna()
        if values.empty:
            continue

        comparison_window = max(1, min(7, len(values) // 2))
        recent_average = float(values.tail(comparison_window).mean())
        if len(values) > comparison_window:
            previous_average = float(
                values.iloc[-2 * comparison_window : -comparison_window].mean()
            )
        else:
            previous_average = recent_average

        point_change = recent_average - previous_average
        growth_percent = (
            (point_change / previous_average) * 100
            if previous_average > 0
            else (100.0 if recent_average > 0 else 0.0)
        )
        if growth_percent >= 10:
            direction = "rising"
        elif growth_percent <= -10:
            direction = "falling"
        else:
            direction = "stable"

        positive_growth = min(max(growth_percent, 0.0), 100.0)
        signal_score = (0.7 * recent_average) + (0.3 * positive_growth)
        records.append(
            {
                "keyword": keyword,
                "current_interest": int(values.iloc[-1]),
                "recent_average": round(recent_average, 2),
                "previous_average": round(previous_average, 2),
                "point_change": round(point_change, 2),
                "growth_percent": round(growth_percent, 2),
                "trend_direction": direction,
                "signal_score": round(signal_score, 2),
            }
        )

    return pd.DataFrame(records).sort_values("signal_score", ascending=False).reset_index(drop=True)


def _related_query_score(value: object, query_type: str, maximum_rising: float) -> float:
    if isinstance(value, str) and value.casefold() == "breakout":
        return 100.0
    try:
        numeric_value = max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0

    if query_type == "top":
        return min(numeric_value, 100.0)
    if maximum_rising <= 0:
        return 0.0
    return min(100.0, 100 * math.log1p(numeric_value) / math.log1p(maximum_rising))


def _build_agent_signals(
    momentum: pd.DataFrame,
    related_queries: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "rank",
        "signal_type",
        "topic",
        "source_keyword",
        "trend_direction",
        "current_interest",
        "growth_percent",
        "related_query_value",
        "signal_score",
    ]
    records: list[dict] = []

    for row in momentum.to_dict(orient="records"):
        records.append(
            {
                "signal_type": "keyword_momentum",
                "topic": row["keyword"],
                "source_keyword": row["keyword"],
                "trend_direction": row["trend_direction"],
                "current_interest": row["current_interest"],
                "growth_percent": row["growth_percent"],
                "related_query_value": None,
                "signal_score": row["signal_score"],
            }
        )

    rising_values = pd.to_numeric(
        related_queries.loc[related_queries["query_type"] == "rising", "value"],
        errors="coerce",
    )
    maximum_rising = float(rising_values.max()) if not rising_values.empty else 0.0
    if math.isnan(maximum_rising):
        maximum_rising = 0.0

    for row in related_queries.to_dict(orient="records"):
        query_type = row["query_type"]
        records.append(
            {
                "signal_type": f"{query_type}_related_query",
                "topic": row["query"],
                "source_keyword": row["keyword"],
                "trend_direction": "rising" if query_type == "rising" else "popular",
                "current_interest": None,
                "growth_percent": row["value"] if query_type == "rising" else None,
                "related_query_value": row["value"],
                "signal_score": round(
                    _related_query_score(row["value"], query_type, maximum_rising),
                    2,
                ),
            }
        )

    signals = pd.DataFrame(records)
    if signals.empty:
        return pd.DataFrame(columns=columns)

    signals = (
        signals.sort_values("signal_score", ascending=False)
        .drop_duplicates(subset=["topic"], keep="first")
        .reset_index(drop=True)
    )
    signals.insert(0, "rank", range(1, len(signals) + 1))
    return signals[columns]


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
    momentum = _summarize_keyword_momentum(interest)
    agent_signals = _build_agent_signals(momentum, related)
    return TrendReport(
        interest_over_time=interest,
        related_queries=related,
        keyword_momentum=momentum,
        agent_signals=agent_signals,
    )


def trend_report_paths(output_folder: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_folder / INTEREST_FILE_NAME,
        output_folder / RELATED_FILE_NAME,
        output_folder / MOMENTUM_FILE_NAME,
        output_folder / AGENT_SIGNALS_FILE_NAME,
    )


@traceable(name="Load Cached Google Trends", run_type="tool")
def load_cached_trend_report(output_folder: Path) -> TrendReport | None:
    interest_path, related_path, momentum_path, agent_signals_path = trend_report_paths(
        output_folder
    )
    if not all(
        path.exists()
        for path in (interest_path, related_path, momentum_path, agent_signals_path)
    ):
        return None

    try:
        interest = pd.read_csv(interest_path, parse_dates=["date"]).set_index("date")
        interest.index.name = "date"
        return TrendReport(
            interest_over_time=interest,
            related_queries=pd.read_csv(related_path),
            keyword_momentum=pd.read_csv(momentum_path),
            agent_signals=pd.read_csv(agent_signals_path),
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.warn(
            f"Cached Google Trends files could not be read: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def trend_cache_is_fresh(
    output_folder: Path,
    *,
    maximum_age: timedelta = TREND_CACHE_MAX_AGE,
    now: datetime | None = None,
) -> bool:
    paths = trend_report_paths(output_folder)
    if not all(path.exists() for path in paths):
        return False

    try:
        newest_allowed_time = (now or datetime.now(timezone.utc)) - maximum_age
        oldest_file_time = min(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            for path in paths
        )
        return oldest_file_time >= newest_allowed_time
    except OSError:
        return False


def save_trend_report(report: TrendReport, output_folder: Path) -> tuple[Path, Path, Path, Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    interest_path, related_path, momentum_path, agent_signals_path = trend_report_paths(
        output_folder
    )
    report.interest_over_time.to_csv(interest_path)
    report.related_queries.to_csv(related_path, index=False)
    report.keyword_momentum.to_csv(momentum_path, index=False)
    report.agent_signals.to_csv(agent_signals_path, index=False)
    return interest_path, related_path, momentum_path, agent_signals_path
