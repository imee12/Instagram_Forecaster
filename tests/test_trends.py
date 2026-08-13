from datetime import datetime, timedelta, timezone
import os

import pandas as pd
import pytest

from ig_forecaster.trends import (
    load_cached_trend_report,
    retrieve_google_trends,
    save_trend_report,
    trend_cache_is_fresh,
)


class FakeTrendsClient:
    def __init__(self):
        self.payload = None

    def build_payload(self, **kwargs):
        self.payload = kwargs

    def interest_over_time(self):
        return pd.DataFrame(
            {"new music": [40, 60], "isPartial": [False, True]},
            index=pd.to_datetime(["2026-07-01", "2026-07-08"]),
        )

    def related_queries(self):
        return {
            "new music": {
                "top": pd.DataFrame([{"query": "new songs", "value": 100}]),
                "rising": pd.DataFrame([{"query": "summer songs", "value": 250}]),
            }
        }


def test_retrieve_google_trends_normalizes_results():
    client = FakeTrendsClient()

    report = retrieve_google_trends(["new music"], client=client)

    assert client.payload["kw_list"] == ["new music"]
    assert "isPartial" not in report.interest_over_time.columns
    assert set(report.related_queries["query_type"]) == {"top", "rising"}
    assert report.keyword_momentum.iloc[0]["trend_direction"] == "rising"
    assert report.keyword_momentum.iloc[0]["growth_percent"] == 50
    assert report.agent_signals.iloc[0]["topic"] == "new songs"
    assert set(report.agent_signals["signal_type"]) == {
        "keyword_momentum",
        "top_related_query",
        "rising_related_query",
    }


def test_retrieve_google_trends_rejects_more_than_five_keywords():
    with pytest.raises(ValueError, match="at most five"):
        retrieve_google_trends([str(number) for number in range(6)], client=FakeTrendsClient())


def test_save_trend_report_exports_csv_files(tmp_path):
    report = retrieve_google_trends(["new music"], client=FakeTrendsClient())

    interest_path, related_path, momentum_path, agent_signals_path = save_trend_report(
        report,
        tmp_path,
    )

    assert interest_path.exists()
    assert related_path.exists()
    assert momentum_path.exists()
    assert agent_signals_path.exists()

    cached_report = load_cached_trend_report(tmp_path)
    assert cached_report is not None
    assert cached_report.agent_signals.iloc[0]["topic"] == "new songs"
    assert trend_cache_is_fresh(tmp_path)


def test_cached_trend_timeout_returns_no_report(tmp_path, monkeypatch):
    report = retrieve_google_trends(["new music"], client=FakeTrendsClient())
    save_trend_report(report, tmp_path)

    def time_out(*args, **kwargs):
        raise TimeoutError("cloud file unavailable")

    monkeypatch.setattr(pd, "read_csv", time_out)

    with pytest.warns(RuntimeWarning, match="could not be read"):
        assert load_cached_trend_report(tmp_path) is None


def test_trend_cache_detects_stale_files(tmp_path):
    report = retrieve_google_trends(["new music"], client=FakeTrendsClient())
    paths = save_trend_report(report, tmp_path)
    file_time = datetime.now(timezone.utc) - timedelta(days=1)
    for path in paths:
        os.utime(path, (file_time.timestamp(), file_time.timestamp()))

    assert not trend_cache_is_fresh(tmp_path)
