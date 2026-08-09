import pandas as pd
import pytest

from ig_forecaster.trends import retrieve_google_trends, save_trend_report


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


def test_retrieve_google_trends_rejects_more_than_five_keywords():
    with pytest.raises(ValueError, match="at most five"):
        retrieve_google_trends([str(number) for number in range(6)], client=FakeTrendsClient())


def test_save_trend_report_exports_csv_files(tmp_path):
    report = retrieve_google_trends(["new music"], client=FakeTrendsClient())

    interest_path, related_path = save_trend_report(report, tmp_path)

    assert interest_path.exists()
    assert related_path.exists()
