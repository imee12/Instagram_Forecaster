from types import SimpleNamespace

import pytest

from ig_forecaster import gemini_client


class FlakyModels:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return kwargs["result"]


def test_generate_content_retries_transient_503(monkeypatch):
    models = FlakyModels([RuntimeError("503 UNAVAILABLE: high demand")])
    delays = []
    monkeypatch.setattr(gemini_client.time, "sleep", delays.append)

    result = gemini_client.generate_content_with_retry(
        SimpleNamespace(models=models), result="ok"
    )

    assert result == "ok"
    assert models.calls == 2
    assert delays == [2]


def test_generate_content_does_not_retry_non_transient_errors(monkeypatch):
    models = FlakyModels([ValueError("invalid structured response")])
    monkeypatch.setattr(
        gemini_client.time,
        "sleep",
        lambda _: pytest.fail("non-transient errors must not be retried"),
    )

    with pytest.raises(ValueError, match="invalid structured response"):
        gemini_client.generate_content_with_retry(
            SimpleNamespace(models=models), result="unused"
        )

    assert models.calls == 1


def test_generate_content_does_not_retry_quota_errors(monkeypatch):
    models = FlakyModels(
        [RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded; retry in 46s")]
    )
    monkeypatch.setattr(
        gemini_client.time,
        "sleep",
        lambda _: pytest.fail("quota errors must not be retried automatically"),
    )

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        gemini_client.generate_content_with_retry(
            SimpleNamespace(models=models), result="unused"
        )

    assert models.calls == 1
