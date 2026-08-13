import pandas as pd

from ig_forecaster.ui.app import _as_list, _gemini_is_configured, _message_text


def test_message_text_handles_gemini_content_blocks():
    assert _message_text([{"type": "text", "text": "Hello"}]) == "Hello"


def test_as_list_normalizes_saved_values():
    assert _as_list(["music", "fashion"]) == ["music", "fashion"]
    assert _as_list("music") == ["music"]
    assert _as_list(float("nan")) == []


def test_placeholder_gemini_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your-gemini-api-key")

    assert not _gemini_is_configured()
