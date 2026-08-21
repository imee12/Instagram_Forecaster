from __future__ import annotations

import os
import time

try:
    from google import genai
except ImportError:  # pragma: no cover - exercised in lightweight environments
    genai = None

try:
    from google.genai import types
except ImportError:  # pragma: no cover - exercised in lightweight environments
    types = None

try:
    from langsmith import wrappers as langsmith_wrappers
except ImportError:  # pragma: no cover - exercised in lightweight environments
    langsmith_wrappers = None

MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_MAX_ATTEMPTS = 4
GEMINI_RETRY_DELAYS_SECONDS = (2, 4, 8)

client = None


def get_client():
    if genai is None:
        raise ImportError("Install google-genai to use Gemini integration.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Add GEMINI_API_KEY in your environment or Colab secrets.")
    gemini_client = genai.Client(api_key=api_key)
    if langsmith_wrappers is None:
        return gemini_client

    return langsmith_wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "media-analysis"],
            "metadata": {"model": MODEL_NAME},
        },
    )


def get_or_create_client():
    global client
    if client is None:
        client = get_client()
    return client


def _is_retryable_gemini_error(exc: Exception) -> bool:
    # Quota/rate-limit responses (429 / RESOURCE_EXHAUSTED) must not be retried
    # automatically: doing so can waste a limited request budget. The caller can
    # retry explicitly after the provider's stated reset time.
    message = str(exc).upper()
    if "RESOURCE_EXHAUSTED" in message or "QUOTA EXCEEDED" in message:
        return False

    retryable_codes = {500, 502, 503, 504}
    for value in (
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            if int(value) in retryable_codes:
                return True
        except (TypeError, ValueError):
            pass

    return any(
        marker in message
        for marker in (
            "500 INTERNAL",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        )
    )


def generate_content_with_retry(client_instance, **kwargs):
    """Retry transient Gemini capacity and transport failures with backoff."""
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            return client_instance.models.generate_content(**kwargs)
        except Exception as exc:
            if attempt == GEMINI_MAX_ATTEMPTS or not _is_retryable_gemini_error(exc):
                raise
            delay = GEMINI_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                f"[IG Forecaster] Gemini temporarily unavailable; retrying in "
                f"{delay}s (attempt {attempt + 1}/{GEMINI_MAX_ATTEMPTS})."
            )
            time.sleep(delay)


def wait_until_ready(uploaded_file, client_instance, timeout_seconds: int = 600):
    start_time = time.time()
    current_file = uploaded_file

    while True:
        state = getattr(current_file.state, "name", str(current_file.state))

        if state == "ACTIVE":
            return current_file

        if state == "FAILED":
            raise RuntimeError(f"Gemini failed to process {current_file.name}")

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out while processing {current_file.name}")

        time.sleep(5)
        current_file = client_instance.files.get(name=current_file.name)
