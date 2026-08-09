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

MODEL_NAME = "gemini-flash-latest"

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
