from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ..gemini_client import MODEL_NAME
from ..pipeline import PipelineService
from .state import IGForecasterState
from .tools import build_agent_tools


SYSTEM_PROMPT = """
You are the IG Forecaster Agent. Help the user inspect project status, analyze
their media, retrieve historical matches and Google Trends signals, generate
post recommendations, and explain the evidence behind those recommendations.

Rules:
- Call get_project_status before deciding that pipeline work is required.
- Prefer saved artifacts and cached results when they already satisfy the request.
- Never force media reanalysis or a trend refresh unless the user explicitly asks.
- Historical matches require media analyses. Recommendations require media
  analyses, historical matches, and trends.
- Do not invent media contents, trend signals, post metrics, or recommendations.
- When presenting recommendations, include rank, media file, format, concept,
  overall score, and the most important supporting evidence.
- Explain failures clearly and tell the user which prerequisite is missing.
"""


def build_agent(
    service: PipelineService,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    model=None,
):
    chat_model = model or ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
    )
    return create_agent(
        model=chat_model,
        tools=build_agent_tools(service),
        system_prompt=SYSTEM_PROMPT,
        state_schema=IGForecasterState,
        checkpointer=checkpointer,
    )


class IGForecasterAgent:
    """Persistent conversational agent intended for the Streamlit UI."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        model=None,
    ):
        self.service = PipelineService(dataset_path=dataset_path)
        database_path = Path(checkpoint_path or (
            self.service.artifacts.output_folder / "agent_checkpoints.sqlite"
        ))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._connection)
        self.graph = build_agent(
            self.service,
            checkpointer=self.checkpointer,
            model=model,
        )

    def invoke(
        self,
        message: str,
        *,
        thread_id: str,
        user_preferences: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not message.strip():
            raise ValueError("Agent message cannot be empty.")
        state: dict[str, Any] = {"messages": [{"role": "user", "content": message}]}
        if user_preferences:
            state["user_preferences"] = user_preferences
        return self.graph.invoke(
            state,
            config={
                "configurable": {"thread_id": thread_id},
                "metadata": {"thread_id": thread_id},
            },
        )

    def get_state(self, *, thread_id: str):
        return self.graph.get_state({"configurable": {"thread_id": thread_id}})

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
