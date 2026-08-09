import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from ig_forecaster.agent.graph import IGForecasterAgent


class ToolCapableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_persists_messages_by_thread(tmp_path):
    dataset = tmp_path / "IG_Forecaster.csv"
    pd.DataFrame(
        [{"description": "A post", "media_type": "image", "category": "fashion"}]
    ).to_csv(dataset, index=False)
    model = ToolCapableFakeChatModel(
        responses=[AIMessage(content="The project is ready.")]
    )

    with IGForecasterAgent(
        dataset_path=dataset,
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        model=model,
    ) as agent:
        result = agent.invoke("Check project status", thread_id="test-thread")
        snapshot = agent.get_state(thread_id="test-thread")

    assert result["messages"][-1].content == "The project is ready."
    assert len(snapshot.values["messages"]) == 2
