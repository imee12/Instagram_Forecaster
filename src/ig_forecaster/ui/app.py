from __future__ import annotations

from html import escape
import os
from pathlib import Path
import uuid

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

from ig_forecaster.agent import RemoteIGForecasterAgent

st.set_page_config(
    page_title="IG Forecaster",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_STYLES = """
<style>
    .stApp { background: #f5f2ec; color: #171714; }
    [data-testid="stSidebar"] { background: #171714; }
    [data-testid="stSidebar"] * { color: #f7f3eb; }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: #302f2a;
        color: #fffaf1;
        border: 1px solid #5c584f;
        border-radius: 10px;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #444139;
        color: #ffffff;
        border-color: #8b8477;
    }
    [data-testid="stSidebar"] .stButton > button:focus {
        color: #ffffff;
        border-color: #d98365;
        box-shadow: 0 0 0 2px rgba(217, 131, 101, .28);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #bb5b3d;
        color: #ffffff;
        border-color: #d17658;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #d06b4b;
        color: #ffffff;
        border-color: #e38b70;
    }
    [data-testid="stSidebar"] .stButton > button p {
        color: inherit;
    }
    .hero {
        padding: 1.2rem 0 1.4rem;
        border-bottom: 1px solid #d7d0c4;
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: #bb5b3d;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.8rem);
        line-height: .98;
        margin: .3rem 0 .65rem;
        letter-spacing: -.045em;
    }
    .hero p { color: #645f57; max-width: 760px; font-size: 1.03rem; }
    .recommendation-card {
        background: #fffdf8;
        border: 1px solid #ddd5c8;
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        margin-bottom: .8rem;
        box-shadow: 0 10px 28px rgba(44, 36, 25, .055);
    }
    .recommendation-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        color: white;
        background: #bb5b3d;
        font-weight: 800;
        margin-right: .55rem;
    }
    .recommendation-card h3 { display: inline; font-size: 1.18rem; }
    .recommendation-meta { color: #756d62; margin: .6rem 0; font-size: .9rem; }
    .score { color: #356657; font-weight: 800; }
    .evidence-pill {
        display: inline-block;
        padding: .22rem .52rem;
        margin: .18rem .18rem .05rem 0;
        border-radius: 999px;
        background: #e9efe9;
        color: #355649;
        font-size: .78rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
        padding: .65rem;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,.66);
        border: 1px solid #ded7cb;
        border-radius: 14px;
    }
</style>
"""


@st.cache_resource
def get_agent() -> RemoteIGForecasterAgent:
    load_dotenv()
    return RemoteIGForecasterAgent(
        server_url=os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
    )


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def _gemini_is_configured() -> bool:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and key != "your-gemini-api-key")


def _status(agent: RemoteIGForecasterAgent) -> dict:
    service = agent.service
    unavailable = 0

    def count(loader) -> int:
        nonlocal unavailable
        try:
            return len(loader())
        except OSError:
            unavailable += 1
            return 0

    try:
        trends = service.load_saved_trends()
    except OSError:
        trends = None
        unavailable += 1
    return {
        "media": count(service.load_saved_media_analyses),
        "errors": count(service.load_saved_media_errors),
        "history": count(service.load_saved_historical_matches),
        "trends": len(trends.agent_signals) if trends is not None else 0,
        "recommendations": count(service.load_saved_recommendations),
        "unavailable": unavailable,
    }


def _render_recommendation_card(row: pd.Series) -> None:
    trends = _as_list(row.get("supporting_trends"))
    pills = "".join(
        f'<span class="evidence-pill">{escape(topic)}</span>' for topic in trends
    )
    rank = int(row.get("rank", 0))
    score = float(row.get("overall_score", 0))
    st.markdown(
        f"""
        <div class="recommendation-card">
            <span class="recommendation-rank">{rank}</span>
            <h3>{escape(str(row.get('concept', 'Recommendation')))}</h3>
            <div class="recommendation-meta">
                {escape(str(row.get('post_format', 'post')).replace('_', ' ').title())}
                · {escape(str(row.get('media_file', 'No media selected')))}
                · <span class="score">{score:.1f} score</span>
            </div>
            <strong>Hook</strong><br>{escape(str(row.get('hook', '')))}<br><br>
            <strong>Why it works</strong><br>{escape(str(row.get('rationale', '')))}<br>
            <div style="margin-top:.65rem">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Execution details for recommendation {rank}"):
        st.write(row.get("caption_direction", ""))
        notes = _as_list(row.get("execution_notes"))
        if notes:
            st.markdown("\n".join(f"- {note}" for note in notes))
        score_columns = [
            "historical_performance_score",
            "trend_alignment_score",
            "media_quality_score",
            "audience_fit_score",
        ]
        available = {column: row.get(column) for column in score_columns if column in row}
        if available:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Factor": [key.replace("_score", "").replace("_", " ").title() for key in available],
                        "Score": list(available.values()),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


def _render_chat(agent: RemoteIGForecasterAgent, thread_id: str) -> None:
    st.subheader("Talk to your forecaster")
    st.caption("Ask for status, explanations, refreshes, or new recommendations.")
    snapshot = agent.get_state(thread_id=thread_id)
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    for message in messages:
        message_type = (
            message.get("type", "")
            if isinstance(message, dict)
            else getattr(message, "type", "")
        )
        if message_type not in {"human", "ai"}:
            continue
        role = "user" if message_type == "human" else "assistant"
        with st.chat_message(role):
            content = (
                message.get("content", "")
                if isinstance(message, dict)
                else message.content
            )
            st.markdown(_message_text(content))

    prompt = st.chat_input(
        "Ask about your content strategy…",
        disabled=not _gemini_is_configured(),
    )
    if not _gemini_is_configured():
        st.caption("Add a real GEMINI_API_KEY to .env, then restart LangGraph and Streamlit.")
    if prompt:
        with st.status("The agent is working…", expanded=True) as status_box:
            status_box.write("Reviewing project context and deciding which tools are needed.")
            try:
                agent.invoke(prompt, thread_id=thread_id)
                status_box.update(label="Agent response ready", state="complete")
            except Exception as exc:
                status_box.update(label="Agent request failed", state="error")
                st.error(str(exc))
                return
        st.rerun()


def _render_sidebar(
    agent: RemoteIGForecasterAgent,
    status: dict,
    thread_id: str,
) -> None:
    with st.sidebar:
        st.markdown("## IG Forecaster")
        st.caption("Project control room")
        left, right = st.columns(2)
        left.metric("Media", status["media"])
        right.metric("Trends", status["trends"])
        left.metric("Matches", status["history"])
        right.metric("Posts", status["recommendations"])

        st.divider()
        st.markdown("### Pipeline actions")
        if st.button("Analyze new media", use_container_width=True):
            with st.status("Analyzing media…"):
                agent.run_workflow(
                    thread_id=thread_id,
                    force_media_refresh=True,
                )
                st.session_state.last_workflow_thread_id = agent.workflow_thread_id(
                    thread_id
                )
            st.rerun()
        if st.button("Refresh Google Trends", use_container_width=True):
            with st.status("Refreshing trends…"):
                agent.run_workflow(
                    thread_id=thread_id,
                    force_trend_refresh=True,
                )
                st.session_state.last_workflow_thread_id = agent.workflow_thread_id(
                    thread_id
                )
            st.rerun()
        if st.button("Generate recommendations", type="primary", use_container_width=True):
            with st.status("Generating recommendations…"):
                agent.run_workflow(
                    thread_id=thread_id,
                    force_recommendation_refresh=True,
                )
                st.session_state.last_workflow_thread_id = agent.workflow_thread_id(
                    thread_id
                )
            st.rerun()

        if st.session_state.get("last_workflow_thread_id"):
            st.caption("Latest LangGraph workflow thread")
            st.code(st.session_state.last_workflow_thread_id, language=None)
            st.link_button(
                "Open local LangGraph Studio",
                "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024",
                use_container_width=True,
            )

        st.divider()
        st.markdown("### Connections")
        st.write("Gemini", "✓" if _gemini_is_configured() else "Missing key")
        st.write("LangSmith", "✓" if os.getenv("LANGSMITH_API_KEY") else "Not configured")
        if status["errors"]:
            st.warning(f"{status['errors']} media analysis errors")
        if status.get("unavailable"):
            st.warning(
                f"{status['unavailable']} saved artifact sources are temporarily unavailable."
            )


def main() -> None:
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    if "agent_thread_id" not in st.session_state:
        st.session_state.agent_thread_id = str(uuid.uuid4())

    agent = get_agent()
    status = _status(agent)
    _render_sidebar(agent, status, st.session_state.agent_thread_id)

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Creator intelligence workspace</div>
            <h1>Plan the next post<br>with evidence.</h1>
            <p>Review available media, historical performance, and current Google
            Trends signals—then work with the agent to shape the strongest ideas.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    workspace, chat = st.columns([1.45, 1], gap="large")
    with workspace:
        recommendations_tab, media_tab, trends_tab, history_tab, activity_tab = st.tabs(
            ["Recommendations", "Media", "Trends", "History", "Activity"]
        )
        with recommendations_tab:
            recommendations = agent.service.load_saved_recommendations()
            if recommendations.empty:
                st.info("No recommendations yet. Run the pipeline or use Generate recommendations.")
            else:
                for _, row in recommendations.sort_values("rank").iterrows():
                    _render_recommendation_card(row)

        with media_tab:
            analyses = agent.service.load_saved_media_analyses()
            if analyses.empty:
                st.info("No analyzed media found.")
            else:
                display_columns = [
                    column
                    for column in ("file_name", "media_type", "visual_summary", "quality_notes")
                    if column in analyses.columns
                ]
                st.dataframe(analyses[display_columns], hide_index=True, use_container_width=True)
                selected = st.selectbox("Preview media", analyses["file_name"].tolist())
                selected_row = analyses.loc[analyses["file_name"] == selected].iloc[0]
                media_path = Path(selected_row.get("file_path", ""))
                if media_path.exists():
                    if str(selected_row.get("media_type", "")).lower() == "video":
                        st.video(str(media_path))
                    else:
                        st.image(str(media_path), caption=selected)

        with trends_tab:
            report = agent.service.load_saved_trends()
            if report is None:
                st.info("No trend report found.")
            else:
                st.markdown("#### Ranked signals")
                st.dataframe(report.agent_signals.head(30), hide_index=True, use_container_width=True)
                st.markdown("#### Keyword momentum")
                st.dataframe(report.keyword_momentum, hide_index=True, use_container_width=True)

        with history_tab:
            matches = agent.service.load_saved_historical_matches()
            if matches.empty:
                st.info("No historical matches found.")
            else:
                columns = [
                    column
                    for column in (
                        "media_file",
                        "retrieval_rank",
                        "similarity_score",
                        "post_id",
                        "description",
                        "views",
                        "likes",
                        "saves",
                        "shares",
                    )
                    if column in matches.columns
                ]
                st.dataframe(matches[columns], hide_index=True, use_container_width=True)

        with activity_tab:
            artifacts = agent.service.load_project()
            st.markdown("#### Saved artifacts")
            st.code(str(artifacts.output_folder))
            errors = agent.service.load_saved_media_errors()
            if errors.empty:
                st.success("No media analysis errors in the latest run.")
            else:
                st.dataframe(errors, hide_index=True, use_container_width=True)
            st.caption("Detailed model and tool traces are available in LangSmith.")

    with chat:
        _render_chat(agent, st.session_state.agent_thread_id)


if __name__ == "__main__":
    main()
