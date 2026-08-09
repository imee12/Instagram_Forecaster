# IG Forecaster

This project refactors the original Colab notebook into a normal Python project while preserving the existing notebook behavior:

- historical post loading and retrieval text construction,
- Sentence Transformers embeddings,
- FAISS index loading and reuse,
- media asset discovery,
- Gemini media analysis with structured Pydantic output,
- existing Drive-based paths and configuration where possible.

## Layout

- `src/ig_forecaster/` — package modules
- `tests/` — basic regression tests

## Usage

Run the main entrypoint with:

```bash
python -m ig_forecaster.main
```

Pipeline stages can also be run independently for UI or agent integrations:

```python
from ig_forecaster.pipeline import PipelineService

pipeline = PipelineService()
artifacts = pipeline.load_project()          # No external API calls
media, errors = pipeline.analyze_media()
history = pipeline.retrieve_history(media)
trends = pipeline.retrieve_trends()
recommendations = pipeline.generate_recommendations(media, history, trends)
```

The conversational agent wraps those stages with LangGraph tools and persistent
SQLite thread state:

```python
from ig_forecaster.agent import IGForecasterAgent

with IGForecasterAgent() as agent:
    result = agent.invoke(
        "Show me the saved recommendations.",
        thread_id="local-development",
    )
    print(result["messages"][-1].content)
```

Run the local interactive interface with:

```bash
streamlit run src/ig_forecaster/ui/app.py
```

By default, files are read from the Google Drive for desktop project folder:

```text
~/Library/CloudStorage/GoogleDrive-imeetwelve@gmail.com/My Drive/Colab Notebooks/IG_Forecaster
```

Set `IG_FORECASTER_PROJECT_ROOT` to use a different project folder.

## LangSmith tracing

Create a LangSmith API key, then configure tracing in the terminal where the
application will run:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="ig-forecaster-development"
```

Run the pipeline and open the `ig-forecaster-development` project in LangSmith
to inspect the pipeline, historical index, media-analysis, Gemini, and Google
Trends spans.
