# IG Forecaster Agent

## Purpose

The IG Forecaster Agent helps generate high-quality Instagram content recommendations by analyzing historical performance, media assets, and current trends.

The goal is to recommend the three highest-probability Instagram posts for a creator based on available photos/videos and current social media trends.

---

# Tech Stack

- Python
- VS Code
- OpenAI Codex
- LangChain (planned)
- LangSmith (planned)
- Gemini Vision (media analysis)
- Pandas
- Pathlib

Future:
- Vector database for semantic retrieval
- RAG for retrieving previous analyses
- LangGraph for orchestration

---

# Project Structure

```
IG_Forecaster/
│
├── media/
├── analyses/
├── metadata/
├── prompts/
├── output/
├── notebooks/
├── PROJECT_CONTEXT.md
├── requirements.txt
└── main.py
```

---

# Project Root

Google Colab used:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/IG_Forecaster")
```

VS Code uses:

```python
PROJECT_ROOT = Path.home() / "Library/CloudStorage/GoogleDrive-imeetwelve@gmail.com/My Drive/Colab Notebooks/IG_Forecaster"
```

Never generate Colab-specific paths.

Always generate code using `PROJECT_ROOT`.

Example:

```python
MEDIA_DIR = PROJECT_ROOT / "media"
OUTPUT_DIR = PROJECT_ROOT / "output"
```

Never hardcode absolute file paths.

---

# Coding Standards

- Use pathlib instead of os.path.
- Use type hints.
- Prefer dataclasses or Pydantic models.
- Keep functions small.
- Raise informative exceptions.
- Avoid duplicated code.
- Prefer readable code over clever code.

---

# Existing Pipeline

Current workflow:

1. Read media files.
2. Analyze each image/video with Gemini.
3. Store structured analysis.
4. Save analyses to DataFrame.
5. Export CSV.
6. Future retrieval will use vector embeddings.

---

# Planned Agent Workflow

1. Load historical analyses.
2. Retrieve similar posts using semantic search.
3. Analyze new media.
4. Retrieve current Instagram trends.
5. Combine retrieved context.
6. Generate candidate posts.
7. Score candidates.
8. Return the top three recommendations.

---

# Future Architecture

Planned migration:

- LangChain
- LangSmith
- LangGraph

Use modular components that are easy to convert into tools and chains.

Avoid notebook-specific code.

---

# Environment

Development machine:

- macOS
- VS Code
- Google Drive for Desktop
- Python virtual environment

The project is no longer developed in Google Colab except when explicitly requested.

---

# Assistant Instructions

Assume this project is an evolving production system.

When suggesting code:

- preserve existing architecture
- avoid unnecessary rewrites
- explain significant design changes
- prefer maintainability
- prefer reusable functions
- avoid global state

When multiple designs are possible, explain the tradeoffs before making major architectural changes.

Do not delete working code unless requested.

When editing code, preserve comments whenever practical.