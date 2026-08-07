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

By default, files are read from the Google Drive for desktop project folder:

```text
~/Library/CloudStorage/GoogleDrive-imeetwelve@gmail.com/My Drive/Colab Notebooks/IG_Forecaster
```

Set `IG_FORECASTER_PROJECT_ROOT` to use a different project folder.
