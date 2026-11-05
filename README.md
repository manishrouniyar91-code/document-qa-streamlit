# Document Q&A

A Streamlit application for asking questions about uploaded PDF, DOCX, and TXT
documents. It retrieves passages using Gemini embeddings, generates concise
answers grounded in those passages, and displays citations to the source file
and page, section, or line range. Conversations and indexed passages are stored
in SQLite for local development or PostgreSQL when `DATABASE_URL` is configured.

## Requirements

- Python 3.10+
- A Gemini API key with access to the configured chat and embedding models

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`, then start the app:

```powershell
py -m streamlit run app.py
```

The app opens in a browser. Upload one or more supported files in the sidebar
and choose **Index documents** before asking questions. Scanned PDFs without
extractable text must be OCR-processed before they can be indexed. Re-uploading
the same file contents does not create duplicate indexed documents.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required Gemini API key |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | Model used to interpret follow-ups and answer questions |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for semantic search |
| `DATABASE_URL` | — | Optional PostgreSQL connection URL; required for durable hosted storage |
| `DATABASE_PATH` | `data/qa_history.db` | Local SQLite database path when `DATABASE_URL` is not set |

Uploaded file contents are sent to the configured Gemini models for embedding
and answer generation. The app stores extracted passages and their embeddings,
document names, citations, and conversation history in the configured database.
Do not upload documents you are not permitted to process this way.
If you already indexed documents with a different embedding model, the app
re-embeds the stored passages using the configured Gemini embedding model.

## Streamlit Community Cloud

1. Push this project to a GitHub repository and deploy `app.py` from that
   repository in Streamlit Community Cloud.
2. Create a Supabase PostgreSQL database and copy its connection URL. Add
   `DATABASE_URL` and `GEMINI_API_KEY` to the app's **Settings → Secrets** in
   Streamlit Community Cloud. Keep these secrets out of GitHub.
3. Use a Supabase connection URL with SSL enabled (for example, ending in
   `?sslmode=require`). On first start, the app creates the required tables.

The app falls back to local SQLite when `DATABASE_URL` is not set; that local
database is not durable on a hosted instance. Add the hosted database secret
before accepting uploads.

This deployment is a public shared demo: all visitors share its documents,
indexed content, and conversation history. Do not use it for private documents
or sensitive information.

Answers without a supported, cited passage are reported as
“Answer not found in the uploaded documents.” The similarity cutoff is
configured in `app.py` as `MIN_SIMILARITY`.
