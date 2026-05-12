# Project Memory

## Environment

- The project runs in WSL.
- Backend and frontend runtime commands should be executed in WSL.
- Plain Windows/PowerShell console commands may not work for runtime checks.
- Do not install requirements or try to run the application unless explicitly requested.

## Frontend

- Frontend lives in `01_frontend/`.
- The frontend is a React application.
- The chat UI lives in `01_frontend/src/components/Chatbot.js`.
- Chat styling lives in `01_frontend/src/components/Chatbot.css`.
- Backend URL is configured via `REACT_APP_BACKEND_URL`, defaulting to `http://localhost:8000`.

## Backend

- Backend lives in `02_backend/`.
- The backend is a python application.
- FastAPI app entrypoint: `02_backend/main.py`.
- Dependency management uses `uv` with `pyproject.toml` and `uv.lock`.

## RAG Requirements

- RAG should use Qdrant.
- Qdrant is already running via `docker-compose.dev.yml`.
- Uploaded documents should be stored in project-root `data/`.
- Supported upload formats: PDF, TXT, Markdown.

## RAG API

- Keep existing `/chat` endpoint unchanged.
- Add RAG chat copy at `/chat/rag`.
- Document upload: `POST /rag/documents/upload`.
- Document list: `GET /rag/documents/list`.
- Document delete: `DELETE /rag/documents/{documentId}`.
- Keep/add query endpoint: `POST /rag/search`.

## Service Boundaries

- `DocumentProcessor`: file validation, storage, PDF/TXT/MD text extraction, chunking, document metadata.
- `EmbeddingService`: all Ollama embedding logic.
- `QdrantService`: all Qdrant collection, upsert, search, and delete logic.
