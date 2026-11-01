# RAG Document Q&A Platform

An end-to-end, enterprise-ready Retrieval-Augmented Generation (RAG) and Agentic Document Q&A platform built with FastAPI, React, PostgreSQL (`pgvector`), and Celery.

---

## 📑 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Project Directory Structure](#project-directory-structure)
- [Implementation Roadmap](#implementation-roadmap)
- [Evaluation & Quality Metrics](#evaluation--quality-metrics)
- [Security & Reliability](#security--reliability)
- [Getting Started](#getting-started)

---

## 🔍 Overview

This platform provides asynchronous document ingestion, grounded vector retrieval, streaming answers with source citations, and a scope-controlled chat interface. The roadmap extends this into hybrid search, neural reranking, and tool-calling autonomous agents.

---

## 🏛️ System Architecture

```text
┌──────────────────────── Frontend (React / Next.js) ────────────────────────┐
│  Document Uploader   │   Agent Chat (Q&A, streaming)   │  Document History │
└──────────┬───────────────────┬──────────────────────────────┬──────────────┘
           │ REST              │ SSE / WebSocket              │ REST
┌──────────▼───────────────────▼──────────────────────────────▼──────────────┐
│                            API Layer (FastAPI)                             │
│     Auth · Upload API · Chat API · History API · Document Management       │
└──────┬───────────────────────────┬───────────────────────────┬─────────────┘
       │ Enqueue Task              │ Read / Query              │
┌──────▼─────────┐        ┌────────▼─────────┐        ┌────────▼─────────┐
│ Ingestion      │        │ RAG / Agent      │        │ Metadata DB      │
│ Worker (Celery)│        │ Service          │        │ (PostgreSQL)     │
│ parse→chunk→   │        │ rewrite→retrieve │        │ users, docs,     │
│ embed→index    │        │ →rerank→generate │        │ chats, messages  │
└──────┬─────────┘        └────────┬─────────┘        └──────────────────┘
       │                           │
┌──────▼───────────┐      ┌────────▼─────────┐     ┌──────────────────────┐
│ Object Storage   │      │ Vector Store     │     │ LLM & Embeddings     │
│ (S3 / MinIO)     │      │ (pgvector)       │     │ (OpenAI/Gemini/Local)│
└──────────────────┘      └──────────────────┘     └──────────────────────┘
```

---

## ⚙️ Core Features

### 1. Document Ingestion Pipeline
- **Asynchronous Task Processing**: Uploads never block API threads; Celery handles CPU/GPU-heavy operations.
- **Deduplication**: SHA-256 file hashing prevents redundant parsing, embedding, and storage.
- **Smart Chunking**: Structure-aware splitting (500–800 tokens, 10–15% overlap) preserving markdown tables, headings, and paragraph boundaries.
- **Granular Metadata**: Every chunk tracks `doc_id`, `page_number`, `section_title`, `chunk_index`, and `embedding_model`.

### 2. Retrieval & Agentic Q&A
- **Query Rewriting** *(Phase 5)*: Contextual resolution of ambiguous follow-up questions using session history.
- **Hybrid Retrieval** *(Phase 5)*: Combines dense vector similarity (`pgvector` with HNSW cosine index) and sparse keyword search (`tsvector` / BM25).
- **Reranking** *(Phase 5)*: Neural cross-encoders (e.g., Cohere Rerank or BGE Reranker) distill candidates to the top 4–6 high-precision passages.
- **Grounded Generation**: System prompt enforcement restricting responses exclusively to verified context.
- **Clickable Citations**: UI citation chips mapped directly to document and page numbers.
- **Real-time Streaming**: Server-Sent Events (SSE) for low-latency streaming responses.
- **Agentic Tool Loop** *(Phase 6)*: Support for function/tool calling (`search_documents`, `summarize_document`, `compare_documents`, `get_page`).

### 3. Session & Document Management
- Dynamic document scoping: query across all documents or target a selected subset.
- Session-based conversation logs with rename, Markdown export, deletion, and per-answer helpfulness feedback.

---

## 🧪 Ingestion Pipeline (Phase 2)

```text
queued ──► download object ──► parse ──► chunk ──► embed (batched) ──► replace chunks ──► ready
                                                                                       │
                                                                                       └──► failed + dead letter
```

- **Parsers**: Markdown (heading hierarchy, tables, fenced code), HTML, DOCX, plain text, and PDF (per-page text with heading detection by font size). PDFs without a text layer (scans) require OCR, which is out of scope and reported as a failed document.
- **Chunking**: ~700 tokens per chunk with ~90-token (≈13%) overlap, configurable via `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS`. Section headings are prepended, sections are never mixed, and tables stay intact (oversized tables split by row with the header repeated).
- **Embeddings**: `EMBEDDING_PROVIDER=deterministic` (default; offline, hash-based) or `openai` (`text-embedding-3-small` by default, batched). The embedding model name is stored on every chunk for model-version isolation.
- **Indexing**: Chunks are replaced atomically per document (delete + bulk insert), so reindexing is idempotent. The `pgvector` HNSW and `tsvector` indexes are maintained automatically.
- **Reliability**: Transient failures (storage, network, rate limits) retry with exponential backoff (`30s → 60s → 120s`, max 3). Permanent failures (unsupported/corrupt files, no text, bad credentials) fail immediately. Final failures set `documents.status='failed'`, surface `error_message`, and append a JSON record to the Redis dead-letter list (`rag:ingestion:dlq`).
- **Reindex**: `POST /api/v1/documents/{id}/reindex` re-runs the pipeline for an existing object (retry after failures or re-embed after a model upgrade).

---

## 💬 Retrieval & Chat (Phase 3)

```text
question ──► embed ──► pgvector top-k ──► grounded prompt ──► streamed tokens (SSE) ──► message + citations
```

- **Vector retrieval**: the query is embedded with the same provider used at ingestion and matched against ready documents using cosine distance (HNSW index). Sessions can search all of the user's documents or a scoped subset; `RETRIEVAL_TOP_K` controls how many chunks are used.
- **Grounded prompt**: the system prompt restricts answers to the retrieved passages and requires bracketed citations (`[1]`, `[2]`). If nothing relevant is retrieved, the assistant says so instead of inventing an answer.
- **Streaming**: `POST /api/v1/chat/sessions/{id}/ask` returns `text/event-stream` with `meta` (citations + model), `token`, `done` (latency, token estimate) and `error` events.
- **LLM providers**: `LLM_PROVIDER=deterministic` (default; offline extractive answers for development and tests) or `openai` (streaming chat completions via `LLM_MODEL`, default `gpt-4o-mini`).
- **Persistence**: user and assistant messages store citation JSON, token estimates and latency; the session title is derived from the first question.

---

## 🗂️ Document & History Scope (Phase 4)

- **Document management**: tenant-scoped listing and status polling, short-lived presigned downloads, reindexing, and cascade deletion of chunks plus the stored object.
- **Conversation history**: sessions are listed newest-first, reopened with their full message history, renamed, and deleted (messages and feedback cascade).
- **Scoped retrieval**: a session can be created against a validated subset of the user's documents; reopening it restores that scope.
- **Answer feedback**: `POST /api/v1/messages/{id}/feedback` records one verdict per assistant message (`rating` = `1` helpful / `-1` not helpful) with an optional 2,000-character comment. Resubmitting replaces the existing verdict.
- **Markdown export**: `GET /api/v1/chat/sessions/{id}/export` downloads a transcript with citations and feedback as a `.md` attachment.

---

## 🛠️ Tech Stack

| Component | Technology | Alternative / Scale-up |
|---|---|---|
| **Backend API** | Python 3.11+, FastAPI, Uvicorn | FastAPI + Kubernetes |
| **Worker Queue** | Celery, Redis | Temporal, AWS SQS |
| **Relational & Vector DB** | PostgreSQL 16+ with `pgvector` | Qdrant, Milvus, Pinecone |
| **Object Storage** | MinIO (local dev) / AWS S3 | Cloudflare R2, GCS |
| **Document Parsers** | PyMuPDF, `python-docx`, BeautifulSoup | Docling, Azure Doc Intelligence |
| **LLM & Embeddings** | OpenAI (chat + embeddings) / deterministic (offline dev) | Google Gemini, Anthropic, self-hosted (vLLM, Ollama, TEI) |
| **Reranker** | BGE-Reranker-Large, Cohere Rerank | Cross-Encoder HuggingFace |
| **Frontend** | React 18 + Vite + TypeScript (plain CSS) | Tailwind CSS + Shadcn/ui |

---

## 🗄️ Database Schema

### `users`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Unique user identifier |
| `email` | `VARCHAR(255)` | Unique user email |
| `password_hash` | `TEXT` | Hashed password |
| `created_at` | `TIMESTAMPTZ` | Timestamp |

### `documents`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Document ID |
| `user_id` | `UUID` (FK) | Reference to `users.id` |
| `filename` | `VARCHAR(255)` | Original uploaded file name |
| `mime_type` | `VARCHAR(100)` | MIME type |
| `size_bytes` | `BIGINT` | File size |
| `file_hash` | `CHAR(64)` | SHA-256 digest |
| `storage_url` | `TEXT` | S3/MinIO bucket location |
| `status` | `VARCHAR(32)` | `queued` \| `processing` \| `ready` \| `failed` |
| `error_message`| `TEXT` | Error trace if failed |
| `created_at` | `TIMESTAMPTZ` | Timestamp |
| `updated_at` | `TIMESTAMPTZ` | Last status change (Phase 1 extension for polling) |

### `chunks`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Chunk identifier |
| `doc_id` | `UUID` (FK) | Reference to `documents.id` (CASCADE) |
| `chunk_index` | `INTEGER` | Sequential chunk index |
| `page_number` | `INTEGER` | Source document page |
| `section` | `TEXT` | Extracted header/section |
| `content` | `TEXT` | Plain chunk text |
| `embedding` | `vector(1536)` | Vector embedding with HNSW index |
| `tsv` | `tsvector` | Full-text search vector with GIN index |
| `model_name` | `VARCHAR(64)` | Model version used to embed |

### `chat_sessions`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Session identifier |
| `user_id` | `UUID` (FK) | Reference to `users.id` |
| `title` | `VARCHAR(255)` | Conversation title |
| `doc_scope` | `UUID[]` | Scoped document IDs (empty = all) |
| `created_at` | `TIMESTAMPTZ` | Session creation timestamp |

### `messages`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Message identifier |
| `session_id` | `UUID` (FK) | Reference to `chat_sessions.id` |
| `role` | `VARCHAR(16)` | `user` \| `assistant` \| `system` |
| `content` | `TEXT` | Raw text response |
| `citations` | `JSONB` | Source chips `[{doc_id, page, snippet}]` |
| `tokens_used` | `INTEGER` | Token consumption |
| `latency_ms` | `INTEGER` | Response time in milliseconds |
| `created_at` | `TIMESTAMPTZ` | Timestamp |

### `feedback`
| Column | Type | Description |
|---|---|---|
| `id` | `UUID` (PK) | Feedback identifier |
| `message_id` | `UUID` (FK) | Target message reference |
| `rating` | `SMALLINT` | `1` (Upvote) or `-1` (Downvote) |
| `comment` | `TEXT` | Optional user notes |

---

## 📡 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Create an account (`email`, `password`)
- `POST /api/v1/auth/token` - OAuth2 password flow; send `username=<email>` as form data
- `GET /api/v1/auth/me` - Current user profile

### Documents
- `POST /api/v1/documents` - Upload document file (multipart/form-data)
- `GET /api/v1/documents` - List uploaded documents (pagination, status filters)
- `GET /api/v1/documents/{id}` - Get processing status and details (incl. chunk count)
- `POST /api/v1/documents/{id}/reindex` - Re-run ingestion (re-parse, re-chunk, re-embed)
- `GET /api/v1/documents/{id}/download` - Redirect to a short-lived presigned URL
- `DELETE /api/v1/documents/{id}` - Delete document, associated chunks, and stored file

### Chat & Q&A
- `POST /api/v1/chat/sessions` - Create chat session (with optional document scoping)
- `GET /api/v1/chat/sessions` - List active sessions
- `GET /api/v1/chat/sessions/{id}` - Get session message history (including existing feedback)
- `PATCH /api/v1/chat/sessions/{id}` - Rename a session
- `DELETE /api/v1/chat/sessions/{id}` - Delete a session, its messages, and feedback
- `GET /api/v1/chat/sessions/{id}/export` - Download the transcript as Markdown
- `POST /api/v1/chat/sessions/{id}/ask` - Submit query & stream response via SSE
- `POST /api/v1/messages/{id}/feedback` - Submit or update helpfulness feedback (`rating` `1`/`-1`, optional `comment`)

SSE events emitted by the ask endpoint:

| Event | Data |
|---|---|
| `meta` | `session_id`, `message_id`, `model`, `citations[]` |
| `token` | `text` |
| `done` | `message_id`, `latency_ms`, `tokens_used` |
| `error` | `detail`, `message_id` |

### Service
- `GET /` - Service metadata
- `GET /healthz` - Liveness + database connectivity check

### Upload behaviour (Phase 1–2)
- Supported types: `.pdf`, `.md`, `.markdown`, `.txt`, `.docx`, `.html`, `.htm` (configurable)
- Maximum size: 50 MB by default (`MAX_UPLOAD_SIZE_MB`), enforced while streaming
- Files are hashed (SHA-256) during upload; re-uploading identical bytes returns `409 Conflict`
  with the existing document ID in the `X-Document-Id` header
- Objects are stored at `s3://<bucket>/users/<user_id>/documents/<document_id>/<filename>`
- Ingestion is queued automatically on upload: `queued → processing → ready` (or `failed` with `error_message`)

---

## 📁 Project Directory Structure

```text
rag/
├── docker-compose.yml          # PostgreSQL+pgvector, Redis, MinIO, API, worker, web app
├── Makefile                    # Common dev commands
├── .env.example                # Environment template
├── README.md                   # System documentation
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── alembic.ini
│   ├── alembic/                # DB migrations (0001_initial = full schema)
│   │   ├── env.py
│   │   └── versions/
│   ├── tests/                  # Unit tests (pytest)
│   └── app/
│       ├── main.py             # FastAPI entrypoint, /healthz, CORS
│       ├── core/               # App configuration & security
│       │   ├── config.py
│       │   └── security.py     # bcrypt + JWT
│       ├── db/                 # DB engine, session & ORM models
│       │   ├── base.py
│       │   ├── session.py
│       │   └── models.py
│       ├── schemas/            # Pydantic models (auth, documents, chat)
│       ├── services/
│       │   ├── storage.py      # S3/MinIO client & presigned URLs
│       │   ├── upload.py       # Validation, hashing, streaming
│       │   └── export.py       # Markdown transcript rendering
│       ├── rag/                # Retrieval & generation (Phase 3)
│       │   ├── retriever.py    # pgvector cosine search, doc scoping
│       │   └── llm.py          # Grounded prompt + deterministic/OpenAI LLMs
│       ├── workers/            # Ingestion pipeline (Celery)
│       │   ├── celery_app.py   # Broker/queue configuration
│       │   ├── parser.py       # PDF/Markdown/HTML/DOCX/TXT → blocks
│       │   ├── chunker.py      # Token-bounded, section-aware chunking
│       │   ├── embedder.py     # Deterministic + OpenAI providers
│       │   ├── pipeline.py     # download → parse → chunk → embed → index
│       │   ├── tasks.py        # Celery task, retries, failure handling
│       │   └── dlq.py          # Redis dead-letter queue
│       └── api/                # REST endpoints
│           ├── deps.py         # Auth dependencies
│           └── v1/
│               ├── auth.py
│               ├── documents.py
│               ├── chat.py     # Sessions, rename/export/delete, retrieval, SSE
│               ├── messages.py # Answer feedback
│               └── router.py
└── frontend/                   # React 18 + Vite chat UI (Phase 3–4)
    ├── Dockerfile              # Multi-stage build served by nginx
    ├── nginx.conf              # SPA fallback + SSE-friendly /api proxy
    ├── package.json
    └── src/
        ├── App.tsx             # Auth gate, documents, scoped chat, streaming
        ├── api.ts              # REST client + SSE frame parser
        ├── types.ts
        └── styles.css
```

---

## 📅 Implementation Roadmap

| Phase | Milestone | Deliverables |
|---|---|---|
| **Phase 1** | ✅ Foundation & Storage | Database schema, authentication, file storage (S3/MinIO), and upload endpoint |
| **Phase 2** | ✅ Ingestion Engine | Celery pipeline: parsing, recursive chunking, batch embeddings, pgvector indexing |
| **Phase 3** | ✅ Core RAG & Streaming | Vector search, basic prompt template, SSE streaming endpoint, and React chat interface |
| **Phase 4** | ✅ Document & History Scope | Document management, scoped conversation sessions with rename/export/delete, and answer feedback |
| **Phase 5** | Advanced Search | Hybrid search (vector + keyword), query rewriting, cross-encoder reranking |
| **Phase 6** | Agent Loop & Production | Autonomous tool invocation (`search`, `summarize`), automated evaluation suite, rate limits |

---

## 📊 Evaluation & Quality Metrics

A continuous golden test set (30–50 domain-specific question/ground-truth pairs) assesses performance:

1. **Retrieval Precision**:
   - **Hit Rate @ K**: Frequency of the target passage appearing in the top $K$ results.
   - **MRR (Mean Reciprocal Rank)**: Average rank of the first relevant chunk.
2. **Answer Quality (LLM-as-a-Judge)**:
   - **Faithfulness**: Detection of hallucinations or statements unsupported by source chunks.
   - **Answer Relevancy**: Degree to which the answer directly addresses the query.
3. **Operational Metrics**:
   - End-to-end latency (Time-to-First-Token and total generation time).
   - Ingestion throughput (pages per minute) and API cost tracking.

---

## 🔒 Security & Reliability

- **Strict Multi-Tenancy**: All vector queries and document operations enforce explicit `user_id` filtering.
- **Prompt Injection Defense**: Document text is isolated inside delimited context tags and treated as untrusted input.
- **Upload Validation**: Extension, MIME-type, and payload size verification before queuing.
- **Fail-Safe Pipelines**: Celery exponential retries and dead-letter queue (DLQ) for failed document jobs.
- **Model Version Isolation**: Storing `model_name` with chunk embeddings prevents silent similarity degradation across upgrades.

---

## 🚀 Getting Started

### Prerequisites

- Docker Engine + Docker Compose v2
- Python 3.12 (only needed for running tests or the API outside Docker)

### 1. Configure the environment

```bash
cp .env.example .env
# Generate a strong secret (required for production):
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set the generated value as `JWT_SECRET` in `.env`.

### 2. Start the stack

```bash
make up        # = docker compose up -d --build
docker compose ps
```

This starts:

| Service | URL | Credentials |
|---|---|---|
| API | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| Web app | http://localhost:3000 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| PostgreSQL | `localhost:5432` | `rag` / `rag` |
| Redis | `localhost:6379` | — |
| Worker | — | Celery ingestion worker (`ingestion` queue) |

The API container applies Alembic migrations automatically on start. To run them
manually:

```bash
make migrate   # = docker compose exec api alembic upgrade head
```

By default the stack uses offline, deterministic embeddings
(`EMBEDDING_PROVIDER=deterministic`), so ingestion works without any API key.
For real semantic embeddings, set the following in `.env` and restart the API
and worker:

```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
# optional: EMBEDDING_MODEL=text-embedding-3-small (1536 dimensions)
```

The `chunks.embedding` column is fixed at `EMBEDDING_DIMENSION` (1536), matching
`text-embedding-3-small`. Changing the dimension requires a migration.

Answers use the offline extractive provider by default
(`LLM_PROVIDER=deterministic`), which keeps streaming, grounding and citations
working without an API key. For real generation set the following in `.env` and
restart the API:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# optional: LLM_MODEL=gpt-4o-mini, LLM_TEMPERATURE=0.1, LLM_MAX_TOKENS=800
```

The same `OPENAI_API_KEY` covers embeddings and chat when both providers are
`openai`; `EMBEDDING_PROVIDER` and `LLM_PROVIDER` are configured independently.

### 3. Create a user and upload a document

The easiest path is to open **http://localhost:3000**, create an account, and
upload a document from the sidebar. The same flow over HTTP:

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "strong-password-123"}'

# Obtain a token (OAuth2 password flow uses `username` for the email)
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=you@example.com&password=strong-password-123"

# Upload
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer <token>" \
  -F "file=@./handbook.pdf"

# List
curl "http://localhost:8000/api/v1/documents?page=1&page_size=20" \
  -H "Authorization: Bearer <token>"

# Poll ingestion status: queued -> processing -> ready (or failed + error_message)
curl http://localhost:8000/api/v1/documents/<id> \
  -H "Authorization: Bearer <token>"
```

Follow the worker and inspect failed jobs:

```bash
make logs-worker
docker compose exec redis redis-cli LRANGE rag:ingestion:dlq 0 -1
```

Failed documents can be retried without re-uploading:

```bash
curl -X POST http://localhost:8000/api/v1/documents/<id>/reindex \
  -H "Authorization: Bearer <token>"
```

Ask a question once a document is `ready` (empty `document_ids` searches all of
your ready documents):

```bash
# Create a scoped chat session
curl -X POST http://localhost:8000/api/v1/chat/sessions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["<document-id>"]}'

# Stream the grounded answer as Server-Sent Events (-N disables curl buffering)
curl -N -X POST http://localhost:8000/api/v1/chat/sessions/<session-id>/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of paid leave do employees get?"}'
```

### 4. Run without Docker (optional)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

export DATABASE_URL="postgresql+psycopg://rag:rag@localhost:5432/rag"
export S3_ENDPOINT_URL="http://localhost:9000"
export REDIS_URL="redis://localhost:6379/0"

alembic upgrade head
uvicorn app.main:app --reload

# In a second shell: start the ingestion worker
celery -A app.workers.celery_app worker --loglevel=INFO -Q ingestion
```

Run the frontend dev server in a third shell (Vite proxies `/api` to
`http://localhost:8000`):

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### 5. Tests and linting

```bash
cd backend
pytest          # unit tests: auth, upload, parsers, chunking, embeddings, retrieval, prompts, task wiring
ruff check .
```

### Phase 2 notes

- The full schema (users, documents, chunks, chat sessions, messages, feedback)
  was created in Phase 1, including the `pgvector` HNSW index and generated
  `tsvector` column, so ingestion needed no new migrations.
- Uploads enqueue `ingestion.ingest_document` after the database commit. If the
  broker is unavailable the upload still succeeds and the document stays
  `queued`; use the reindex endpoint once Redis is reachable.
- Ingestion is idempotent: chunks for a document are deleted and re-inserted in
  one transaction, and the object-storage file is never modified.
- Transient failures retry with exponential backoff (30s, 60s, 120s; max 3).
  After the final attempt the document is marked `failed` and a JSON record is
  appended to the Redis list `rag:ingestion:dlq` (capped at 1000 entries).
- Scanned/image-only PDFs have no text layer and fail with
  `No extractable text found`; OCR is intentionally out of scope.
- Object storage is created automatically (and by the `minio-init` service),
  so no manual bucket setup is required.
- `S3_PUBLIC_ENDPOINT_URL` controls the host embedded in presigned download
  URLs. Inside Docker the API talks to `http://minio:9000`, while browsers use
  the public value (default `http://localhost:9000`).

### Phase 3 notes

- Retrieval is vector-only in this phase. Hybrid keyword search, query rewriting
  and cross-encoder reranking arrive in Phase 5; the `tsvector` index is already
  in place.
- `LLM_PROVIDER=deterministic` makes the full chat flow work offline by quoting
  the best-matching retrieved sentences with citations. It is an extractive
  development provider, not a general-purpose model.
- Answers stream over SSE (`text/event-stream`), not WebSockets, so they survive
  proxies and work with plain `fetch`. The nginx frontend disables proxy
  buffering and sets a long read timeout so tokens arrive immediately.
- Chat is restricted to the caller's `ready` documents. A session's `doc_scope`
  narrows retrieval to a validated subset; an empty scope searches every ready
  document owned by the user.
- The ask endpoint persists the user message before streaming and opens a fresh
  database session to store the assistant message after generation, so a
  response-scoped session closing early cannot lose the answer. Each assistant
  message stores citation JSON, a token estimate and latency; session titles are
  derived from the first question.
- The React app (Vite + TypeScript) is built in a multi-stage image and served by
  nginx at port 3000. For local development use `npm run dev` and the Vite
  proxy instead.

### Phase 4 notes

- Feedback is one verdict per assistant message (unique `message_id`); posting
  again updates it instead of creating duplicates. Feedback for user or system
  messages is rejected with `400`.
- Deleting a document removes its chunks through the database cascade and then
  deletes the stored object; deleting a session cascades its messages and
  feedback.
- Session titles are derived from the first question but can be renamed to any
  non-blank title up to 255 characters.
- Transcripts are rendered in `app/services/export.py` and streamed as
  `text/markdown` with a sanitised `Content-Disposition` filename.
- The sidebar now exposes document reindex/delete and session
  rename/export/delete; assistant messages have helpful/not-helpful controls
  with an optional comment.


