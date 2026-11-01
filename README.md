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

This platform provides asynchronous document ingestion, multi-stage hybrid retrieval (vector similarity + keyword BM25 search), neural reranking, and grounded generative Q&A with precise source citations. It supports progressive evolution from a deterministic RAG pipeline to tool-calling autonomous agents.

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
- **Query Rewriting**: Contextual resolution of ambiguous follow-up questions using session history.
- **Hybrid Retrieval**: Combines dense vector similarity (`pgvector` with HNSW cosine index) and sparse keyword search (`tsvector` / BM25).
- **Reranking**: Neural cross-encoders (e.g., Cohere Rerank or BGE Reranker) distill candidates to the top 4–6 high-precision passages.
- **Grounded Generation**: System prompt enforcement restricting responses exclusively to verified context.
- **Clickable Citations**: UI citation chips mapped directly to document and page numbers.
- **Real-time Streaming**: Server-Sent Events (SSE) for low-latency streaming responses.
- **Agentic Tool Loop**: Support for function/tool calling (`search_documents`, `summarize_document`, `compare_documents`, `get_page`).

### 3. Session & Document Management
- Dynamic document scoping: query across all documents or target a selected subset.
- Session-based conversation logs with export, rename, and user feedback mechanisms.

---

## 🛠️ Tech Stack

| Component | Technology | Alternative / Scale-up |
|---|---|---|
| **Backend API** | Python 3.11+, FastAPI, Uvicorn | FastAPI + Kubernetes |
| **Worker Queue** | Celery, Redis | Temporal, AWS SQS |
| **Relational & Vector DB** | PostgreSQL 16+ with `pgvector` | Qdrant, Milvus, Pinecone |
| **Object Storage** | MinIO (local dev) / AWS S3 | Cloudflare R2, GCS |
| **Document Parsers** | PyMuPDF, `pdfplumber`, `unstructured` | Docling, Azure Doc Intelligence |
| **LLM & Embeddings** | OpenAI / Google Gemini / Anthropic | Self-hosted (vLLM, Ollama, TEI) |
| **Reranker** | BGE-Reranker-Large, Cohere Rerank | Cross-Encoder HuggingFace |
| **Frontend** | React (Next.js / Vite), Tailwind CSS | React + Shadcn/ui |

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
- `GET /api/v1/documents/{id}/download` - Redirect to a short-lived presigned URL
- `DELETE /api/v1/documents/{id}` - Delete document, associated chunks, and stored file
- `POST /api/v1/documents/{id}/reindex` - Trigger re-chunking and re-embedding *(Phase 4)*

### Chat & Q&A *(Phase 3+)*
- `POST /api/v1/chat/sessions` - Create chat session (with optional document scoping)
- `GET /api/v1/chat/sessions` - List active sessions
- `GET /api/v1/chat/sessions/{id}` - Get session message history
- `POST /api/v1/chat/sessions/{id}/ask` - Submit query & stream response via SSE
- `POST /api/v1/messages/{id}/feedback` - Submit helpfulness rating / comments

### Service
- `GET /` - Service metadata
- `GET /healthz` - Liveness + database connectivity check

### Upload behaviour (Phase 1)
- Supported types: `.pdf`, `.md`, `.markdown`, `.txt`, `.docx`, `.html`, `.htm` (configurable)
- Maximum size: 50 MB by default (`MAX_UPLOAD_SIZE_MB`), enforced while streaming
- Files are hashed (SHA-256) during upload; re-uploading identical bytes returns `409 Conflict`
  with the existing document ID in the `X-Document-Id` header
- Objects are stored at `s3://<bucket>/users/<user_id>/documents/<document_id>/<filename>`

---

## 📁 Project Directory Structure

```text
rag/
├── docker-compose.yml          # PostgreSQL+pgvector, Redis, MinIO, API
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
│       ├── schemas/            # Pydantic request/response models
│       ├── services/
│       │   ├── storage.py      # S3/MinIO client & presigned URLs
│       │   └── upload.py       # Validation, hashing, streaming
│       └── api/                # REST endpoints
│           ├── deps.py         # Auth dependencies
│           └── v1/
│               ├── auth.py
│               ├── documents.py
│               └── router.py
├── workers/                    # Ingestion pipeline (Phase 2)
└── frontend/                   # React app (Phase 3)
```

---

## 📅 Implementation Roadmap

| Phase | Milestone | Deliverables |
|---|---|---|
| **Phase 1** | ✅ Foundation & Storage | Database schema, authentication, file storage (S3/MinIO), and upload endpoint |
| **Phase 2** | Ingestion Engine | Celery pipeline: parsing, recursive chunking, batch embeddings, pgvector indexing |
| **Phase 3** | Core RAG & Streaming | Vector search, basic prompt template, SSE streaming endpoint, and React chat interface |
| **Phase 4** | Document & History Scope | Document management, conversation sessions, reindexing, and document-specific scoping |
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
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| PostgreSQL | `localhost:5432` | `rag` / `rag` |
| Redis | `localhost:6379` | — |

The API container applies Alembic migrations automatically on start. To run them
manually:

```bash
make migrate   # = docker compose exec api alembic upgrade head
```

### 3. Create a user and upload a document

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
```

### 4. Run without Docker (optional)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

export DATABASE_URL="postgresql+psycopg://rag:rag@localhost:5432/rag"
export S3_ENDPOINT_URL="http://localhost:9000"

alembic upgrade head
uvicorn app.main:app --reload
```

### 5. Tests and linting

```bash
cd backend
pytest          # unit tests: auth, upload validation, OpenAPI/DDL checks
ruff check .
```

### Phase 1 notes

- The full schema (users, documents, chunks, chat sessions, messages, feedback)
  is created up front, including the `pgvector` HNSW index and generated
  `tsvector` column, so Phase 2 can index chunks without further migrations.
- Uploaded documents remain in `queued` status until the Celery ingestion
  worker from Phase 2 picks them up.
- Object storage is created automatically (and by the `minio-init` service),
  so no manual bucket setup is required.
- `S3_PUBLIC_ENDPOINT_URL` controls the host embedded in presigned download
  URLs. Inside Docker the API talks to `http://minio:9000`, while browsers use
  the public value (default `http://localhost:9000`).


