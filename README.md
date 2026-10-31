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

### Documents
- `POST /api/v1/documents` - Upload document file (multipart/form-data)
- `GET /api/v1/documents` - List uploaded documents (pagination, status filters)
- `GET /api/v1/documents/{id}` - Get processing status and details
- `DELETE /api/v1/documents/{id}` - Delete document, associated chunks, and vectors
- `POST /api/v1/documents/{id}/reindex` - Trigger re-chunking and re-embedding

### Chat & Q&A
- `POST /api/v1/chat/sessions` - Create chat session (with optional document scoping)
- `GET /api/v1/chat/sessions` - List active sessions
- `GET /api/v1/chat/sessions/{id}` - Get session message history
- `POST /api/v1/chat/sessions/{id}/ask` - Submit query & stream response via SSE
- `POST /api/v1/messages/{id}/feedback` - Submit helpfulness rating / comments

---

## 📁 Project Directory Structure

```text
rag/
├── docker-compose.yml          # PostgreSQL, pgvector, Redis, MinIO
├── README.md                   # System documentation
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic/                # DB migrations
│   └── app/
│       ├── main.py             # FastAPI entrypoint
│       ├── core/               # App configuration & security
│       │   ├── config.py
│       │   └── security.py
│       ├── db/                 # DB engine, session & models
│       │   ├── session.py
│       │   └── models.py
│       ├── api/                # REST endpoints
│       │   └── v1/
│       │       ├── auth.py
│       │       ├── documents.py
│       │       └── chat.py
│       ├── workers/            # Ingestion pipeline (Celery)
│       │   ├── celery_app.py
│       │   ├── parser.py
│       │   ├── chunker.py
│       │   └── embedder.py
│       └── rag/                # Retrieval, reranking & LLM integration
│           ├── retriever.py
│           ├── reranker.py
│           ├── prompt.py
│           └── agent.py
└── frontend/
    ├── package.json
    ├── src/
    │   ├── components/
    │   │   ├── uploader/
    │   │   ├── chat/
    │   │   └── history/
    │   ├── hooks/
    │   └── services/
```

---

## 📅 Implementation Roadmap

| Phase | Milestone | Deliverables |
|---|---|---|
| **Phase 1** | Foundation & Storage | Database schema, authentication, file storage (S3/MinIO), and upload endpoint |
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

*(Detailed installation, Docker orchestration, and environment setup instructions will be updated here as Phase 1 commences.)*

