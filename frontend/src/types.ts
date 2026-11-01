export type DocumentStatus = "queued" | "processing" | "ready" | "failed";

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  file_hash: string;
  status: DocumentStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentDetail extends DocumentItem {
  chunk_count: number;
}

export interface DocumentList {
  items: DocumentItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface Citation {
  index: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  chunk_index: number;
  page_number: number | null;
  section: string | null;
  score: number;
  snippet: string;
}

export interface Feedback {
  id: string;
  message_id: string;
  rating: -1 | 1;
  comment: string | null;
  created_at: string;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations: Citation[];
  tokens_used: number | null;
  latency_ms: number | null;
  feedback: Feedback | null;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  doc_scope: string[];
  created_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: Message[];
}

export interface SessionList {
  items: ChatSession[];
  total: number;
  page: number;
  page_size: number;
}

export interface RetrievalInfo {
  rewritten: boolean;
  hybrid: boolean;
  reranked: boolean;
  matches: number;
}

export interface MetaEvent {
  session_id: string;
  message_id: string;
  model: string;
  citations: Citation[];
  query?: string;
  retrieval?: RetrievalInfo;
}

export interface DoneEvent {
  message_id: string;
  latency_ms: number;
  tokens_used: number;
}
