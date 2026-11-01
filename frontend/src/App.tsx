import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, KeyboardEvent } from "react";

import {
  api,
  askQuestion,
  exportSession,
  getToken,
  login,
  register,
  setToken,
} from "./api";
import type {
  ChatSession,
  Citation,
  DocumentItem,
  Message,
  RetrievalInfo,
  ToolCall,
  User,
} from "./types";

type LocalMessage = Message & {
  streaming?: boolean;
  error?: string;
  model?: string;
  retrieval?: RetrievalInfo;
  rewrittenQuery?: string;
};

type FeedbackHandler = (
  messageId: string,
  rating: -1 | 1,
  comment: string | null,
) => Promise<void>;

export default function App() {
  const [token, setTokenState] = useState<string | null>(() => getToken());
  const [user, setUser] = useState<User | null>(null);
  const [booting, setBooting] = useState(Boolean(getToken()));

  useEffect(() => {
    const handleLogout = () => {
      setTokenState(null);
      setUser(null);
    };
    window.addEventListener("rag:logout", handleLogout);
    return () => window.removeEventListener("rag:logout", handleLogout);
  }, []);

  useEffect(() => {
    if (!token) {
      setUser(null);
      setBooting(false);
      return;
    }

    let cancelled = false;
    setBooting(true);
    api
      .me()
      .then((profile) => {
        if (!cancelled) {
          setUser(profile);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setToken(null);
          setTokenState(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setBooting(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  if (!token) {
    return <AuthScreen onAuthenticated={() => setTokenState(getToken())} />;
  }

  if (booting) {
    return (
      <div className="boot">
        <span className="spinner" />
        <span>Loading workspace…</span>
      </div>
    );
  }

  return (
    <Workspace
      user={user}
      onLogout={() => {
        setToken(null);
        setTokenState(null);
      }}
    />
  );
}

function AuthScreen({ onAuthenticated }: { onAuthenticated: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") {
        await register(email, password);
      }
      await login(email, password);
      onAuthenticated();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-shell">
      <section className="auth-art">
        <span className="brand-mark large">RAG</span>
        <h1>
          Ask your documents.
          <br />
          Trust the answer.
        </h1>
        <p>
          Upload PDFs, Markdown, DOCX and more. Get grounded answers with
          page-level citations, streamed in real time.
        </p>
        <ul>
          <li>Structure-aware ingestion and pgvector retrieval</li>
          <li>Answers restricted to your documents</li>
          <li>Clickable citations with source snippets</li>
        </ul>
      </section>

      <form className="auth-card" onSubmit={submit}>
        <h2>{mode === "login" ? "Welcome back" : "Create an account"}</h2>
        <p className="muted">
          {mode === "login"
            ? "Sign in to your workspace."
            : "Your documents stay scoped to your account."}
        </p>

        <label>
          <span>Email</span>
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
          />
        </label>
        <label>
          <span>Password</span>
          <input
            type="password"
            autoComplete={
              mode === "login" ? "current-password" : "new-password"
            }
            required
            minLength={8}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="At least 8 characters"
          />
        </label>

        {error && <div className="form-error">{error}</div>}

        <button className="primary" type="submit" disabled={busy}>
          {busy
            ? "Please wait…"
            : mode === "login"
              ? "Sign in"
              : "Create account"}
        </button>

        <p className="auth-switch">
          {mode === "login" ? "New here?" : "Already have an account?"}{" "}
          <button
            type="button"
            className="link"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
          >
            {mode === "login" ? "Create an account" : "Sign in"}
          </button>
        </p>
      </form>
    </div>
  );
}

function Workspace({
  user,
  onLogout,
}: {
  user: User | null;
  onLogout: () => void;
}) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [scope, setScope] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [docActionId, setDocActionId] = useState<string | null>(null);
  const [renamingSession, setRenamingSession] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [sessionAction, setSessionAction] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const list = await api.documents();
      setDocuments(list.items);
    } catch (err) {
      setError(errorText(err));
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.sessions();
      setSessions(list.items);
    } catch (err) {
      setError(errorText(err));
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
    void refreshSessions();
  }, [refreshDocuments, refreshSessions]);

  const hasPendingDocuments = documents.some(
    (document) =>
      document.status === "queued" || document.status === "processing",
  );

  useEffect(() => {
    if (!hasPendingDocuments) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshDocuments();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [hasPendingDocuments, refreshDocuments]);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages]);

  function toggleScope(documentId: string) {
    setScope((previous) =>
      previous.includes(documentId)
        ? previous.filter((id) => id !== documentId)
        : [...previous, documentId],
    );
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    setUploading(true);
    setError(null);
    try {
      await api.uploadDocument(file);
      await refreshDocuments();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleReindex(documentId: string) {
    setDocActionId(documentId);
    setError(null);
    try {
      await api.reindexDocument(documentId);
      await refreshDocuments();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setDocActionId(null);
    }
  }

  async function handleDeleteDocument(document: DocumentItem) {
    if (
      !window.confirm(
        `Delete "${document.filename}" and its indexed chunks? This cannot be undone.`,
      )
    ) {
      return;
    }

    setDocActionId(document.id);
    setError(null);
    try {
      await api.deleteDocument(document.id);
      setScope((previous) => previous.filter((id) => id !== document.id));
      await refreshDocuments();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setDocActionId(null);
    }
  }

  function startRename() {
    if (!activeSession) {
      return;
    }
    setRenameValue(activeSession.title);
    setRenamingSession(true);
  }

  async function saveRename(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const title = renameValue.trim();
    if (!title || !activeSessionId) {
      return;
    }

    setSessionAction(true);
    setError(null);
    try {
      const updated = await api.renameSession(activeSessionId, title);
      setSessions((previous) =>
        previous.map((session) =>
          session.id === updated.id ? updated : session,
        ),
      );
      setRenamingSession(false);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSessionAction(false);
    }
  }

  async function handleDeleteSession() {
    if (!activeSession) {
      return;
    }
    if (
      !window.confirm(
        `Delete "${activeSession.title}"? Its messages and feedback will be removed.`,
      )
    ) {
      return;
    }

    setSessionAction(true);
    setError(null);
    try {
      await api.deleteSession(activeSession.id);
      setSessions((previous) =>
        previous.filter((session) => session.id !== activeSession.id),
      );
      setActiveSessionId(null);
      setMessages([]);
      setRenamingSession(false);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSessionAction(false);
    }
  }

  async function handleExportSession() {
    if (!activeSession) {
      return;
    }

    setSessionAction(true);
    setError(null);
    try {
      const { filename, markdown } = await exportSession(activeSession.id);
      downloadText(filename, markdown);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSessionAction(false);
    }
  }

  async function handleFeedback(
    messageId: string,
    rating: -1 | 1,
    comment: string | null,
  ) {
    try {
      const saved = await api.feedback(messageId, rating, comment);
      setMessages((previous) =>
        previous.map((message) =>
          message.id === messageId ? { ...message, feedback: saved } : message,
        ),
      );
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function startNewChat() {
    setLoadingSession(true);
    setError(null);
    try {
      const session = await api.createSession(scope);
      setSessions((previous) => [session, ...previous]);
      setActiveSessionId(session.id);
      setMessages([]);
      setRenamingSession(false);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoadingSession(false);
    }
  }

  async function openSession(sessionId: string) {
    if (sessionId === activeSessionId || streaming) {
      return;
    }
    setLoadingSession(true);
    setError(null);
    try {
      const detail = await api.session(sessionId);
      setActiveSessionId(sessionId);
      setMessages(detail.messages);
      setScope(detail.doc_scope);
      setRenamingSession(false);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoadingSession(false);
    }
  }

  async function send() {
    const text = question.trim();
    if (!text || streaming) {
      return;
    }

    setError(null);
    setQuestion("");
    setStreaming(true);
    setLoadingSession(false);

    let optimisticIds: string[] = [];
    let assistantKey: string | null = null;

    try {
      let sessionId = activeSessionId;
      if (!sessionId) {
        const session = await api.createSession(scope);
        sessionId = session.id;
        setSessions((previous) => [session, ...previous]);
        setActiveSessionId(session.id);
      }

      const userMessage: LocalMessage = {
        id: `local-user-${Date.now()}`,
        session_id: sessionId,
        role: "user",
        content: text,
        citations: [],
        tool_calls: [],
        tokens_used: null,
        latency_ms: null,
        feedback: null,
        created_at: new Date().toISOString(),
      };
      const pendingId = `pending-${Date.now()}`;
      assistantKey = pendingId;
      optimisticIds = [userMessage.id, pendingId];

      setMessages((previous) => [
        ...previous,
        userMessage,
        {
          id: pendingId,
          session_id: sessionId,
          role: "assistant",
          content: "",
          citations: [],
          tool_calls: [],
          tokens_used: null,
          latency_ms: null,
          feedback: null,
          created_at: new Date().toISOString(),
          streaming: true,
        },
      ]);

      let answer = "";
      await askQuestion(sessionId, text, {
        onMeta: (meta) => {
          assistantKey = meta.message_id;
          setMessages((previous) =>
            previous.map((message) =>
              message.id === pendingId
                ? {
                    ...message,
                    id: meta.message_id,
                    session_id: meta.session_id,
                    citations: meta.citations,
                    tool_calls: meta.tools ?? [],
                    model: meta.model,
                    retrieval: meta.retrieval,
                    rewrittenQuery: meta.retrieval?.rewritten
                      ? meta.query
                      : undefined,
                  }
                : message,
            ),
          );
        },
        onToken: (token) => {
          answer += token;
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantKey
                ? { ...message, content: answer }
                : message,
            ),
          );
        },
        onDone: (done) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantKey
                ? {
                    ...message,
                    streaming: false,
                    tokens_used: done.tokens_used,
                    latency_ms: done.latency_ms,
                  }
                : message,
            ),
          );
          void refreshSessions();
        },
        onError: (detail) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantKey
                ? {
                    ...message,
                    streaming: false,
                    error: detail,
                    content:
                      answer ||
                      "The assistant could not complete the answer.",
                  }
                : message,
            ),
          );
          setError(detail);
        },
      });
    } catch (err) {
      setMessages((previous) =>
        previous.filter(
          (message) =>
            !optimisticIds.includes(message.id) &&
            message.id !== assistantKey,
        ),
      );
      setQuestion(text);
      setError(errorText(err));
    } finally {
      setStreaming(false);
    }
  }

  function handleComposerKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  }

  const activeSession =
    sessions.find((session) => session.id === activeSessionId) ?? null;
  const scopeLabel = scope.length
    ? `${scope.length} document${scope.length === 1 ? "" : "s"} in scope`
    : "All ready documents";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">RAG</span>
          <div>
            <h1>Document Q&amp;A</h1>
            <p>Grounded answers with citations</p>
          </div>
        </div>
        <div className="topbar-right">
          <span className="scope-pill">{scopeLabel}</span>
          {user && <span className="user-email">{user.email}</span>}
          <button className="ghost" onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <section className="sidebar-section">
            <div className="section-heading">
              <h2>Documents</h2>
              <button
                className="ghost small"
                onClick={() => fileRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? "Uploading…" : "Upload"}
              </button>
              <input
                ref={fileRef}
                type="file"
                hidden
                accept=".pdf,.md,.markdown,.txt,.docx,.html,.htm"
                onChange={handleUpload}
              />
            </div>
            <p className="hint">
              Tick ready documents to scope a new chat. Leave all unticked to
              search everything.
            </p>
            <ul className="doc-list">
              {documents.length === 0 && (
                <li className="empty-list">No documents yet.</li>
              )}
              {documents.map((document) => (
                <li
                  key={document.id}
                  className={`doc-item status-${document.status}`}
                >
                  <label>
                    <input
                      type="checkbox"
                      checked={scope.includes(document.id)}
                      disabled={document.status !== "ready" || streaming}
                      onChange={() => toggleScope(document.id)}
                    />
                    <span className="doc-name" title={document.filename}>
                      {document.filename}
                    </span>
                    <span
                      className={`status-dot ${document.status}`}
                      title={document.status}
                    />
                  </label>
                  <div className="doc-meta">
                    <span>{document.status}</span>
                    <span>{formatBytes(document.size_bytes)}</span>
                  </div>
                  {document.status === "failed" && document.error_message && (
                    <p className="doc-error" title={document.error_message}>
                      {document.error_message}
                    </p>
                  )}
                  <div className="doc-actions">
                    {(document.status === "ready" ||
                      document.status === "failed") && (
                      <button
                        className="ghost small"
                        onClick={() => void handleReindex(document.id)}
                        disabled={docActionId === document.id || streaming}
                      >
                        Reindex
                      </button>
                    )}
                    <button
                      className="ghost small danger"
                      onClick={() => void handleDeleteDocument(document)}
                      disabled={docActionId === document.id || streaming}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <section className="sidebar-section grow">
            <div className="section-heading">
              <h2>Chats</h2>
              <button
                className="ghost small"
                onClick={() => void startNewChat()}
                disabled={streaming || loadingSession}
              >
                New
              </button>
            </div>
            {activeSession && (
              <div className="session-actions">
                {renamingSession ? (
                  <form className="rename-form" onSubmit={saveRename}>
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      maxLength={255}
                      aria-label="Session title"
                      disabled={sessionAction}
                    />
                    <button
                      className="ghost small"
                      type="submit"
                      disabled={sessionAction || !renameValue.trim()}
                    >
                      Save
                    </button>
                    <button
                      className="ghost small"
                      type="button"
                      onClick={() => setRenamingSession(false)}
                      disabled={sessionAction}
                    >
                      Cancel
                    </button>
                  </form>
                ) : (
                  <>
                    <button
                      className="ghost small"
                      onClick={startRename}
                      disabled={streaming || sessionAction}
                    >
                      Rename
                    </button>
                    <button
                      className="ghost small"
                      onClick={() => void handleExportSession()}
                      disabled={sessionAction}
                    >
                      Export
                    </button>
                    <button
                      className="ghost small danger"
                      onClick={() => void handleDeleteSession()}
                      disabled={streaming || sessionAction}
                    >
                      Delete
                    </button>
                  </>
                )}
              </div>
            )}
            <ul className="session-list">
              {sessions.length === 0 && (
                <li className="empty-list">No conversations yet.</li>
              )}
              {sessions.map((session) => (
                <li key={session.id}>
                  <button
                    className={`session-item ${
                      session.id === activeSessionId ? "active" : ""
                    }`}
                    onClick={() => void openSession(session.id)}
                    disabled={streaming || loadingSession}
                  >
                    <span className="session-title">{session.title}</span>
                    <span className="session-date">
                      {formatDate(session.created_at)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </aside>

        <main className="main">
          <div className="messages" ref={scrollRef}>
            {loadingSession ? (
              <div className="boot inline">
                <span className="spinner" />
                <span>Loading conversation…</span>
              </div>
            ) : messages.length === 0 ? (
              <EmptyState onPick={setQuestion} />
            ) : (
              messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  onCitation={setActiveCitation}
                  onFeedback={handleFeedback}
                />
              ))
            )}
          </div>

          <footer className="composer">
            {error && (
              <div className="error-banner">
                <span>{error}</span>
                <button
                  className="ghost small"
                  onClick={() => setError(null)}
                >
                  Dismiss
                </button>
              </div>
            )}
            <div className="composer-row">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={
                  activeSessionId
                    ? "Ask a question about the selected documents…"
                    : "Ask a question to start a new chat…"
                }
                rows={2}
                maxLength={4000}
                disabled={streaming}
              />
              <button
                className="send"
                onClick={() => void send()}
                disabled={streaming || !question.trim()}
              >
                {streaming ? "Answering…" : "Send"}
              </button>
            </div>
            <p className="composer-hint">
              Enter to send · Shift+Enter for a new line · {scopeLabel}
            </p>
          </footer>
        </main>
      </div>

      {activeCitation && (
        <CitationModal
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
        />
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  const suggestions = [
    "Summarise the key policies in my documents.",
    "What are the most important deadlines?",
    "Which document mentions security requirements?",
  ];

  return (
    <div className="empty-state">
      <div className="empty-mark">✦</div>
      <h2>Ask your documents</h2>
      <p>
        Upload a document, wait for it to finish processing, then ask a
        question. Answers are grounded in retrieved passages and cite their
        sources.
      </p>
      <div className="empty-grid">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            className="suggestion"
            onClick={() => onPick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

function toolLabel(tool: ToolCall): string {
  const page = tool.arguments.page_number;
  if (
    tool.name === "get_page" &&
    (typeof page === "number" || typeof page === "string")
  ) {
    return `page ${page}`;
  }
  switch (tool.name) {
    case "search_documents":
      return "search";
    case "summarize_document":
      return "summarize";
    case "compare_documents":
      return "compare";
    default:
      return tool.name.replaceAll("_", " ");
  }
}

function MessageBubble({
  message,
  onCitation,
  onFeedback,
}: {
  message: LocalMessage;
  onCitation: (citation: Citation) => void;
  onFeedback: FeedbackHandler;
}) {
  const isUser = message.role === "user";

  return (
    <article className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "You" : "AI"}</div>
      <div className="message-body">
        <div className="message-head">
          <span>{isUser ? "You" : "Assistant"}</span>
          {message.model && (
            <span className="model-badge">{message.model}</span>
          )}
          {message.retrieval?.hybrid && (
            <span
              className="model-badge alt"
              title="Dense vector and keyword retrieval"
            >
              hybrid
            </span>
          )}
          {message.retrieval?.reranked && (
            <span
              className="model-badge alt"
              title="Candidates re-scored before answering"
            >
              reranked
            </span>
          )}
          {message.rewrittenQuery && (
            <span
              className="model-badge alt"
              title={`Follow-up rewritten for retrieval: ${message.rewrittenQuery}`}
            >
              rewritten
            </span>
          )}
          {message.retrieval?.agent && (
            <span
              className="model-badge alt"
              title="Agent planned tool calls for this answer"
            >
              agent
            </span>
          )}
        </div>

        <div className="message-content">
          {message.content}
          {message.streaming && <span className="stream-cursor" />}
          {!message.content && message.streaming && !isUser && (
            <span className="thinking">Retrieving context…</span>
          )}
        </div>

        {!isUser && message.citations.length > 0 && (
          <div className="citations">
            {message.citations.map((citation) => (
              <button
                key={`${citation.chunk_id}-${citation.index}`}
                className="citation-chip"
                onClick={() => onCitation(citation)}
                title={citation.snippet}
              >
                <span className="citation-index">[{citation.index}]</span>
                <span>{citation.filename}</span>
                {citation.page_number !== null && (
                  <span className="citation-page">
                    p. {citation.page_number}
                  </span>
                )}
              </button>
            ))}
          </div>
        )}

        {!isUser && message.tool_calls.length > 0 && (
          <div className="tool-trace" aria-label="Agent tool calls">
            {message.tool_calls.map((tool, index) => (
              <span
                key={`${tool.name}-${index}`}
                className={`tool-chip ${tool.status === "error" ? "error" : ""}`}
                title={
                  tool.status === "error"
                    ? tool.detail ?? "Tool failed"
                    : `${JSON.stringify(tool.arguments)} · ${tool.matches} matches`
                }
              >
                <span className="tool-dot" />
                {toolLabel(tool)}
                {tool.status === "ok" && tool.matches > 0 && (
                  <span className="tool-matches">{tool.matches}</span>
                )}
              </span>
            ))}
          </div>
        )}

        {message.error && (
          <div className="message-error">{message.error}</div>
        )}

        {!isUser && !message.streaming && message.latency_ms !== null && (
          <div className="message-foot">
            {message.tokens_used ?? 0} tokens · {message.latency_ms} ms
          </div>
        )}

        {!isUser && !message.streaming && !message.error && (
          <FeedbackBar message={message} onFeedback={onFeedback} />
        )}
      </div>
    </article>
  );
}

function FeedbackBar({
  message,
  onFeedback,
}: {
  message: LocalMessage;
  onFeedback: FeedbackHandler;
}) {
  const existing = message.feedback;
  const [comment, setComment] = useState(existing?.comment ?? "");
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(rating: -1 | 1, nextComment: string | null) {
    setBusy(true);
    try {
      await onFeedback(message.id, rating, nextComment);
    } finally {
      setBusy(false);
    }
  }

  async function saveComment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!existing) {
      return;
    }
    await submit(existing.rating, comment.trim() || null);
    setEditing(false);
  }

  const pendingComment = editing
    ? comment.trim() || null
    : (existing?.comment ?? null);

  return (
    <div className="feedback">
      <div className="feedback-bar">
        <span className="feedback-label">Was this helpful?</span>
        <button
          className={`feedback-btn ${existing?.rating === 1 ? "active" : ""}`}
          onClick={() => void submit(1, pendingComment)}
          disabled={busy}
        >
          Helpful
        </button>
        <button
          className={`feedback-btn ${existing?.rating === -1 ? "active" : ""}`}
          onClick={() => void submit(-1, pendingComment)}
          disabled={busy}
        >
          Not helpful
        </button>
        {existing && !editing && (
          <button
            className="link feedback-comment-toggle"
            type="button"
            onClick={() => {
              setComment(existing.comment ?? "");
              setEditing(true);
            }}
          >
            {existing.comment ? "Edit comment" : "Add comment"}
          </button>
        )}
      </div>

      {existing?.comment && !editing && (
        <p className="feedback-saved">“{existing.comment}”</p>
      )}

      {editing && (
        <form className="feedback-form" onSubmit={saveComment}>
          <input
            autoFocus
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            maxLength={2000}
            placeholder="What worked or what was missing?"
            aria-label="Feedback comment"
            disabled={busy}
          />
          <button className="ghost small" type="submit" disabled={busy}>
            Save
          </button>
          <button
            className="ghost small"
            type="button"
            onClick={() => setEditing(false)}
            disabled={busy}
          >
            Cancel
          </button>
        </form>
      )}
    </div>
  );
}

function CitationModal({
  citation,
  onClose,
}: {
  citation: Citation;
  onClose: () => void;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="citation-modal"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-head">
          <h3>Source [{citation.index}]</h3>
          <button className="ghost small" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="citation-file">
          {citation.filename}
          {citation.page_number !== null
            ? ` · page ${citation.page_number}`
            : ""}
          {citation.section ? ` · ${citation.section}` : ""}
        </p>
        <blockquote>{citation.snippet}</blockquote>
        <p className="citation-score">
          Relevance {Math.max(0, Math.round(citation.score * 100))}% · chunk #
          {citation.chunk_index}
        </p>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function downloadText(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
