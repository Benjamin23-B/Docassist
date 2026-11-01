import type {
  ChatSession,
  ChatSessionDetail,
  DocumentDetail,
  DocumentList,
  DoneEvent,
  Feedback,
  MetaEvent,
  SessionList,
  User,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";
const TOKEN_KEY = "rag.access_token";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getToken(): string | null {
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) {
    window.localStorage.setItem(TOKEN_KEY, token);
  } else {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return String(item);
        })
        .join("; ");
    }
  } catch {
    // Fall through to the generic message.
  }
  return `Request failed (${response.status})`;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    if (response.status === 401) {
      setToken(null);
    }
    throw new ApiError(response.status, await errorMessage(response));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export async function register(
  email: string,
  password: string,
): Promise<void> {
  await request<User>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function login(
  email: string,
  password: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  const payload = (await response.json()) as { access_token: string };
  setToken(payload.access_token);
}

export const api = {
  me: () => request<User>("/auth/me"),
  documents: () => request<DocumentList>("/documents?page=1&page_size=100"),
  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<DocumentDetail>("/documents", { method: "POST", body });
  },
  reindexDocument: (documentId: string) =>
    request<DocumentDetail>(`/documents/${documentId}/reindex`, {
      method: "POST",
    }),
  deleteDocument: (documentId: string) =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),
  sessions: () =>
    request<SessionList>("/chat/sessions?page=1&page_size=50"),
  createSession: (documentIds: string[]) =>
    request<ChatSession>("/chat/sessions", {
      method: "POST",
      body: JSON.stringify({ document_ids: documentIds }),
    }),
  session: (sessionId: string) =>
    request<ChatSessionDetail>(`/chat/sessions/${sessionId}`),
  renameSession: (sessionId: string, title: string) =>
    request<ChatSession>(`/chat/sessions/${sessionId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteSession: (sessionId: string) =>
    request<void>(`/chat/sessions/${sessionId}`, { method: "DELETE" }),
  feedback: (messageId: string, rating: -1 | 1, comment: string | null) =>
    request<Feedback>(`/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ rating, comment }),
    }),
};

export async function exportSession(
  sessionId: string,
): Promise<{ filename: string; markdown: string }> {
  const headers = new Headers();
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(
    `${API_BASE}/chat/sessions/${sessionId}/export`,
    { headers },
  );
  if (!response.ok) {
    if (response.status === 401) {
      setToken(null);
    }
    throw new ApiError(response.status, await errorMessage(response));
  }

  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(disposition);
  return {
    filename: match?.[1] ?? "chat-session.md",
    markdown: await response.text(),
  };
}

export interface AskHandlers {
  onMeta?: (event: MetaEvent) => void;
  onToken?: (text: string) => void;
  onDone?: (event: DoneEvent) => void;
  onError?: (detail: string) => void;
}

export async function askQuestion(
  sessionId: string,
  question: string,
  handlers: AskHandlers,
): Promise<void> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(
    `${API_BASE}/chat/sessions/${sessionId}/ask`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
    },
  );

  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  if (!response.body) {
    throw new ApiError(500, "Streaming is not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleFrame = (frame: string): void => {
    let event = "message";
    const dataLines: string[] = [];

    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trimStart());
      }
    }
    if (!dataLines.length) {
      return;
    }

    const payload = JSON.parse(dataLines.join("\n")) as Record<
      string,
      unknown
    >;
    switch (event) {
      case "meta":
        handlers.onMeta?.(payload as unknown as MetaEvent);
        break;
      case "token":
        handlers.onToken?.(String(payload.text ?? ""));
        break;
      case "done":
        handlers.onDone?.(payload as unknown as DoneEvent);
        break;
      case "error":
        handlers.onError?.(
          String(payload.detail ?? "The assistant could not answer."),
        );
        break;
      default:
        break;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary).replace(/\r/g, "");
      buffer = buffer.slice(boundary + 2);
      if (frame.trim()) {
        handleFrame(frame);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    handleFrame(buffer.replace(/\r/g, ""));
  }
}
