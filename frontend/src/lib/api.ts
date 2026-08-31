// Thin wrapper over the backend API. Every call sends cookies, which is how the
// session travels.

export type ChapterMeta = {
  slug: string;
  title: string;
  part: string;
  summary: string;
  minutes?: number;
  objectives: string[];
  lab?: string;
  gpu: boolean;
};

export type Lab = {
  id: string;
  title: string;
  brief: string;
  gpu: string;
  needs_gpu: boolean;
  timeout: number;
  hints: string[];
  metrics: string[];
  starter: string;
  draft?: string | null;
  runs?: Run[];
};

export type Chapter = ChapterMeta & {
  body: string;
  prev: string | null;
  next: string | null;
  note: string;
  lab_detail?: Lab;
};

export type Run = {
  id: string;
  lab: string;
  provider: string;
  status: string;
  passed: boolean | null;
  metrics: Record<string, unknown>;
  started_at: number;
  finished_at: number | null;
};

export type ProviderInfo = {
  name: string;
  available: boolean;
  reason: string;
  preferred: boolean;
};

export type InfraAction = {
  action: string;
  label: string;
  confirm?: string;
};

export type Instance = {
  provider: string;
  kind: string;
  id: string;
  label: string;
  status: string;
  active: boolean;
  gpu: string;
  started_at: number | null;
  age_label: string;
  detail: string;
  actions: InfraAction[];
};

export type InfraVolume = {
  provider: string;
  id: string;
  name: string;
  kind: string;
  detail: string;
  size_gb: number | null;
  region: string;
  browsable: boolean;
  console_url: string;
  primary: boolean;
};

export type Fact = { label: string; value: string };

export type ProviderInfra = {
  name: string;
  available: boolean;
  reason: string;
  default: boolean;
  console_url: string;
  instances: Instance[];
  volumes: InfraVolume[];
  notices: string[];
  facts: Fact[];
  functions?: { name: string; deployed: boolean; backlog?: number; containers?: number }[];
  account?: Record<string, number>;
  jobs?: Record<string, number>;
};

export type StrandedRun = {
  id: string;
  user: string;
  lab: string;
  provider: string;
  status: string;
  job_id: string | null;
  started_at: number;
  age: number;
  cancellable: boolean;
  hint: string;
};

export type Infra = {
  generated_at: number;
  active: number;
  providers: ProviderInfra[];
  runs: StrandedRun[];
};

export type VolumeEntry = {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  modified_at: number | null;
};

export type VolumeListing = {
  provider: string;
  volume: string;
  path: string;
  parent: string | null;
  bytes: number;
  entries: VolumeEntry[];
};

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // The body was not JSON; the status text is the best we have.
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  me: () =>
    request<{ name: string | null; model: string; small_model: string; gpu: string }>(
      "/api/me",
    ),
  login: (name: string, password: string) =>
    request<{ name: string }>("/api/login", {
      method: "POST",
      body: JSON.stringify({ name, password }),
    }),
  logout: () => request<{ ok: boolean }>("/api/logout", { method: "POST" }),
  chapters: () =>
    request<{ chapters: ChapterMeta[]; progress: Record<string, string> }>(
      "/api/chapters",
    ),
  chapter: (slug: string) => request<Chapter>(`/api/chapters/${slug}`),
  lab: (id: string) => request<Lab>(`/api/labs/${id}`),
  solution: (id: string) => request<{ solution: string }>(`/api/labs/${id}/solution`),
  setProgress: (chapter: string, status: "in_progress" | "done") =>
    request<{ ok: boolean }>("/api/progress", {
      method: "POST",
      body: JSON.stringify({ chapter, status }),
    }),
  saveDraft: (lab: string, code: string) =>
    request<{ ok: boolean }>("/api/drafts", {
      method: "POST",
      body: JSON.stringify({ lab, code }),
    }),
  saveNote: (chapter: string, body: string) =>
    request<{ ok: boolean }>("/api/notes", {
      method: "POST",
      body: JSON.stringify({ chapter, body }),
    }),
  runs: (lab?: string) =>
    request<{ runs: Run[] }>(`/api/runs${lab ? `?lab=${encodeURIComponent(lab)}` : ""}`),
  providers: () =>
    request<{ default: string; providers: ProviderInfo[] }>("/api/providers"),
  infra: () => request<Infra>("/api/infra"),
  infraAction: (provider: string, action: string, target: string) =>
    request<{ message: string }>("/api/infra/action", {
      method: "POST",
      body: JSON.stringify({ provider, action, target }),
    }),
  volume: (provider: string, name: string, path: string) =>
    request<VolumeListing>(
      `/api/infra/volume?provider=${encodeURIComponent(provider)}` +
        `&name=${encodeURIComponent(name)}&path=${encodeURIComponent(path)}`,
    ),
};

export type RunEvent =
  | { type: "start"; run_id: string; provider: string }
  | { type: "log"; line: string }
  | {
      type: "result";
      passed: boolean;
      checks: { name: string; passed: boolean; detail: string }[];
      metrics: Record<string, unknown>;
    }
  | { type: "done"; run_id: string; passed: boolean; seconds: number }
  | { type: "error"; message: string }
  | { type: "out_of_credits"; provider: string; message: string; hint: string };

/**
 * Streams a lab run. The backend answers with server-sent events, so this reads
 * the response body rather than using EventSource, which cannot POST.
 */
export async function runLab(
  lab: string,
  code: string,
  provider: string | null,
  onEvent: (event: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/run", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lab, code, provider }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const payload = frame.replace(/^data: /, "");
      if (payload) {
        try {
          onEvent(JSON.parse(payload) as RunEvent);
        } catch {
          onEvent({ type: "log", line: payload });
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
