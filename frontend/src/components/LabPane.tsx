import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api, runLab, type Lab, type ProviderInfo, type RunEvent } from "../lib/api";
import SplitPane from "./SplitPane";

// Monaco is heavy, so the editor is its own chunk and loads with the first lab.
const CodeEditor = lazy(() => import("./CodeEditor"));

type Check = { name: string; passed: boolean; detail: string };
type SaveState = "idle" | "dirty" | "saving" | "saved" | "failed";
type Tab = "console" | "checks" | "hints" | "solution";

type Props = {
  lab: Lab;
  onPassed: () => void;
};

export default function LabPane({ lab, onPassed }: Props) {
  const [code, setCode] = useState(lab.draft || lab.starter);
  const [logs, setLogs] = useState<string[]>([]);
  const [checks, setChecks] = useState<Check[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<"idle" | "running" | "passed" | "failed" | "error">(
    "idle",
  );
  const [message, setMessage] = useState("");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [creditWarning, setCreditWarning] = useState("");
  const [solution, setSolution] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>(lab.draft ? "saved" : "idle");
  const [tab, setTab] = useState<Tab>("console");

  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimer = useRef<number | null>(null);

  useEffect(() => {
    setCode(lab.draft || lab.starter);
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setStatus("idle");
    setMessage("");
    setSolution("");
    setTab("console");
    setSaveState(lab.draft ? "saved" : "idle");
  }, [lab.id, lab.draft, lab.starter]);

  useEffect(() => {
    api.providers().then((result) => {
      setProviders(result.providers);
      setProvider(result.default);
    });
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs]);

  useEffect(() => {
    if (status !== "running") return;
    const started = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      500,
    );
    return () => window.clearInterval(timer);
  }, [status]);

  // Autosave the draft a second after typing stops.
  const scheduleSave = useCallback(
    (next: string) => {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
      setSaveState("dirty");
      saveTimer.current = window.setTimeout(() => {
        setSaveState("saving");
        api
          .saveDraft(lab.id, next)
          .then(() => setSaveState("saved"))
          .catch(() => setSaveState("failed"));
      }, 1000);
    },
    [lab.id],
  );

  function handleChange(next: string) {
    setCode(next);
    scheduleSave(next);
  }

  function resetToStarter() {
    if (code === lab.starter) return;
    if (!window.confirm("Replace your code with the starter file? This overwrites your draft.")) {
      return;
    }
    handleChange(lab.starter);
  }

  function loadSolution() {
    if (!solution || code === solution) return;
    if (!window.confirm("Replace your code with the worked solution? This overwrites your draft.")) {
      return;
    }
    handleChange(solution);
  }

  async function run() {
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setMessage("");
    setCreditWarning("");
    setStatus("running");
    setElapsed(0);
    setTab("console");

    const controller = new AbortController();
    abortRef.current = controller;

    const handle = (event: RunEvent) => {
      switch (event.type) {
        case "start":
          setLogs((lines) => [...lines, `— running on ${event.provider} —`]);
          break;
        case "log":
          setLogs((lines) => [...lines, event.line]);
          break;
        case "result":
          setChecks(event.checks);
          setMetrics(event.metrics);
          break;
        case "done":
          setStatus(event.passed ? "passed" : "failed");
          setMessage(`Finished in ${event.seconds}s`);
          setTab("checks");
          if (event.passed) onPassed();
          break;
        case "out_of_credits":
          setStatus("error");
          setCreditWarning(event.hint);
          setMessage(event.message);
          setProvider("runpod");
          break;
        case "error":
          setStatus("error");
          setMessage(event.message);
          break;
      }
    };

    try {
      await runLab(lab.id, code, provider, handle, controller.signal);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setStatus("error");
        setMessage(String(error));
      }
    } finally {
      abortRef.current = null;
      setStatus((current) => (current === "running" ? "idle" : current));
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setStatus("idle");
    setMessage("Cancelled. The GPU job may still be finishing.");
  }

  async function openSolution() {
    setTab("solution");
    if (!solution) {
      const result = await api.solution(lab.id);
      setSolution(result.solution);
    }
  }

  const runnable = providers.find((p) => p.name === provider);
  const canRun = status !== "running" && Boolean(runnable?.available);
  const passedCount = checks.filter((c) => c.passed).length;

  const saveLabel: Record<SaveState, string> = {
    idle: "Autosaves as you type",
    dirty: "Unsaved edits",
    saving: "Saving…",
    saved: "Draft saved",
    failed: "Could not save the draft",
  };

  const tabs: { id: Tab; label: string; badge?: string }[] = [
    { id: "console", label: "Console" },
    { id: "checks", label: "Checks", badge: checks.length ? `${passedCount}/${checks.length}` : undefined },
    { id: "hints", label: "Hints", badge: lab.hints.length ? String(lab.hints.length) : undefined },
    { id: "solution", label: "Solution" },
  ];

  const editorPane = (
    <>
      <div className="flex shrink-0 items-center gap-3 border-b border-ink-800 bg-ink-950/40 px-3 py-1.5">
        <span className="flex items-center gap-1.5 rounded border border-ink-800 bg-ink-900 px-2 py-0.5 font-mono text-[11px] text-ink-300">
          <span className="h-1.5 w-1.5 rounded-full bg-flame-500/80" aria-hidden />
          solution.py
        </span>
        <span
          className={`truncate text-[11px] ${
            saveState === "failed"
              ? "text-rose-450"
              : saveState === "dirty"
                ? "text-ink-400"
                : "text-ink-600"
          }`}
          aria-live="polite"
        >
          {saveLabel[saveState]}
        </span>
        <div className="ml-auto flex items-center gap-3">
          <span className="hidden items-center gap-1 text-[11px] text-ink-600 xl:flex">
            <kbd className="rounded border border-ink-700 bg-ink-900 px-1.5 py-0.5 font-mono text-[10px] text-ink-400">
              ⌘⏎
            </kbd>
            run
          </span>
          <button
            onClick={resetToStarter}
            disabled={code === lab.starter}
            className="text-[11px] font-medium text-ink-400 transition hover:text-flame-300 disabled:cursor-default disabled:opacity-40 disabled:hover:text-ink-400"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center bg-ink-950 text-xs text-ink-600">
              Loading the editor…
            </div>
          }
        >
          <CodeEditor
            fill
            path={`${lab.id}/solution.py`}
            value={code}
            onChange={handleChange}
            onRun={() => {
              if (canRun) run();
            }}
          />
        </Suspense>
      </div>
    </>
  );

  const consolePane = (
    <>
      <div className="flex shrink-0 items-stretch gap-px border-y border-ink-800 bg-ink-900">
        {tabs.map((item) => (
          <button
            key={item.id}
            onClick={() => (item.id === "solution" ? openSolution() : setTab(item.id))}
            className={`flex items-center gap-1.5 border-b-2 px-3.5 py-2 text-[12px] font-medium transition ${
              tab === item.id
                ? "border-flame-500 text-ink-100"
                : "border-transparent text-ink-500 hover:text-ink-200"
            }`}
          >
            {item.label}
            {item.badge && (
              <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-ink-400">
                {item.badge}
              </span>
            )}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 px-3 text-[11px]">
          {status === "running" && (
            <span className="flex items-center gap-1.5 text-flame-400">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-flame-500" />
              {elapsed}s
            </span>
          )}
          {status === "passed" && <span className="text-mint-400">All checks passed</span>}
          {status === "failed" && <span className="text-rose-450">Checks failed</span>}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto bg-ink-950">
        {tab === "console" && (
          <div ref={logRef} className="h-full overflow-y-auto px-4 py-3 font-mono text-xs leading-relaxed">
            {logs.length === 0 && status !== "running" && (
              <p className="text-ink-600">
                Run the lab to stream the GPU output here.
              </p>
            )}
            {logs.map((line, index) => (
              <div
                key={index}
                className={
                  line.startsWith("[PASS]")
                    ? "text-mint-400"
                    : line.startsWith("[FAIL]")
                      ? "text-rose-450"
                      : line.startsWith("—")
                        ? "text-ink-600"
                        : "whitespace-pre-wrap text-ink-300"
                }
              >
                {line}
              </div>
            ))}
            {status === "running" && <div className="text-ink-600">▍ waiting on the GPU…</div>}
            {status === "error" && message && (
              <div className="mt-3 rounded-lg border border-rose-450/30 bg-rose-450/10 px-3 py-2 text-rose-450">
                {message}
              </div>
            )}
          </div>
        )}

        {tab === "checks" && (
          <div className="px-4 py-3.5">
            {checks.length === 0 ? (
              <p className="text-xs text-ink-600">No results yet. Run the lab.</p>
            ) : (
              <>
                <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
                  <span
                    className={
                      status === "passed"
                        ? "rounded-lg border border-mint-400/25 bg-mint-400/10 px-2.5 py-1 font-semibold text-mint-400"
                        : "rounded-lg border border-rose-450/25 bg-rose-450/10 px-2.5 py-1 font-semibold text-rose-450"
                    }
                  >
                    {passedCount}/{checks.length} checks passed
                  </span>
                  {message && <span className="text-ink-500">{message}</span>}
                </div>
                <ul className="space-y-2">
                  {checks.map((check) => (
                    <li key={check.name} className="flex gap-2.5 text-xs">
                      <span
                        className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] ${
                          check.passed
                            ? "bg-mint-400/15 text-mint-400"
                            : "bg-rose-450/15 text-rose-450"
                        }`}
                      >
                        {check.passed ? "✓" : "×"}
                      </span>
                      <span className="leading-4 text-ink-300">
                        {check.name}
                        {check.detail && <span className="text-ink-500"> — {check.detail}</span>}
                      </span>
                    </li>
                  ))}
                </ul>

                {Object.keys(metrics).length > 0 && (
                  <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-ink-800 bg-ink-800 sm:grid-cols-3">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="bg-ink-900 px-3 py-2.5">
                        <div className="truncate text-[10px] font-bold uppercase tracking-[0.12em] text-ink-600">
                          {key}
                        </div>
                        <div className="mt-0.5 truncate font-mono text-sm text-ink-100">
                          {String(value)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {tab === "hints" && (
          <div className="px-4 py-3.5">
            {lab.hints.length === 0 ? (
              <p className="text-xs text-ink-600">This lab ships without hints.</p>
            ) : (
              <ul className="space-y-2.5">
                {lab.hints.map((hint, index) => (
                  <li key={hint} className="flex gap-2.5 text-xs leading-5 text-ink-300">
                    <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded bg-flame-500/12 font-mono text-[10px] text-flame-400">
                      {index + 1}
                    </span>
                    {hint}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {tab === "solution" && (
          <div className="px-4 py-3.5">
            {!solution ? (
              <p className="text-xs text-ink-600">Loading the worked solution…</p>
            ) : (
              <>
                <button
                  onClick={loadSolution}
                  disabled={code === solution}
                  className="mb-3 rounded-lg border border-ink-700 bg-ink-900 px-3 py-1.5 text-[11px] font-semibold text-ink-300 transition hover:border-flame-500/60 hover:text-flame-300 disabled:cursor-default disabled:opacity-40"
                >
                  Load into the editor
                </button>
                <pre className="overflow-auto rounded-lg border border-ink-800 bg-ink-900 p-3.5 font-mono text-xs leading-relaxed text-ink-300">
                  {solution}
                </pre>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );

  return (
    <section className="flex h-full min-h-0 flex-col bg-ink-900/60">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-ink-800 bg-ink-900 px-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-flame-400">
              Lab
            </span>
            <span className="hidden rounded border border-ink-800 bg-ink-950/50 px-1.5 py-0.5 font-mono text-[10px] text-ink-400 sm:inline">
              {lab.gpu}
            </span>
          </div>
          <h2 className="truncate text-[13px] font-semibold tracking-tight text-ink-100" title={lab.brief}>
            {lab.title}
          </h2>
        </div>

        <select
          value={provider ?? ""}
          onChange={(event) => setProvider(event.target.value)}
          className="rounded-lg border border-ink-800 bg-ink-950/60 px-2 py-1.5 text-[11px] text-ink-200 outline-none focus:border-flame-500"
          aria-label="GPU provider"
        >
          {providers.map((item) => (
            <option key={item.name} value={item.name} disabled={!item.available}>
              {item.name}
              {item.available ? "" : " (not configured)"}
            </option>
          ))}
        </select>

        {status === "running" ? (
          <button
            onClick={cancel}
            className="rounded-lg border border-rose-450/45 bg-rose-450/10 px-3 py-1.5 text-[11px] font-bold text-rose-450 transition hover:bg-rose-450/20"
          >
            Stop · {elapsed}s
          </button>
        ) : (
          <button
            onClick={run}
            disabled={!canRun}
            className="rounded-lg bg-flame-500 px-3.5 py-1.5 text-[11px] font-bold text-ink-950 transition hover:bg-flame-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Run ▸
          </button>
        )}
      </header>

      {runnable && !runnable.available && (
        <p className="shrink-0 border-b border-ink-800 bg-ink-950/50 px-3 py-2 text-[11px] text-ink-400">
          {runnable.reason}
        </p>
      )}
      {creditWarning && (
        <p className="shrink-0 border-b border-ink-800 bg-flame-500/10 px-3 py-2 text-[11px] text-flame-300">
          {creditWarning}
        </p>
      )}

      <SplitPane
        direction="column"
        storageKey="li.lab.console"
        initial={62}
        min={20}
        max={88}
        className="flex-1"
        label="Resize the editor and the console"
        first={editorPane}
        second={consolePane}
      />
    </section>
  );
}
