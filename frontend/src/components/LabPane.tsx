import { useCallback, useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { api, runLab, type Lab, type ProviderInfo, type RunEvent } from "../lib/api";

type Check = { name: string; passed: boolean; detail: string };

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
  const [showSolution, setShowSolution] = useState(false);
  const [solution, setSolution] = useState("");
  const [elapsed, setElapsed] = useState(0);

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
    setShowSolution(false);
    setSolution("");
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
      saveTimer.current = window.setTimeout(() => {
        api.saveDraft(lab.id, next).catch(() => undefined);
      }, 1000);
    },
    [lab.id],
  );

  function handleChange(next: string | undefined) {
    const value = next ?? "";
    setCode(value);
    scheduleSave(value);
  }

  async function run() {
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setMessage("");
    setCreditWarning("");
    setStatus("running");
    setElapsed(0);

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

  async function revealSolution() {
    if (!solution) {
      const result = await api.solution(lab.id);
      setSolution(result.solution);
    }
    setShowSolution((open) => !open);
  }

  const runnable = providers.find((p) => p.name === provider);

  return (
    <section className="rounded-2xl border border-ink-800 bg-ink-900">
      <header className="flex flex-wrap items-center gap-3 border-b border-ink-800 px-5 py-3.5">
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-sm font-semibold text-ink-100">
            Lab · {lab.title}
          </h2>
          <p className="mt-0.5 text-xs text-ink-400">{lab.brief}</p>
        </div>

        <select
          value={provider ?? ""}
          onChange={(event) => setProvider(event.target.value)}
          className="rounded-lg border border-ink-700 bg-ink-850 px-2.5 py-1.5 text-xs text-ink-200 outline-none focus:border-flame-500"
          aria-label="GPU provider"
        >
          {providers.map((item) => (
            <option key={item.name} value={item.name} disabled={!item.available}>
              {item.name}
              {item.available ? "" : " (not configured)"}
            </option>
          ))}
        </select>

        <span className="rounded-md border border-ink-700 px-2 py-1 font-mono text-[11px] text-ink-400">
          {lab.gpu}
        </span>

        {status === "running" ? (
          <button
            onClick={cancel}
            className="rounded-lg border border-ink-700 px-3.5 py-1.5 text-xs font-semibold text-ink-200 transition hover:border-rose-450 hover:text-rose-450"
          >
            Stop · {elapsed}s
          </button>
        ) : (
          <button
            onClick={run}
            disabled={!runnable?.available}
            className="rounded-lg bg-flame-500 px-4 py-1.5 text-xs font-semibold text-ink-950 transition hover:bg-flame-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Run
          </button>
        )}
      </header>

      {runnable && !runnable.available && (
        <p className="border-b border-ink-800 bg-ink-850 px-5 py-2 text-xs text-ink-400">
          {runnable.reason}
        </p>
      )}

      {creditWarning && (
        <p className="border-b border-ink-800 bg-flame-500/10 px-5 py-2.5 text-xs text-flame-300">
          {creditWarning}
        </p>
      )}

      <div className="border-b border-ink-800">
        <Editor
          height="420px"
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={handleChange}
          options={{
            fontSize: 13,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            padding: { top: 14, bottom: 14 },
            fontFamily: 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace',
            renderLineHighlight: "line",
            tabSize: 4,
            rulers: [80],
          }}
        />
      </div>

      {lab.hints.length > 0 && (
        <details className="border-b border-ink-800 px-5 py-3">
          <summary className="cursor-pointer text-xs font-medium text-ink-300 hover:text-ink-100">
            Hints ({lab.hints.length})
          </summary>
          <ul className="mt-2.5 space-y-1.5 pl-4 text-xs text-ink-400">
            {lab.hints.map((hint) => (
              <li key={hint} className="list-disc">
                {hint}
              </li>
            ))}
          </ul>
        </details>
      )}

      {(logs.length > 0 || status === "running" || message) && (
        <div className="border-b border-ink-800">
          <div
            ref={logRef}
            className="max-h-80 overflow-y-auto bg-ink-950 px-5 py-3 font-mono text-xs leading-relaxed"
          >
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
            {status === "running" && (
              <div className="text-ink-600">▍ waiting on the GPU…</div>
            )}
          </div>
        </div>
      )}

      {checks.length > 0 && (
        <div className="border-b border-ink-800 px-5 py-4">
          <div className="mb-2.5 flex items-center gap-2 text-xs font-semibold">
            <span
              className={
                status === "passed"
                  ? "rounded-md bg-mint-400/15 px-2 py-0.5 text-mint-400"
                  : "rounded-md bg-rose-450/15 px-2 py-0.5 text-rose-450"
              }
            >
              {checks.filter((c) => c.passed).length}/{checks.length} checks passed
            </span>
            {message && <span className="text-ink-500">{message}</span>}
          </div>
          <ul className="space-y-1">
            {checks.map((check) => (
              <li key={check.name} className="flex gap-2 text-xs">
                <span className={check.passed ? "text-mint-400" : "text-rose-450"}>
                  {check.passed ? "✓" : "✕"}
                </span>
                <span className="text-ink-300">
                  {check.name}
                  {check.detail && (
                    <span className="text-ink-500"> — {check.detail}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {Object.keys(metrics).length > 0 && (
        <div className="grid grid-cols-2 gap-px border-b border-ink-800 bg-ink-800 sm:grid-cols-4">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="bg-ink-900 px-4 py-3">
              <div className="truncate text-[11px] text-ink-500">{key}</div>
              <div className="truncate font-mono text-sm text-ink-100">
                {String(value)}
              </div>
            </div>
          ))}
        </div>
      )}

      {status === "error" && message && !creditWarning && (
        <p className="border-b border-ink-800 bg-rose-450/10 px-5 py-2.5 font-mono text-xs text-rose-450">
          {message}
        </p>
      )}

      <div className="px-5 py-3">
        <button
          onClick={revealSolution}
          className="text-xs text-ink-500 underline underline-offset-4 transition hover:text-ink-200"
        >
          {showSolution ? "Hide the worked solution" : "Show the worked solution"}
        </button>
        {showSolution && (
          <pre className="mt-3 max-h-96 overflow-auto rounded-lg border border-ink-800 bg-ink-950 p-4 font-mono text-xs leading-relaxed text-ink-300">
            {solution}
          </pre>
        )}
      </div>
    </section>
  );
}
