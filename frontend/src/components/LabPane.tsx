import { lazy, Suspense, useCallback, useEffect, useId, useRef, useState } from "react";
import { api, runLab, type Lab, type ProviderInfo, type RunEvent } from "../lib/api";
import { useEditorLook } from "../lib/editorPrefs";
import { useTheme } from "../lib/theme";
import { useIsWide } from "../lib/useMediaQuery";
import { OutputHeader, useOutputCollapse } from "./CollapsibleOutput";
import EditorToolbar, { ToolbarPanel, type ToolbarAction } from "./EditorToolbar";
import { modKey } from "./shellIcons";
import SplitPane from "./SplitPane";
import { Button, Kbd, Notice, Tag } from "./ui";
import { CheckIcon, CloseIcon } from "./icons";

// Monaco is heavy, so the editor is its own chunk and loads with the first lab.
const CodeEditor = lazy(() => import("./CodeEditor"));

/** "wall_seconds" reads as "Wall seconds". */
function metricLabel(key: string): string {
  const words = key.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

type Check = { name: string; passed: boolean; detail: string };
type SaveState = "idle" | "dirty" | "saving" | "saved" | "failed";
type Tab = "console" | "checks";
type Status = "idle" | "running" | "passed" | "failed" | "error";

type Props = {
  lab: Lab;
  onPassed: () => void;
  /** Folds the lab away so the chapter takes the whole width. */
  onCollapse?: () => void;
  /** The shortcut that folds it, shown in the collapse button's tooltip. */
  collapseShortcut?: string;
};

export default function LabPane({ lab, onPassed, onCollapse, collapseShortcut }: Props) {
  const [code, setCode] = useState(lab.draft || lab.starter);
  const [logs, setLogs] = useState<string[]>([]);
  const [checks, setChecks] = useState<Check[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [provider, setProvider] = useState<string | null>(null);
  const [creditWarning, setCreditWarning] = useState("");
  const [solution, setSolution] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>(lab.draft ? "saved" : "idle");
  const [tab, setTab] = useState<Tab>("console");
  // Hints and the solution are opened on purpose, one step at a time, so a
  // learner takes only as much help as they need. Either one shows in a
  // panel under the toolbar, never both.
  const [revealed, setRevealed] = useState(0);
  const [panel, setPanel] = useState<"hints" | "solution" | null>(null);
  // Counts runs, so the output panel opens when one starts.
  const [runId, setRunId] = useState(0);
  const labId = useRef(lab.id);
  labId.current = lab.id;

  const logRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimer = useRef<number | null>(null);
  const wide = useIsWide();
  const look = useEditorLook(useTheme());
  const output = useOutputCollapse(runId);
  const outputId = useId();
  const panelId = useId();
  const mod = modKey();

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
    setRevealed(0);
    setPanel(null);
  }, [lab.id]);

  useEffect(() => {
    api.providers().then((result) => {
      setProviders(result.providers);
      // The configured default, unless it is not set up on this deployment, in
      // which case pick something that can actually run.
      const usable = result.providers.filter((item) => item.available);
      const preset = usable.find((item) => item.name === result.default);
      setProvider((preset ?? usable[0])?.name ?? result.default);
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

  const runnable = providers.find((p) => p.name === provider);
  const canRun = status !== "running" && Boolean(runnable?.available);

  async function run() {
    if (!canRun) return;
    setLogs([]);
    setChecks([]);
    setMetrics({});
    setMessage("");
    setCreditWarning("");
    setStatus("running");
    setElapsed(0);
    setTab("console");
    setRunId((count) => count + 1);

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

  // The solution panel only explains itself at first; the code is fetched and
  // shown once the learner asks for it.
  async function fetchSolution() {
    if (solution) return;
    const id = lab.id;
    const result = await api.solution(id);
    // The reader may have moved to another chapter while this loaded.
    if (labId.current === id) setSolution(result.solution);
  }

  const passedCount = checks.filter((c) => c.passed).length;

  const saveLabel: Record<SaveState, string> = {
    idle: "",
    dirty: "Unsaved",
    saving: "Saving…",
    saved: "Saved",
    failed: "Not saved",
  };

  const outcome =
    status === "passed"
      ? `${passedCount}/${checks.length} checks passed`
      : status === "failed"
        ? checks.length
          ? `${passedCount}/${checks.length} checks passed`
          : "Checks failed"
        : status === "error"
          ? "Error"
          : message
            ? "Stopped"
            : undefined;

  const actions: ToolbarAction[] = [
    ...(lab.hints.length > 0
      ? [
          {
            id: "hints",
            label: panel === "hints" ? "Hide hints" : "Show hints",
            icon: "bulb" as const,
            pressed: panel === "hints",
            controls: panelId,
            onClick: () => setPanel((open) => (open === "hints" ? null : "hints")),
          },
        ]
      : []),
    {
      id: "reset",
      label: "Reset to the starter file",
      icon: "reset",
      priority: "low",
      disabled: code === lab.starter,
      onClick: resetToStarter,
    },
    {
      id: "solution",
      label: panel === "solution" ? "Hide the solution" : "Show the solution",
      icon: "key",
      priority: "low",
      pressed: panel === "solution",
      controls: panelId,
      onClick: () => setPanel((open) => (open === "solution" ? null : "solution")),
    },
  ];

  const header = (collapsed: boolean) => (
    <OutputHeader
      label="Output"
      detail={
        status === "running"
          ? `${provider ?? ""} · ${elapsed}s`
          : outcome && message && status !== "error"
            ? message
            : undefined
      }
      running={status === "running"}
      status={
        outcome && (
          <span className={status === "passed" ? "text-ok" : status === "idle" ? "" : "text-bad"}>
            {outcome}
          </span>
        )
      }
      collapsed={collapsed}
      onToggle={output.toggle}
      controls={outputId}
    />
  );

  const editorPane = (
    <div className="min-h-0 flex-1 bg-code">
      <Suspense
        fallback={
          <div className="flex h-full items-center justify-center bg-code text-[12.5px] text-fg-subtle">
            Loading the editor…
          </div>
        }
      >
        <CodeEditor fill path={`${lab.id}/solution.py`} value={code} onChange={handleChange} onRun={run} />
      </Suspense>
    </div>
  );

  const tabs: { id: Tab; label: string; badge?: string }[] = [
    { id: "console", label: "Console" },
    { id: "checks", label: "Checks", badge: checks.length ? `${passedCount}/${checks.length}` : undefined },
  ];

  const outputPane = (
    <>
      {header(false)}
      <div id={outputId} className="flex min-h-0 flex-1 flex-col">
        <div className="flex shrink-0 items-stretch border-b border-line bg-paper px-1" role="tablist" aria-label="Lab output">
          {tabs.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={tab === item.id}
              onClick={() => setTab(item.id)}
              className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-2.5 py-1.5 font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] transition-colors ${
                tab === item.id ? "border-accent text-fg" : "border-transparent text-fg-subtle hover:text-fg"
              }`}
            >
              {item.label}
              {item.badge && (
                <span className="rounded bg-tint px-1 font-mono text-[10px] normal-case leading-4 tabular-nums tracking-normal text-fg-muted">
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto bg-paper" data-testid="lab-output">
          {tab === "console" && (
            <div ref={logRef} className="h-full overflow-y-auto px-4 py-3 font-mono text-[12px] leading-[1.75]">
              {logs.length === 0 && status !== "running" && (
                <div className="max-w-md font-sans text-[13px] leading-5 text-fg-subtle">
                  <p>
                    Run the lab to send your code to a GPU. Output streams here as it runs, and the
                    checks open when it finishes.
                  </p>
                  <p className="mt-2 flex items-center gap-1.5 text-[12px] text-fg-faint">
                    <Kbd>{mod} ↵</Kbd> runs it from the editor.
                  </p>
                </div>
              )}
              {logs.map((line, index) => (
                <div
                  key={index}
                  className={
                    line.startsWith("[PASS]")
                      ? "text-ok"
                      : line.startsWith("[FAIL]")
                        ? "text-bad"
                        : line.startsWith("—")
                          ? "text-fg-subtle"
                          : "whitespace-pre-wrap text-fg-muted"
                  }
                >
                  {line}
                </div>
              ))}
              {status === "running" && <div className="text-fg-subtle">▍ waiting on the GPU…</div>}
              {status === "error" && message && (
                <Notice tone="bad" className="mt-3 font-sans">
                  {message}
                </Notice>
              )}
            </div>
          )}

          {tab === "checks" && (
            <div className="px-4 py-3.5">
              {checks.length === 0 ? (
                <p className="text-[13px] text-fg-subtle">No results yet. Run the lab.</p>
              ) : (
                <>
                  <div className="mb-3 flex flex-wrap items-center gap-2">
                    <Tag tone={status === "passed" ? "ok" : "bad"} className="tabular-nums">
                      {passedCount}/{checks.length} checks passed
                    </Tag>
                    {message && <span className="text-[12px] tabular-nums text-fg-subtle">{message}</span>}
                  </div>
                  <ul className="space-y-2">
                    {checks.map((check) => (
                      <li key={check.name} className="flex gap-2.5 text-[13px] leading-5">
                        <span
                          className={`mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full ${
                            check.passed ? "bg-ok/12 text-ok" : "bg-bad/12 text-bad"
                          }`}
                        >
                          {check.passed ? <CheckIcon className="size-3" /> : <CloseIcon className="size-3" />}
                          <span className="sr-only">{check.passed ? "Passed:" : "Failed:"}</span>
                        </span>
                        <span className="min-w-0 text-fg">
                          {check.name}
                          {check.detail && <span className="text-fg-subtle"> — {check.detail}</span>}
                        </span>
                      </li>
                    ))}
                  </ul>

                  {Object.keys(metrics).length > 0 && (
                    <dl className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-3">
                      {Object.entries(metrics).map(([key, value]) => (
                        <div key={key} className="min-w-0 bg-card px-3 py-2">
                          <dt className="truncate text-[12px] text-fg-subtle" title={key}>
                            {metricLabel(key)}
                          </dt>
                          <dd className="mt-0.5 truncate font-mono text-[13px] tabular-nums text-fg">
                            {String(value)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );

  return (
    <section
      className={`slab flex h-full min-h-0 flex-col ${look === "paper" ? "code-light sh-slab-paper" : ""}`}
      aria-label={`Lab: ${lab.title}`}
    >
      <EditorToolbar
        title={lab.title}
        info={
          <>
            <p>{lab.brief}</p>
            <p className="mt-2 text-[12px] text-fg-subtle">
              Runs on {lab.gpu}, with a {lab.timeout}s limit.
            </p>
          </>
        }
        meta={lab.gpu}
        status={saveLabel[saveState]}
        running={status === "running"}
        onRun={run}
        onStop={cancel}
        runLabel="Run"
        actions={actions}
        onCollapse={onCollapse}
        collapseShortcut={collapseShortcut}
      >
        {/* With one provider there is nothing to pick, so the select stays
            hidden. A narrow screen marks an unusable provider instead of
            spelling out why; the reason is a line of its own under the bar. */}
        {providers.length > 1 && (
          <select
            value={provider ?? ""}
            onChange={(event) => setProvider(event.target.value)}
            className="h-7 min-w-0 max-w-28 shrink rounded-[5px] border border-line bg-paper px-1.5 text-[12px] text-fg-muted outline-none transition-colors hover:border-line-strong hover:text-fg focus-visible:border-accent"
            aria-label="GPU provider"
            title="GPU provider"
          >
            {providers.map((item) => (
              <option key={item.name} value={item.name} disabled={!item.available}>
                {item.name}
                {item.available ? "" : wide ? " (not configured)" : " ⚠"}
              </option>
            ))}
          </select>
        )}
      </EditorToolbar>

      {runnable && !runnable.available && (
        <p className="shrink-0 border-b border-line px-3 py-1.5 text-[12px] text-fg-muted">{runnable.reason}</p>
      )}
      {creditWarning && (
        <p className="shrink-0 border-b border-line bg-warn/8 px-3 py-1.5 text-[12px] text-warn">{creditWarning}</p>
      )}

      {panel === "hints" && (
        <ToolbarPanel id={panelId} title="Hints" onClose={() => setPanel(null)}>
          {revealed === 0 && (
            <p className="mb-2.5 max-w-md text-[13px] leading-5 text-fg-subtle">
              Each hint is a nudge, not the answer. Open one, try again, and open the next only if
              you're still stuck.
            </p>
          )}
          {revealed > 0 && (
            <ol className="mb-2.5 space-y-2">
              {lab.hints.slice(0, revealed).map((hint, index) => (
                <li key={hint} className="flex gap-2.5 text-[13px] leading-5 text-fg-muted">
                  <span className="w-12 shrink-0 font-mono text-[10px] font-medium uppercase leading-5 tracking-[0.08em] text-fg-subtle">
                    Hint {index + 1}
                  </span>
                  <span className="min-w-0">{hint}</span>
                </li>
              ))}
            </ol>
          )}
          {revealed < lab.hints.length ? (
            <button
              type="button"
              onClick={() => setRevealed((count) => count + 1)}
              className="text-[12.5px] font-medium text-accent hover:underline"
            >
              Show hint {revealed + 1} of {lab.hints.length}
            </button>
          ) : (
            <p className="text-[12px] text-fg-faint">That's every hint for this lab.</p>
          )}
        </ToolbarPanel>
      )}

      {panel === "solution" && (
        <ToolbarPanel
          id={panelId}
          title="Worked solution"
          onClose={() => setPanel(null)}
          footer={
            solution ? (
              <Button size="sm" onClick={loadSolution} disabled={code === solution}>
                Load into the editor
              </Button>
            ) : undefined
          }
        >
          {!solution ? (
            <>
              <p className="mb-2.5 max-w-md text-[13px] leading-5 text-fg-subtle">
                The worked solution passes every check. Looking at it before you've tried the lab
                skips the part that teaches.
              </p>
              <Button size="sm" onClick={() => void fetchSolution()}>
                Show the solution
              </Button>
            </>
          ) : (
            <pre className="overflow-auto rounded-md border border-line bg-card p-3.5 font-mono text-[12px] leading-relaxed text-fg">
              {solution}
            </pre>
          )}
        </ToolbarPanel>
      )}

      <SplitPane
        direction="column"
        storageKey="li.lab.console"
        initial={62}
        min={20}
        max={88}
        className="min-h-0 flex-1"
        label="Resize the editor and the output"
        collapsed={output.collapsed}
        rail={header(true)}
        first={editorPane}
        second={outputPane}
      />
    </section>
  );
}
