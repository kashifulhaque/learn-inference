import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, type ChapterMeta } from "./lib/api";
import ComputeBadge from "./components/ComputeBadge";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import ChapterPage from "./pages/ChapterPage";
import Compute from "./pages/Compute";
import Dashboard from "./pages/Dashboard";

const COLLAPSED_KEY = "li.sidebar.collapsed";

function wasCollapsed(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

/** Pages that flow down the screen scroll inside the shell, not with it. */
function Scroller({ children }: { children: ReactNode }) {
  return <div className="h-full overflow-y-auto">{children}</div>;
}

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [chapters, setChapters] = useState<ChapterMeta[]>([]);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [model, setModel] = useState("");
  const [gpu, setGpu] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(wasCollapsed);

  useEffect(() => {
    api
      .me()
      .then((me) => {
        setName(me.name);
        setModel(me.model);
        setGpu(me.gpu);
      })
      .finally(() => setReady(true));
  }, []);

  useEffect(() => {
    if (!name) return;
    api.chapters().then((result) => {
      setChapters(result.chapters);
      setProgress(result.progress);
    });
  }, [name]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // The sidebar still collapses, it just forgets between visits.
    }
  }, [collapsed]);

  // Cmd/Ctrl+B gives the chapter and its lab the whole screen, the way an
  // editor hides its file tree.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "b") {
        event.preventDefault();
        setCollapsed((current) => !current);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const changeProgress = useCallback(
    (slug: string, status: "in_progress" | "done") => {
      setProgress((current) => ({ ...current, [slug]: status }));
      api.setProgress(slug, status).catch(() => undefined);
    },
    [],
  );

  async function signOut() {
    await api.logout();
    setName(null);
  }

  if (!ready) {
    return <div className="p-10 text-sm text-ink-500">Loading…</div>;
  }
  if (!name) {
    return <Login onSignedIn={setName} />;
  }

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        chapters={chapters}
        progress={progress}
        open={menuOpen}
        collapsed={collapsed}
        onNavigate={() => setMenuOpen(false)}
        onToggleCollapsed={() => setCollapsed((current) => !current)}
      />

      {menuOpen && (
        <button
          aria-label="Close the menu"
          onClick={() => setMenuOpen(false)}
          className="fixed inset-0 z-20 bg-ink-950/75 backdrop-blur-sm lg:hidden"
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-ink-800 bg-ink-900/70 px-3 backdrop-blur-xl lg:px-4">
          <button
            onClick={() => setMenuOpen((open) => !open)}
            aria-label="Open the course list"
            className="flex h-8 items-center gap-2 rounded-lg border border-ink-800 bg-ink-850 px-2.5 text-xs font-semibold text-ink-200 transition hover:border-ink-600 lg:hidden"
          >
            <span className="text-sm leading-none">☰</span>
            Chapters
          </button>

          <div className="hidden items-center gap-2 text-[11px] text-ink-500 lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-flame-500" />
            Learning workspace
            <kbd className="ml-1 rounded border border-ink-800 bg-ink-950/60 px-1.5 py-0.5 font-mono text-[10px] text-ink-600">
              ⌘B
            </kbd>
          </div>

          <div className="ml-auto flex items-center gap-2.5">
            <ComputeBadge />
            <span className="rounded-lg border border-ink-800 bg-ink-850 px-2.5 py-1.5 text-[11px] font-medium text-ink-200">
              {name}
            </span>
            <button
              onClick={signOut}
              className="text-[11px] font-medium text-ink-500 transition hover:text-flame-400"
            >
              Sign out
            </button>
          </div>
        </header>

        <main className="min-h-0 min-w-0 flex-1 overflow-hidden">
          <Routes>
            <Route
              path="/"
              element={
                <Scroller>
                  <Dashboard
                    chapters={chapters}
                    progress={progress}
                    model={model}
                    gpu={gpu}
                  />
                </Scroller>
              }
            />
            <Route
              path="/c/:slug"
              element={
                <ChapterPage progress={progress} onProgress={changeProgress} />
              }
            />
            <Route
              path="/compute"
              element={
                <Scroller>
                  <Compute />
                </Scroller>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
