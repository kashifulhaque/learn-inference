import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, type ChapterMeta } from "./lib/api";
import ComputeBadge from "./components/ComputeBadge";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import ChapterPage from "./pages/ChapterPage";
import Compute from "./pages/Compute";
import Dashboard from "./pages/Dashboard";

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [chapters, setChapters] = useState<ChapterMeta[]>([]);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [model, setModel] = useState("");
  const [gpu, setGpu] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

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
        onNavigate={() => setMenuOpen(false)}
      />

      {menuOpen && (
        <button
          aria-label="Close the menu"
          onClick={() => setMenuOpen(false)}
          className="fixed inset-0 z-20 bg-ink-950/75 backdrop-blur-sm lg:hidden"
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-ink-800/80 bg-ink-900/65 px-5 backdrop-blur-xl lg:px-8">
          <div className="flex items-center gap-3 lg:hidden">
            <button
              onClick={() => setMenuOpen((open) => !open)}
              className="flex h-9 items-center gap-2 rounded-lg border border-ink-700 bg-ink-850 px-3 text-xs font-semibold text-ink-200 transition hover:border-ink-600"
            >
              <span className="text-base leading-none">☰</span>
              Course
            </button>
          </div>
          <div className="hidden items-center gap-2 text-xs text-ink-500 lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-flame-500" />
            Learning workspace
          </div>
          <div className="flex items-center gap-3">
            <ComputeBadge />
            <span className="hidden text-xs text-ink-500 sm:inline">Signed in as</span>
            <span className="rounded-lg border border-ink-700/70 bg-ink-850 px-2.5 py-1.5 text-xs font-medium text-ink-200">
              {name}
            </span>
            <button
              onClick={signOut}
              className="text-xs font-medium text-ink-500 transition hover:text-flame-400"
            >
              Sign out
            </button>
          </div>
        </header>

        <main className="min-w-0 flex-1 overflow-y-auto">
          <Routes>
            <Route
              path="/"
              element={
                <Dashboard
                  chapters={chapters}
                  progress={progress}
                  model={model}
                  gpu={gpu}
                />
              }
            />
            <Route
              path="/c/:slug"
              element={
                <ChapterPage progress={progress} onProgress={changeProgress} />
              }
            />
            <Route path="/compute" element={<Compute />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
