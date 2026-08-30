import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, type ChapterMeta } from "./lib/api";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import ChapterPage from "./pages/ChapterPage";
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
    <div className="flex h-full">
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
          className="fixed inset-0 z-20 bg-ink-950/70 lg:hidden"
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-800 bg-ink-900/80 px-5 py-3 backdrop-blur lg:justify-end">
          <button
            onClick={() => setMenuOpen((open) => !open)}
            className="rounded-lg border border-ink-700 px-3 py-1.5 text-xs text-ink-300 lg:hidden"
          >
            Chapters
          </button>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-ink-400">{name}</span>
            <button
              onClick={signOut}
              className="text-ink-500 underline underline-offset-4 transition hover:text-ink-200"
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
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
