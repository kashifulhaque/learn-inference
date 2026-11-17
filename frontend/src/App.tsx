import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api, type ChapterMeta } from "./lib/api";
import { chapterNumbers, groupByPart, splitPart } from "./lib/chapters";
import { doneCount, partNumber, useSearchEntries } from "./lib/shellData";
import { setThemeChoice, useThemeChoice } from "./lib/theme";
import AppSidebar, { type SidebarSection, type SidebarUtility } from "./components/AppSidebar";
import { Mark } from "./components/Brand";
import { computeLabel, computeTitle, useActiveCompute } from "./components/ComputeBadge";
import { useBook } from "./components/DownloadBook";
import SearchDialog from "./components/SearchDialog";
import StatusMark from "./components/StatusMark";
import { Loading } from "./components/ui";
import AgentsPage from "./pages/AgentsPage";
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
  return <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>;
}

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [chapters, setChapters] = useState<ChapterMeta[] | null>(null);
  const [progress, setProgress] = useState<Record<string, string>>({});
  const [model, setModel] = useState("");
  const [gpu, setGpu] = useState("");
  // menuOpen drives the drawer on a narrow screen; collapsed folds the
  // sidebar to its icon rail on a wide one.
  const [menuOpen, setMenuOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(wasCollapsed);
  const [searchOpen, setSearchOpen] = useState(false);
  const location = useLocation();
  const themeChoice = useThemeChoice();
  const active = useActiveCompute(Boolean(name));
  const book = useBook(Boolean(name));

  useEffect(() => {
    const redirectToSso = () => {
      const returnTo = encodeURIComponent(window.location.href);
      window.location.assign(`https://auth.ifkash.dev/?rd=${returnTo}`);
    };
    api
      .me()
      .then((me) => {
        if (!me.name) {
          redirectToSso();
          return;
        }
        setName(me.name);
        setModel(me.model);
        setGpu(me.gpu);
        setReady(true);
      })
      .catch(redirectToSso);
  }, []);

  useEffect(() => {
    if (!name) return;
    api.chapters().then((result) => {
      setChapters(result.chapters);
      setProgress(result.progress);
    });
  }, [name]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // The sidebar still collapses, it just forgets between visits.
    }
  }, [collapsed]);

  // Cmd or Ctrl with B folds the sidebar to its rail on a wide screen and
  // opens or closes the drawer on a narrow one.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key !== "b" && event.key !== "B") return;
      event.preventDefault();
      if (window.matchMedia("(min-width: 1024px)").matches) setCollapsed((v) => !v);
      else setMenuOpen((v) => !v);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // "/" opens the search when the keyboard is not busy in a field or the
  // editor. Cmd or Ctrl with K opens it from anywhere.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
          !!target.closest(".monaco-editor"));
      const slash = event.key === "/" && !event.metaKey && !event.ctrlKey && !event.altKey && !typing;
      const commandK =
        (event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k";
      if (!slash && !commandK) return;
      event.preventDefault();
      setMenuOpen(false);
      setSearchOpen(true);
    }
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

  const list = useMemo(() => chapters ?? [], [chapters]);
  const entries = useSearchEntries(list, searchOpen);
  const currentSlug = location.pathname.startsWith("/c/")
    ? decodeURIComponent(location.pathname.slice("/c/".length))
    : null;

  const sections = useMemo<SidebarSection[]>(() => {
    const numbers = chapterNumbers(list);
    return groupByPart(list).map(([part, items], index) => ({
      id: part,
      number: partNumber(part, index),
      title: splitPart(part).name,
      lessons: items.map((chapter) => ({
        id: chapter.slug,
        to: `/c/${chapter.slug}`,
        title: chapter.title,
        number: numbers.get(chapter.slug),
        done: progress[chapter.slug] === "done",
        current: chapter.slug === currentSlug,
        mark: <StatusMark state={progress[chapter.slug]} />,
      })),
    }));
  }, [list, progress, currentSlug]);

  const utilities = useMemo<SidebarUtility[]>(
    () =>
      book
        ? [
            {
              id: "book",
              label: "Download book",
              detail: `PDF · ${book.pages} pages · ${(book.bytes / 1e6).toFixed(1)} MB`,
              icon: "download",
              href: book.url,
              download: true,
            },
          ]
        : [],
    [book],
  );

  if (!ready || !name) {
    return <Loading />;
  }

  const done = doneCount(list, progress);

  return (
    <div className="sh-app">
      <AppSidebar
        brand={{ name: "learn/inference", kicker: "Field notes", mark: <Mark className="size-[30px]" /> }}
        searchLabel="Find a chapter"
        onSearch={() => {
          setMenuOpen(false);
          setSearchOpen(true);
        }}
        nav={[
          { to: "/", label: "Course overview", icon: "grid", active: location.pathname === "/" },
          {
            to: "/compute",
            label: "Compute",
            icon: "chip",
            count: active ? active : undefined,
            active: location.pathname === "/compute",
          },
          { to: "/agents", label: "Agent access", icon: "key", active: location.pathname === "/agents" },
        ]}
        sections={sections}
        storageKey="inference.sidebar"
        progress={{
          done,
          total: list.length,
          label: "Chapters done",
          caption: done === list.length && list.length > 0 ? "Course complete." : `Saved for ${name}`,
        }}
        utilities={utilities}
        status={{
          label: `GPU ${computeLabel(active).toLowerCase()}`,
          title: computeTitle(active),
          tone: active === null ? "neutral" : active > 0 ? "alert" : "ok",
        }}
        theme={{ mode: themeChoice, onChange: setThemeChoice }}
        account={{ username: name, signOutHref: "https://auth.ifkash.dev/logout" }}
        open={menuOpen}
        onOpen={() => setMenuOpen(true)}
        onClose={() => setMenuOpen(false)}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((current) => !current)}
      />

      <main className="sh-main" id="main">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <Routes>
            <Route
              path="/"
              element={<Dashboard chapters={chapters} progress={progress} model={model} gpu={gpu} />}
            />
            <Route
              path="/c/:slug"
              element={
                <div className="min-h-0 flex-1">
                  <ChapterPage chapters={list} progress={progress} onProgress={changeProgress} />
                </div>
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
            <Route
              path="/agents"
              element={
                <Scroller>
                  <AgentsPage />
                </Scroller>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>

      <SearchDialog
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        entries={entries}
        noun="chapter"
        placeholder="Try a topic, such as “paged cache”, “RoPE”, or “batching”"
      />
    </div>
  );
}
