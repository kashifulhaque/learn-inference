import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";

const POLL_MS = 60_000;

/**
 * Header indicator for anything running on a GPU provider.
 *
 * The point is that a container nobody remembers starting is visible from every
 * page, not only from the compute panel.
 */
export default function ComputeBadge() {
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    let live = true;

    const read = () => {
      api
        .infra()
        .then((result) => {
          if (live) setActive(result.active);
        })
        .catch(() => {
          if (live) setActive(null);
        });
    };

    // The first read happens even in a background tab, so the count is right
    // the moment the page is looked at.
    read();
    const onVisible = () => {
      if (!document.hidden) read();
    };
    document.addEventListener("visibilitychange", onVisible);
    const timer = window.setInterval(() => {
      if (!document.hidden) read();
    }, POLL_MS);

    return () => {
      live = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const running = (active ?? 0) > 0;

  return (
    <Link
      to="/compute"
      title={
        running
          ? "Something is running on a GPU provider"
          : "Nothing is running on a GPU provider"
      }
      className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
        running
          ? "border-rose-450/45 bg-rose-450/10 text-rose-450 hover:bg-rose-450/15"
          : "border-ink-700/70 bg-ink-850 text-ink-400 hover:border-ink-600 hover:text-ink-200"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          active === null
            ? "bg-ink-600"
            : running
              ? "animate-pulse bg-rose-450"
              : "bg-flame-500"
        }`}
      />
      {active === null ? "GPU ?" : running ? `${active} running` : "GPU idle"}
    </Link>
  );
}
