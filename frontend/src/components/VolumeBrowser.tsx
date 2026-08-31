import { useEffect, useState } from "react";
import { api, type VolumeListing } from "../lib/api";
import { formatBytes } from "../lib/format";

type Props = {
  provider: string;
  volume: string;
};

/**
 * Lists one directory of a provider volume at a time. The weight cache is tens
 * of gigabytes across nested Hugging Face directories, so this walks it rather
 * than fetching the tree.
 */
export default function VolumeBrowser({ provider, volume }: Props) {
  const [path, setPath] = useState("/");
  const [listing, setListing] = useState<VolumeListing | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    api
      .volume(provider, volume, path)
      .then((result) => {
        if (!live) return;
        setListing(result);
        setError("");
      })
      .catch((problem) => {
        if (live) setError(String((problem as Error).message ?? problem));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [provider, volume, path]);

  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-ink-800 bg-ink-950">
      <div className="flex items-center gap-3 border-b border-ink-800 bg-ink-900/60 px-3 py-2">
        <button
          onClick={() => setPath(listing?.parent ?? "/")}
          disabled={!listing?.parent}
          className="rounded-md border border-ink-700 px-2 py-0.5 text-[11px] text-ink-300 transition hover:border-flame-500/50 disabled:opacity-30"
        >
          ↑ up
        </button>
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-400">
          {volume}:{listing?.path ?? path}
        </span>
        {listing && listing.bytes > 0 && (
          <span className="text-[11px] text-ink-600">
            {formatBytes(listing.bytes)} here
          </span>
        )}
      </div>

      {loading && <p className="px-4 py-4 text-[11px] text-ink-600">Listing…</p>}
      {error && (
        <p className="px-4 py-4 font-mono text-[11px] leading-5 text-rose-450">{error}</p>
      )}
      {!loading && !error && listing?.entries.length === 0 && (
        <p className="px-4 py-4 text-[11px] text-ink-600">
          Empty. Nothing has been written here yet, so the first lab that needs
          weights will download them.
        </p>
      )}

      {listing && listing.entries.length > 0 && (
        <ul className="max-h-72 divide-y divide-ink-800/70 overflow-y-auto">
          {listing.entries.map((entry) => (
            <li key={entry.path}>
              <button
                onClick={() => entry.is_dir && setPath(`/${entry.path}`)}
                disabled={!entry.is_dir}
                className={`flex w-full items-center gap-3 px-4 py-2 text-left font-mono text-[11px] transition ${
                  entry.is_dir
                    ? "text-ink-200 hover:bg-ink-900/70 hover:text-flame-300"
                    : "cursor-default text-ink-400"
                }`}
              >
                <span className={entry.is_dir ? "text-flame-400/70" : "text-ink-700"}>
                  {entry.is_dir ? "▸" : "·"}
                </span>
                <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                <span className="text-ink-600">
                  {entry.is_dir ? "dir" : formatBytes(entry.size)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
