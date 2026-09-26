import { useEffect, useState } from "react";
import { api, type VolumeListing } from "../lib/api";
import { formatBytes } from "../lib/format";
import { IconButton } from "./ui";
import { ArrowLeftIcon, FileIcon, FolderIcon } from "./icons";

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
    <div className="mt-3 overflow-hidden rounded-md border border-line bg-card">
      <div className="flex items-center gap-2 border-b border-line bg-well py-1 pl-1 pr-3">
        <IconButton
          size="sm"
          onClick={() => setPath(listing?.parent ?? "/")}
          disabled={!listing?.parent}
          aria-label="Up one directory"
          title="Up one directory"
        >
          <ArrowLeftIcon className="size-3.5" />
        </IconButton>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg-muted">
          <span className="text-fg-faint">{volume}:</span>
          {listing?.path ?? path}
        </span>
        {listing && listing.bytes > 0 && (
          <span className="flex-none font-mono text-[12px] tabular-nums text-fg-subtle">
            {formatBytes(listing.bytes)} here
          </span>
        )}
      </div>

      {loading && (
        <p className="flex items-center gap-2 px-3 py-3 text-[12.5px] text-fg-subtle" role="status">
          <span
            className="size-3 animate-spin rounded-full border-[1.5px] border-line-strong border-t-fg-subtle"
            aria-hidden
          />
          Listing…
        </p>
      )}
      {error && (
        <p className="break-words px-3 py-3 font-mono text-[12px] leading-5 text-bad">{error}</p>
      )}
      {!loading && !error && listing?.entries.length === 0 && (
        <p className="px-3 py-4 text-[12.5px] leading-5 text-fg-subtle">
          Empty. Nothing has been written here yet, so the first lab that needs
          weights will download them.
        </p>
      )}

      {listing && listing.entries.length > 0 && (
        <ul className="max-h-72 divide-y divide-line overflow-y-auto">
          {listing.entries.map((entry) => (
            <li key={entry.path}>
              <button
                type="button"
                onClick={() => entry.is_dir && setPath(`/${entry.path}`)}
                disabled={!entry.is_dir}
                className={`flex w-full items-center gap-2.5 px-3 py-1.5 text-left font-mono text-[12px] transition-colors ${
                  entry.is_dir
                    ? "text-fg hover:bg-tint"
                    : "cursor-default text-fg-muted"
                }`}
              >
                {entry.is_dir ? (
                  <FolderIcon className="size-3.5 text-fg-subtle" />
                ) : (
                  <FileIcon className="size-3.5 text-fg-faint" />
                )}
                <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                <span className="flex-none tabular-nums text-fg-faint">
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
