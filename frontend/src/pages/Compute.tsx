import { useCallback, useEffect, useState } from "react";
import {
  api,
  type Infra,
  type InfraVolume,
  type Instance,
  type ProviderInfra,
} from "../lib/api";
import VolumeBrowser from "../components/VolumeBrowser";
import { formatAge } from "../lib/format";

const REFRESH_MS = 20_000;

const PROVIDER_LABELS: Record<string, string> = { modal: "Modal", runpod: "RunPod" };

export default function Compute() {
  const [infra, setInfra] = useState<Infra | null>(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [, setClock] = useState(0);

  const load = useCallback(async () => {
    try {
      setInfra(await api.infra());
      setError("");
    } catch (problem) {
      setError(String((problem as Error).message ?? problem));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while the tab is visible, so a forgotten GPU shows up without a click.
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) load();
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [live, load]);

  // Re-render once a second so the "up for" clocks keep counting.
  useEffect(() => {
    const timer = window.setInterval(() => setClock((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  async function act(
    provider: string,
    action: string,
    target: string,
    confirmText?: string,
  ) {
    if (confirmText && !window.confirm(confirmText)) return;
    const key = `${provider}:${action}:${target}`;
    setBusy(key);
    setMessage("");
    try {
      const result = await api.infraAction(provider, action, target);
      setMessage(result.message);
      await load();
    } catch (problem) {
      setError(String((problem as Error).message ?? problem));
    } finally {
      setBusy(null);
    }
  }

  const active = infra?.active ?? 0;
  const facts = (infra?.providers ?? []).flatMap((provider) =>
    provider.facts
      .filter((fact) => ["Balance", "Burn rate", "Spent, 24h"].includes(fact.label))
      .map((fact) => ({ ...fact, provider: provider.name })),
  );

  return (
    <div className="mx-auto max-w-6xl px-5 py-8 sm:px-8 lg:px-12 lg:py-11">
      <section className="surface-grid panel-glow relative overflow-hidden rounded-[2rem] border border-ink-700/70 bg-ink-900/70 px-6 py-7 sm:px-9 sm:py-9">
        <div
          className={`pointer-events-none absolute -right-20 -top-24 h-80 w-80 rounded-full blur-3xl ${
            active ? "bg-rose-450/15" : "bg-flame-500/10"
          }`}
        />
        <div className="relative flex flex-wrap items-start justify-between gap-5">
          <div className="max-w-2xl">
            <div className="mb-4 flex items-center gap-3">
              <span
                className={`h-2 w-2 rounded-full ${
                  active ? "animate-pulse bg-rose-450" : "bg-flame-500"
                }`}
              />
              <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-flame-400">
                Compute
              </span>
            </div>
            <h1 className="text-3xl font-semibold tracking-[-0.04em] text-ink-100 sm:text-4xl">
              {active
                ? `${active} thing${active === 1 ? "" : "s"} running.`
                : "Nothing is running."}
            </h1>
            <p className="mt-4 text-sm leading-6 text-ink-400">
              Every container, pod, and worker pool this app can start, and the
              storage the labs read weights from. Stopping something here does not
              delete it: the RunPod endpoint stays deployed and costs nothing while
              idle, and the volumes keep their weight cache.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-xs text-ink-300">
              <input
                type="checkbox"
                checked={live}
                onChange={(event) => setLive(event.target.checked)}
                className="accent-flame-500"
              />
              Auto-refresh
            </label>
            <button
              onClick={load}
              className="rounded-lg bg-flame-500 px-4 py-2 text-xs font-bold text-ink-950 transition hover:bg-flame-400"
            >
              Refresh
            </button>
          </div>
        </div>

        {infra && (
          <p className="relative mt-5 text-[11px] text-ink-600">
            Read {formatAge((Date.now() - infra.generated_at * 1000) / 1000)} ago
            {live ? ` · refreshing every ${REFRESH_MS / 1000}s` : " · auto-refresh off"}
          </p>
        )}
      </section>

      {facts.length > 0 && (
        <section className="relative -mt-1 grid gap-3 border-x border-b border-ink-800/80 bg-ink-900/35 p-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-ink-800 bg-ink-900/70 px-4 py-4">
            <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-ink-500">
              Active now
            </div>
            <div
              className={`mt-2 text-2xl font-semibold tracking-[-0.03em] ${
                active ? "text-rose-450" : "text-ink-100"
              }`}
            >
              {active}
            </div>
            <div className="mt-1 text-xs text-ink-500">containers, pods, workers</div>
          </div>
          {facts.map((fact) => (
            <div
              key={`${fact.provider}-${fact.label}`}
              className="rounded-xl border border-ink-800 bg-ink-900/70 px-4 py-4"
            >
              <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-ink-500">
                {fact.label}
              </div>
              <div className="mt-2 truncate text-2xl font-semibold tracking-[-0.03em] text-ink-100">
                {fact.value}
              </div>
              <div className="mt-1 text-xs text-ink-500">{fact.provider}</div>
            </div>
          ))}
        </section>
      )}

      {message && (
        <p className="mt-6 rounded-xl border border-mint-400/30 bg-mint-400/10 px-4 py-3 text-xs leading-5 text-mint-400">
          {message}
        </p>
      )}
      {error && (
        <p className="mt-6 rounded-xl border border-rose-450/30 bg-rose-450/10 px-4 py-3 font-mono text-xs leading-5 text-rose-450">
          {error}
        </p>
      )}
      {loading && !infra && (
        <p className="mt-8 text-sm text-ink-500">Asking the providers…</p>
      )}

      {infra && infra.runs.length > 0 && (
        <section className="panel-glow mt-8 rounded-2xl border border-flame-500/30 bg-ink-900/70 p-5 sm:p-6">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.18em] text-flame-400">
            Unfinished lab runs
          </div>
          <h2 className="text-lg font-semibold tracking-tight text-ink-100">
            Runs this app never saw finish
          </h2>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-ink-500">
            A run stays here when the browser tab closed mid-stream. The GPU job
            may still be going.
          </p>
          <ul className="mt-4 space-y-2">
            {infra.runs.map((run) => (
              <li
                key={run.id}
                className="flex flex-wrap items-center gap-3 rounded-xl border border-ink-800 bg-ink-950/40 px-4 py-3"
              >
                <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-flame-500" />
                <span className="text-sm font-medium text-ink-100">{run.lab}</span>
                <span className="font-mono text-[11px] text-ink-500">
                  {run.provider} · {run.user} · {formatAge(run.age)}
                </span>
                <span className="flex-1 text-[11px] text-ink-600">{run.hint}</span>
                <button
                  onClick={() =>
                    act(
                      run.provider,
                      "cancel-run",
                      run.id,
                      "Cancel this run? Any job still on the GPU is cancelled too.",
                    )
                  }
                  disabled={busy !== null}
                  className="rounded-lg border border-rose-450/45 bg-rose-450/8 px-3 py-1.5 text-[11px] font-bold text-rose-450 transition hover:bg-rose-450/15 disabled:opacity-40"
                >
                  Cancel run
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {infra?.providers.map((provider) => (
        <ProviderSection
          key={provider.name}
          provider={provider}
          busy={busy}
          onAct={act}
        />
      ))}
    </div>
  );
}

type SectionProps = {
  provider: ProviderInfra;
  busy: string | null;
  onAct: (
    provider: string,
    action: string,
    target: string,
    confirmText?: string,
  ) => void;
};

function ProviderSection({ provider, busy, onAct }: SectionProps) {
  const [showOthers, setShowOthers] = useState(false);

  // The workspace holds volumes from other projects. The course's own volume is
  // the one that matters here, so the rest wait behind a click.
  const mine = provider.volumes.filter((volume) => volume.primary);
  const others = provider.volumes.filter((volume) => !volume.primary);
  const volumes = mine.length ? [...mine, ...(showOthers ? others : [])] : others;

  return (
    <section className="panel-glow mt-8 overflow-hidden rounded-2xl border border-ink-800 bg-ink-900/70">
      <header className="flex flex-wrap items-center gap-3 border-b border-ink-800 bg-ink-850/50 px-5 py-4 sm:px-6">
        <h2 className="text-base font-semibold tracking-tight text-ink-100">
          {PROVIDER_LABELS[provider.name] ?? provider.name}
        </h2>
        {provider.default && (
          <span className="rounded-md border border-flame-500/35 bg-flame-500/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-flame-400">
            default
          </span>
        )}
        <span
          className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            provider.available
              ? "border-mint-400/30 bg-mint-400/10 text-mint-400"
              : "border-ink-700 bg-ink-850 text-ink-500"
          }`}
        >
          {provider.available ? "configured" : "unavailable"}
        </span>
        <div className="flex-1" />
        {provider.console_url && (
          <a
            href={provider.console_url}
            target="_blank"
            rel="noreferrer"
            className="text-xs font-medium text-ink-500 underline decoration-ink-700 underline-offset-4 transition hover:text-flame-300"
          >
            Open console ↗
          </a>
        )}
      </header>

      {!provider.available && (
        <p className="border-b border-ink-800 bg-ink-950/40 px-5 py-3 text-xs text-ink-400 sm:px-6">
          {provider.reason}
        </p>
      )}

      {provider.notices.map((notice) => (
        <p
          key={notice}
          className="border-b border-ink-800 bg-flame-500/8 px-5 py-3 text-xs leading-5 text-flame-300 sm:px-6"
        >
          {notice}
        </p>
      ))}

      {provider.facts.length > 0 && (
        <div className="grid grid-cols-2 gap-px border-b border-ink-800 bg-ink-800 sm:grid-cols-3 lg:grid-cols-4">
          {provider.facts.map((fact) => (
            <div key={fact.label} className="bg-ink-900 px-4 py-3">
              <div className="truncate text-[10px] font-bold uppercase tracking-[0.12em] text-ink-600">
                {fact.label}
              </div>
              <div className="mt-1 truncate font-mono text-xs text-ink-200">
                {fact.value}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="px-5 py-5 sm:px-6">
        <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500">
          Instances
        </div>
        {provider.instances.length === 0 ? (
          <p className="rounded-xl border border-dashed border-ink-800 px-4 py-6 text-center text-xs text-ink-600">
            Nothing running on {PROVIDER_LABELS[provider.name] ?? provider.name}.
          </p>
        ) : (
          <ul className="space-y-2">
            {provider.instances.map((instance) => (
              <InstanceRow
                key={`${instance.kind}-${instance.id}`}
                instance={instance}
                busy={busy}
                onAct={onAct}
              />
            ))}
          </ul>
        )}
      </div>

      {volumes.length > 0 && (
        <div className="border-t border-ink-800 px-5 py-5 sm:px-6">
          <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.18em] text-ink-500">
            Storage
          </div>
          <ul className="space-y-2">
            {volumes.map((volume) => (
              <VolumeRow key={volume.id} volume={volume} />
            ))}
          </ul>
          {mine.length > 0 && others.length > 0 && (
            <button
              onClick={() => setShowOthers((value) => !value)}
              className="mt-3 text-xs font-medium text-ink-500 underline decoration-ink-700 underline-offset-4 transition hover:text-flame-300"
            >
              {showOthers
                ? "Hide the other volumes"
                : `Show ${others.length} other volume${
                    others.length === 1 ? "" : "s"
                  } in this workspace`}
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function InstanceRow({
  instance,
  busy,
  onAct,
}: {
  instance: Instance;
  busy: string | null;
  onAct: SectionProps["onAct"];
}) {
  const uptime = instance.started_at
    ? formatAge(Date.now() / 1000 - instance.started_at)
    : null;

  return (
    <li className="rounded-xl border border-ink-800 bg-ink-950/40 px-4 py-3.5">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={`h-2.5 w-2.5 shrink-0 rounded-full ${
            instance.active
              ? "animate-pulse bg-rose-450 shadow-[0_0_10px_rgba(255,118,144,0.7)]"
              : "bg-ink-700"
          }`}
          aria-hidden
        />
        <span className="text-sm font-medium text-ink-100">{instance.label}</span>
        <span className="rounded-md border border-ink-700 bg-ink-850 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-400">
          {instance.kind}
        </span>
        <span className="text-xs text-ink-400">{instance.status}</span>
        {uptime && (
          <span className="text-xs text-ink-600">
            {instance.age_label} {uptime}
          </span>
        )}
        <div className="flex-1" />
        {instance.actions.map((action) => (
          <button
            key={action.action}
            onClick={() =>
              onAct(instance.provider, action.action, instance.id, action.confirm)
            }
            disabled={busy !== null}
            className="rounded-lg border border-rose-450/45 bg-rose-450/8 px-3 py-1.5 text-[11px] font-bold text-rose-450 transition hover:bg-rose-450/15 disabled:opacity-40"
          >
            {busy === `${instance.provider}:${action.action}:${instance.id}`
              ? "Working…"
              : action.label}
          </button>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 pl-6 text-[11px] leading-5 text-ink-500">
        <span className="font-mono text-ink-600">{instance.id}</span>
        {instance.gpu && <span className="font-mono text-flame-400/80">{instance.gpu}</span>}
        <span>{instance.detail}</span>
      </div>
    </li>
  );
}

function VolumeRow({ volume }: { volume: InfraVolume }) {
  const [open, setOpen] = useState(false);

  return (
    <li
      className={`rounded-xl border bg-ink-950/40 px-4 py-3.5 ${
        volume.primary ? "border-flame-500/30" : "border-ink-800"
      }`}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-flame-400/70">▤</span>
        <span className="text-sm font-medium text-ink-100">{volume.name}</span>
        <span className="rounded-md border border-ink-700 bg-ink-850 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-400">
          {volume.kind}
        </span>
        {volume.size_gb !== null && (
          <span className="text-xs text-ink-400">{volume.size_gb} GB</span>
        )}
        {volume.region && <span className="text-xs text-ink-600">{volume.region}</span>}
        <div className="flex-1" />
        {volume.browsable && (
          <button
            onClick={() => setOpen((value) => !value)}
            className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-1.5 text-[11px] font-semibold text-ink-200 transition hover:border-flame-500/50 hover:text-flame-300"
          >
            {open ? "Hide files" : "Browse files"}
          </button>
        )}
        {volume.console_url && (
          <a
            href={volume.console_url}
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-1.5 text-[11px] font-semibold text-ink-200 transition hover:border-flame-500/50 hover:text-flame-300"
          >
            View in console ↗
          </a>
        )}
      </div>
      <p className="mt-2 pl-6 text-[11px] leading-5 text-ink-500">
        <span className="font-mono text-ink-600">{volume.id}</span> · {volume.detail}
        {!volume.browsable && " Files are only readable from a machine that mounts it."}
      </p>
      {open && (
        <VolumeBrowser provider={volume.provider} volume={volume.name} />
      )}
    </li>
  );
}
