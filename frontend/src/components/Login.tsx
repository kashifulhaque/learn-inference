import { useState } from "react";
import { api } from "../lib/api";

export default function Login({ onSignedIn }: { onSignedIn: (name: string) => void }) {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api.login(name.trim(), password);
      onSignedIn(result.name);
    } catch {
      setError("That password is not right.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="surface-grid relative flex min-h-full items-center justify-center overflow-hidden px-6 py-12">
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-125 w-125 -translate-x-1/2 -translate-y-1/2 rounded-full bg-flame-500/8 blur-3xl" />
      <form
        onSubmit={submit}
        className="panel-glow relative w-full max-w-md rounded-[1.75rem] border border-ink-700/70 bg-ink-900/85 p-7 backdrop-blur sm:p-9"
      >
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-flame-500 text-lg font-black tracking-tighter text-ink-950 shadow-lg shadow-flame-500/15">
            λ
          </div>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-[0.22em] text-flame-400">
              Learn inference
            </div>
            <div className="mt-0.5 text-sm text-ink-400">Systems course workspace</div>
          </div>
        </div>

        <h1 className="text-3xl font-semibold tracking-[-0.04em] text-ink-100">
          Build the stack.
        </h1>
        <p className="mt-3 max-w-sm text-sm leading-6 text-ink-400">
          Work from model weights to a production inference engine, one lab at a time.
        </p>

        <div className="my-7 h-px bg-ink-800" />

        <label className="mb-2 block text-xs font-semibold text-ink-300" htmlFor="name">
          Name
        </label>
        <input
          id="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          required
          maxLength={40}
          autoComplete="username"
          className="mb-5 w-full rounded-xl border border-ink-700 bg-ink-950/50 px-3.5 py-3 text-sm text-ink-100 outline-none transition placeholder:text-ink-600 focus:border-flame-500 focus:bg-ink-950"
          placeholder="Used to keep your progress separate"
        />

        <label className="mb-2 block text-xs font-semibold text-ink-300" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          required
          autoComplete="current-password"
          className="mb-5 w-full rounded-xl border border-ink-700 bg-ink-950/50 px-3.5 py-3 text-sm text-ink-100 outline-none transition focus:border-flame-500 focus:bg-ink-950"
        />

        {error && (
          <div className="mb-5 rounded-xl border border-rose-450/30 bg-rose-450/10 px-3.5 py-3 text-sm text-rose-450">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={busy || !name.trim() || !password}
          className="group flex w-full items-center justify-center gap-2 rounded-xl bg-flame-500 px-4 py-3 text-sm font-bold text-ink-950 transition hover:bg-flame-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Signing in…" : "Enter course"}
          {!busy && <span className="transition-transform group-hover:translate-x-0.5">→</span>}
        </button>
      </form>
    </div>
  );
}
