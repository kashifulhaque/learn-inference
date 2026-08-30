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
    <div className="flex min-h-full items-center justify-center px-6">
      <form
        onSubmit={submit}
        className="w-full max-w-sm rounded-2xl border border-ink-800 bg-ink-900 p-8 shadow-2xl"
      >
        <div className="mb-1 text-xs font-medium uppercase tracking-[0.2em] text-flame-500">
          learn-inference
        </div>
        <h1 className="mb-1 text-2xl font-semibold text-ink-100">Build an engine</h1>
        <p className="mb-7 text-sm text-ink-400">
          Twenty-one chapters and twenty-one labs on an A100.
        </p>

        <label className="mb-1.5 block text-xs font-medium text-ink-300" htmlFor="name">
          Your name
        </label>
        <input
          id="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          required
          maxLength={40}
          autoComplete="username"
          className="mb-4 w-full rounded-lg border border-ink-700 bg-ink-850 px-3 py-2.5 text-sm text-ink-100 outline-none transition placeholder:text-ink-600 focus:border-flame-500"
          placeholder="Used to keep your progress separate"
        />

        <label className="mb-1.5 block text-xs font-medium text-ink-300" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          required
          autoComplete="current-password"
          className="mb-5 w-full rounded-lg border border-ink-700 bg-ink-850 px-3 py-2.5 text-sm text-ink-100 outline-none transition focus:border-flame-500"
        />

        {error && (
          <div className="mb-4 rounded-lg border border-rose-450/40 bg-rose-450/10 px-3 py-2 text-sm text-rose-450">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={busy || !name.trim() || !password}
          className="w-full rounded-lg bg-flame-500 px-4 py-2.5 text-sm font-semibold text-ink-950 transition hover:bg-flame-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
