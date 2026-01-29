'use client';

import { useEffect, useState } from 'react';

type Note = {
  id: string;
  text: string;
  createdAt: number;
  embedding?: number[];
};

function cosineSimilarity(a: number[], b: number[]) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

const STORAGE_KEY = 'memu-lite-notes-v1';

export default function Home() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [draft, setDraft] = useState('');
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<string>('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) setNotes(parsed);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
    } catch {
      // ignore
    }
  }, [notes]);

  async function embed(text: string): Promise<number[]> {
    const res = await fetch('/api/embed', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      let errText = `HTTP ${res.status}`;
      try {
        const data = (await res.json()) as { error?: string; stack?: string };
        errText = data.error ?? errText;
        if (data.stack) console.error(data.stack);
      } catch {
        // ignore
      }
      throw new Error(errText);
    }

    const data = (await res.json()) as { embedding: number[] };
    if (!data?.embedding?.length) throw new Error('Empty embedding returned');
    return data.embedding;
  }

  async function onAddNote() {
    const text = draft.trim();
    if (!text) return;
    setBusy(true);
    setStatus('Embedding note…');
    try {
      const embedding = await embed(text);
      const newNote: Note = {
        id: crypto.randomUUID(),
        text,
        createdAt: Date.now(),
        embedding,
      };
      setNotes((prev) => [newNote, ...prev]);
      setDraft('');
      setStatus('Added.');
    } catch (e: unknown) {
      setStatus(`Failed to embed note: ${(e instanceof Error ? e.message : String(e))}`);
    } finally {
      setBusy(false);
    }
  }

  async function onReembedAll() {
    if (!notes.length) return;
    setBusy(true);
    try {
      setStatus(`Re-embedding ${notes.length} notes…`);
      const updated: Note[] = [];
      for (const n of notes) {
        const embedding = await embed(n.text);
        updated.push({ ...n, embedding });
      }
      setNotes(updated);
      setStatus('Done.');
    } catch (e: unknown) {
      setStatus(`Failed to re-embed: ${(e instanceof Error ? e.message : String(e))}`);
    } finally {
      setBusy(false);
    }
  }

  const [showClearConfirm, setShowClearConfirm] = useState(false);

  function onClear() {
    setShowClearConfirm(true);
  }

  function confirmClear() {
    setNotes([]);
    setDraft('');
    setQuery('');
    setStatus('Cleared.');
    setShowClearConfirm(false);
  }

  function cancelClear() {
    setShowClearConfirm(false);
  }

  async function runSearch() {
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    setStatus('Embedding query + ranking…');
    try {
      const qEmb = await embed(q);
      const scored = notes
        .filter((n) => n.embedding && n.embedding.length)
        .map((n) => ({
          note: n,
          score: cosineSimilarity(qEmb, n.embedding as number[]),
        }))
        .sort((a, b) => b.score - a.score);

      // keep any notes missing embeddings at the bottom
      const missing = notes
        .filter((n) => !n.embedding || !n.embedding.length)
        .map((n) => ({ note: n, score: -1 }));

      setStatus('Ranked.');
      // store the ordering by rewriting notes
      setNotes([...scored.map((x) => x.note), ...missing.map((x) => x.note)]);
    } catch (e: unknown) {
      setStatus(`Search failed: ${(e instanceof Error ? e.message : String(e))}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">memU-lite</h1>
        <p className="text-sm text-neutral-600">
          A tiny “memory bank” for proactive agents — inspired by the trending idea of persistent memory for agents.
          Runs fully in your browser using a local embedding model (no API keys).
        </p>
      </div>

      <div className="rounded-lg border p-4 mb-6">
        <div className="flex gap-2">
          <input
            className="w-full rounded-md border px-3 py-2"
            placeholder="Add a note (idea, snippet, link summary, etc.)"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') onAddNote();
            }}
            disabled={busy}
          />
          <button
            className="rounded-md bg-black text-white px-4 py-2 disabled:opacity-50"
            onClick={onAddNote}
            disabled={busy || !draft.trim()}
          >
            Add
          </button>
        </div>
        <div className="mt-2 text-xs text-neutral-600">
          Tip: Press <span className="font-mono">⌘/Ctrl</span>+<span className="font-mono">Enter</span> to add.
        </div>
      </div>

      <div className="rounded-lg border p-4 mb-6">
        <div className="flex gap-2">
          <input
            className="w-full rounded-md border px-3 py-2"
            placeholder="Search your notes (semantic search)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') runSearch();
            }}
            disabled={busy}
          />
          <button
            className="rounded-md bg-neutral-900 text-white px-4 py-2 disabled:opacity-50"
            onClick={runSearch}
            disabled={busy || !query.trim()}
          >
            Search
          </button>
        </div>
        <div className="mt-3 flex gap-2">
          <button className="text-sm underline disabled:opacity-50" onClick={onReembedAll} disabled={busy || !notes.length}>
            Re-embed all
          </button>
          <button className="text-sm underline disabled:opacity-50" onClick={onClear} disabled={busy || !notes.length}>
            Clear
          </button>
        </div>
      </div>

      {showClearConfirm ? (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4">
          <div className="text-sm font-medium text-red-900">Clear all notes?</div>
          <div className="mt-1 text-sm text-red-800">This only affects this browser (localStorage).</div>
          <div className="mt-3 flex gap-2">
            <button
              className="rounded-md bg-red-600 px-3 py-2 text-sm text-white"
              onClick={confirmClear}
            >
              Yes, clear
            </button>
            <button className="rounded-md border px-3 py-2 text-sm" onClick={cancelClear}>
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {status ? <div className="mb-4 text-sm text-neutral-700">{status}</div> : null}

      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-lg font-semibold">Notes</h2>
        <div className="text-xs text-neutral-500">{notes.length} total</div>
      </div>

      <ul className="space-y-3">
        {notes.map((n) => (
          <li key={n.id} className="rounded-lg border p-4">
            <div className="text-sm whitespace-pre-wrap">{n.text}</div>
            <div className="mt-2 text-xs text-neutral-500">
              {new Date(n.createdAt).toLocaleString()} · {n.embedding?.length ? 'embedded' : 'no embedding'}
            </div>
          </li>
        ))}
        {!notes.length ? (
          <li className="rounded-lg border border-dashed p-8 text-center text-sm text-neutral-500">
            Add a couple notes, then search by meaning.
          </li>
        ) : null}
      </ul>

      <footer className="mt-10 text-xs text-neutral-500">
        <p>
          No server, no database, no keys. Your notes live in <span className="font-mono">localStorage</span>. For a real agent, this would be backed by a DB + access controls.
        </p>
      </footer>
    </main>
  );
}
