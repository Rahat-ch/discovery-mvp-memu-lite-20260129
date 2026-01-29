# memU-lite (discovery MVP)

A tiny, browser-only “memory bank” for notes with **semantic search**, inspired by the idea of persistent memory for proactive agents.

- No server
- No database
- No API keys
- Notes stored in `localStorage`
- Embeddings computed locally in the browser via `@xenova/transformers` (MiniLM)

## Why
Agent workflows keep trending, but most “agent memory” demos require infra + keys. This MVP shows the core UX loop (capture → recall) with **zero setup**.

## Run locally

```bash
pnpm install
pnpm dev
```

Open http://localhost:3000

## How it works
- Add notes → we compute an embedding (client-side)
- Search → we embed your query and rank notes by cosine similarity

## Limitations
- Client-only, `localStorage` persistence only
- First run downloads the embedding model into browser cache (can take a bit)
- Not optimized for huge note sets

## Next steps
- Tagging + collections
- Export/import
- Server-backed storage + auth
- Background “agent memory” ingestion from RSS/email/etc (with explicit permissions)
