export const runtime = 'nodejs';

import type { NextRequest } from 'next/server';

type Embedder = (
  text: string,
  opts: { pooling: 'mean'; normalize: boolean }
) => Promise<unknown>;

async function getEmbedder(): Promise<Embedder> {
  const g = globalThis as unknown as {
    __memuEmbedder?: Embedder;
  };

  if (g.__memuEmbedder) return g.__memuEmbedder;

  const { pipeline } = await import('@xenova/transformers');
  const embedder = (await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  })) as Embedder;

  g.__memuEmbedder = embedder;
  return embedder;
}

function coerceEmbedding(output: unknown): number[] {
  if (output == null) throw new Error('Embedding model returned no output');

  // Array outputs
  if (Array.isArray(output)) {
    const first = output[0] as unknown;
    if (typeof first === 'number') return output as number[];
    if (Array.isArray(first) && typeof first[0] === 'number') return first as number[];
    const maybeObj = first as { data?: unknown };
    if (maybeObj.data && maybeObj.data instanceof Float32Array) return Array.from(maybeObj.data);
    if (maybeObj.data && Array.isArray(maybeObj.data) && typeof maybeObj.data[0] === 'number') return maybeObj.data as number[];
  }

  const outObj = output as { data?: unknown; tolist?: () => Promise<unknown> };
  if (outObj.data && outObj.data instanceof Float32Array) return Array.from(outObj.data);
  if (outObj.data && Array.isArray(outObj.data) && typeof outObj.data[0] === 'number') return outObj.data as number[];

  throw new Error('Unexpected embedding output shape');
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as { text?: string };
    const text = (body.text ?? '').trim();
    if (!text) {
      return Response.json({ error: 'Missing text' }, { status: 400 });
    }

    const embedder = await getEmbedder();
    const output = await embedder(text, { pooling: 'mean', normalize: true });
    const embedding = coerceEmbedding(output);

    return Response.json({ embedding });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    const stack = e instanceof Error ? e.stack : undefined;
    return Response.json({ error: message, stack }, { status: 500 });
  }
}
