import { NextResponse } from "next/server";

/**
 * Inference proxy → the SLM-125M Hugging Face ZeroGPU Space (a Gradio app).
 *
 * Gradio's REST API is two calls: POST /gradio_api/call/<fn> returns an
 * event_id, then GET /gradio_api/call/<fn>/<event_id> streams the result as SSE.
 * We do both server-side so the browser only sees a simple { completion }.
 *
 * The Space is public, so no secret is needed. Override the target with the
 * INFERENCE_URL env var if you point the demo at a different Gradio Space.
 */
export const runtime = "nodejs";
export const maxDuration = 60; // seconds (Vercel cap); a warm generation takes a few

const SPACE_URL = (
  process.env.INFERENCE_URL ?? "https://opening-dry-call-commissioners.trycloudflare.com"
).replace(/\/$/, "");
const INFERENCE_BACKEND =
  process.env.INFERENCE_BACKEND ?? (SPACE_URL.includes(".hf.space") ? "gradio" : "fastapi");
const INFERENCE_SECRET = process.env.INFERENCE_SECRET;
const HF_TOKEN = process.env.HF_TOKEN;
const FN = "generate"; // Gradio api_name

export async function POST(request: Request) {
  let body: {
    prompt?: string;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    topK?: number;
  };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ready: false, message: "Invalid request." }, { status: 400 });
  }

  const prompt = String(body.prompt ?? "").trim();
  if (!prompt) {
    return NextResponse.json({ ready: false, message: "Enter a prompt." }, { status: 400 });
  }
  const generation = {
    prompt,
    temperature: Number(body.temperature ?? 0.8),
    maxTokens: Number(body.maxTokens ?? 90),
    topP: Number(body.topP ?? 0.95),
    topK: Number(body.topK ?? 50),
  };

  try {
    if (INFERENCE_BACKEND === "fastapi") {
      const response = await fetch(`${SPACE_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(INFERENCE_SECRET ? { Authorization: `Bearer ${INFERENCE_SECRET}` } : {}),
        },
        body: JSON.stringify(generation),
        signal: AbortSignal.timeout(55_000),
      });

      const result = (await response.json().catch(() => null)) as {
        ready?: boolean;
        completion?: string;
        detail?: string;
      } | null;

      if (!response.ok || !result?.ready) {
        return NextResponse.json(
          {
            ready: false,
            message: result?.detail ?? "The local model server returned an error.",
          },
          { status: 502 }
        );
      }

      return NextResponse.json({ ready: true, completion: result.completion ?? "" });
    }

    const data = [
      generation.prompt,
      generation.temperature,
      generation.maxTokens,
      generation.topP,
      generation.topK,
    ];

    // 1) enqueue the call
    const enqueue = await fetch(`${SPACE_URL}/gradio_api/call/${FN}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {}),
      },
      body: JSON.stringify({ data }),
      signal: AbortSignal.timeout(45_000),
    });
    if (!enqueue.ok) {
      return NextResponse.json(
        {
          ready: false,
          message: `The model Space returned ${enqueue.status} — it may be waking from idle. Try again in ~30s.`,
        },
        { status: 502 }
      );
    }
    const { event_id } = (await enqueue.json()) as { event_id?: string };
    if (!event_id) {
      return NextResponse.json(
        { ready: false, message: "Could not start generation." },
        { status: 502 }
      );
    }

    // 2) read the result stream (SSE); Gradio closes it after the 'complete' event
    const stream = await fetch(`${SPACE_URL}/gradio_api/call/${FN}/${event_id}`, {
      signal: AbortSignal.timeout(55_000),
    });
    const text = await stream.text();

    let full: string | null = null;
    let sawError = false;
    for (const ev of text.split(/\n\n/)) {
      const dataLine = ev.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const payload = dataLine.slice(5).trim();
      if (ev.includes("event: complete")) {
        try {
          const parsed = JSON.parse(payload);
          full = Array.isArray(parsed) ? String(parsed[0] ?? "") : String(parsed);
        } catch {
          /* ignore malformed frame */
        }
        break;
      }
      if (ev.includes("event: error")) sawError = true;
    }

    if (full === null) {
      return NextResponse.json(
        {
          ready: false,
          message: sawError
            ? "The model Space reported an error."
            : "The model Space is waking up (cold start). Try again in ~30s.",
        },
        { status: 502 }
      );
    }

    // The Space returns prompt + continuation; strip the prompt so the UI can
    // render it separately.
    let completion = full;
    if (completion.startsWith(prompt)) completion = completion.slice(prompt.length);
    completion = completion.replace(/^\s+/, "");

    return NextResponse.json({ ready: true, completion });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "TimeoutError";
    return NextResponse.json(
      {
        ready: false,
        message: aborted
          ? "The model Space is waking up (cold start). Give it ~30s and try again."
          : "Could not reach the model Space.",
      },
      { status: 502 }
    );
  }
}
