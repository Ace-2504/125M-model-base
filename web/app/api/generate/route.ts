import { NextResponse } from "next/server";

/**
 * Inference proxy.
 *
 * Forwards generation requests to the inference server named by
 * INFERENCE_URL (e.g. http://localhost:8000 for local testing, or a deployed
 * Modal / HF endpoint later), attaching the shared secret from INFERENCE_SECRET
 * so the model endpoint stays server-side and access-controlled.
 *
 * If INFERENCE_URL is unset, returns 503 so the UI shows its "not live yet"
 * state instead of erroring.
 */
export const runtime = "nodejs";

const INFERENCE_URL = process.env.INFERENCE_URL;
const INFERENCE_SECRET = process.env.INFERENCE_SECRET ?? "";

export async function POST(request: Request) {
  if (!INFERENCE_URL) {
    return NextResponse.json(
      {
        ready: false,
        message:
          "Inference is not connected — set INFERENCE_URL to a running model server.",
      },
      { status: 503 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ message: "Invalid JSON body." }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${INFERENCE_URL.replace(/\/$/, "")}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(INFERENCE_SECRET ? { Authorization: `Bearer ${INFERENCE_SECRET}` } : {}),
      },
      body: JSON.stringify(body),
      // 125M on CPU can take a few seconds; allow a generous cold-start window.
      signal: AbortSignal.timeout(120_000),
    });

    const data = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      return NextResponse.json(
        {
          ready: false,
          message:
            (data as { detail?: string; message?: string }).detail ??
            (data as { message?: string }).message ??
            `Inference server returned ${upstream.status}.`,
        },
        { status: upstream.status }
      );
    }
    return NextResponse.json(data);
  } catch (err) {
    const aborted = err instanceof Error && err.name === "TimeoutError";
    return NextResponse.json(
      {
        ready: false,
        message: aborted
          ? "The model server timed out (it may be cold-starting). Try again."
          : "Could not reach the inference server.",
      },
      { status: 502 }
    );
  }
}
