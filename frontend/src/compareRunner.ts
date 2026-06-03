import {
  ApiErrorPayload,
  FetchLike,
  PanelSide,
  VisionPayload,
  VisionRequest,
  fetchJobStatus,
  normalizeApiError,
  submitVisionRequest
} from "./api";

export interface PanelRunRequest extends VisionRequest {
  side: PanelSide;
}

export interface PanelSuccess {
  side: PanelSide;
  state: "done";
  payload: VisionPayload;
  jobId?: string;
}

export interface PanelQueued {
  side: PanelSide;
  state: "queued";
  payload: VisionPayload;
  jobId?: string;
}

export interface PanelFailure {
  side: PanelSide;
  state: "error";
  error: ApiErrorPayload;
  jobId?: string;
}

export type PanelResult = PanelSuccess | PanelQueued | PanelFailure;

export interface CompareRequest {
  apiKey: string;
  left: VisionRequest;
  right: VisionRequest;
}

export interface RunnerOptions {
  fetchImpl?: FetchLike;
  pollIntervalMs?: number;
  maxPolls?: number;
}

export async function runComparison(request: CompareRequest, options: RunnerOptions = {}) {
  const [left, right] = await Promise.all([
    runPanel({ ...request.left, side: "A" }, request.apiKey, options),
    runPanel({ ...request.right, side: "B" }, request.apiKey, options)
  ]);
  return { left, right };
}

export async function runPanel(
  request: PanelRunRequest,
  apiKey: string,
  options: RunnerOptions = {}
): Promise<PanelResult> {
  try {
    const initial = await submitVisionRequest(apiKey, request, options.fetchImpl);
    if (initial.status === 202) {
      const queued = initial.payload as VisionPayload;
      if (!queued.status_url) {
        return { side: request.side, state: "queued", payload: queued, jobId: queued.job_id };
      }
      return pollQueuedPanel(request.side, apiKey, queued, options);
    }
    if (initial.status < 200 || initial.status >= 300) {
      return {
        side: request.side,
        state: "error",
        error: normalizeApiError(initial.payload, initial.status)
      };
    }
    return { side: request.side, state: "done", payload: initial.payload as VisionPayload };
  } catch (error) {
    return {
      side: request.side,
      state: "error",
      error: normalizeThrownError(error)
    };
  }
}

async function pollQueuedPanel(
  side: PanelSide,
  apiKey: string,
  queued: VisionPayload,
  options: RunnerOptions
): Promise<PanelResult> {
  const maxPolls = options.maxPolls ?? 120;
  const pollIntervalMs = options.pollIntervalMs ?? 1000;
  for (let attempt = 0; attempt < maxPolls; attempt += 1) {
    if (pollIntervalMs > 0 || attempt > 0) {
      await delay(pollIntervalMs);
    }
    const job = await fetchJobStatus(apiKey, queued.status_url as string, options.fetchImpl);
    if (job.status === "done" && job.result) {
      return { side, state: "done", payload: job.result, jobId: job.job_id };
    }
    if (job.status === "failed" || job.status === "canceled") {
      return {
        side,
        state: "error",
        error: job.error || { code: `JOB_${job.status.toUpperCase()}`, message: `Job ${job.status}.` },
        jobId: job.job_id
      };
    }
  }
  return { side, state: "queued", payload: queued, jobId: queued.job_id };
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeThrownError(error: unknown): ApiErrorPayload {
  if (error && typeof error === "object" && "code" in error && "message" in error) {
    return error as ApiErrorPayload;
  }
  if (error instanceof Error) {
    return { code: "REQUEST_FAILED", message: error.message };
  }
  return { code: "REQUEST_FAILED", message: "Request failed." };
}
