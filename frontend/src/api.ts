export type Task = "detect" | "segment";
export type PanelSide = "A" | "B";

export interface ModelInfo {
  name: string;
  task: Task;
  model_path: string;
  default_conf: number;
  half: boolean;
  device: string;
  input_modes: string[];
  ready: boolean;
  busy: boolean;
  last_error: string | null;
  active: boolean;
  default: boolean;
  class_names: string[] | null;
}

export interface ModelCatalog {
  defaults: Record<Task, string>;
  models: ModelInfo[];
}

export interface PromptPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export interface PromptBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface VisionControls {
  conf: number;
  classes: string;
  overlay: string;
  outputFormats: string[];
  prompt: string;
  points: PromptPoint[];
  boxes: PromptBox[];
}

export interface VisionRequest {
  task: Task;
  model: string;
  inputModes: string[];
  image: Blob;
  filename?: string;
  controls: VisionControls;
}

export interface ApiErrorPayload {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  request_id?: string | null;
}

export interface DetectionItem {
  id: number;
  score: number;
  bbox: number[];
  class_id?: number;
  class_name?: string;
  mask_url?: string;
  alpha_url?: string;
  polygon?: number[][];
  rle?: unknown;
}

export interface VisionPayload {
  request_id?: string;
  task?: Task;
  detect_model?: string;
  segment_model?: string;
  cached?: boolean;
  classes?: Array<string | number>;
  prompt?: string | null;
  output_formats?: string[];
  image_meta?: { width: number; height: number };
  detections?: DetectionItem[];
  overlay_url?: string | null;
  timing_ms?: Record<string, number>;
  cache_key?: string;
  status?: string;
  mode?: string;
  job_id?: string;
  status_url?: string;
}

export interface JobPayload {
  job_id: string;
  status: "queued" | "running" | "canceling" | "done" | "failed" | "canceled";
  task: Task;
  created_at?: string;
  updated_at?: string;
  result?: VisionPayload;
  error?: ApiErrorPayload;
}

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function authHeaders(apiKey: string): HeadersInit {
  return { "X-API-Key": apiKey };
}

export function buildVisionFormData(request: VisionRequest): FormData {
  const form = new FormData();
  const filename = request.filename || getBlobName(request.image) || "upload.png";
  form.append("image", request.image, filename);

  const controls = request.controls;
  const modes = new Set(request.inputModes);
  form.append("conf", String(controls.conf));
  if (controls.overlay) {
    form.append("overlay", controls.overlay);
  }

  if (request.task === "detect") {
    form.append("detect_model", request.model);
    if (modes.has("classes")) appendIfPresent(form, "classes", controls.classes);
    return form;
  }

  form.append("segment_model", request.model);
  if (controls.outputFormats.length > 0) {
    form.append("output_formats", JSON.stringify(controls.outputFormats));
  }
  if (modes.has("prompt")) appendIfPresent(form, "prompt", controls.prompt);
  if (modes.has("points") && controls.points.length > 0) {
    form.append("points", JSON.stringify(controls.points.map((p) => [p.x, p.y])));
    form.append("point_labels", JSON.stringify(controls.points.map((p) => p.label)));
  }
  if (modes.has("boxes") && controls.boxes.length > 0) {
    form.append(
      "boxes",
      JSON.stringify(controls.boxes.map((b) => [b.x1, b.y1, b.x2, b.y2]))
    );
  }
  if (modes.has("classes")) appendIfPresent(form, "classes", controls.classes);
  return form;
}

export async function fetchModelClasses(
  apiKey: string,
  modelName: string,
  fetchImpl: FetchLike = fetch
): Promise<string[] | null> {
  const response = await fetchImpl(`/v1/models/${encodeURIComponent(modelName)}/classes`, {
    headers: authHeaders(apiKey)
  });
  const payload = await readJson(response);
  if (!response.ok) throw normalizeApiError(payload, response.status);
  return (payload as { class_names: string[] | null }).class_names;
}

export async function fetchModelCatalog(apiKey: string, fetchImpl: FetchLike = fetch): Promise<ModelCatalog> {
  const response = await fetchImpl("/v1/models", { headers: authHeaders(apiKey) });
  const payload = await readJson(response);
  if (!response.ok) {
    throw normalizeApiError(payload, response.status);
  }
  return payload as ModelCatalog;
}

export async function submitVisionRequest(
  apiKey: string,
  request: VisionRequest,
  fetchImpl: FetchLike = fetch
): Promise<{ status: number; payload: VisionPayload | ApiErrorPayload }> {
  const endpoint = request.task === "detect" ? "/v1/detect" : "/v1/segment";
  const response = await fetchImpl(endpoint, {
    method: "POST",
    headers: authHeaders(apiKey),
    body: buildVisionFormData(request)
  });
  const payload = (await readJson(response)) as VisionPayload | ApiErrorPayload;
  return { status: response.status, payload };
}

export async function fetchJobStatus(
  apiKey: string,
  statusUrl: string,
  fetchImpl: FetchLike = fetch
): Promise<JobPayload> {
  const response = await fetchImpl(statusUrl, { headers: authHeaders(apiKey) });
  const payload = await readJson(response);
  if (!response.ok) {
    throw normalizeApiError(payload, response.status);
  }
  return payload as JobPayload;
}

export function normalizeApiError(payload: unknown, status = 0): ApiErrorPayload {
  if (payload && typeof payload === "object" && "code" in payload && "message" in payload) {
    return payload as ApiErrorPayload;
  }
  return {
    code: status ? `HTTP_${status}` : "REQUEST_FAILED",
    message: "Request failed.",
    details: typeof payload === "undefined" ? {} : { payload: payload as unknown }
  };
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { code: `HTTP_${response.status}`, message: text };
  }
}

function appendIfPresent(form: FormData, key: string, value: string): void {
  const cleaned = value.trim();
  if (cleaned) {
    form.append(key, cleaned);
  }
}

function getBlobName(blob: Blob): string | undefined {
  if ("name" in blob && typeof blob.name === "string") {
    return blob.name;
  }
  return undefined;
}
