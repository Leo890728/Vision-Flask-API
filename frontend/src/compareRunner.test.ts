import { describe, expect, it } from "vitest";
import { runPanel } from "./compareRunner";
import type { FetchLike, VisionControls } from "./api";

const controls: VisionControls = {
  conf: 0.25,
  classes: "",
  overlay: "bbox",
  outputFormats: ["mask_png"],
  prompt: "",
  points: [],
  boxes: []
};

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

describe("runPanel", () => {
  it("returns done for successful sync responses", async () => {
    const fetchImpl: FetchLike = async () =>
      response({ request_id: "req-1", task: "detect", detect_model: "yolo26n", detections: [] });

    const result = await runPanel(
      {
        side: "A",
        task: "detect",
        model: "yolo26n",
        image: new Blob(["image"], { type: "image/png" }),
        filename: "sample.png",
        controls
      },
      "key",
      { fetchImpl }
    );

    expect(result.state).toBe("done");
    expect(result.side).toBe("A");
  });

  it("returns error for API errors", async () => {
    const fetchImpl: FetchLike = async () =>
      response({ code: "INVALID_DETECT_MODEL", message: "bad model" }, 400);

    const result = await runPanel(
      {
        side: "B",
        task: "detect",
        model: "bad",
        image: new Blob(["image"], { type: "image/png" }),
        filename: "sample.png",
        controls
      },
      "key",
      { fetchImpl }
    );

    expect(result.state).toBe("error");
    if (result.state === "error") {
      expect(result.error.code).toBe("INVALID_DETECT_MODEL");
    }
  });

  it("polls queued jobs until done", async () => {
    const calls: string[] = [];
    const fetchImpl: FetchLike = async (input) => {
      calls.push(String(input));
      if (String(input) === "/v1/detect") {
        return response({ job_id: "job-1", status_url: "/v1/jobs/job-1", status: "queued" }, 202);
      }
      return response({
        job_id: "job-1",
        status: "done",
        task: "detect",
        result: { request_id: "req-queued", task: "detect", detect_model: "yolo11n", detections: [] }
      });
    };

    const result = await runPanel(
      {
        side: "A",
        task: "detect",
        model: "yolo11n",
        image: new Blob(["image"], { type: "image/png" }),
        filename: "sample.png",
        controls
      },
      "key",
      { fetchImpl, pollIntervalMs: 0, maxPolls: 3 }
    );

    expect(calls).toEqual(["/v1/detect", "/v1/jobs/job-1"]);
    expect(result.state).toBe("done");
    if (result.state === "done") {
      expect(result.jobId).toBe("job-1");
      expect(result.payload.detect_model).toBe("yolo11n");
    }
  });
});
