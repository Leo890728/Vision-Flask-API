import { describe, expect, it } from "vitest";
import { buildVisionFormData, type VisionControls } from "./api";

const baseControls: VisionControls = {
  conf: 0.33,
  classes: "person, car",
  overlay: "bbox",
  outputFormats: ["mask_png"],
  prompt: "person",
  points: [],
  boxes: []
};

describe("buildVisionFormData", () => {
  it("builds detect requests with detect_model", () => {
    const form = buildVisionFormData({
      task: "detect",
      model: "yolo11n",
      inputModes: ["classes"],
      image: new Blob(["image"], { type: "image/png" }),
      filename: "sample.png",
      controls: baseControls
    });

    expect(form.get("detect_model")).toBe("yolo11n");
    expect(form.get("conf")).toBe("0.33");
    expect(form.get("classes")).toBe("person, car");
    expect(form.get("overlay")).toBe("bbox");
    expect(form.has("image")).toBe(true);
  });

  it("builds SAM3 segment requests with visual prompts", () => {
    const form = buildVisionFormData({
      task: "segment",
      model: "sam3",
      inputModes: ["prompt", "points", "boxes"],
      image: new Blob(["image"], { type: "image/png" }),
      filename: "sample.png",
      controls: {
        ...baseControls,
        overlay: "both",
        points: [
          { x: 10, y: 20, label: 1 },
          { x: 30, y: 40, label: 0 }
        ],
        boxes: [{ x1: 5, y1: 6, x2: 50, y2: 60 }],
        outputFormats: ["mask_png", "polygon"]
      }
    });

    expect(form.get("segment_model")).toBe("sam3");
    expect(form.get("prompt")).toBe("person");
    expect(form.get("points")).toBe("[[10,20],[30,40]]");
    expect(form.get("point_labels")).toBe("[1,0]");
    expect(form.get("boxes")).toBe("[[5,6,50,60]]");
    expect(form.get("output_formats")).toBe("[\"mask_png\",\"polygon\"]");
  });

  it("does not send SAM3-only prompts to yolo_seg", () => {
    const form = buildVisionFormData({
      task: "segment",
      model: "yolo_seg",
      inputModes: ["classes"],
      image: new Blob(["image"], { type: "image/png" }),
      filename: "sample.png",
      controls: {
        ...baseControls,
        overlay: "mask",
        points: [{ x: 10, y: 20, label: 1 }],
        boxes: [{ x1: 5, y1: 6, x2: 50, y2: 60 }]
      }
    });

    expect(form.get("segment_model")).toBe("yolo_seg");
    expect(form.get("classes")).toBe("person, car");
    expect(form.has("points")).toBe(false);
    expect(form.has("boxes")).toBe(false);
  });
});
