import { describe, expect, it } from "vitest";
import { displayBoxToNatural, displayToNaturalPoint, naturalBoxToDisplay } from "./promptGeometry";

const rect = { left: 10, top: 20, width: 200, height: 100 };
const natural = { width: 1000, height: 500 };

describe("promptGeometry", () => {
  it("maps display coordinates to natural image coordinates", () => {
    const point = displayToNaturalPoint(110, 70, rect, natural, 0);
    expect(point).toEqual({ x: 500, y: 250, label: 0 });
  });

  it("clamps points to the image bounds", () => {
    const point = displayToNaturalPoint(-50, 999, rect, natural, 1);
    expect(point).toEqual({ x: 0, y: 500, label: 1 });
  });

  it("normalizes boxes from drag direction", () => {
    const box = displayBoxToNatural(210, 120, 10, 20, rect, natural);
    expect(box).toEqual({ x1: 0, y1: 0, x2: 1000, y2: 500 });
  });

  it("maps natural boxes back to display percentages", () => {
    const view = naturalBoxToDisplay({ x1: 250, y1: 100, x2: 750, y2: 400 }, natural);
    expect(view).toEqual({ left: 25, top: 20, width: 50, height: 60 });
  });
});
