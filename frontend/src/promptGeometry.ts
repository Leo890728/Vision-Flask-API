import type { PromptBox, PromptPoint } from "./api";

export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface NaturalSize {
  width: number;
  height: number;
}

export function displayToNaturalPoint(
  clientX: number,
  clientY: number,
  rect: RectLike,
  natural: NaturalSize,
  label: 0 | 1 = 1
): PromptPoint {
  const xRatio = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
  const yRatio = rect.height > 0 ? (clientY - rect.top) / rect.height : 0;
  return {
    x: roundCoord(clamp(xRatio, 0, 1) * natural.width),
    y: roundCoord(clamp(yRatio, 0, 1) * natural.height),
    label
  };
}

export function displayBoxToNatural(
  startClientX: number,
  startClientY: number,
  endClientX: number,
  endClientY: number,
  rect: RectLike,
  natural: NaturalSize
): PromptBox {
  const start = displayToNaturalPoint(startClientX, startClientY, rect, natural);
  const end = displayToNaturalPoint(endClientX, endClientY, rect, natural);
  return {
    x1: Math.min(start.x, end.x),
    y1: Math.min(start.y, end.y),
    x2: Math.max(start.x, end.x),
    y2: Math.max(start.y, end.y)
  };
}

export function naturalPointToDisplay(point: PromptPoint, natural: NaturalSize): { x: number; y: number } {
  return {
    x: natural.width > 0 ? (point.x / natural.width) * 100 : 0,
    y: natural.height > 0 ? (point.y / natural.height) * 100 : 0
  };
}

export function naturalBoxToDisplay(box: PromptBox, natural: NaturalSize) {
  const left = natural.width > 0 ? (box.x1 / natural.width) * 100 : 0;
  const top = natural.height > 0 ? (box.y1 / natural.height) * 100 : 0;
  const right = natural.width > 0 ? (box.x2 / natural.width) * 100 : 0;
  const bottom = natural.height > 0 ? (box.y2 / natural.height) * 100 : 0;
  return {
    left,
    top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top)
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function roundCoord(value: number) {
  return Math.round(value * 100) / 100;
}
