// Client-side YOLOv8 helpers: letterbox preprocessing and NMS postprocessing.
// Pure functions (no DOM) so they can be unit-tested in Node. They mirror what
// Ultralytics does on the server: letterbox to 640 (pad 114), conf 0.5, per-class NMS at IoU 0.7.

export const INPUT_SIZE = 640;
export const NUM_CLASSES = 6;
export const CLASS_NAMES = [
  'fall-armyworm-egg',
  'fall-armyworm-frass',
  'fall-armyworm-larva',
  'fall-armyworm-larval-damage',
  'healthy-maize',
  'maize-streak-disease',
];

/** Letterbox geometry for an image of size (w, h), matching Ultralytics. */
export function letterboxParams(w, h, size = INPUT_SIZE) {
  const ratio = Math.min(size / h, size / w);
  const newW = Math.round(w * ratio);
  const newH = Math.round(h * ratio);
  const dw = (size - newW) / 2;
  const dh = (size - newH) / 2;
  const padLeft = Math.round(dw - 0.1);
  const padTop = Math.round(dh - 0.1);
  return { ratio, newW, newH, padLeft, padTop, srcW: w, srcH: h };
}

/**
 * Convert RGBA pixels of an already letterboxed size x size image to a
 * normalised CHW Float32Array (RGB, 0-1).
 */
export function rgbaToTensor(rgba, size = INPUT_SIZE) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = rgba[p] / 255;
    out[plane + i] = rgba[p + 1] / 255;
    out[2 * plane + i] = rgba[p + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-9);
}

/**
 * Decode raw YOLOv8 output [1, 4 + C, N] (channel-major) into detections.
 * Returns objects shaped like the server's detections (class_id, class_name,
 * confidence, bbox normalised to the original image, bbox_pixel).
 */
export function postprocess(output, lb, { conf = 0.5, iouThreshold = 0.7, numClasses = NUM_CLASSES } = {}) {
  const n = output.length / (4 + numClasses);
  const candidates = [];

  for (let i = 0; i < n; i++) {
    let best = -1;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = output[(4 + c) * n + i];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (bestScore < conf) continue;

    const cx = output[i];
    const cy = output[n + i];
    const w = output[2 * n + i];
    const h = output[3 * n + i];
    candidates.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score: bestScore, cls: best });
  }

  candidates.sort((a, b) => b.score - a.score);

  // Per-class NMS
  const kept = [];
  for (const c of candidates) {
    let ok = true;
    for (const k of kept) {
      if (k.cls === c.cls && iou(k.box, c.box) > iouThreshold) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push(c);
  }

  return kept.map(({ box, score, cls }) => {
    const x1 = clamp((box[0] - lb.padLeft) / lb.ratio, 0, lb.srcW);
    const y1 = clamp((box[1] - lb.padTop) / lb.ratio, 0, lb.srcH);
    const x2 = clamp((box[2] - lb.padLeft) / lb.ratio, 0, lb.srcW);
    const y2 = clamp((box[3] - lb.padTop) / lb.ratio, 0, lb.srcH);
    return {
      class_id: cls,
      class_name: CLASS_NAMES[cls] ?? `unknown_class_${cls}`,
      confidence: score,
      bbox: [x1 / lb.srcW, y1 / lb.srcH, x2 / lb.srcW, y2 / lb.srcH],
      bbox_pixel: [x1, y1, x2, y2],
    };
  });
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
