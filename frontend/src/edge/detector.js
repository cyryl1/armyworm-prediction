// Browser-side detector: loads the ONNX model with onnxruntime-web and runs
// letterbox -> inference -> NMS. Spike code; see src/pages/Bench.jsx.
import * as ort from 'onnxruntime-web/webgpu';
import { INPUT_SIZE, letterboxParams, rgbaToTensor, postprocess } from './yolo';

ort.env.wasm.wasmPaths = '/ort/';

export const MODEL_URL = '/models/best.onnx';

/** Load the model, returning a session plus the time spent fetching vs. initialising. */
export async function loadModel(backend = 'wasm') {
  const t0 = performance.now();
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`Model fetch failed: ${res.status}`);
  const buf = await res.arrayBuffer();
  const t1 = performance.now();

  const session = await ort.InferenceSession.create(buf, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  const t2 = performance.now();
  return { session, bytes: buf.byteLength, fetchMs: t1 - t0, initMs: t2 - t1 };
}

let scratch = null;
function getScratch() {
  if (!scratch) {
    scratch = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
      : Object.assign(document.createElement('canvas'), { width: INPUT_SIZE, height: INPUT_SIZE });
  }
  return scratch;
}

/** Letterbox a drawable (video/img/canvas/bitmap) into a 640x640 tensor. */
export function preprocess(source, srcW, srcH) {
  const lb = letterboxParams(srcW, srcH);
  const canvas = getScratch();
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, lb.padLeft, lb.padTop, lb.newW, lb.newH);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  return { tensor: rgbaToTensor(data), lb };
}

/** Run one frame; returns detections plus per-stage timings in ms. */
export async function detect(session, source, srcW, srcH, opts) {
  const t0 = performance.now();
  const { tensor, lb } = preprocess(source, srcW, srcH);
  const t1 = performance.now();

  const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const data = out[session.outputNames[0]].data;
  const t2 = performance.now();

  const detections = postprocess(data, lb, opts);
  const t3 = performance.now();

  return { detections, timings: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 } };
}

export { ort };
