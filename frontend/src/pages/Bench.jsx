import { useRef, useState } from 'react';
import { loadModel, detect, ort } from '../edge/detector';

const RUNS = 30;

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const p95 = (a) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * 0.95))];
const fmt = (n) => n.toFixed(1);

/** Spike: measure in-browser inference speed on this device. Route: /bench */
export default function Bench() {
  const [backend, setBackend] = useState('wasm');
  const [imgSrc, setImgSrc] = useState(null);
  const [log, setLog] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const say = (m) => setLog((l) => [...l, m]);

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (f) setImgSrc(URL.createObjectURL(f));
  };

  const run = async () => {
    setBusy(true);
    setResult(null);
    setLog([]);
    try {
      const img = imgRef.current;
      if (!img?.naturalWidth) throw new Error('Pick an image first');
      const w = img.naturalWidth;
      const h = img.naturalHeight;

      say(`Loading model on ${backend}...`);
      const { session, bytes, fetchMs, initMs } = await loadModel(backend);
      say(`Model ${(bytes / 1e6).toFixed(1)} MB: fetch ${fmt(fetchMs)} ms, init ${fmt(initMs)} ms`);

      say('Warm-up...');
      await detect(session, img, w, h);

      const t = { pre: [], infer: [], post: [], total: [] };
      let last;
      for (let i = 0; i < RUNS; i++) {
        last = await detect(session, img, w, h);
        for (const k of Object.keys(t)) t[k].push(last.timings[k]);
      }

      const stats = Object.fromEntries(Object.entries(t).map(([k, v]) => [k, { med: median(v), p95: p95(v) }]));
      setResult({ backend, stats, detections: last.detections, bytes, fetchMs, initMs });
      drawBoxes(last.detections, img);
      say('Done.');
    } catch (err) {
      say(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  const drawBoxes = (dets, img) => {
    const c = canvasRef.current;
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = Math.max(2, c.width / 300);
    ctx.font = `${Math.max(14, c.width / 40)}px sans-serif`;
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.bbox_pixel;
      ctx.strokeStyle = '#4ade80';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = '#4ade80';
      ctx.fillText(`${d.class_name} ${(d.confidence * 100).toFixed(0)}%`, x1 + 4, y1 + 18);
    }
  };

  return (
    <div className="p-lg md:ml-sidebar-width space-y-md text-on-surface">
      <h2 className="text-xl font-bold">Edge inference benchmark</h2>
      <div className="text-xs space-y-1 glass-panel rounded-lg p-md">
        <div>onnxruntime-web {ort.env.versions?.web}</div>
        <div>WebGPU available: {String(!!navigator.gpu)}</div>
        <div>crossOriginIsolated (WASM threads): {String(window.crossOriginIsolated)}</div>
        <div>hardwareConcurrency: {navigator.hardwareConcurrency}</div>
        <div className="break-all">UA: {navigator.userAgent}</div>
      </div>

      <div className="flex flex-wrap gap-md items-center">
        <input type="file" accept="image/*" onChange={onFile} />
        <select value={backend} onChange={(e) => setBackend(e.target.value)} className="bg-surface-variant p-2 rounded">
          <option value="wasm">WASM (CPU)</option>
          <option value="webgpu">WebGPU</option>
        </select>
        <button onClick={run} disabled={busy || !imgSrc} className="px-md py-2 rounded bg-primary-container text-on-primary-container disabled:opacity-50">
          {busy ? 'Running...' : `Run ${RUNS} frames`}
        </button>
      </div>

      {imgSrc && <img ref={imgRef} src={imgSrc} alt="" className="hidden" />}
      <canvas ref={canvasRef} className="w-full max-w-xl rounded-lg" />

      {log.length > 0 && <pre className="text-xs whitespace-pre-wrap">{log.join('\n')}</pre>}

      {result && (
        <table className="text-sm">
          <thead><tr><th className="pr-md text-left">Stage</th><th className="pr-md text-left">median ms</th><th className="text-left">p95 ms</th></tr></thead>
          <tbody>
            {['pre', 'infer', 'post', 'total'].map((k) => (
              <tr key={k}><td className="pr-md">{k}</td><td className="pr-md">{fmt(result.stats[k].med)}</td><td>{fmt(result.stats[k].p95)}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      {result && (
        <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(result.detections.map((d) => ({ c: d.class_name, p: +d.confidence.toFixed(3) })), null, 1)}</pre>
      )}
    </div>
  );
}
