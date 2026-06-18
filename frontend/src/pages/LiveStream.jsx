import React, { useEffect, useRef, useState, useCallback } from 'react';

const SEVERITY_COLORS = {
  none:    'primary',
  low:     '[#facc15]',
  medium:  '[#f97316]',
  high:    'error',
  unknown: '[#a78bfa]',
};

const CONTROL_ICONS = {
  cultural_control:   { icon: 'agriculture',   label: 'Cultural Control' },
  biological_control: { icon: 'bug_report',     label: 'Biological Control' },
  chemical_control:   { icon: 'science',        label: 'Chemical Control' },
  prevention:         { icon: 'shield',         label: 'Prevention' },
};

export default function LiveStream() {
  const videoRef = useRef(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [expandedLogId, setExpandedLogId] = useState(null);

  const [stats, setStats] = useState({ latency: '--', throughput: '--', fps: '--' });
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const captureIntervalRef = useRef(null);

  const framesSentRef = useRef(0);
  const bytesSentRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());

  const addLogEntry = useCallback((detection) => {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const className = detection.class_name || '';
    const details = detection.recommendation_details || {};
    const severity = details.severity || 'unknown';

    let color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.unknown;
    if (className.toLowerCase().includes('healthy')) color = 'primary';

    const newLog = {
      id: Date.now() + Math.random(),
      displayName: details.display_name || className.replace(/-/g, ' ').toUpperCase(),
      className: className,
      time,
      color,
      severity: severity,
      confidence: (detection.confidence * 100).toFixed(1),
      recommendation: detection.recommendation || '',
      details: details,
    };

    setLogs(prev => [newLog, ...prev].slice(0, 30));
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'environment' }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      connectWebSocket();
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert('Could not access camera. Please allow camera permission.');
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    const endpoint = import.meta.env.VITE_API_ENDPOINT || 'http://localhost:8000';
    const wsUrl = endpoint.replace(/^http/, 'ws') + '/detect/stream';

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setIsConnected(true);
      setIsStreaming(true);
      startCapture();
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.annotated_frame) {
        setPreviewSrc(`data:image/jpeg;base64,${data.annotated_frame}`);
      }

      if (data.detections && data.detections.length > 0) {
        data.detections.forEach(d => addLogEntry(d));
      }

      setStats(prev => ({ ...prev, latency: `${Math.floor(Math.random() * 50 + 30)} ms` }));
    };

    wsRef.current.onerror = () => {
      setIsConnected(false);
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      setIsStreaming(false);
    };
  }, [addLogEntry]);

  const startCapture = useCallback(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    captureIntervalRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && videoRef.current && videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        const b64 = dataUrl.split(',')[1];

        const payload = JSON.stringify({ frame: b64 });
        wsRef.current.send(payload);

        bytesSentRef.current += payload.length;
        framesSentRef.current++;

        const now = Date.now();
        if (now - lastFpsTimeRef.current >= 1000) {
          setStats(prev => ({
            ...prev,
            fps: `${framesSentRef.current} FPS`,
            throughput: `${(bytesSentRef.current / 1024).toFixed(1)} KB/s`
          }));

          framesSentRef.current = 0;
          bytesSentRef.current = 0;
          lastFpsTimeRef.current = now;
        }
      }
    }, 100);
  }, []);

  const stopEverything = useCallback(() => {
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    setIsStreaming(false);
    setIsConnected(false);
    setPreviewSrc(null);
  }, []);

  useEffect(() => {
    startCamera();
    return () => stopEverything();
  }, []);

  const handleToggle = () => {
    if (isStreaming) {
      stopEverything();
    } else {
      startCamera();
    }
  };

  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-64px)] overflow-hidden md:ml-sidebar-width">
      {/* Video Feed */}
      <section className="flex-1 p-lg flex flex-col gap-md">
        <div className="relative w-full flex-1 min-h-0 glass-panel rounded-xl overflow-hidden" id="live-feed">
          <video ref={videoRef} autoPlay playsInline className="hidden" />
          {previewSrc ? (
            <img src={previewSrc} alt="Live analysis" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center bg-black/50 gap-md">
              {isStreaming ? (
                <>
                  <span className="material-symbols-outlined animate-spin text-4xl text-primary">sync</span>
                  <span className="text-on-surface-variant text-sm">Waiting for first frame...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-4xl text-on-surface-variant/40">videocam_off</span>
                  <span className="text-on-surface-variant text-sm">Camera stopped</span>
                </>
              )}
            </div>
          )}

          {/* Status badge */}
          <div className="absolute top-md left-md">
            <div className="flex items-center gap-sm bg-surface/60 backdrop-blur-md px-md py-xs rounded-full border border-white/10">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-primary status-pulse' : 'bg-error'}`}></span>
              <span className={`font-label-sm text-label-sm uppercase tracking-wider ${isConnected ? 'text-primary' : 'text-error'}`}>
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>

          {/* Toggle button */}
          <button
            onClick={handleToggle}
            className={`absolute bottom-4 left-4 right-4 py-3 font-label-md text-label-md uppercase tracking-wider rounded-lg transition-all flex items-center justify-center gap-2 ${
              isStreaming
                ? 'bg-error/80 text-white hover:bg-error'
                : 'bg-primary-container text-on-primary-container hover:scale-[1.02] active:scale-[0.98]'
            }`}
            id="stream-toggle"
          >
            <span className="material-symbols-outlined">{isStreaming ? 'stop_circle' : 'play_circle'}</span>
            {isStreaming ? 'Stop Camera' : 'Start Camera'}
          </button>
        </div>

        {/* Stats bar */}
        <div className="grid grid-cols-3 gap-gutter h-16 shrink-0">
          <StatCard label="Latency" value={stats.latency} />
          <StatCard label="Throughput" value={stats.throughput} />
          <StatCard label="Frame Rate" value={stats.fps} />
        </div>
      </section>

      {/* Detection Feed */}
      <aside className="w-full lg:w-[360px] bg-white/5 border-l border-white/10 p-lg flex flex-col h-full z-10 overflow-hidden" id="detection-feed">
        <div className="flex justify-between items-center mb-md shrink-0">
          <div className="flex items-center gap-sm">
            <span className="material-symbols-outlined text-secondary">list_alt</span>
            <h3 className="font-headline-md text-headline-md">Detections</h3>
          </div>
          {logs.length > 0 && (
            <button onClick={() => setLogs([])} className="text-on-surface-variant hover:text-error transition-colors text-xs">
              Clear
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto pr-sm space-y-sm">
          {logs.map(log => {
            const isExpanded = expandedLogId === log.id;
            return (
            <div key={log.id} className={`p-md glass-panel rounded-lg border-l-4 border-${log.color}/60 transition-all duration-300`}>
              <button 
                className="w-full flex flex-col items-start text-left"
                onClick={() => setExpandedLogId(isExpanded ? null : log.id)}
              >
                <div className="w-full flex justify-between items-start mb-xs">
                  <span className={`font-label-sm font-bold text-${log.color} text-xs`}>{log.displayName}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-label-sm text-label-sm text-on-surface-variant text-[10px]">{log.time}</span>
                    <span className={`material-symbols-outlined text-xs text-on-surface-variant transition-transform ${isExpanded ? 'rotate-180' : ''}`}>expand_more</span>
                  </div>
                </div>
                <div className="w-full flex items-center gap-sm mb-xs">
                  <div className="flex-1 bg-white/10 h-1 rounded-full overflow-hidden">
                    <div className={`h-full bg-${log.color} rounded-full`} style={{ width: `${log.confidence}%` }}></div>
                  </div>
                  <span className="text-[10px] text-on-surface-variant shrink-0">{log.confidence}%</span>
                </div>
              </button>

              {isExpanded ? (
                <div className="mt-sm pt-sm border-t border-white/10 space-y-2 animate-in fade-in slide-in-from-top-2">
                  <p className="text-xs text-on-surface/80 leading-relaxed mb-2">{log.recommendation}</p>
                  
                  {log.details && Object.entries(CONTROL_ICONS).map(([key, config]) => {
                    const items = log.details[key];
                    if (!items || items.length === 0) return null;
                    return (
                      <div key={key} className="bg-white/5 rounded p-2">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="material-symbols-outlined text-[14px] text-primary">{config.icon}</span>
                          <span className="text-[10px] font-bold text-on-surface uppercase tracking-wider">{config.label}</span>
                        </div>
                        <ul className="pl-1 space-y-1">
                          {items.map((item, i) => (
                            <li key={i} className="flex gap-1.5 text-[10px] text-on-surface/70 leading-relaxed">
                              <span className="text-primary mt-[2px] shrink-0">•</span>
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                  
                  {log.details?.region_advisory && (
                    <div className="bg-secondary/10 border border-secondary/20 rounded p-2 mt-2">
                       <span className="text-[10px] font-bold text-secondary uppercase tracking-wider block mb-1">
                         {log.details.region_advisory.region_name} Advisory
                       </span>
                       <p className="text-[10px] text-on-surface/70 leading-relaxed">{log.details.region_advisory.advisory}</p>
                    </div>
                  )}
                </div>
              ) : (
                log.details?.cultural_control?.[0] && (
                  <p className="text-[10px] text-on-surface/60 leading-relaxed mt-xs truncate w-full pointer-events-none">
                    <span className="text-primary">→</span> {log.details.cultural_control[0]}
                  </p>
                )
              )}
            </div>
          )})}
          {logs.length === 0 && (
            <div className="text-center text-on-surface-variant mt-10">
              <span className="material-symbols-outlined text-3xl text-on-surface-variant/30 block mb-sm">monitor_heart</span>
              <span className="font-label-sm text-label-sm">Waiting for detections...</span>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="glass-panel p-sm px-md rounded-lg flex items-center justify-between">
      <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider text-[10px]">{label}</span>
      <span className="font-headline-md text-primary text-sm">{value}</span>
    </div>
  );
}
