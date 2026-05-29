import React, { useState, useRef, useEffect } from 'react';

interface LiveStreamProps {
  apiUrl: string;
  apiKey: string;
}

export default function LiveStream({ apiUrl, apiKey }: LiveStreamProps) {
  const [connected, setConnected] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [throttleInterval, setThrottleInterval] = useState<number>(600); // ms between frames
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<any>(null);

  // Convert HTTP API url to WS url
  const getWsUrl = () => {
    const cleanUrl = apiUrl.trim().replace(/\/+$/, '');
    const wsBase = cleanUrl.replace(/^https:\/\//i, 'wss://').replace(/^http:\/\//i, 'ws://');
    const wsUrl = wsBase.endsWith('/detect/stream') ? wsBase : `${wsBase}/detect/stream`;
    
    if (apiKey) {
      const separator = wsUrl.includes('?') ? '&' : '?';
      return `${wsUrl}${separator}api_key=${encodeURIComponent(apiKey)}`;
    }
    return wsUrl;
  };

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'environment' }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsStreaming(true);
      connectWebSocket();
    } catch (err) {
      alert('Failed to launch webcam. Please check your camera permissions.');
    }
  };

  const stopStream = () => {
    setIsStreaming(false);
    
    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    // Stop timers
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    // Close WS
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    setAnnotatedFrame(null);
    setDetections([]);
  };

  const connectWebSocket = () => {
    try {
      const wsUrl = getWsUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        // Start streaming frames
        startFramePipeline();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.annotated_frame) {
            setAnnotatedFrame(data.annotated_frame);
          }
          if (data.detections) {
            setDetections(data.detections);
          }
        } catch (e) {
          console.error('Error parsing WS frame', e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        stopStream();
      };

      ws.onerror = (e) => {
        console.error('WS stream error', e);
        setConnected(false);
        stopStream();
      };
    } catch (err) {
      alert('WebSocket streaming connection failed.');
      stopStream();
    }
  };

  const startFramePipeline = () => {
    if (timerRef.current) clearInterval(timerRef.current);

    timerRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && videoRef.current) {
        sendFrame();
      }
    }, throttleInterval);
  };

  const sendFrame = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = 480; // Downscale frame slightly for better performance
    canvas.height = 360;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
      const base64Content = dataUrl.split(',')[1];
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ frame: base64Content }));
      }
    }
  };

  // Re-run pipeline if interval changes
  useEffect(() => {
    if (isStreaming && connected) {
      startFramePipeline();
    }
  }, [throttleInterval]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="stream-panel">
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div className="stream-controls">
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            {!isStreaming ? (
              <button onClick={startStream} className="btn-primary">
                🎥 Start Live Stream
              </button>
            ) : (
              <button onClick={stopStream} className="btn-danger">
                🛑 Stop Stream
              </button>
            )}
            
            <div className="server-status-badge" style={{ display: isStreaming ? 'flex' : 'none' }}>
              <div className={`status-dot ${connected ? 'online' : 'offline'}`} />
              <span className="status-label">{connected ? 'Streaming Pipeline Active' : 'Connecting WebSocket...'}</span>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Processing Speed:</span>
            <select
              value={throttleInterval}
              onChange={(e) => setThrottleInterval(Number(e.target.value))}
              className="text-input"
              style={{ padding: '6px 12px', background: 'var(--bg-primary)', border: '1px solid var(--border-light)', borderRadius: '8px' }}
              disabled={isStreaming && !connected}
            >
              <option value={300}>High-Speed (300ms)</option>
              <option value={600}>Normal (600ms)</option>
              <option value={1000}>Eco-Throttled (1000ms)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="stream-screen-split">
        {/* Left Side - Client Video Source */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '15px', marginBottom: '12px' }}>Camera Capture Source</h3>
          <div className="stream-view">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="webcam-video"
              style={{ transform: 'scaleX(-1)' }} // Mirror video locally
            />
            {!isStreaming && (
              <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                Camera stream is inactive. Click "Start Live Stream".
              </div>
            )}
            {isStreaming && (
              <div className="stream-live-indicator">
                <span style={{ display: 'inline-block', width: '6px', height: '6px', background: '#fff', borderRadius: '50%' }} />
                SOURCE RAW
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Processed Bounding Frames */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '15px', marginBottom: '12px' }}>Annotated AI Stream</h3>
          <div className="stream-view" style={{ borderColor: annotatedFrame ? 'var(--color-green)' : 'var(--border-light)' }}>
            {annotatedFrame ? (
              <img
                src={`data:image/jpeg;base64,${annotatedFrame}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                alt="AI stream detections"
              />
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '0 20px' }}>
                {isStreaming
                  ? 'Connecting websocket... awaiting first annotated frame.'
                  : 'Webcam crop AI annotations will display here in real-time.'}
              </div>
            )}
            
            {annotatedFrame && (
              <div className="stream-live-indicator" style={{ backgroundColor: 'var(--color-green)', boxShadow: '0 0 10px rgba(16,185,129,0.4)' }}>
                🌽 LIVE DIAGNOSING
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Real-time detections list beneath cameras */}
      {detections.length > 0 && (
        <div className="glass-panel">
          <h3 style={{ fontSize: '16px', marginBottom: '16px' }}>Real-time Detection Classes ({detections.length})</h3>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {detections.map((d, idx) => {
              const name = d.class_name || 'pest';
              const isHealthy = name.toLowerCase().includes('healthy');
              const color = isHealthy ? 'var(--color-green)' : 'var(--color-red)';
              
              return (
                <span
                  key={idx}
                  className="match-badge"
                  style={{
                    backgroundColor: `${color}15`,
                    color: color,
                    border: `1px solid ${color}30`,
                    padding: '8px 14px',
                    fontSize: '13px',
                    borderRadius: '10px',
                    fontWeight: '600',
                    textTransform: 'capitalize'
                  }}
                >
                  {isHealthy ? '🟢' : '🐛'} {name.replace(/-/g, ' ')} ({Math.round(d.confidence * 100)}%)
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
