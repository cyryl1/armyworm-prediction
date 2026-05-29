import React, { useState, useRef, useEffect } from 'react';

interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  recommendation: string;
  recommendation_details?: string | null;
  gps_latitude?: number | null;
  gps_longitude?: number | null;
  detection_timestamp: string;
}

interface DetectionResponse {
  detections: Detection[];
  annotated_image: string | null;
}

interface DiagnosticHubProps {
  apiUrl: string;
  apiKey: string;
  checkServerHealth: () => void;
}

export default function DiagnosticHub({ apiUrl, apiKey, checkServerHealth }: DiagnosticHubProps) {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  
  // Webcam capture state
  const [showWebcam, setShowWebcam] = useState<boolean>(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // File Upload Handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const processFile = (file: File) => {
    // Revoke previous url if exists
    if (selectedImage && !selectedImage.startsWith('data:')) {
      URL.revokeObjectURL(selectedImage);
    }
    
    const localUrl = URL.createObjectURL(file);
    setSelectedImage(localUrl);
    setResult(null);
    analyzeCropImage(file);
  };

  // API Call to FastAPI
  const analyzeCropImage = async (file: File) => {
    setIsAnalyzing(true);
    console.log(`🔍 Diagnostic: Analyzing image "${file.name}" with API key: ${apiKey ? 'present' : 'missing'}`);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiUrl}/detect`, {
        method: 'POST',
        body: formData,
        headers: {
          'accept': 'application/json',
          ...(apiKey ? { 'x-api-key': apiKey } : {}),
        },
      });

      console.log(`📡 Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ API Error (${response.status}):`, errorText);
        throw new Error(`Server error (${response.status}): ${errorText || 'Unknown error'}`);
      }

      const data: DetectionResponse = await response.json();
      console.log(`✅ Diagnostic complete:`, data);
      setResult(data);
    } catch (error: any) {
      console.error('🚨 Diagnostic failed:', error);
      alert(`Diagnostic Failed: ${error?.message || 'Check your server connection & settings.'}`);
      setSelectedImage(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Webcam Handlers
  const startWebcam = async () => {
    setShowWebcam(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      setWebcamStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert('Camera access denied or unavailable.');
      setShowWebcam(false);
    }
  };

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
    }
    setWebcamStream(null);
    setShowWebcam(false);
  };

  const captureSnapshot = () => {
    if (!videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'captured_leaf.jpg', { type: 'image/jpeg' });
          processFile(file);
          stopWebcam();
        }
      }, 'image/jpeg', 0.92);
    }
  };

  const resetScanner = () => {
    if (selectedImage && !selectedImage.startsWith('data:')) {
      URL.revokeObjectURL(selectedImage);
    }
    setSelectedImage(null);
    setResult(null);
  };

  // Color mappings
  const getCategoryColor = (className: string) => {
    const name = className.toLowerCase();
    if (name.includes('healthy')) return '#10b981'; // Green
    if (name.includes('larva') || name.includes('egg')) return '#ef4444'; // Red
    if (name.includes('damage') || name.includes('frass')) return '#f97316'; // Orange
    return '#3b82f6'; // Blue
  };

  const getCategoryIcon = (className: string) => {
    const name = className.toLowerCase();
    if (name.includes('healthy')) return '🟢';
    if (name.includes('larva')) return '🐛';
    if (name.includes('egg')) return '🥚';
    if (name.includes('damage')) return '⚠️';
    return '🔍';
  };

  return (
    <div className="diagnostic-grid">
      {/* Left Pane - Image Preview & Interaction Canvas */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        {!selectedImage ? (
          <div
            className={`upload-zone ${dragActive ? 'dragging' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
          >
            <div className="upload-icon-circle">🌱</div>
            <h3 className="upload-title">Scan Maize Leaf</h3>
            <p className="upload-desc">
              Drag & drop a crop image, or capture directly using your device camera to diagnose pests.
            </p>
            <div className="btn-row">
              <button onClick={triggerFileInput} className="btn-primary">
                📁 Select Photo
              </button>
              <button onClick={startWebcam} className="btn-secondary">
                📷 Use Camera
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileInput}
              style={{ display: 'none' }}
              accept="image/*"
            />
          </div>
        ) : (
          <div>
            <div className="scanner-container">
              <div className="canvas-wrapper">
                <img
                  src={
                    result?.annotated_image
                      ? `data:image/jpeg;base64,${result.annotated_image}`
                      : selectedImage
                  }
                  className="display-canvas"
                  alt="Crop Scan Preview"
                />
                
                {/* Neon pulsing scanning bar */}
                {isAnalyzing && <div className="scanning-pulse" />}
                
                {/* Inference loading mask */}
                {isAnalyzing && (
                  <div className="analyzing-overlay">
                    <div className="spinner" />
                    <span style={{ fontSize: '15px', fontWeight: 'bold' }}>AI Diagnostics Processing...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Actions row under canvas */}
            {!isAnalyzing && (
              <div className="scanner-actions">
                <button onClick={resetScanner} className="btn-danger">
                  🗑️ Clear Scan
                </button>
                <button onClick={triggerFileInput} className="btn-primary">
                  🔄 Scan Another
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Pane - Results Info and Recommendations */}
      <div className="diagnostics-panel">
        <div className="glass-panel" style={{ minHeight: '400px' }}>
          <h3 style={{ fontSize: '18px', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '12px' }}>
            Diagnostic Results
          </h3>

          {!selectedImage && (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '60px 0' }}>
              <span style={{ fontSize: '40px', display: 'block', marginBottom: '16px' }}>📋</span>
              Upload an image to start crop pest and disease detection.
            </div>
          )}

          {selectedImage && isAnalyzing && (
            <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '60px 0' }}>
              <span style={{ fontSize: '32px', display: 'block', marginBottom: '16px', animation: 'pulse 1s infinite' }}>🧠</span>
              AI Model analyzing leaf patterns...
            </div>
          )}

          {!isAnalyzing && result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {result.detections.length === 0 ? (
                <div className="healthy-alert">
                  <div className="healthy-icon">✓</div>
                  <div>
                    <h4 className="alert-title">Crop Healthy!</h4>
                    <p className="alert-desc">
                      No fall armyworms, larvae, egg clusters, or streak disease symptoms detected in this maize leaf scan. Keep monitoring.
                    </p>
                  </div>
                </div>
              ) : (
                result.detections.map((d, i) => {
                  const color = getCategoryColor(d.class_name);
                  const icon = getCategoryIcon(d.class_name);
                  
                  return (
                    <div
                      key={i}
                      className="detection-card"
                      style={{ borderLeftColor: color }}
                    >
                      <div className="card-header">
                        <div className="class-title-group">
                          <span className="class-icon">{icon}</span>
                          <span className="class-name">{d.class_name.replace(/-/g, ' ')}</span>
                        </div>
                        <span
                          className="match-badge"
                          style={{ backgroundColor: `${color}25`, color: color }}
                        >
                          {Math.round(d.confidence * 100)}% Confidence
                        </span>
                      </div>

                      {/* Progress Confidence Bar */}
                      <div className="progress-container">
                        <div
                          className="progress-fill"
                          style={{ width: `${d.confidence * 100}%`, backgroundColor: color }}
                        />
                      </div>

                      {/* Reco box */}
                      <div className="reco-box">
                        <div className="reco-label">Management Recommendation:</div>
                        <div className="reco-text">{d.recommendation}</div>
                        {d.recommendation_details && (
                          <div className="reco-details">{d.recommendation_details}</div>
                        )}
                      </div>

                      {/* Metadata coordinate details */}
                      {(d.gps_latitude || d.gps_longitude) && (
                        <div className="meta-row">
                          <span>📍</span>
                          <span>
                            Coordinates: {d.gps_latitude?.toFixed(5)}, {d.gps_longitude?.toFixed(5)}
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>

      {/* Web Camera Snapshot Modal */}
      {showWebcam && (
        <div className="webcam-modal-backdrop">
          <div className="webcam-modal">
            <div className="webcam-header">
              <h3>Webcam Crop Capture</h3>
              <button onClick={stopWebcam} className="btn-secondary" style={{ padding: '6px 12px' }}>✕ Close</button>
            </div>
            <div className="webcam-viewport">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="webcam-video"
              />
            </div>
            <div className="webcam-footer">
              <button onClick={captureSnapshot} className="btn-primary" style={{ padding: '12px 30px' }}>
                📸 Capture Diagnostic Frame
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
