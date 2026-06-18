import React, { useState, useRef, useEffect } from 'react';

const SEVERITY_CONFIG = {
  none:    { color: 'bg-primary',    text: 'text-primary',    border: 'border-primary/30',    label: 'HEALTHY' },
  low:     { color: 'bg-[#facc15]',  text: 'text-[#facc15]',  border: 'border-[#facc15]/30',  label: 'LOW RISK' },
  medium:  { color: 'bg-[#f97316]',  text: 'text-[#f97316]',  border: 'border-[#f97316]/30',  label: 'MEDIUM RISK' },
  high:    { color: 'bg-error',      text: 'text-error',      border: 'border-error/30',      label: 'HIGH RISK' },
  unknown: { color: 'bg-[#a78bfa]',  text: 'text-[#a78bfa]',  border: 'border-[#a78bfa]/30',  label: 'REVIEW' },
};

const CONTROL_ICONS = {
  cultural_control:   { icon: 'agriculture',   label: 'Cultural Control' },
  biological_control: { icon: 'bug_report',     label: 'Biological Control' },
  chemical_control:   { icon: 'science',        label: 'Chemical Control' },
  prevention:         { icon: 'shield',         label: 'Prevention' },
};

export default function DiagnosticHub() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [expandedCard, setExpandedCard] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});
  const [showTutorial, setShowTutorial] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const hasSeen = localStorage.getItem('hasSeenScannerTutorial');
    if (!hasSeen) {
      setShowTutorial(true);
    }
  }, []);

  const closeTutorial = () => {
    localStorage.setItem('hasSeenScannerTutorial', 'true');
    setShowTutorial(false);
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedImage(URL.createObjectURL(file));
    setAnnotatedImage(null);
    setResults(null);
    setIsProcessing(true);
    setExpandedCard(null);
    setExpandedSections({});

    const formData = new FormData();
    formData.append('file', file);

    const endpoint = localStorage.getItem('apiEndpoint') || 'http://localhost:8000';

    try {
      const res = await fetch(`${endpoint}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();

      if (data.annotated_image) {
        setAnnotatedImage(`data:image/jpeg;base64,${data.annotated_image}`);
      }
      setResults(data);
      // Auto-expand first detection
      if (data.detections?.length > 0) setExpandedCard(0);
    } catch (err) {
      console.error(err);
      alert('Detection failed. Make sure the backend is running at ' + endpoint);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-primary/60', 'bg-primary/5');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-primary/60', 'bg-primary/5');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-primary/60', 'bg-primary/5');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const toggleSection = (cardIdx, sectionKey) => {
    const key = `${cardIdx}-${sectionKey}`;
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="p-lg md:p-xl md:ml-sidebar-width flex flex-col lg:flex-row gap-xl max-w-[1600px] mx-auto relative">
      {/* Tutorial Modal */}
      {showTutorial && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-[#121212] border border-white/10 rounded-2xl max-w-md w-full p-xl shadow-2xl animate-in zoom-in-95 duration-300">
            <div className="flex items-center gap-3 mb-6">
              <span className="material-symbols-outlined text-primary text-3xl">info</span>
              <h2 className="font-headline-md text-2xl font-bold text-on-surface">For Best Results</h2>
            </div>
            
            <div className="space-y-6 mb-8">
              <div className="flex gap-4 items-start">
                <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-primary">check_circle</span>
                </div>
                <div>
                  <h3 className="font-bold text-on-surface mb-1 text-lg">DO: Get Close</h3>
                  <p className="text-on-surface-variant text-sm">Get very close (10-20 cm) to the damaged leaf or pest. The damage should fill the screen.</p>
                </div>
              </div>
              
              <div className="flex gap-4 items-start">
                <div className="w-12 h-12 rounded-full bg-error/20 flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-error">cancel</span>
                </div>
                <div>
                  <h3 className="font-bold text-on-surface mb-1 text-lg">DON'T: Take Wide Shots</h3>
                  <p className="text-on-surface-variant text-sm">Do not take wide pictures of the whole farm or field. The AI cannot see tiny details from far away.</p>
                </div>
              </div>
            </div>
            
            <button 
              onClick={closeTutorial}
              className="w-full py-3 px-4 bg-primary text-on-primary rounded-xl font-bold hover:bg-primary/90 transition-colors"
            >
              Got it, let's scan!
            </button>
          </div>
        </div>
      )}

      {/* Upload & Preview Area */}
      <section className="flex-1 flex flex-col gap-md">
        <div className="mb-sm">
          <h2 className="font-headline-lg text-headline-lg font-semibold text-on-surface mb-xs">Capture & Identify</h2>
          <p className="text-on-surface-variant font-body-md">Upload a photo of your maize to detect fall armyworm and get management advice.</p>
        </div>

        <div
          className="relative w-full aspect-[4/3] rounded-2xl border-2 border-dashed border-white/20 glass-panel flex flex-col items-center justify-center transition-all duration-300 overflow-hidden cursor-pointer hover:border-primary/40 hover:bg-white/5"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          id="capture-dropzone"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            capture="environment"
            className="hidden"
            id="capture-input"
          />

          {!selectedImage && !annotatedImage && (
            <div className="text-center p-lg pointer-events-none">
              <div className="w-20 h-20 mx-auto mb-lg rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <span className="material-symbols-outlined text-primary text-4xl">photo_camera</span>
              </div>
              <h3 className="font-headline-md text-headline-md font-bold text-on-surface mb-sm">Take or Upload Photo</h3>
              <p className="text-on-surface-variant text-sm max-w-xs mx-auto">Snap a photo of the affected leaf or drag an image here to begin analysis.</p>
            </div>
          )}

          {(selectedImage || annotatedImage) && (
            <img
              src={annotatedImage || selectedImage}
              alt="Analysis preview"
              className={`w-full h-full object-contain ${isProcessing ? 'opacity-50 blur-sm' : ''}`}
            />
          )}

          {isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="flex flex-col items-center gap-4 bg-surface/80 p-6 rounded-xl backdrop-blur-md">
                <span className="material-symbols-outlined text-primary text-5xl animate-spin">sync</span>
                <span className="font-headline-md text-primary animate-pulse">Analysing...</span>
              </div>
            </div>
          )}
        </div>

        {/* GPS Info if available */}
        {results?.detections?.[0]?.gps_latitude && (
          <div className="flex items-center gap-sm text-on-surface-variant font-label-sm text-label-sm glass-panel rounded-lg px-md py-sm">
            <span className="material-symbols-outlined text-sm">location_on</span>
            <span>GPS: {results.detections[0].gps_latitude.toFixed(4)}°, {results.detections[0].gps_longitude.toFixed(4)}°</span>
          </div>
        )}
      </section>

      {/* Results & Management Panel */}
      <aside className="w-full lg:w-[460px] flex flex-col gap-md" id="results-panel">
        <div className="flex items-center gap-sm mb-sm">
          <span className="material-symbols-outlined text-secondary">biotech</span>
          <h3 className="font-headline-md text-headline-md text-on-surface">Analysis Results</h3>
        </div>

        {isProcessing ? (
          <div className="glass-panel rounded-xl p-lg space-y-4 animate-pulse">
            <div className="h-4 bg-white/10 rounded w-3/4"></div>
            <div className="h-4 bg-white/10 rounded w-1/2"></div>
            <div className="h-32 bg-white/5 rounded-lg mt-4"></div>
          </div>
        ) : results ? (
          <div className="space-y-md overflow-y-auto max-h-[calc(100vh-200px)] pr-1">
            {results.detections && results.detections.length > 0 ? (
              results.detections.map((det, idx) => {
                const details = det.recommendation_details || {};
                const severity = SEVERITY_CONFIG[details.severity] || SEVERITY_CONFIG.unknown;
                const isExpanded = expandedCard === idx;

                return (
                  <div
                    key={idx}
                    className={`glass-panel rounded-xl overflow-hidden transition-all duration-300 ${severity.border} border ${isExpanded ? 'ring-1 ring-white/10' : ''}`}
                    id={`detection-card-${idx}`}
                  >
                    {/* Card Header — always visible */}
                    <button
                      onClick={() => setExpandedCard(isExpanded ? null : idx)}
                      className="w-full p-md flex items-start gap-md text-left hover:bg-white/5 transition-colors"
                    >
                      <div className={`w-3 h-3 rounded-full mt-1.5 shrink-0 ${severity.color}`}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-sm mb-xs">
                          <h4 className={`font-headline-md text-sm font-bold ${severity.text} truncate`}>
                            {details.display_name || det.class_name.replace(/-/g, ' ').toUpperCase()}
                          </h4>
                          <span className={`font-label-sm text-label-sm px-2 py-0.5 rounded-full ${severity.color}/20 ${severity.text} shrink-0`}>
                            {severity.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-md mb-xs">
                          <div className="flex-1 bg-white/10 h-1.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${severity.color}`}
                              style={{ width: `${det.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="font-label-sm text-label-sm text-on-surface-variant shrink-0">
                            {(det.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-on-surface-variant text-xs leading-relaxed line-clamp-2">
                          {details.description || det.recommendation}
                        </p>
                      </div>
                      <span className={`material-symbols-outlined text-on-surface-variant transition-transform duration-300 mt-1 ${isExpanded ? 'rotate-180' : ''}`}>
                        expand_more
                      </span>
                    </button>

                    {/* Expandable Management Details */}
                    {isExpanded && (
                      <div className="border-t border-white/10 p-md space-y-sm animate-in fade-in slide-in-from-top-2 duration-300">
                        {Object.entries(CONTROL_ICONS).map(([key, config]) => {
                          const items = details[key];
                          if (!items || items.length === 0) return null;
                          const sectionKey = `${idx}-${key}`;
                          const isSectionOpen = expandedSections[sectionKey] !== false; // default open

                          return (
                            <div key={key} className="rounded-lg bg-white/3 border border-white/5 overflow-hidden">
                              <button
                                onClick={(e) => { e.stopPropagation(); toggleSection(idx, key); }}
                                className="w-full flex items-center gap-sm p-sm px-md hover:bg-white/5 transition-colors"
                              >
                                <span className="material-symbols-outlined text-sm text-primary">{config.icon}</span>
                                <span className="font-label-md text-label-md text-on-surface flex-1 text-left">{config.label}</span>
                                <span className={`material-symbols-outlined text-on-surface-variant text-sm transition-transform duration-200 ${isSectionOpen ? 'rotate-180' : ''}`}>
                                  expand_more
                                </span>
                              </button>
                              {isSectionOpen && (
                                <ul className="px-md pb-sm space-y-xs">
                                  {items.map((item, i) => (
                                    <li key={i} className="flex gap-sm text-xs text-on-surface/85 leading-relaxed">
                                      <span className="text-primary mt-0.5 shrink-0">•</span>
                                      <span>{item}</span>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          );
                        })}

                        {/* Region Advisory */}
                        {details.region_advisory && (
                          <div className="mt-sm p-sm px-md rounded-lg bg-secondary/5 border border-secondary/20">
                            <div className="flex items-center gap-xs mb-xs">
                              <span className="material-symbols-outlined text-secondary text-sm">public</span>
                              <span className="font-label-sm text-label-sm text-secondary">
                                {details.region_advisory.region_name} Advisory
                              </span>
                            </div>
                            <p className="text-xs text-on-surface/70 leading-relaxed">{details.region_advisory.advisory}</p>
                          </div>
                        )}

                        {/* Sources */}
                        {details.sources && details.sources.length > 0 && (
                          <div className="pt-sm border-t border-white/5">
                            <span className="font-label-sm text-label-sm text-on-surface-variant block mb-xs">Sources</span>
                            {details.sources.map((src, i) => (
                              <p key={i} className="text-[10px] text-on-surface-variant/60 leading-relaxed">— {src}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="glass-panel rounded-xl p-lg text-center border border-primary/20">
                <span className="material-symbols-outlined text-primary text-4xl mb-sm block">search</span>
                <p className="font-headline-md text-primary mb-xs">No Pests Detected</p>
                <p className="text-on-surface-variant text-sm mb-4">Make sure you are standing very close to the plant (10-20cm) and the image is not blurry.</p>
                <p className="text-on-surface-variant text-sm font-semibold">If your photo was a wide shot of the field, please try again closer! Otherwise, continue routine scouting.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="glass-panel rounded-xl p-xl flex flex-col items-center justify-center text-center min-h-[300px]">
            <span className="material-symbols-outlined text-4xl text-on-surface-variant/40 mb-md">data_usage</span>
            <p className="font-body-md text-on-surface-variant">Upload a photo to begin analysis.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
