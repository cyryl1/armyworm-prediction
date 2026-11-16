import { useState, useRef } from 'react';

const THRESHOLD = 20; // 20% infestation threshold

export default function ScoutingMode() {
  const [sessionActive, setSessionActive] = useState(false);
  const [scans, setScans] = useState({ total: 0, healthy: 0, infested: 0, pests: {} });
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [sessionSaved, setSessionSaved] = useState(false);
  
  const fileInputRef = useRef(null);
  const endpoint = import.meta.env.VITE_API_ENDPOINT || 'http://localhost:8000';

  const infestationRate = scans.total === 0 ? 0 : Math.round((scans.infested / scans.total) * 100);
  const isOverThreshold = infestationRate >= THRESHOLD;

  const startSession = () => {
    setScans({ total: 0, healthy: 0, infested: 0, pests: {} });
    setSessionActive(true);
    setSessionSaved(false);
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setCurrentImage(URL.createObjectURL(file));
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${endpoint}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      
      let isHealthy = true;
      let primaryPest = null;

      if (data.detections && data.detections.length > 0) {
        // Check if any detection is not healthy
        const pests = data.detections.filter(d => !d.class_name.toLowerCase().includes('healthy'));
        if (pests.length > 0) {
          isHealthy = false;
          primaryPest = pests[0]; // Just take the highest confidence one
        }
      }

      setScans(prev => {
        const newPests = { ...prev.pests };
        if (primaryPest) {
          const name = primaryPest.recommendation_details?.display_name || primaryPest.class_name;
          newPests[name] = (newPests[name] || 0) + 1;
        }
        
        return {
          total: prev.total + 1,
          healthy: prev.healthy + (isHealthy ? 1 : 0),
          infested: prev.infested + (isHealthy ? 0 : 1),
          pests: newPests
        };
      });

    } catch (err) {
      console.error(err);
      alert('Detection failed. Ensure backend is running.');
    } finally {
      setIsProcessing(false);
      // Optional: keep image briefly then clear, or keep latest
      setTimeout(() => setCurrentImage(null), 1500);
    }
  };

  const finishSession = async () => {
    try {
      const primaryRec = Object.keys(scans.pests).length > 0 
        ? Object.keys(scans.pests).reduce((a, b) => scans.pests[a] > scans.pests[b] ? a : b) 
        : "None";

      const res = await fetch(`${endpoint}/scouting-sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_plants: scans.total,
          healthy_plants: scans.healthy,
          infested_plants: scans.infested,
          primary_recommendation: primaryRec
        }),
      });

      if (!res.ok) throw new Error("Failed to save session");
      
      setSessionActive(false);
      setSessionSaved(true);
    } catch (err) {
      console.error(err);
      alert('Could not save session. Please try again.');
    }
  };

  return (
    <div className="p-lg md:p-xl md:ml-sidebar-width max-w-[1200px] mx-auto min-h-[calc(100vh-64px)] flex flex-col items-center">
      
      <div className="text-center mb-xl w-full">
        <h2 className="font-headline-lg text-2xl md:text-4xl font-bold text-on-surface mb-xs">Field Scouting</h2>
        <p className="text-on-surface-variant text-base md:text-lg">Track field-level action thresholds for Integrated Pest Management.</p>
      </div>

      {!sessionActive && !sessionSaved && (
        <div className="glass-panel p-xl rounded-3xl flex flex-col items-center text-center max-w-lg w-full animate-in zoom-in-95 duration-500">
          <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-md border-2 border-primary/20">
            <span className="material-symbols-outlined text-primary text-5xl">explore</span>
          </div>
          <h3 className="text-xl md:text-2xl font-bold mb-sm">Start a Scouting Session</h3>
          <p className="text-on-surface-variant text-sm md:text-base mb-xl">
            Walk your field in a "W" pattern. Scan at least 50 random plants. The app will calculate the infestation percentage and alert you if chemical/biological intervention is required.
          </p>
          <button 
            onClick={startSession}
            className="w-full py-4 bg-primary text-on-primary font-bold text-lg rounded-xl hover:bg-primary/90 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_20px_rgba(var(--color-primary),0.3)]"
          >
            Start Scouting
          </button>
        </div>
      )}

      {sessionActive && (
        <div className="w-full flex flex-col lg:flex-row gap-xl animate-in fade-in duration-500">
          {/* Main Action Area */}
          <div className="flex-1 flex flex-col items-center">
            
            {/* Dynamic Alert Banner */}
            <div className={`w-full mb-lg rounded-2xl p-md flex items-center gap-md transition-all duration-500 border ${
              scans.total === 0 ? 'bg-surface/50 border-white/5' :
              isOverThreshold ? 'bg-error/10 border-error/30' : 'bg-primary/10 border-primary/30'
            }`}>
              <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 ${
                scans.total === 0 ? 'bg-white/10' :
                isOverThreshold ? 'bg-error/20' : 'bg-primary/20'
              }`}>
                <span className={`material-symbols-outlined text-2xl ${
                  scans.total === 0 ? 'text-on-surface-variant' :
                  isOverThreshold ? 'text-error' : 'text-primary'
                }`}>
                  {scans.total === 0 ? 'info' : isOverThreshold ? 'warning' : 'verified'}
                </span>
              </div>
              <div>
                <h4 className={`font-bold text-base md:text-lg ${
                  scans.total === 0 ? 'text-on-surface-variant' :
                  isOverThreshold ? 'text-error' : 'text-primary'
                }`}>
                  {scans.total === 0 ? 'Waiting for first scan...' : 
                   isOverThreshold ? `ACTION REQUIRED: Threshold Exceeded (> ${THRESHOLD}%)` : 
                   'Below Action Threshold'}
                </h4>
                <p className="text-on-surface-variant text-xs md:text-sm">
                  {scans.total === 0 ? 'Scan random plants across the field.' :
                   isOverThreshold ? 'Intervention (Biological/Chemical) is justified to prevent economic loss.' :
                   'Continue routine monitoring. No chemical intervention needed.'}
                </p>
              </div>
            </div>

            {/* Camera / Capture Button */}
            <div 
              className={`relative w-full max-w-md aspect-square rounded-3xl border-2 border-dashed glass-panel flex flex-col items-center justify-center transition-all duration-300 overflow-hidden cursor-pointer hover:bg-white/5 ${
                isProcessing ? 'border-primary/60 bg-primary/5' : 'border-white/20'
              }`}
              onClick={() => !isProcessing && fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                capture="environment"
                className="hidden"
              />

              {currentImage ? (
                <img src={currentImage} alt="Current scan" className="w-full h-full object-cover opacity-60" />
              ) : (
                <div className="text-center pointer-events-none p-lg">
                  <span className="material-symbols-outlined text-5xl md:text-6xl text-primary/60 mb-sm">add_a_photo</span>
                  <p className="font-bold text-lg md:text-xl text-on-surface">Tap to Scan Plant</p>
                  <p className="text-on-surface-variant text-xs md:text-sm mt-2">Scan {Math.max(0, 50 - scans.total)} more for high accuracy</p>
                </div>
              )}

              {isProcessing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                  <div className="flex flex-col items-center">
                    <span className="material-symbols-outlined text-primary text-5xl animate-spin mb-2">sync</span>
                    <span className="font-bold text-primary animate-pulse tracking-widest">ANALYSING</span>
                  </div>
                </div>
              )}
            </div>

          </div>

          {/* Stats Panel */}
          <aside className="w-full lg:w-[400px] flex flex-col gap-md">
            
            <div className="glass-panel p-xl rounded-3xl flex flex-col items-center justify-center relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-white/10">
                <div className={`h-full transition-all duration-1000 ${isOverThreshold ? 'bg-error' : 'bg-primary'}`} style={{ width: `${infestationRate}%` }}></div>
              </div>
              
              <p className="text-on-surface-variant uppercase tracking-widest text-xs md:text-sm font-bold mb-2">Infestation Rate</p>
              <div className="flex items-baseline gap-1 mb-6">
                <span className={`text-6xl md:text-7xl font-black ${isOverThreshold ? 'text-error' : 'text-primary'}`}>
                  {infestationRate}
                </span>
                <span className={`text-2xl md:text-3xl font-bold ${isOverThreshold ? 'text-error/50' : 'text-primary/50'}`}>%</span>
              </div>

              <div className="w-full grid grid-cols-3 gap-2 text-center border-t border-white/10 pt-6">
                <div>
                  <p className="text-2xl md:text-3xl font-bold text-on-surface">{scans.total}</p>
                  <p className="text-[10px] md:text-xs text-on-surface-variant uppercase mt-1">Total</p>
                </div>
                <div>
                  <p className="text-2xl md:text-3xl font-bold text-[#facc15]">{scans.healthy}</p>
                  <p className="text-[10px] md:text-xs text-on-surface-variant uppercase mt-1">Healthy</p>
                </div>
                <div>
                  <p className="text-2xl md:text-3xl font-bold text-error">{scans.infested}</p>
                  <p className="text-[10px] md:text-xs text-on-surface-variant uppercase mt-1">Infested</p>
                </div>
              </div>
            </div>

            <button
              onClick={finishSession}
              disabled={scans.total === 0}
              className={`w-full py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 ${
                scans.total === 0 
                  ? 'bg-surface border border-white/10 text-on-surface-variant opacity-50 cursor-not-allowed'
                  : 'bg-white text-black hover:bg-white/90 shadow-xl'
              }`}
            >
              <span className="material-symbols-outlined">save</span>
              Finish & Save Session
            </button>
            
          </aside>
        </div>
      )}

      {sessionSaved && (
        <div className="glass-panel p-xl rounded-3xl flex flex-col items-center text-center max-w-lg w-full animate-in zoom-in-95 duration-500">
          <div className="w-24 h-24 bg-green-500/20 rounded-full flex items-center justify-center mb-md border-2 border-green-500/40">
            <span className="material-symbols-outlined text-green-500 text-5xl">check_circle</span>
          </div>
          <h3 className="text-2xl md:text-3xl font-bold mb-sm">Session Saved!</h3>
          <div className="w-full bg-white/5 rounded-xl p-md mb-xl flex justify-around">
            <div>
              <p className="text-xs md:text-sm text-on-surface-variant uppercase">Rate</p>
              <p className="text-xl md:text-2xl font-bold text-primary">{infestationRate}%</p>
            </div>
            <div>
              <p className="text-xs md:text-sm text-on-surface-variant uppercase">Total</p>
              <p className="text-xl md:text-2xl font-bold text-on-surface">{scans.total}</p>
            </div>
          </div>
          <button 
            onClick={() => setSessionSaved(false)}
            className="w-full py-3 bg-surface border border-white/10 text-on-surface font-bold rounded-xl hover:bg-white/5 transition-colors"
          >
            Back to Scouting
          </button>
        </div>
      )}

    </div>
  );
}
