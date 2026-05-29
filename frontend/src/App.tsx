import React, { useState, useEffect } from 'react';
import DiagnosticHub from './components/DiagnosticHub';
import LiveStream from './components/LiveStream';
import HistoryLog from './components/HistoryLog';
import Encyclopedia from './components/Encyclopedia';

export type TabName = 'diagnostic' | 'stream' | 'history' | 'encyclopedia';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabName>('diagnostic');
  const [apiUrl, setApiUrl] = useState<string>(() => {
    return localStorage.getItem('apiUrl') || 'http://localhost:8000';
  });
  const [apiKey, setApiKey] = useState<string>(() => {
    return localStorage.getItem('apiKey') || '';
  });
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);

  // Check backend server health
  const checkServerHealth = async (urlToCheck?: string, keyToUse?: string) => {
    const targetUrl = urlToCheck || apiUrl;
    const targetKey = keyToUse !== undefined ? keyToUse : apiKey;
    setServerStatus('checking');
    
    try {
      const response = await fetch(`${targetUrl}/health`, {
        headers: {
          'accept': 'application/json',
          ...(targetKey ? { 'x-api-key': targetKey } : {}),
        },
      });
      if (response.ok) {
        const data = await response.json();
        setServerStatus('online');
        setModelLoaded(!!data.model_loaded);
      } else {
        setServerStatus('offline');
        setModelLoaded(false);
      }
    } catch (error) {
      setServerStatus('offline');
      setModelLoaded(false);
    }
  };

  useEffect(() => {
    checkServerHealth(apiUrl, apiKey);
  }, []);

  const handleSaveSettings = (newUrl: string, newKey: string) => {
    const cleanUrl = newUrl.trim().replace(/\/+$/, '');
    const cleanKey = newKey.trim();
    setApiUrl(cleanUrl);
    setApiKey(cleanKey);
    localStorage.setItem('apiUrl', cleanUrl);
    localStorage.setItem('apiKey', cleanKey);
    checkServerHealth(cleanUrl, cleanKey);
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">🌽</div>
          <h1 className="brand-name">Pest Detect AI</h1>
        </div>

        <nav className="nav-menu">
          <button
            onClick={() => setActiveTab('diagnostic')}
            className={`nav-item ${activeTab === 'diagnostic' ? 'active' : ''}`}
          >
            <span className="nav-icon">📊</span>
            <span>Diagnostic Hub</span>
          </button>

          <button
            onClick={() => setActiveTab('stream')}
            className={`nav-item ${activeTab === 'stream' ? 'active' : ''}`}
          >
            <span className="nav-icon">🎥</span>
            <span>Live Stream</span>
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
          >
            <span className="nav-icon">📜</span>
            <span>History Log</span>
          </button>

          <button
            onClick={() => setActiveTab('encyclopedia')}
            className={`nav-item ${activeTab === 'encyclopedia' ? 'active' : ''}`}
          >
            <span className="nav-icon">🐛</span>
            <span>Pest Encyclopedia</span>
          </button>
        </nav>

        {/* Sidebar Footer Info */}
        <div className="sidebar-footer">
          <div className="server-status-badge">
            <div className={`status-dot ${serverStatus}`} />
            <span className="status-label">
              {serverStatus === 'checking' && 'Connecting to AI...'}
              {serverStatus === 'online' &&
                (modelLoaded ? 'AI Server Online' : 'AI Loading Model...')}
              {serverStatus === 'offline' && 'AI Server Offline'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="main-content">
        {/* Page Title & settings bar */}
        <div className="page-header">
          <div className="page-title-group">
            {activeTab === 'diagnostic' && (
              <>
                <h2>maize Diagnostic Hub</h2>
                <p>Upload maize leaf photos or use your camera to diagnose fall armyworms and streak diseases instantly.</p>
              </>
            )}
            {activeTab === 'stream' && (
              <>
                <h2>Real-time Stream</h2>
                <p>Connect your webcam to process real-time frames over high-speed WebSocket channels.</p>
              </>
            )}
            {activeTab === 'history' && (
              <>
                <h2>Detection History Logs</h2>
                <p>Browse through historically saved crop scans, confidence records, and management logs.</p>
              </>
            )}
            {activeTab === 'encyclopedia' && (
              <>
                <h2>Pest & Disease Encyclopedia</h2>
                <p>Detailed guide on identifying supported armyworm states and maize stripe viruses with preventive controls.</p>
              </>
            )}
          </div>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="settings-trigger"
            title="Configure Server Settings"
          >
            ⚙️
          </button>
        </div>

        {/* Server settings drop block */}
        {showSettings && (
          <div className="settings-drawer">
            <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>API Server Configuration</h3>
            <div className="settings-row">
              <div className="input-group">
                <span className="input-label">Server End-point</span>
                <input
                  type="text"
                  className="text-input"
                  value={apiUrl}
                  onChange={(e) => handleSaveSettings(e.target.value, apiKey)}
                  placeholder="e.g. http://localhost:8000"
                />
              </div>
              <div className="input-group">
                <span className="input-label">x-api-key Header Token (optional)</span>
                <input
                  type="password"
                  className="text-input"
                  value={apiKey}
                  onChange={(e) => handleSaveSettings(apiUrl, e.target.value)}
                  placeholder="Enter API Key token if required"
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button
                  onClick={() => checkServerHealth(apiUrl, apiKey)}
                  className="btn-secondary"
                  style={{ height: '38px', padding: '0 16px' }}
                >
                  🔄 Reconnect
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Pages injection */}
        {activeTab === 'diagnostic' && (
          <DiagnosticHub apiUrl={apiUrl} apiKey={apiKey} checkServerHealth={() => checkServerHealth(apiUrl, apiKey)} />
        )}
        {activeTab === 'stream' && (
          <LiveStream apiUrl={apiUrl} apiKey={apiKey} />
        )}
        {activeTab === 'history' && (
          <HistoryLog apiUrl={apiUrl} apiKey={apiKey} />
        )}
        {activeTab === 'encyclopedia' && (
          <Encyclopedia apiUrl={apiUrl} />
        )}
      </main>
    </div>
  );
}
