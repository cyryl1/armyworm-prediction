import React, { useState, useEffect } from 'react';

interface HistoryRecord {
  id?: string;
  detection_timestamp: string;
  class_name: string;
  confidence: number;
  recommendation: string;
  recommendation_details?: string | null;
  gps_latitude?: number | null;
  gps_longitude?: number | null;
  detections?: Array<{
    class_name: string;
    confidence: number;
    recommendation: string;
  }>;
}

interface HistoryLogProps {
  apiUrl: string;
  apiKey: string;
}

export default function HistoryLog({ apiUrl, apiKey }: HistoryLogProps) {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterClass, setFilterClass] = useState<string>('all');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const headers: any = { accept: 'application/json' };
      if (apiKey) headers['x-api-key'] = apiKey;
      
      const response = await fetch(`${apiUrl}/history`, { headers });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setRecords(data.records || []);
    } catch (error) {
      console.warn('Failed to load detection logs history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [apiUrl, apiKey]);

  // Format iso timestamp to user readable local date-time
  const formatTimestamp = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    } catch (e) {
      return isoString;
    }
  };

  // Get color for tags
  const getCategoryColor = (className: string) => {
    const name = className.toLowerCase();
    if (name.includes('healthy')) return '#10b981';
    if (name.includes('larva') || name.includes('egg')) return '#ef4444';
    if (name.includes('damage') || name.includes('frass')) return '#f97316';
    return '#3b82f6';
  };

  // Filtering Logic
  const filteredRecords = records.filter(record => {
    // A history record might have a main class, or a nested `detections` array depending on schema
    const detectionsList = record.detections || [
      { class_name: record.class_name, recommendation: record.recommendation }
    ];

    const matchSearch = detectionsList.some(d => {
      const label = (d.class_name || '').toLowerCase();
      const reco = (d.recommendation || '').toLowerCase();
      const term = searchTerm.toLowerCase();
      return label.includes(term) || reco.includes(term);
    });

    const matchFilter = filterClass === 'all' || detectionsList.some(d => {
      const name = (d.class_name || '').toLowerCase();
      if (filterClass === 'healthy') return name.includes('healthy');
      if (filterClass === 'pest') return name.includes('larva') || name.includes('egg');
      if (filterClass === 'damage') return name.includes('damage') || name.includes('frass');
      return true;
    });

    return matchSearch && matchFilter;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Search and Filters box */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div className="history-filter-row">
          <div className="search-input-wrapper">
            <input
              type="text"
              placeholder="🔍 Search history by pest name, tag, or recommendation..."
              className="text-input"
              style={{ width: '100%', paddingLeft: '14px' }}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Type Filter:</span>
            <button
              onClick={() => setFilterClass('all')}
              className={filterClass === 'all' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '8px 14px', fontSize: '12px', borderRadius: '8px' }}
            >
              All Scans
            </button>
            <button
              onClick={() => setFilterClass('pest')}
              className={filterClass === 'pest' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '8px 14px', fontSize: '12px', borderRadius: '8px' }}
            >
              🐛 Pests
            </button>
            <button
              onClick={() => setFilterClass('damage')}
              className={filterClass === 'damage' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '8px 14px', fontSize: '12px', borderRadius: '8px' }}
            >
              ⚠️ Damage
            </button>
            <button
              onClick={() => setFilterClass('healthy')}
              className={filterClass === 'healthy' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '8px 14px', fontSize: '12px', borderRadius: '8px' }}
            >
              🟢 Healthy
            </button>
          </div>
          
          <button onClick={fetchHistory} className="btn-secondary" style={{ padding: '8px 14px', fontSize: '12px', borderRadius: '8px' }}>
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Logs Display */}
      {loading ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 0' }}>
          <div className="spinner" style={{ margin: '0 auto 16px auto' }} />
          <span>Synchronizing historical scans...</span>
        </div>
      ) : filteredRecords.length === 0 ? (
        <div className="glass-panel history-empty">
          <div className="empty-icon">📁</div>
          <h3>No Diagnostic Records Found</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '6px' }}>
            {records.length === 0 
              ? "You haven't run any maize leaf diagnostics yet."
              : "No records match your selected query filters."}
          </p>
        </div>
      ) : (
        <div className="history-grid">
          {filteredRecords.map((record, index) => {
            // Support single detection format & legacy schemas
            const detectionsList = record.detections || [
              {
                class_name: record.class_name,
                confidence: record.confidence,
                recommendation: record.recommendation
              }
            ];

            return (
              <div key={index} className="history-card">
                <div className="history-header">
                  <span className="history-date">
                    📅 {formatTimestamp(record.detection_timestamp)}
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    #{index + 1}
                  </span>
                </div>

                <div className="history-meta">
                  <div className="history-tag-list">
                    {detectionsList.map((d, dIdx) => {
                      const color = getCategoryColor(d.class_name || 'healthy');
                      return (
                        <span
                          key={dIdx}
                          className="match-badge"
                          style={{
                            backgroundColor: `${color}15`,
                            color: color,
                            border: `1px solid ${color}25`,
                            textTransform: 'capitalize',
                            display: 'inline-block',
                            marginBottom: '6px'
                          }}
                        >
                          {d.class_name?.replace(/-/g, ' ')} ({Math.round((d.confidence || 0) * 100)}%)
                        </span>
                      );
                    })}
                  </div>

                  {detectionsList.map((d, dIdx) => (
                    <div
                      key={dIdx}
                      className="reco-box"
                      style={{ margin: '8px 0 0 0', backgroundColor: 'rgba(0,0,0,0.15)' }}
                    >
                      <div className="reco-label" style={{ fontSize: '12px' }}>AI recommendation:</div>
                      <div className="reco-text" style={{ fontSize: '13px' }}>{d.recommendation}</div>
                    </div>
                  ))}

                  {/* GPS Metadata */}
                  {(record.gps_latitude || record.gps_longitude) && (
                    <div className="meta-row" style={{ marginTop: '12px' }}>
                      <span>📍</span>
                      <span style={{ fontSize: '11px' }}>
                        Coordinates: {record.gps_latitude?.toFixed(4)}, {record.gps_longitude?.toFixed(4)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
