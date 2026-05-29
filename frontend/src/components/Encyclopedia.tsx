import React, { useState, useEffect } from 'react';

interface ClassDetails {
  className: string;
  displayName: string;
  emoji: string;
  severity: 'high' | 'medium' | 'low' | 'none';
  managementTier: 'chemical' | 'integrated' | 'biological' | 'cultural' | 'monitoring';
  description: string;
  symptoms: string[];
  preventiveTips: string[];
  primaryAction: string;
  secondaryAction: string;
}

const LOCAL_ENCYCLOPEDIA_DATA: Record<string, Omit<ClassDetails, 'className'>> = {
  'fall-armyworm-egg': {
    displayName: 'Fall Armyworm Egg Cluster',
    emoji: '🥚',
    severity: 'low',
    managementTier: 'cultural',
    description: 'Clusters of 100-200 dome-shaped eggs laid on leaves, covered in protective mold-like wooly scales from the female moth\'s body.',
    symptoms: [
      'Greenish-white fuzzy clusters on upper or lower leaf surfaces.',
      'Eggs turn brown/dark grey right before hatching.',
      'Fuzzy hair-like covering that acts as protection against predators.'
    ],
    preventiveTips: [
      'Hand-pick and squash egg masses during early scouting.',
      'Encourage natural egg predators such as earwigs and ladybird beetles.',
      'Utilize trap cropping (e.g., Napier grass) around the maize crop field.'
    ],
    primaryAction: 'Scout fields daily and remove egg masses where feasible.',
    secondaryAction: 'Use threshold-based intervention if infestation expands.'
  },
  'fall-armyworm-frass': {
    displayName: 'Fall Armyworm Frass',
    emoji: '🪵',
    severity: 'medium',
    managementTier: 'biological',
    description: 'Coarse, moist, sawdust-like golden-brown fecal waste deposited deep within the leaf whorl by feeding caterpillars.',
    symptoms: [
      'Large piles of moist, yellowish-brown coarse powder in leaf whorls.',
      'Sticky substance on leaf surfaces indicating digestive waste.',
      'Bacterial leaf decay/rot when moisture mixes with frass piles.'
    ],
    preventiveTips: [
      'Apply sand, sawdust, or soil mixed with wood ash into whorls to suffocate larvae.',
      'Deploy biological control agents like Bacillus thuringiensis (Bt) or entomopathogenic fungi.',
      'Maintain adequate farm weeding to expose larvae to birds and predatory insects.'
    ],
    primaryAction: 'Inspect surrounding leaves for active larvae and damage.',
    secondaryAction: 'Consider biological control before chemical treatment.'
  },
  'fall-armyworm-larva': {
    displayName: 'Fall Armyworm Larva',
    emoji: '🐛',
    severity: 'high',
    managementTier: 'chemical',
    description: 'The highly destructive caterpillar stage of Spodoptera frugiperda, identified by an inverted white "Y" on the dark head capsule and four black dots forming a square on the tail.',
    symptoms: [
      'Caterpillars ranging from light green to dark brown/black inside whorls.',
      'Inverted white "Y" line pattern prominently visible on the head.',
      'Active voracious chewing, complete skeletonization of leaves, and hollowed whorls.'
    ],
    preventiveTips: [
      'Rotate insecticides with different modes of action to prevent chemical resistance.',
      'Plant streak/pest-resistant Bt maize hybrids where legally permitted.',
      'Sow crops early in the season to avoid high pest populations.'
    ],
    primaryAction: 'Apply targeted control promptly using label-approved products.',
    secondaryAction: 'Rotate modes of action to reduce resistance pressure.'
  },
  'fall-armyworm-larval-damage': {
    displayName: 'Fall Armyworm Feeding Damage',
    emoji: '🌽',
    severity: 'medium',
    managementTier: 'integrated',
    description: 'Characteristic ragged leaf holes, torn margins, and windowpane patterns (scraping of chlorophyll leaving translucent window-like skins) caused by hungry caterpillars.',
    symptoms: [
      'Elongated, dry, papery "windowpane" spots on young leaves.',
      'Deep, irregular circular holes along leaf margins and inside the whorl.',
      'Severed crop stalks and damaged tassels or ears during late-stage feeding.'
    ],
    preventiveTips: [
      'Use the Push-Pull integrated strategy: plant Desmodium (repels moths) and Napier grass (attracts them).',
      'Assess damage severity index (1-9 scale) to decide economic threshold sprays.',
      'Maintain strong nitrogen and phosphorus fertilization to help maize plants outgrow damage.'
    ],
    primaryAction: 'Assess live larvae before spraying and map affected areas.',
    secondaryAction: 'Escalate treatment if fresh feeding is still active.'
  },
  'healthy-maize': {
    displayName: 'Healthy Maize Leaf',
    emoji: '🟢',
    severity: 'none',
    managementTier: 'monitoring',
    description: 'Vibrant, dark green, structurally sound maize leaves free of insect damage, yellow streaks, or pest-induced distress.',
    symptoms: [
      'Uniform dark green coloration across all leaves.',
      'Intact whorl and leaves with smooth, solid margins.',
      'Strong, upright stalk and regular internode development.'
    ],
    preventiveTips: [
      'Perform regular visual inspection twice a week in a W-shaped scouting pattern.',
      'Adhere to optimal irrigation schedules to prevent crop moisture stress.',
      'Practice clean weeding and clear old plant residues from preceding seasons.'
    ],
    primaryAction: 'No treatment required.',
    secondaryAction: 'Continue routine scouting.'
  },
  'maize-streak-disease': {
    displayName: 'Maize Streak Disease',
    emoji: '🦠',
    severity: 'high',
    managementTier: 'integrated',
    description: 'A major crop virus transmitted by the Cicadulina leafhopper vector, characterized by severe stunting and yellowing of the crop leaves.',
    symptoms: [
      'Narrow, broken, pale yellow parallel streaks running parallel to leaf veins.',
      'Premature leaf drying and severe stunting of infected plants.',
      'Deformed ears or complete failure to produce tassels and grain.'
    ],
    preventiveTips: [
      'Rogue (pull up and burn) early infected plants immediately to prevent virus spread.',
      'Apply systemic seed dressing insecticides to control the leafhopper vector early.',
      'Select certified, virus-tolerant maize varieties for all future seed purchase cycles.'
    ],
    primaryAction: 'Rogue infected plants and control vector pressure.',
    secondaryAction: 'Use resistant varieties for future plantings.'
  }
};

interface EncyclopediaProps {
  apiUrl: string;
}

export default function Encyclopedia({ apiUrl }: EncyclopediaProps) {
  const [classes, setClasses] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [expandedClass, setExpandedClass] = useState<string | null>(null);

  useEffect(() => {
    const fetchClasses = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${apiUrl}/classes`, {
          headers: { accept: 'application/json' },
        });
        if (response.ok) {
          const data = await response.json();
          setClasses(data.classes || {});
        } else {
          // Fallback to offline keys if backend fails
          setClasses({
            '0': 'fall-armyworm-egg',
            '1': 'fall-armyworm-frass',
            '2': 'fall-armyworm-larva',
            '3': 'fall-armyworm-larval-damage',
            '4': 'healthy-maize',
            '5': 'maize-streak-disease'
          });
        }
      } catch (error) {
        console.warn('Could not contact backend server for class inventory:', error);
        setClasses({
          '0': 'fall-armyworm-egg',
          '1': 'fall-armyworm-frass',
          '2': 'fall-armyworm-larva',
          '3': 'fall-armyworm-larval-damage',
          '4': 'healthy-maize',
          '5': 'maize-streak-disease'
        });
      } finally {
        setLoading(false);
      }
    };

    fetchClasses();
  }, [apiUrl]);

  // Combine fetched classes and our rich local data
  const encyclopediaData: ClassDetails[] = Object.values(classes).map((className) => {
    const local = LOCAL_ENCYCLOPEDIA_DATA[className] || {
      displayName: className.replace(/-/g, ' '),
      emoji: '🔍',
      severity: 'none',
      managementTier: 'monitoring',
      description: 'No detailed information is currently available for this plant state.',
      symptoms: ['Verify symptoms against agricultural standards.'],
      preventiveTips: ['Inspect crops regularly.'],
      primaryAction: 'Scout crops daily.',
      secondaryAction: 'Consult with agronomists.'
    };

    return {
      className,
      ...local,
    };
  });

  // Severity specific color mappings
  const getSeverityBadgeColor = (severity: string) => {
    switch (severity) {
      case 'high': return { bg: 'rgba(239, 68, 68, 0.15)', text: '#ef4444', border: 'rgba(239, 68, 68, 0.25)', leftBorder: '#ef4444' };
      case 'medium': return { bg: 'rgba(249, 115, 22, 0.15)', text: '#f97316', border: 'rgba(249, 115, 22, 0.25)', leftBorder: '#f97316' };
      case 'low': return { bg: 'rgba(59, 130, 246, 0.15)', text: '#3b82f6', border: 'rgba(59, 130, 246, 0.25)', leftBorder: '#3b82f6' };
      default: return { bg: 'rgba(16, 185, 129, 0.15)', text: '#10b981', border: 'rgba(16, 185, 129, 0.25)', leftBorder: '#10b981' };
    }
  };

  const getTierIcon = (tier: string) => {
    switch (tier) {
      case 'chemical': return '🧪';
      case 'integrated': return '🔄';
      case 'biological': return '🐝';
      case 'cultural': return '👨‍🌾';
      default: return '👁️';
    }
  };

  // Filter items
  const filteredData = encyclopediaData.filter((item) => {
    const matchesSearch = 
      item.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.symptoms.some(s => s.toLowerCase().includes(searchTerm.toLowerCase()));

    const matchesSeverity = selectedSeverity === 'all' || item.severity === selectedSeverity;
    const matchesTier = selectedTier === 'all' || item.managementTier === selectedTier;

    return matchesSearch && matchesSeverity && matchesTier;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Filtering Drawer */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Search Row */}
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <input
              type="text"
              placeholder="🔍 Search encyclopedia by name, symptoms, or preventive guidelines..."
              className="text-input"
              style={{ flexGrow: 1, paddingLeft: '14px' }}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          {/* Filters pills row */}
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'center', fontSize: '13px' }}>
            
            {/* Severity Filters */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Severity:</span>
              {['all', 'high', 'medium', 'low', 'none'].map((sev) => (
                <button
                  key={sev}
                  onClick={() => setSelectedSeverity(sev)}
                  className={selectedSeverity === sev ? 'btn-primary' : 'btn-secondary'}
                  style={{
                    padding: '6px 12px',
                    fontSize: '11px',
                    borderRadius: '8px',
                    textTransform: 'capitalize'
                  }}
                >
                  {sev === 'none' ? 'Healthy/None' : sev}
                </button>
              ))}
            </div>

            {/* Management Tier Filters */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Management:</span>
              {['all', 'chemical', 'integrated', 'biological', 'cultural', 'monitoring'].map((tier) => (
                <button
                  key={tier}
                  onClick={() => setSelectedTier(tier)}
                  className={selectedTier === tier ? 'btn-primary' : 'btn-secondary'}
                  style={{
                    padding: '6px 12px',
                    fontSize: '11px',
                    borderRadius: '8px',
                    textTransform: 'capitalize'
                  }}
                >
                  {tier}
                </button>
              ))}
            </div>
            
          </div>
        </div>
      </div>

      {/* Grid Display */}
      {loading ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 0' }}>
          <div className="spinner" style={{ margin: '0 auto 16px auto' }} />
          <span>Synchronizing pest catalogue...</span>
        </div>
      ) : filteredData.length === 0 ? (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 0' }}>
          <span style={{ fontSize: '48px', display: 'block', marginBottom: '16px' }}>📭</span>
          <h3>No Encyclopedia Entries Found</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '6px' }}>
            Try refining your search terms or filters above.
          </p>
        </div>
      ) : (
        <div className="encyclopedia-grid">
          {filteredData.map((item) => {
            const colors = getSeverityBadgeColor(item.severity);
            const isExpanded = expandedClass === item.className;

            return (
              <div
                key={item.className}
                className="encyclopedia-card"
                style={{
                  borderTopColor: colors.leftBorder,
                  cursor: 'pointer',
                  height: 'auto',
                }}
                onClick={() => setExpandedClass(isExpanded ? null : item.className)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '28px' }}>{item.emoji}</span>
                    <h3 className="encyclopedia-name" style={{ margin: 0 }}>{item.displayName}</h3>
                  </div>
                  <span
                    className="impact-tag"
                    style={{
                      backgroundColor: colors.bg,
                      color: colors.text,
                      border: `1px solid ${colors.border}`,
                    }}
                  >
                    {item.severity} severity
                  </span>
                </div>

                <p className="encyclopedia-desc">{item.description}</p>

                {/* Expanded Details Section */}
                {isExpanded && (
                  <div
                    style={{
                      borderTop: '1px solid rgba(255,255,255,0.05)',
                      marginTop: '16px',
                      paddingTop: '16px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '16px',
                      animation: 'fadeIn 0.25s ease-out',
                    }}
                  >
                    {/* Symptoms Guide */}
                    <div>
                      <h4 style={{ fontSize: '13px', color: '#fff', marginBottom: '6px', fontWeight: '600' }}>
                        📋 Identification & Symptoms
                      </h4>
                      <ul style={{ paddingLeft: '18px', margin: 0, fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {item.symptoms.map((sym, idx) => (
                          <li key={idx} style={{ lineHeight: '1.4' }}>{sym}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Integrated Control & Preventive Measures */}
                    <div>
                      <h4 style={{ fontSize: '13px', color: '#fff', marginBottom: '6px', fontWeight: '600' }}>
                        💡 Preventive & Control Measures
                      </h4>
                      <ul style={{ paddingLeft: '18px', margin: 0, fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {item.preventiveTips.map((tip, idx) => (
                          <li key={idx} style={{ lineHeight: '1.4' }}>{tip}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Recommendation Actions Box */}
                    <div className="reco-box" style={{ margin: 0, backgroundColor: 'rgba(0,0,0,0.18)' }}>
                      <div className="reco-label" style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        🛡️ Recommended Management Response
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.4' }}>
                        <strong>Action Plan:</strong> {item.primaryAction}
                      </div>
                      {item.secondaryAction && (
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: '4px', lineHeight: '1.4' }}>
                          <strong>Resistance mitigation:</strong> {item.secondaryAction}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="encyclopedia-impact">
                  <span style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? '👆 Click to collapse' : '👇 Click to inspect symptom guide & prevention'}
                  </span>
                  <span
                    className="match-badge"
                    style={{
                      backgroundColor: 'rgba(255,255,255,0.03)',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border-light)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      fontSize: '10px'
                    }}
                  >
                    {getTierIcon(item.managementTier)} {item.managementTier.toUpperCase()}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
