import React, { useEffect, useState } from 'react';
import { fetchBenchmark, fetchHealth } from '../services/orchestrator';

// Shows real, measured acceleration from the quantum core/VM layer.
// No simulated JQLD/LVVM "boost" is displayed here.
const SystemBoostPanel: React.FC = () => {
  const [parallel, setParallel] = useState('—');
  const [vectorised, setVectorised] = useState('—');
  const [health, setHealth] = useState('—');
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then(report => { if (active) setHealth(`${report.available_count}/${report.total_count} ${report.status}`); })
      .catch(() => { if (active) setError('orchestrator offline'); });
    fetchBenchmark()
      .then(report => {
        if (!active) return;
        setParallel(`${report.parallel.measured_speedup}x`);
        setVectorised(`${report.vectorised.measured_speedup}x`);
      })
      .catch(() => { if (active) setError('benchmark unavailable'); });
    return () => { active = false; };
  }, []);

  return (
    <div style={{ border: '1px solid #00d5ff', borderRadius: 6, padding: 10, marginBottom: 12 }}>
      <h4 style={{ marginTop: 0 }}>Core / VM — measured acceleration</h4>
      <div>Multiprocessing: <b>{parallel}</b></div>
      <div>Vectorised math: <b>{vectorised}</b></div>
      <div>Services: <b>{health}</b></div>
      {error && <div style={{ color: '#ff6b35', fontSize: 12 }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.7 }}>
        Wall-clock measured on this device; simulated JQLD/LVVM metrics are excluded.
      </div>
    </div>
  );
};

export default SystemBoostPanel;
