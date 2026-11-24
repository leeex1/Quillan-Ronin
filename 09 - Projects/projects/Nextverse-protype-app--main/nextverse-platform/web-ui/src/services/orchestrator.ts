// Client for the local orchestrator API. Surfaces *measured* core/VM numbers.
export interface BenchmarkReport {
  parallel: { measured_speedup: number; correctness: string };
  vectorised: { measured_speedup: number; correctness: string };
}

const ORCHESTRATOR_URL = 'http://127.0.0.1:3006';

export async function fetchBenchmark(): Promise<BenchmarkReport> {
  const response = await fetch(`${ORCHESTRATOR_URL}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'benchmark', payload: {} }),
  });
  if (!response.ok) throw new Error(`orchestrator returned ${response.status}`);
  const payload = await response.json();
  if (payload.status !== 'success') throw new Error(payload.message || 'benchmark failed');
  return payload.result as BenchmarkReport;
}

export async function fetchHealth(): Promise<{ status: string; available_count: number; total_count: number }> {
  const response = await fetch(`${ORCHESTRATOR_URL}/health`);
  if (!response.ok) throw new Error(`orchestrator returned ${response.status}`);
  return response.json();
}
