const t = Date.now();
try {
  const r = await fetch('http://127.0.0.1:8188/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: {} }),
    signal: AbortSignal.timeout(12000)
  });
  const txt = await r.text();
  console.log(`POST localhost:8188 -> ${r.status} in ${Date.now() - t}ms :: ${txt.slice(0, 120)}`);
} catch (e) {
  console.log(`POST localhost FAILED after ${Date.now() - t}ms: ${e.message}`);
}
const t2 = Date.now();
try {
  const g = await fetch('http://127.0.0.1:8188/system_stats', { signal: AbortSignal.timeout(8000) });
  console.log(`GET localhost:8188 -> ${g.status} in ${Date.now() - t2}ms`);
} catch (e) {
  console.log(`GET localhost FAILED after ${Date.now() - t2}ms: ${e.message}`);
}
process.exit(0);
