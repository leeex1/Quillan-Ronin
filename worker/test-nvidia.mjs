const t = Date.now();
const base = 'https://integrate.api.nvidia.com/v1';
const key = process.env.NVIDIA_API_KEY;
console.log('key present:', !!key);

try {
  const r0 = await fetch(`${base}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(20000) });
  console.log(`GET /models -> ${r0.status} in ${Date.now() - t}ms`);
} catch (e) {
  console.log(`GET /models FAILED after ${Date.now() - t}ms: ${e.message}`);
}

const t2 = Date.now();
try {
  const r = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: 'meta/llama-3.3-70b-instruct', messages: [{ role: 'user', content: 'Say exactly: KERNEL ONLINE' }], max_tokens: 12 }),
    signal: AbortSignal.timeout(90000)
  });
  const body = await r.text();
  console.log(`POST /chat/completions -> ${r.status} in ${Date.now() - t2}ms`);
  console.log(body.slice(0, 400));
} catch (e) {
  console.log(`POST FAILED after ${Date.now() - t2}ms: ${e.message}`);
}
