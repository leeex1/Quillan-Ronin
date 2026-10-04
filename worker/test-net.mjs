import net from 'node:net';
import dns from 'node:dns/promises';

const t0 = Date.now();
try {
  const res = await dns.lookup('integrate.api.nvidia.com', { all: true });
  console.log(`DNS ok in ${Date.now() - t0}ms:`, res.map(r => r.address).join(', '));
} catch (e) {
  console.log(`DNS FAIL ${Date.now() - t0}ms:`, e.message);
}

const t1 = Date.now();
const sock = net.connect({ host: '75.2.113.119', port: 443 });
sock.setTimeout(12000);
sock.on('connect', () => { console.log(`TCP connect OK in ${Date.now() - t1}ms`); sock.destroy(); process.exit(0); });
sock.on('timeout', () => { console.log(`TCP connect TIMEOUT after ${Date.now() - t1}ms`); sock.destroy(); process.exit(1); });
sock.on('error', (e) => { console.log(`TCP ERROR after ${Date.now() - t1}ms: ${e.message}`); process.exit(1); });
