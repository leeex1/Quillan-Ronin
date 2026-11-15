import { createServer } from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import os from 'node:os';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const ROOT = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(ROOT, '..');
const SPIDER_DIR = join(REPO_ROOT, '09 - Projects', 'mcp', 'quillan_spider');
const MEMORY_JSON = join(REPO_ROOT, 'memory.json');
const MEMGRAPH_JSON = join(SPIDER_DIR, 'memory_graph.json');
const TASKS_JSON = join(SPIDER_DIR, 'tasks.json');
const START_TIME = Date.now();

function computeThresholds(memPct, sovOnline) {
  const risk = Math.min(0.95, Math.round(((memPct / 100) * 0.45 + (sovOnline ? 0 : 0.20)) * 10000) / 10000);
  const headroom = Math.max(0.05, Math.round((1 - risk) * 10000) / 10000);
  const phi = Math.min(0.98, Math.max(0.70, Math.round((0.82 + risk * 0.14) * 10000) / 10000));
  return {
    evidence_min: phi,
    aesthetics_min: Math.round(Math.max(5.5, Math.min(9.5, 6.2 + headroom * 1.8)) * 100) / 100,
    drift_ceiling: Math.round(Math.max(0.10, Math.min(0.50, 0.42 - risk * 0.25)) * 10000) / 10000,
    coherence_target: Math.round(Math.max(0.72, Math.min(0.99, 0.78 + headroom * 0.16)) * 10000) / 10000,
    risk,
    headroom
  };
}

// Live data cache
const liveData = {
  agents: new Map(),
  events: [],
  gitnexus: { nodes: [], edges: [], lastUpdate: null },
  obsidian: { entities: [], chunks: 0, lastUpdate: null },
  thresholds: computeThresholds(os.totalmem() > 0 ? ((os.totalmem() - os.freemem()) / os.totalmem()) * 100 : 0, false),
  systemMetrics: { cpu: 0, memory: 0, activeAgents: 0 }
};

// WebSocket clients
const clients = new Set();

function broadcast(data) {
  const msg = JSON.stringify(data, (k, v) => (v instanceof Map ? Object.fromEntries(v) : v));
  clients.forEach(client => {
    if (client.readyState === client.OPEN) {
      client.send(msg);
    }
  });
}

function readJsonSafe(filePath, fallback) {
  try {
    if (!existsSync(filePath)) return fallback;
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch {
    return fallback;
  }
}

async function fetchGitNexusData() {
  const mg = readJsonSafe(MEMGRAPH_JSON, null);
  if (mg && mg.nodes) {
    const entries = Object.entries(mg.nodes).slice(0, 24);
    const idxMap = new Map();
    const nodes = entries.map(([p, meta], i) => {
      const id = `node-${i}`;
      idxMap.set(p, id);
      const label = p.split(/[\\/]/).pop() || p;
      return {
        id,
        type: meta.kind || 'file',
        label,
        x: 80 + (i % 6) * 110,
        y: 60 + Math.floor(i / 6) * 85,
        path: p
      };
    });
    const edges = [];
    for (const [k, e] of Object.entries(mg.edges || {})) {
      const parts = k.split('\u2192');
      if (parts.length === 2 && idxMap.has(parts[0]) && idxMap.has(parts[1])) {
        edges.push({ from: idxMap.get(parts[0]), to: idxMap.get(parts[1]), action: e.wire ? 'wire' : 'link', strength: e.wire ? 0.9 : 0.6 });
      }
    }
    return { nodes, edges, timestamp: new Date().toISOString() };
  }
  return { nodes: [], edges: [], timestamp: new Date().toISOString() };
}

async function fetchObsidianData() {
  const entities = [];
  const mem = readJsonSafe(MEMORY_JSON, []);
  if (Array.isArray(mem)) {
    for (const item of mem.slice(0, 32)) {
      const val = item?.value;
      const obs = Array.isArray(val) ? val.map(String) : (val != null ? [String(val)] : []);
      entities.push({
        id: item.key || `mem-${entities.length}`,
        type: item.category || 'concept',
        label: String(item.key || '').replace(/_/g, ' '),
        connections: obs.length + 1,
        observations: obs.slice(0, 3)
      });
    }
  }
  const mg = readJsonSafe(MEMGRAPH_JSON, null);
  if (mg && mg.nodes) {
    for (const [p, node] of Object.entries(mg.nodes).slice(0, 12)) {
      entities.push({
        id: p.split(/[\\/]/).pop() || p,
        type: node.kind || 'module',
        label: p.split(/[\\/]/).pop() || p,
        connections: node.visits || 1,
        observations: [`Visits: ${node.visits || 1}`, `Last action: ${node.last_action || 'READ'}`]
      });
    }
  }
  return { entities, chunks: entities.length * 45, timestamp: new Date().toISOString() };
}

async function probeHttpOk(url) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(800) });
    return res.ok;
  } catch {
    return false;
  }
}

async function refreshLiveAgents() {
  const [sovOk, workerOk, ollamaOk] = await Promise.all([
    probeHttpOk('http://127.0.0.1:8080/health'),
    probeHttpOk('http://127.0.0.1:7777/api/state'),
    probeHttpOk('http://127.0.0.1:11434/api/version')
  ]);
  const now = new Date().toISOString();
  const coreNodes = [
    { id: 'sovereign_server', name: 'Quillan Sovereign (:8080)', role: 'OpenAI/Ollama API Gateway', status: sovOk ? 'online' : 'offline' },
    { id: 'quillan_worker', name: 'QuillanWorker (:7777)', role: 'MCP + Extension Bridge', status: workerOk ? 'online' : 'offline' },
    { id: 'ollama_engine', name: 'Ollama Runtime (:11434)', role: 'Local GGUF Model Engine', status: ollamaOk ? 'online' : 'offline' }
  ];
  for (const node of coreNodes) {
    const prev = liveData.agents.get(node.id) || {};
    liveData.agents.set(node.id, {
      ...prev,
      ...node,
      lastUpdate: now
    });
  }
}

// Update live data periodically from real filesystem & OS state
async function updateLiveData() {
  const [gnData, obsData] = await Promise.all([
    fetchGitNexusData(),
    fetchObsidianData(),
    refreshLiveAgents()
  ]);
  liveData.gitnexus = gnData;
  liveData.obsidian = obsData;

  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const memPct = totalMem > 0 ? Math.round(((totalMem - freeMem) / totalMem) * 1000) / 10 : 0;
  const load = os.loadavg()[0] || 0;
  const cpuCores = Math.max(1, os.cpus().length);
  const sovOnline = liveData.agents.get('sovereign_server')?.status === 'online';
  liveData.thresholds = computeThresholds(memPct, sovOnline);
  liveData.systemMetrics = {
    cpu: Math.min(100, Math.round((load / cpuCores) * 1000) / 10),
    memory: memPct,
    activeAgents: Array.from(liveData.agents.values()).filter(a => a.status === 'online' || a.status === 'active').length
  };

  broadcast({
    type: 'live-update',
    data: {
      ...liveData,
      agents: Array.from(liveData.agents.values())
    }
  });
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost`);
  
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }
  
  // API endpoints
  if (url.pathname === '/api/live-data' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ...liveData, agents: Array.from(liveData.agents.values()) }));
  }

  if ((url.pathname === '/health' || url.pathname === '/api/health') && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      status: 'ok',
      hub: 'ws-node',
      uptime_s: Math.round((Date.now() - START_TIME) / 100) / 10,
      events_count: liveData.events.length,
      active_agents_count: liveData.agents.size,
      thresholds: liveData.thresholds || {},
      systemMetrics: liveData.systemMetrics
    }));
  }

  if (url.pathname === '/api/gitnexus' && req.method === 'GET') {
    const data = await fetchGitNexusData();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(data));
  }

  if (url.pathname === '/api/obsidian' && req.method === 'GET') {
    const data = await fetchObsidianData();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(data));
  }

  if (url.pathname === '/api/agents' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ agents: Array.from(liveData.agents.values()) }));
  }

  if (url.pathname === '/api/events' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ events: liveData.events, thresholds: liveData.thresholds || {} }));
  }

  if (url.pathname === '/api/stats' && req.method === 'GET') {
    const tasks = readJsonSafe(TASKS_JSON, []);
    const tasksDone = Array.isArray(tasks) ? tasks.filter(t => ['done', 'completed', 'ok'].includes(String(t?.status || '').toLowerCase())).length : 0;
    const mg = readJsonSafe(MEMGRAPH_JSON, {});
    const nodes = mg?.nodes || {};
    const edges = mg?.edges || {};
    const uptimeS = Math.max(0.1, Math.round((Date.now() - START_TIME) / 100) / 10);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      tasks_done: tasksDone,
      files: Object.keys(nodes).length,
      findings: Object.values(nodes).reduce((acc, n) => acc + (Array.isArray(n?.findings) ? n.findings.length : 0), 0),
      wires: Object.values(edges).filter(e => e?.wire).length,
      events_count: liveData.events.length,
      uptime_s: uptimeS,
      watch_on: true
    }));
  }

  if (url.pathname === '/api/events' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const event = JSON.parse(body || '{}');
        if (!event.agent || !event.type) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ error: 'agent and type required' }));
        }
        event.ts = event.ts || new Date().toISOString();
        event.detail = event.detail || '';
        liveData.events.push(event);
        if (liveData.events.length > 1000) liveData.events.shift();
        const prev = liveData.agents.get(event.agent) || {};
        liveData.agents.set(event.agent, {
          id: event.agent,
          name: event.agent,
          status: 'active',
          last_action: event.type,
          last_detail: String(event.detail).slice(0, 120),
          events: (prev.events || 0) + 1,
          lastUpdate: event.ts
        });
        broadcast({ type: 'event', event });
        res.writeHead(202, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ accepted: true }));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'invalid event' }));
      }
    });
    return;
  }

  // Serve static files (for serving the dashboard)
  if (url.pathname === '/' || url.pathname === '/dashboard') {
    try {
      const html = readFileSync(join(ROOT, 'quillan-dashboard.html'), 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      return res.end(html);
    } catch (e) {
      res.writeHead(404);
      return res.end('Dashboard file not found');
    }
  }
  
  res.writeHead(404);
  res.end('Not found');
});

// WebSocket server for real-time updates
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
  clients.add(ws);
  console.log('Client connected, total clients:', clients.size);
  
  // Send initial data
  ws.send(JSON.stringify({
    type: 'initial',
    data: { ...liveData, agents: Array.from(liveData.agents.values()) }
  }));
  
  ws.on('close', () => {
    clients.delete(ws);
    console.log('Client disconnected, total clients:', clients.size);
  });
  
  ws.on('error', (e) => {
    console.error('WebSocket error:', e);
    clients.delete(ws);
  });
});

const PORT = Number(process.env.HUB_PORT || process.env.PORT || 8777);
server.listen(PORT, '127.0.0.1', () => {
  console.log(`Quillan Hub Server running on http://127.0.0.1:${PORT}`);
  console.log('WebSocket server ready for live streaming');
  console.log(`Dashboard available at http://127.0.0.1:${PORT}/dashboard`);
});

// Start live data updates
setInterval(updateLiveData, 2000);
updateLiveData();
