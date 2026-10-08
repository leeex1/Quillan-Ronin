import { createServer } from 'node:http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(fileURLToPath(import.meta.url));

// Live data cache
const liveData = {
  agents: new Map(),
  gitnexus: { nodes: [], edges: [], lastUpdate: null },
  obsidian: { entities: [], chunks: 0, lastUpdate: null },
  systemMetrics: { cpu: 0, memory: 0, activeAgents: 0 }
};

// WebSocket clients
const clients = new Set();

function broadcast(data) {
  const msg = JSON.stringify(data);
  clients.forEach(client => {
    if (client.readyState === client.OPEN) {
      client.send(msg);
    }
  });
}

// MCP connection functions (these would connect to actual MCP servers)
async function fetchGitNexusData() {
  try {
    // In production, this would call the actual GitNexus MCP
    // For now, return structure that matches GitNexus graph format
    return {
      nodes: [
        { id: 'agent-1', type: 'agent', label: 'CodeWeaver', x: 100, y: 200, cluster: 'communication' },
        { id: 'agent-2', type: 'agent', label: 'Predator', x: 300, y: 150, cluster: 'systems' },
        { id: 'agent-3', type: 'agent', label: 'Logos', x: 200, y: 350, cluster: 'cognitive' },
        { id: 'file-1', type: 'file', label: 'server.js', x: 400, y: 250, path: 'QuillanWorker/server.js' },
        { id: 'file-2', type: 'file', label: 'scanner.mjs', x: 500, y: 300, path: 'QuillanWorker/scanner.mjs' },
        { id: 'file-3', type: 'file', label: 'quillan-dashboard.html', x: 350, y: 400, path: '06 - Deployment & Platforms/quillan-dashboard.html' }
      ],
      edges: [
        { from: 'agent-1', to: 'file-1', action: 'editing', strength: 0.8 },
        { from: 'agent-2', to: 'file-2', action: 'analyzing', strength: 0.6 },
        { from: 'agent-3', to: 'agent-1', action: 'collaborating', strength: 0.9 },
        { from: 'agent-1', to: 'file-3', action: 'refactoring', strength: 0.7 }
      ],
      timestamp: new Date().toISOString()
    };
  } catch (e) {
    console.error('GitNexus fetch error:', e);
    return { nodes: [], edges: [], timestamp: new Date().toISOString() };
  }
}

async function fetchObsidianData() {
  try {
    // In production, this would call the Memory MCP
    // For now, return structure that matches Obsidian brain format
    return {
      entities: [
        { id: 'entity-1', type: 'concept', label: 'Quillan-Ronin', connections: 5, observations: ['Advanced Cognitive Engine', '34 Council Experts'] },
        { id: 'entity-2', type: 'task', label: 'Code Refactoring', connections: 3, observations: ['Active refactoring in progress'] },
        { id: 'entity-3', type: 'agent', label: 'CodeWeaver', connections: 4, observations: ['AST analysis', 'Code architecture'] },
        { id: 'entity-4', type: 'agent', label: 'Predator', connections: 2, observations: ['Security analysis', 'Adversarial testing'] }
      ],
      chunks: 360,
      timestamp: new Date().toISOString()
    };
  } catch (e) {
    console.error('Obsidian fetch error:', e);
    return { entities: [], chunks: 0, timestamp: new Date().toISOString() };
  }
}

// Simulate agent movements with more realistic behavior
function simulateAgentMovement() {
  const agents = [
    { id: 'antigravity', name: 'Antigravity', role: 'Lead Systems Architect', avatar: '⚡' },
    { id: 'quillan_trainer', name: 'Quillan Model Forge', role: 'BitNet Trainer', avatar: '🧠' },
    { id: 'codeweaver', name: 'CodeWeaver (C10)', role: 'AST Architect', avatar: '🛠️' },
    { id: 'predator', name: 'Predator (C34)', role: 'Security Hunter', avatar: '⚔️' }
  ];
  
  const files = [
    'quillan-dashboard.html',
    'server.js',
    'scanner.mjs',
    'agent-browser.mjs',
    'quillan_collector.py'
  ];
  
  const actions = ['editing', 'analyzing', 'debugging', 'refactoring', 'testing', 'reviewing'];
  
  agents.forEach((agent, i) => {
    const file = files[Math.floor(Math.random() * files.length)];
    const action = actions[Math.floor(Math.random() * actions.length)];
    const x = 100 + Math.random() * 400;
    const y = 100 + Math.random() * 300;
    
    const existing = liveData.agents.get(agent.id) || {};
    
    liveData.agents.set(agent.id, {
      id: agent.id,
      name: agent.name,
      role: agent.role,
      avatar: agent.avatar,
      x,
      y,
      targetFile: file,
      action: action,
      status: 'online',
      toolCalls: (existing.toolCalls || 0) + Math.floor(Math.random() * 3),
      tokens: (existing.tokens || 0) + Math.floor(Math.random() * 1000),
      lastUpdate: new Date().toISOString()
    });
  });
}

// Update live data periodically
async function updateLiveData() {
  // Fetch GitNexus
  const gnData = await fetchGitNexusData();
  liveData.gitnexus = gnData;
  
  // Fetch Obsidian
  const obsData = await fetchObsidianData();
  liveData.obsidian = obsData;
  
  // Simulate agent movements
  simulateAgentMovement();
  
  // Update metrics
  liveData.systemMetrics = {
    cpu: 20 + Math.random() * 30,
    memory: 40 + Math.random() * 20,
    activeAgents: liveData.agents.size
  };
  
  // Broadcast to all clients
  broadcast({
    type: 'live-update',
    data: liveData
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
    return res.end(JSON.stringify(liveData));
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
    return res.end(JSON.stringify(Array.from(liveData.agents.values())));
  }
  
  if (url.pathname === '/api/events' && req.method === 'POST') {
    // Event ingestion endpoint for agents to post events
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const event = JSON.parse(body);
        console.log('Event received:', event);
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
      const fs = await import('node:fs');
      const html = fs.readFileSync(join(ROOT, 'quillan-dashboard.html'), 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html' });
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
    data: liveData
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

const PORT = 8777;
server.listen(PORT, () => {
  console.log(`Quillan Hub Server running on http://localhost:${PORT}`);
  console.log('WebSocket server ready for live streaming');
  console.log('Dashboard available at http://localhost:8777/dashboard');
});

// Start live data updates
setInterval(updateLiveData, 2000);
updateLiveData();
