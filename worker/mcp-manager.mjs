import { McpStdioClient } from './mcp-client.mjs';
import { readFileSync, existsSync } from 'node:fs';
import { execSync, exec } from 'node:child_process';

const CONFIG_PATH = 'C:\\02_QUILLAN\\mcp\\quillan-mcp-config.json';
const clients = new Map();
let reaperInterval = null;

function resolveCmd(cmd) {
  const base = cmd.split(/[\\/]/).pop().toLowerCase();
  try {
    const p = execSync(`where ${base}`, { encoding: 'utf8', timeout: 8000 }).split(/\r?\n/)[0].trim();
    if (p) return p;
  } catch {}
  return cmd;
}

export async function loadMcpConfig() {
  const { existsSync } = await import('node:fs');
  const { readFile } = await import('node:fs/promises');
  if (!existsSync(CONFIG_PATH)) throw new Error('config not found: ' + CONFIG_PATH);
  const raw = (await readFile(CONFIG_PATH, 'utf8')).replace(/^\uFEFF/, '');
  return JSON.parse(raw).mcpServers || {};
}

export function mcpStatus() {
  const out = {};
  for (const [name, c] of clients) {
    out[name] = { ready: c.ready, pid: c.pid, tools: c.tools.map(t => t.name), lastUsed: c.lastUsed };
  }
  return out;
}

export async function ensureServer(name) {
  if (clients.has(name)) {
    const c = clients.get(name);
    if (!c.ready) await c.connect();
    return c;
  }
  const cfg = await loadMcpConfig();
  const entry = cfg[name];
  if (!entry || entry.url) throw new Error(`server "${name}" not configured as stdio`);
  let cmd = entry.command;
  if (['uvx', 'npx'].includes(cmd)) cmd = resolveCmd(cmd);
  const client = new McpStdioClient(cmd, entry.args || [], entry.env || {});
  clients.set(name, client);
  await client.connect();
  ensureReaper();
  return client;
}

export async function ensureAll(names) {
  const results = {};
  for (const name of names) {
    try { await ensureServer(name); results[name] = 'ready'; }
    catch (e) { results[name] = 'failed: ' + e.message.slice(0, 80); }
  }
  return results;
}

export async function callTool(serverName, toolName, args = {}, timeoutMs = 120000) {
  const client = await ensureServer(serverName);
  return client.callTool(toolName, args, timeoutMs);
}

export function stopServer(name) {
  if (clients.has(name)) {
    const c = clients.get(name);
    c.kill();
    clients.delete(name);
    return true;
  }
  return false;
}

export function stopAllServers() {
  let count = 0;
  for (const [name, client] of clients) {
    try {
      client.kill();
      count++;
    } catch {}
  }
  clients.clear();
  return count;
}

export async function cleanGhostProcesses() {
  try {
    if (process.platform === 'win32') {
      // Fast, non-blocking taskkill targeting orphaned windows-mcp instances
      await new Promise((resolve) => {
        exec('taskkill /F /IM windows-mcp.exe', (err) => resolve({ err }));
      });
    }
    return { cleaned: true };
  } catch (e) {
    return { cleaned: false, error: e.message };
  }
}

function ensureReaper() {
  if (reaperInterval) return;
  // Prune idle MCP servers inactive for > 15 minutes to save RAM/CPU
  reaperInterval = setInterval(() => {
    const now = Date.now();
    const IDLE_LIMIT = 15 * 60 * 1000;
    for (const [name, client] of clients) {
      if (now - client.lastUsed > IDLE_LIMIT) {
        console.log(`[MCP Manager] Reaping idle server: ${name} (pid ${client.pid})`);
        client.kill();
        clients.delete(name);
      }
    }
  }, 60000);
}

export async function configServerNames() {
  try { return Object.keys(await loadMcpConfig()); } catch (err) { return []; }
}

// Global process lifecycle hooks to prevent orphan MCP processes on server shutdown
const cleanup = () => {
  stopAllServers();
};
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(0); });
process.on('SIGTERM', () => { cleanup(); process.exit(0); });
process.on('SIGBREAK', () => { cleanup(); process.exit(0); });

