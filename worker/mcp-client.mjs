import { spawn, exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

export class McpStdioClient {
  constructor(command, args = [], env = {}) {
    this.command = command;
    this.args = args;
    this.env = env;
    this.proc = null;
    this.pid = null;
    this.nextId = 1;
    this.pending = new Map();
    this.tools = [];
    this.ready = false;
    this.buffer = '';
    this.lastUsed = Date.now();
  }

  async connect(timeoutMs = 20000) {
    if (this.ready && this.proc && !this.proc.killed) {
      this.lastUsed = Date.now();
      return true;
    }
    
    this.kill(); // Ensure previous instance is fully terminated

    this.proc = spawn(this.command, this.args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, ...this.env },
      windowsHide: true
    });
    
    this.pid = this.proc.pid;
    this.lastUsed = Date.now();

    this.proc.on('error', (err) => {
      console.error(`[MCP Client ${this.command}] Process error:`, err.message);
      this.ready = false;
    });

    this.proc.stdout.setEncoding('utf8');
    this.proc.stdout.on('data', chunk => this._onData(chunk));
    
    this.proc.stderr.setEncoding('utf8');
    this.proc.stderr.on('data', (data) => {
      // Keep stderr drained to prevent pipe blocking
    });

    if (this.proc.stdin) {
      this.proc.stdin.on('error', () => {});
    }

    this.proc.on('exit', () => {
      this.ready = false;
      this.proc = null;
      this.pid = null;
    });

    try {
      await this._request('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'quillan-worker', version: '1.5.0' }
      }, timeoutMs);
      
      this._notify('notifications/initialized', {});
      const tools = await this._request('tools/list', {}, timeoutMs);
      this.tools = (tools.tools || []).map(t => ({ name: t.name, description: t.description || '' }));
      this.ready = true;
      this.lastUsed = Date.now();
      return true;
    } catch (err) {
      this.kill();
      throw err;
    }
  }

  _onData(chunk) {
    this.lastUsed = Date.now();
    this.buffer += chunk;
    let idx;
    while ((idx = this.buffer.indexOf('\n')) !== -1) {
      const line = this.buffer.slice(0, idx).trim();
      this.buffer = this.buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve, reject } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          if (msg.error) reject(new Error(msg.error.message || JSON.stringify(msg.error)));
          else resolve(msg.result);
        }
      } catch {}
    }
  }

  _notify(method, params) {
    if (!this.proc || !this.proc.stdin || !this.proc.stdin.writable) return;
    this.proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n');
  }

  _request(method, params, timeoutMs = 60000) {
    const id = this.nextId++;
    if (!this.proc || !this.proc.stdin || !this.proc.stdin.writable) {
      return Promise.reject(new Error('MCP process not writable'));
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP ${method} timed out`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: v => { clearTimeout(timer); resolve(v); },
        reject: e => { clearTimeout(timer); reject(e); }
      });
      try {
        this.proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
      } catch (e) {
        this.pending.delete(id);
        clearTimeout(timer);
        reject(e);
      }
    });
  }

  async callTool(name, args = {}, timeoutMs = 120000) {
    this.lastUsed = Date.now();
    const r = await this._request('tools/call', { name, arguments: args }, timeoutMs);
    const text = (r.content || []).filter(c => c.type === 'text').map(c => c.text).join('\n');
    return { ok: !r.isError, text };
  }

  kill() {
    this.ready = false;
    const targetPid = this.pid;
    if (this.proc) {
      try {
        this.proc.stdin?.destroy();
        this.proc.stdout?.destroy();
        this.proc.stderr?.destroy();
        this.proc.kill('SIGTERM');
      } catch {}
      this.proc = null;
    }
    
    // On Windows, cleanly tree-kill child processes (npx, python, etc.) to prevent orphans
    if (targetPid && process.platform === 'win32') {
      try {
        execAsync(`taskkill /PID ${targetPid} /T /F`).catch(() => {});
      } catch {}
    }
    this.pid = null;
  }
}
