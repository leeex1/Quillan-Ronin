import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// 1. Refactor scanner.mjs
const scannerPath = './scanner.mjs';
let scanner = readFileSync(scannerPath, 'utf8');

scanner = scanner.replace(
  /export function loadJSON\(name\) \{[\s\S]*?return JSON\.parse\(raw\);\n\}/,
  `export function loadJSON(name) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  const raw = readFileSync(new URL(\`data/\${safeName}.json\`, import.meta.url), 'utf8').replace(/^\\uFEFF/, '');
  return JSON.parse(raw);
}

export async function loadJSONAsync(name) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  const { readFile } = await import('node:fs/promises');
  const raw = (await readFile(new URL(\`data/\${safeName}.json\`, import.meta.url), 'utf8')).replace(/^\\uFEFF/, '');
  return JSON.parse(raw);
}`
);

scanner = scanner.replace(
  /const KERNEL_PATHS = \[[\s\S]*?\];/,
  `const KERNEL_PATHS = [
  'C:\\\\02_QUILLAN\\\\system prompts\\\\Quillan-Samurai.md',
  'C:\\\\Users\\\\Admin\\\\Quillan-Ronin\\\\system prompts\\\\Quillan-Samurai.md',
  'C:\\\\02_QUILLAN\\\\SOUL.md',
  'C:\\\\02_QUILLAN\\\\AGENTS.md',
  'C:\\\\02_QUILLAN\\\\CLAUDE.md'
].filter(p => !p.includes('..'));`
);

writeFileSync(scannerPath, scanner);


// 2. Refactor server.js
const serverPath = './server.js';
let server = readFileSync(serverPath, 'utf8');

server = server.replace(
  /import \{ loadJSON, analyzeLead, dailyBrief/,
  `import { loadJSON, loadJSONAsync, analyzeLead, dailyBrief`
);

server = server.replace(
  /if \(personaCache\.size > 50\) personaCache\.clear\(\);/,
  `if (personaCache.size >= 50) personaCache.delete(personaCache.keys().next().value);`
);

server = server.replace(
  /const ROOT = dirname\(fileURLToPath\(import\.meta\.url\)\);\nconst DATA = join\(ROOT, 'data'\);/,
  `const ROOT = dirname(fileURLToPath(import.meta.url));
const DATA = join(ROOT, 'data');

function logError(context, err) {
  try {
    const msg = \`\${new Date().toISOString()} ERROR [\${context}]: \${err?.stack || err}\\n\`;
    require('node:fs').appendFileSync(join(DATA, 'error.log'), msg);
  } catch (_) {}
}`
);

server = server.replace(/\} catch \{\}/g, `} catch (err) { logError('SilentCatch', err); }`);

server = server.replace(
  /async function saveJSON\(name, obj\) \{[\s\S]*?\}/,
  `async function saveJSON(name, obj) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  await writeFile(join(DATA, \`\${safeName}.json\`), JSON.stringify(obj, null, 2));
}`
);

server = server.replace(
  /if \(!\/\^\[\\\\w\\\\s\\\\-\\\\\\\\:\\\\\/\.\=\]\+\$\/i\.test\(String\(tool\.command \|\| ''\)\)\) return 'refused: shell command has forbidden characters';/,
  `const cmdStr = String(tool.command || '').trim();
        if (!/^(echo|dir|ipconfig|ping|whoami|systeminfo|node)(\\s+[\\w\\s\\-\\:\\/\\.=]+)?$/i.test(cmdStr)) {
          return 'refused: shell command not in allowlist or contains forbidden characters';
        }`
);
server = server.replace(
  /const \{ stdout \} = await execAsync\(String\(tool\.command\), \{ timeout: 20000 \}\);/,
  `const { stdout } = await execAsync(cmdStr, { timeout: 20000 });`
);

server = server.replace(
  /res\.setHeader\('Access-Control-Allow-Origin', '\*'\);/,
  `const origin = req.headers.origin || '*';
    if (origin.startsWith('chrome-extension://') || origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1')) {
      res.setHeader('Access-Control-Allow-Origin', origin);
    } else {
      res.setHeader('Access-Control-Allow-Origin', 'http://localhost');
    }`
);

server = server.replace(/const settings = loadJSON\('settings'\);/g, `const settings = await loadJSONAsync('settings');`);
server = server.replace(/const opps = loadJSON\('opportunities'\);/g, `const opps = await loadJSONAsync('opportunities');`);
server = server.replace(/const ledger = loadJSON\('ledger'\);/g, `const ledger = await loadJSONAsync('ledger');`);

server = server.replace(
  /function ledgerSummaryShort\(\) \{[\s\S]*?return '\$0\.00 earned'; \n\}/,
  `async function ledgerSummaryShort() {
  try {
    const l = await loadJSONAsync('ledger');
    return \`\$\${l.entries.reduce((s, e) => s + Number(e.amount || 0), 0).toFixed(2)} earned\`;
  } catch (err) { logError('ledgerSummaryShort', err); return '$0.00 earned'; }
}`
);

server = server.replace(
  /return \`QuillanWorker up; model \$\{s\.nvidiaModel\}; earnings \$\{ledgerSummaryShort\(\)\}\`;/,
  `return \`QuillanWorker up; model \$\{s.nvidiaModel\}; earnings \$\{await ledgerSummaryShort()\}\`;`
);

server = server.replace(
  /const earned = ledger\.entries\.reduce\(\(s, e\) => s \+ Number\(e\.amount \|\| 0\), 0\);\n\s+res\.writeHead\(200, \{ 'Content-Type': 'application\/json' \}\);\n\s+return res\.end\(JSON\.stringify\(\{ settings, opportunities: opps, ledger, earned \}\)\);/,
  `const earned = ledger.entries.reduce((s, e) => s + Number(e.amount || 0), 0);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ settings, opportunities: opps, ledger, earned }));`
);

writeFileSync(serverPath, server);

// 3. Refactor mcp-manager.mjs
const mcpPath = './mcp-manager.mjs';
let mcp = readFileSync(mcpPath, 'utf8');
mcp = mcp.replace(
  /export function loadMcpConfig\(\) \{[\s\S]*?return JSON\.parse\(raw\)\.mcpServers \|\| \{\};\n\}/,
  `export async function loadMcpConfig() {
  const { existsSync } = await import('node:fs');
  const { readFile } = await import('node:fs/promises');
  if (!existsSync(CONFIG_PATH)) throw new Error('config not found: ' + CONFIG_PATH);
  const raw = (await readFile(CONFIG_PATH, 'utf8')).replace(/^\\uFEFF/, '');
  return JSON.parse(raw).mcpServers || {};
}`
);

mcp = mcp.replace(/const cfg = loadMcpConfig\(\);/, `const cfg = await loadMcpConfig();`);
mcp = mcp.replace(/export function configServerNames\(\) \{/, `export async function configServerNames() {`);
mcp = mcp.replace(/try \{ return Object\.keys\(loadMcpConfig\(\)\); \} catch \{ return \[\]; \}/, `try { return Object.keys(await loadMcpConfig()); } catch (err) { return []; }`);

writeFileSync(mcpPath, mcp);

console.log("Refactoring complete.");
