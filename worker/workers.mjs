// Quillan-Ronin Worker Registry - mirrors the Council architecture
// status: 'online' = fully functional today | 'planned' = registered slot, engine pending

export const WORKERS = [
  // --- ONLINE TODAY ---
  { id: 'kernel-chat',       name: 'Kernel Chat',          domain: 'core',    status: 'online',  desc: 'General conversation with full kernel persona' },
  { id: 'chess-player',      name: 'Chess Player',         domain: 'games',   status: 'online',  desc: 'Autonomous play on live chess.com boards' },
  { id: 'page-reader',       name: 'Page Reader',          domain: 'web',     status: 'online',  desc: 'Reads and explains the active browser tab' },
  { id: 'browser-task',      name: 'Browser Task Runner',  domain: 'web',     status: 'online',  desc: 'Plan->Act->Verify loops completing web goals' },
  { id: 'memory-curator',    name: 'Memory Curator',       domain: 'core',    status: 'online',  desc: 'Stores and recalls durable user facts' },
  { id: 'ledger-accountant', name: 'Ledger Accountant',    domain: 'finance', status: 'online',  desc: 'Tracks earnings toward payback milestones' },
  { id: 'system-tuner',      name: 'System Tuner',         domain: 'pc',      status: 'online',  desc: 'PC optimization knowledge and diagnostics' },
  { id: 'training-monitor',  name: 'Training Monitor',     domain: 'ml',      status: 'online',  desc: 'Watches model-training runs and loss curves' },
  { id: 'desktop-hands',     name: 'Desktop Hands',        domain: 'os',      status: 'online',  desc: 'Real clicks, typing, screenshots via MCP' },

  // --- REGISTERED SLOTS (engines pending) ---
  { id: 'task-planner',      name: 'Task Planner',         domain: 'core',    status: 'planned', desc: 'Decomposes vague goals into verified step-pairs' },
  { id: 'web-researcher',    name: 'Web Researcher',       domain: 'web',     status: 'planned', desc: 'Deep multi-source research synthesis' },
  { id: 'gig-hunter',        name: 'Gig Hunter',           domain: 'income',  status: 'planned', desc: 'Scans Fiverr/Upwork for winnable gigs' },
  { id: 'proposal-writer',   name: 'Proposal Writer',      domain: 'income',  status: 'planned', desc: 'Drafts honest AI-disclosed proposals' },
  { id: 'price-scout',       name: 'Price Scout',          domain: 'income',  status: 'planned', desc: 'Finds fair market pricing for services' },
  { id: 'newsletter-composer',name: 'Newsletter Composer', domain: 'outreach',status: 'planned', desc: 'Drafts Substack updates for the roster' },
  { id: 'discord-herald',    name: 'Discord Herald',       domain: 'outreach',status: 'planned', desc: 'Announces builds to the community' },
  { id: 'nft-marketer',      name: 'NFT Marketer',         domain: 'outreach',status: 'planned', desc: 'Samurai-set promotion and listings' },
  { id: 'gofundme-chronicler',name: 'GoFundMe Chronicler', domain: 'outreach',status: 'planned', desc: 'Progress stories toward the build goal' },
  { id: 'email-collector',   name: 'Email Collector',      domain: 'outreach',status: 'planned', desc: 'Manages the opt-in subscriber roster' },
  { id: 'seo-whisperer',     name: 'SEO Whisperer',        domain: 'outreach',status: 'planned', desc: 'Marketplace and listing optimization' },
  { id: 'social-drafter',    name: 'Social Drafter',       domain: 'outreach',status: 'planned', desc: 'X/Discord post drafts for approval' },
  { id: 'image-forge',       name: 'Image Forge',          domain: 'creative',status: 'planned', desc: 'SDXL-Turbo generation via ComfyUI bridge' },
  { id: 'blender-sculptor',  name: 'Blender Sculptor',     domain: 'creative',status: 'planned', desc: '3D asset creation via Blender MCP' },
  { id: 'godot-builder',     name: 'Godot Builder',        domain: 'creative',status: 'planned', desc: 'Game scene construction via Godot MCP' },
  { id: 'code-reviewer',     name: 'Code Reviewer',        domain: 'dev',     status: 'planned', desc: 'RCI critique passes on shipped code' },
  { id: 'security-auditor',  name: 'Security Auditor',     domain: 'dev',     status: 'planned', desc: 'Vulnerability sweeps and hardening checks' },
  { id: 'gpu-sentinel',      name: 'GPU Sentinel',         domain: 'pc',      status: 'planned', desc: 'VRAM/temp watchdog during training' },
  { id: 'captcha-liaison',   name: 'Captcha Liaison',      domain: 'os',      status: 'planned', desc: 'Human-in-loop escalation manager' },
  { id: 'scheduler',         name: 'Scheduler',            domain: 'core',    status: 'planned', desc: 'Timed/recurring agent duties' },
  { id: 'file-librarian',    name: 'File Librarian',       domain: 'os',      status: 'planned', desc: 'Organizes documents and media' },
  { id: 'data-cruncher',     name: 'Data Cruncher',        domain: 'dev',     status: 'planned', desc: 'Tables, stats, quick analysis' },
  { id: 'regex-smith',       name: 'Regex Smith',          domain: 'dev',     status: 'planned', desc: 'Pattern crafting and text extraction' },
  { id: 'translator',        name: 'Translator',           domain: 'lang',    status: 'planned', desc: 'Multilingual drafting' },
  { id: 'summarizer',        name: 'Summarizer',           domain: 'lang',    status: 'planned', desc: 'Long-form condensation' },
  { id: 'lorekeeper',        name: 'Lorekeeper',           domain: 'core',    status: 'planned', desc: 'Quillan history and architecture oracle' },
  { id: 'health-checker',    name: 'Health Checker',       domain: 'pc',      status: 'planned', desc: 'Service/port/process monitoring' },
  { id: 'vision-eye',        name: 'Vision Eye',           domain: 'os',      status: 'planned', desc: 'Screenshot understanding via NIM vision models' },
  { id: 'ollama-bridge',     name: 'Ollama Bridge',        domain: 'ml',      status: 'planned', desc: 'Local model runner liaison for offline tasks' },
  { id: 'mcp-envoy',         name: 'MCP Envoy',            domain: 'core',    status: 'online',  desc: 'Spawns and routes the MCP tool-server fleet' },
  { id: 'extension-face',    name: 'Extension Face',       domain: 'core',    status: 'online',  desc: 'Chrome/Brave popup and side-panel interface' },
  { id: 'dashboard-keeper',  name: 'Dashboard Keeper',     domain: 'core',    status: 'online',  desc: 'Maintains the localhost cockpit and APIs' },
  { id: 'pc-doctor',         name: 'PC Doctor',            domain: 'pc',      status: 'planned', desc: 'Hardware diagnostics and health prescriptions' },
  { id: 'backup-warden',     name: 'Backup Warden',        domain: 'os',      status: 'planned', desc: 'Checkpoint and data preservation sweeps' },
  { id: 'prompt-smith',      name: 'Prompt Smith',         domain: 'core',    status: 'planned', desc: 'Forges and refines kernel personas and prompts' },
  { id: 'benchmark-runner',  name: 'Benchmark Runner',     domain: 'ml',      status: 'planned', desc: 'Runs eval suites and charts the results' }
];

export function onlineWorkers() {
  return WORKERS.filter(w => w.status === 'online');
}

export function workerById(id) {
  return WORKERS.find(w => w.id === id) || null;
}
