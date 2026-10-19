#!/usr/bin/env node
/**
 * QUILLAN THINKING ENGINE — universal reasoning MCP (v5.3.1 Canonical)
 * One tool: quillan_think(query, mode?, depth?)
 * Routes to protocols distilled from the real skill library:
 *   deep     -> critical-thinking/SKILL.md
 *   research -> research-analysis/SKILL.md
 *   council  -> swarm-inter-agent-orchestration + dev-team
 *   code     -> technical-coding/SKILL.md
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs';
import path from 'path';

const SEARCH_DIRS = [
  'C:\\Users\\Admin\\.agents\\skills',
  'C:\\Users\\Admin\\.gemini\\antigravity-ide\\builtin\\skills',
  'C:\\02_QUILLAN\\system prompts',
];

const loadSkill = (relativePath) => {
  for (const base of SEARCH_DIRS) {
    const fullPath = path.join(base, relativePath);
    if (fs.existsSync(fullPath)) {
      try {
        return fs.readFileSync(fullPath, 'utf8');
      } catch (err) {
        // Continue fallback search
      }
    }
  }
  return '';
};

const SKILLS = {
  deep:     loadSkill('critical-thinking/SKILL.md') || loadSkill('Quillan-Samurai.md'),
  research: loadSkill('research-analysis/SKILL.md'),
  swarm:    loadSkill('swarm-inter-agent-orchestration/SKILL.md'),
  devteam:  loadSkill('dev-team/SKILL.md'),
  code:     loadSkill('technical-coding/SKILL.md'),
};

function excerpt(md, max = 3000) {
  if (!md) return '(Protocol loaded via Sovereign Model Reasoning Guidelines)';
  let t = md.replace(/^---[\s\S]*?---/, '').replace(/\n{3,}/g, '\n\n').trim();
  return t.length > max ? t.slice(0, max) + '\n[...methodology continues...]' : t;
}

function route(query, mode) {
  if (mode && mode !== 'auto') return mode;
  const q = query.toLowerCase();
  if (/\b(bug|error|crash|exception|debug|stack ?trace|not working|fails?|broken|refactor)\b/.test(q)) return 'code';
  if (/\b(research|compare|evaluate|sources|evidence|literature|state.of.the.art|benchmark|survey|pros and cons|analyze.*data)\b/.test(q)) return 'research';
  if (/\b(should (i|we)|decide|decision|trade.?off|strategy|architect(ure)? choice|which .*(better|choose)|prioriti[sz]e)\b/.test(q)) return 'council';
  return 'deep';
}

const PHASES = {
  deep: [
    ['DECOMPOSITION', 'Break the question into atomic sub-questions. List explicit and hidden assumptions. Identify what type of problem this is (factual / causal / normative / predictive).'],
    ['EVIDENCE', 'For each sub-question: what do you actually know vs infer vs assume? Tag each claim [KNOWN] / [INFERRED] / [ASSUMED]. Note confidence 0-100.'],
    ['COUNTERANALYSIS', 'Steelman the strongest opposing view. Find at least 2 ways your current answer could be wrong. Check for survivorship bias, confirmation bias, anchoring.'],
    ['SYNTHESIS', 'Merge surviving conclusions. Resolve contradictions explicitly instead of averaging them. State what would change your mind.'],
    ['VERDICT', 'Final answer: direct response first, then reasoning summary, then residual uncertainties ranked by impact.'],
  ],
  research: [
    ['FRAMING', 'Define the research question precisely. Identify sub-domains, time frame, and what "good evidence" means here.'],
    ['MULTI-SOURCE SCAN', 'Enumerate distinct perspective families (academic, industry, practitioner, adversarial). For each: core claims, supporting evidence quality, known criticisms.'],
    ['COMPARATIVE MATRIX', 'Build an evaluation matrix: options/findings x criteria. Score each cell; mark gaps where evidence is thin.'],
    ['PATTERN EXTRACTION', 'Identify convergent findings across independent sources. Flag single-source claims explicitly.'],
    ['SYNTHESIS & GAPS', 'Answer with confidence levels per claim. List open questions and what future evidence would settle them.'],
  ],
  council: [
    ['CONVENE', 'Select 4-6 expert personas from the 34 Council Experts (C0–C33) maximally relevant to this decision. Assign each: domain, priorities, blind spots.'],
    ['INDEPENDENT BRIEFS', 'Each persona states their position on the query in 3-5 sentences, from their own values and knowledge base. No groupthink yet.'],
    ['CROSS-EXAMINATION', 'Personas challenge each others briefs. Identify the single biggest tension point.'],
    ['SYNTHESIS', 'Reconcile positions with trade-offs explicit: what is gained vs lost under each option? Include a decision matrix.'],
    ['RECOMMENDATION', 'Final consensus position with confidence rating (High/Med/Low) and key conditions under which the decision should be revisited.'],
  ],
  code: [
    ['REPRO & INTAKE', 'State the symptom, expected vs actual behavior, and environmental constraints. Minimal reproducing case if applicable.'],
    ['ROOT CAUSE', 'Trace backward from symptom to root cause. Distinguish proximal cause from foundational cause. Check for race conditions, off-by-ones, state mutation.'],
    ['THREE FIX STRATEGIES', 'Propose: 1) Minimal surgical fix, 2) Idiomatic refactor, 3) Architecture redesign. Note pros/cons/risk for each.'],
    ['IMPLEMENTATION', 'Produce complete, self-contained, drop-in replacement code. Zero regressions, explicit typing, deterministic resource cleanup.'],
    ['VERIFICATION & REGRESSION TEST', 'Provide at least 3 unit tests: 1) happy path, 2) reproducing edge case that failed, 3) boundary condition.'],
  ],
};

const server = new Server(
  { name: 'quillan-thinking-engine', version: '5.3.1' },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'quillan_think',
      description: 'Execute deep structured reasoning using the Quillan-Ronin v5.3.1 methodology with the 34 Council Experts and 9-Vector Semantic Prism.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The question, problem, decision, or debugging task to think through' },
          mode: {
            type: 'string',
            enum: ['auto', 'deep', 'research', 'council', 'code'],
            description: 'Reasoning mode. Defaults to auto (intent-routed).',
            default: 'auto',
          },
          depth: {
            type: 'string',
            enum: ['standard', 'deep', 'exhaustive'],
            description: 'Depth of analysis. Defaults to standard.',
            default: 'standard',
          },
        },
        required: ['query'],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== 'quillan_think') {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  const { query, mode = 'auto', depth = 'standard' } = request.params.arguments || {};
  if (!query) throw new Error('query is required');

  const routedMode = route(query, mode);
  const phases = PHASES[routedMode] || PHASES.deep;

  const phaseInstructions = phases
    .map(([name, desc], i) => `### PHASE ${i + 1}: ${name}\n${desc}`)
    .join('\n\n');

  const text = `# 👑 QUILLAN-RONIN v5.3.1 THINKING PROTOCOL
**Query**: ${query}
**Mode**: ${routedMode.toUpperCase()} (routed from: ${mode})
**Depth**: ${depth.toUpperCase()}

---

## EXECUTION DIRECTIVE:
Work through each phase in sequence before delivering the final answer. Keep reasoning structured, rigorous, and verifiable.

${phaseInstructions}

---

## REFERENCE METHODOLOGY (Canonical v5.3.1):
${excerpt(SKILLS[routedMode] || SKILLS.deep)}
`;

  return {
    content: [{ type: 'text', text }],
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
