import { ensureActive } from './agent-browser.mjs';
const s = await ensureActive();
const p = s.page;
const info = await p.evaluate(() => ({
  pieces: document.querySelectorAll('[class*="piece"][class*="square-"]').length,
  highlights: [...document.querySelectorAll('[class*="highlight"]')].map(h => h.className.toString().match(/square-\d\d/)?.[0] || '?'),
  bodyHint: (document.body.innerText || '').split('\n').filter(l => /to move|thinking|check|resign|draw|Checkmate|Game/i.test(l)).slice(0, 6)
}));
console.log(JSON.stringify(info, null, 2));
process.exit(0);
