import { ensureActive } from './agent-browser.mjs';
const s = await ensureActive();
const p = s.page;
console.log('url:', p.url());
const info = await p.evaluate(() => {
  const btns = [...document.querySelectorAll('button, [role="button"], a')]
    .filter(b => b.innerText && b.innerText.toLowerCase().includes('start'))
    .map(b => ({ txt: b.innerText.trim().slice(0, 40), visible: b.offsetParent !== null, cls: b.className.toString().slice(0, 60) }));
  return { btns, boardVisible: !!document.querySelector('[class*="board"]'), pieces: document.querySelectorAll('[class*="piece"][class*="square-"]').length };
});
console.log(JSON.stringify(info, null, 2));
process.exit(0);
