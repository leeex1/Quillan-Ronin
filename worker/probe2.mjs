import { ensureActive, pickBestPage } from './agent-browser.mjs';
const s = await ensureActive();
console.log('contexts:', s.browser.contexts().length);
for (let ci = 0; ci < s.browser.contexts().length; ci++) {
  const ctx = s.browser.contexts()[ci];
  const pages = ctx.pages ? ctx.pages() : [];
  console.log(`ctx[${ci}] pages: ${pages.length}`);
  for (let pi = 0; pi < pages.length; pi++) {
    try {
      const u = pages[pi].url();
      const n = await pages[pi].evaluate(() => document.querySelectorAll('[class*="piece"][class*="square-"]').length).catch(e => 'ERR:' + e.message.slice(0, 40));
      console.log(`  [${ci}][${pi}] pieces=${n} ${u.slice(0, 70)}`);
    } catch (e) { console.log(`  [${ci}][${pi}] ERR ${e.message.slice(0, 50)}`); }
  }
}
console.log('after pickBestPage, s.page:', s.page ? s.page.url().slice(0, 60) : 'NULL');
process.exit(0);
