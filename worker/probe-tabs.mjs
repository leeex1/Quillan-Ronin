import { chromium } from 'playwright-core';
const require2 = (await import('node:module')).createRequire(import.meta.url);
const browser = await chromium.connectOverCDP('http://127.0.0.1:9223', { timeout: 10000 });
for (let ci = 0; ci < browser.contexts().length; ci++) {
  const ctx = browser.contexts()[ci];
  for (let pi = 0; pi < ctx.pages().length; pi++) {
    const pg = ctx.pages()[pi];
    try {
      const n = await pg.evaluate(() => document.querySelectorAll('[class*="piece"][class*="square-"]').length).catch(e => 'ERR');
      const title = await pg.title().catch(() => '');
      console.log(`[${ci}][${pi}] pieces=${n} | ${title.slice(0, 40)} | ${pg.url().slice(0, 80)}`);
    } catch (e) { console.log(`[${ci}][${pi}] ERR ${e.message.slice(0, 50)}`); }
  }
}
process.exit(0);
