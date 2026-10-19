import { ensureActive } from './agent-browser.mjs';
try {
  const s = await ensureActive();
  console.log('session page:', s && s.page ? 'EXISTS' : 'MISSING');
  if (s.page) {
    console.log('url:', await s.page.url());
    const b = await s.page.evaluate(() => document.querySelectorAll('[class*="square-"]').length);
    console.log('square elements:', b);
  }
  process.exit(0);
} catch (e) {
  console.log('PROBE ERR:', e.message);
  process.exit(1);
}
