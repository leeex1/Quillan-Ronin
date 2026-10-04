import { nativeScreenshot } from './agent-browser.mjs';
import { visionDescribe, loadJSON } from './scanner.mjs';
const t = Date.now();
try {
  const b64 = await nativeScreenshot();
  console.log('shot: ' + b64.length + 'b in ' + (Date.now()-t) + 'ms');
  const settings = loadJSON('settings');
  const desc = await visionDescribe(b64, 'Describe what is on this screen. What windows, apps, and UI elements are visible?', settings);
  console.log('VISION SAYS: ' + desc.slice(0, 500));
  process.exit(0);
} catch (e) { console.log('ERR: ' + e.message); process.exit(1); }
