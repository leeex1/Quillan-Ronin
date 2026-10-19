import { nativeScreenshot } from './agent-browser.mjs';
const t = Date.now();
try {
  const b64 = await nativeScreenshot();
  console.log(`captured in ${Date.now()-t}ms, base64 length: ${b64.length}`);
  console.log('first 60 chars: ' + b64.slice(0, 60));
  process.exit(0);
} catch (e) {
  console.log(`FAILED in ${Date.now()-t}ms: ${e.message}`);
  process.exit(1);
}
