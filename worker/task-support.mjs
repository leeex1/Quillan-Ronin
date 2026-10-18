import { callNVIDIA, loadJSON } from './scanner.mjs';
import { callTool } from './mcp-manager.mjs';

export { callNVIDIA, loadJSON };

export async function callToolViaMcp(server, tool, args = {}, timeoutMs = 60000) {
  return callTool(server, tool, args, timeoutMs);
}
