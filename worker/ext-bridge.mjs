/**
 * Extension Bridge Module
 * 
 * Provides a driver interface for the QuillanWorker task engine to communicate
 * with the Chrome extension via the command queue system.
 * 
 * Usage:
 *   import { sendCmd } from './ext-bridge.mjs';
 *   const result = await sendCmd('read', { tabId: 123 });
 */

const SERVER_URL = 'http://localhost:7777';
const AGENT_ID = 'quillan-ronin';
const POLL_INTERVAL = 100; // ms
const MAX_POLL_TIME = 30000; // 30 seconds timeout

/**
 * Send a command to the extension and wait for the result
 * @param {string} op - Operation name (navigate, eval, read, click, screenshot)
 * @param {object} params - Operation parameters
 * @returns {Promise<object>} Result from the extension
 */
export async function sendCmd(op, params = {}, timeoutMs = 6000) {
  // Queue the command
  const queueResponse = await fetch(`${SERVER_URL}/api/ext/queue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ op, params })
  });

  if (!queueResponse.ok) {
    throw new Error(`Failed to queue command: ${queueResponse.statusText}`);
  }

  const { commandId } = await queueResponse.json();
  
  // Poll for result
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
    
    const resultResponse = await fetch(`${SERVER_URL}/api/ext/result?commandId=${commandId}`);
    if (!resultResponse.ok) continue;
    
    const resultData = await resultResponse.json();
    if (resultData.result !== undefined) {
      return resultData.result;
    }
  }
  
  throw new Error(`Command ${op} timed out after ${timeoutMs}ms`);
}

/**
 * Convenience methods for common operations
 */
export const extDriver = {
  /**
   * Navigate to a URL
   */
  async navigate(tabId, url) {
    return await sendCmd('navigate', { tabId, url });
  },
  
  /**
   * Evaluate JavaScript in the active tab
   */
  async eval(tabId, expression) {
    return await sendCmd('eval', { tabId, expression });
  },
  
  /**
   * Read the page text content
   */
  async read(tabId) {
    return await sendCmd('read', { tabId });
  },
  
  /**
   * Click an element by selector
   */
  async click(tabId, selector) {
    return await sendCmd('click', { tabId, selector });
  },
  
  /**
   * Capture a screenshot
   */
  async screenshot(tabId) {
    return await sendCmd('screenshot', { tabId });
  },
  
  /**
   * Get context for a specific tab
   */
  async getContext(tabId) {
    return await sendCmd('getContext', { tabId });
  },
  
  /**
   * Get all tab contexts
   */
  async getAllContexts() {
    return await sendCmd('getAllContexts', {});
  },
  
  /**
   * Enable autonomy for a tab
   */
  async enableAutonomy(tabId) {
    return await sendCmd('enableAutonomy', { tabId });
  },
  
  /**
   * Disable autonomy for a tab
   */
  async disableAutonomy(tabId) {
    return await sendCmd('disableAutonomy', { tabId });
  },
  
  /**
   * Check if autonomy is enabled for a tab
   */
  async isAutonomyEnabled(tabId) {
    return await sendCmd('isAutonomyEnabled', { tabId });
  },
  
  /**
   * Ping the extension to check if it's alive
   */
  async ping() {
    return await sendCmd('ping', {});
  }
};

export default extDriver;
