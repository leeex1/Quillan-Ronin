/**
 * Test script for extension bridge
 * 
 * This script tests the command queue system by:
 * 1. Starting the server (if not running)
 * 2. Queueing a test command
 * 3. Verifying the extension can poll and execute it
 */

import { sendCmd } from './ext-bridge.mjs';

console.log('Testing Extension Bridge...\n');

// Test 1: Queue a simple read command
console.log('Test 1: Queueing a read command...');
try {
  const result = await sendCmd('read', { tabId: null });
  console.log('✓ Read command result:', result);
} catch (error) {
  console.log('✗ Read command failed:', error.message);
  console.log('  (This is expected if extension is not loaded yet)');
}

// Test 2: Queue an eval command
console.log('\nTest 2: Queueing an eval command...');
try {
  const result = await sendCmd('eval', { tabId: null, expression: 'document.title' });
  console.log('✓ Eval command result:', result);
} catch (error) {
  console.log('✗ Eval command failed:', error.message);
}

// Test 3: Queue a navigate command
console.log('\nTest 3: Queueing a navigate command...');
try {
  const result = await sendCmd('navigate', { tabId: null, url: 'https://example.com' });
  console.log('✓ Navigate command result:', result);
} catch (error) {
  console.log('✗ Navigate command failed:', error.message);
}

console.log('\n---');
console.log('To test with the actual extension:');
console.log('1. Load the extension in Chrome/Brave');
console.log('2. Open a tab');
console.log('3. Run this test again');
console.log('4. Check browser console for extension logs');
