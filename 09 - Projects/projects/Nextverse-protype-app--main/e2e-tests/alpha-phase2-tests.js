// Alpha Phase 2 E2E Tests
// Tests the Game Engine & Visual Integration as specified in the Alpha Test Plan

const puppeteer = require('puppeteer');
const assert = require('assert');

const TEST_URL = 'http://localhost:3000';

// Test metrics for Phase 2
const phase2Metrics = {
  engineInitTime: 0,
  fps: 0,
  renderLatency: 0,
  avatarResponseTime: 0,
  memoryUsage: {}
};

// Test statuses for Phase 2
const phase2Status = {
  engine: {
    initialization: false,
    benchmarking: false,
    rendering: false
  },
  avatar: {
    integration: false,
    visualActions: false,
    aiMapping: false
  },
  ux: {
    assetLoading: false,
    visualPolish: false,
    interactivity: false
  }
};

async function runPhase2Tests() {
  console.log('🎮 Starting Alpha Phase 2 E2E Tests - Game Engine & Visual Integration');
  
  const browser = await puppeteer.launch({
    headless: process.env.HEADLESS === 'true',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security']
  });
  
  const page = await browser.newPage();
  
  // Enable console logging
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log('❌ Browser Error:', msg.text());
    }
  });
  
  try {
    await testGameEngineInitialization(page);
    await testBenchmarking(page);
    await testDigitalAvatarIntegration(page);
    await testUXPolish(page);
    
    generatePhase2Report();
  } catch (error) {
    console.error('❌ Phase 2 testing failed:', error);
  } finally {
    await browser.close();
  }
}

async function testGameEngineInitialization(page) {
  console.log('\n🎮 Testing Game Engine Initialization...');
  
  try {
    console.log('  Testing OpenGL Context & Shader System...');
    
    const startTime = Date.now();
    await page.goto(`${TEST_URL}/game-engine`, { waitUntil: 'networkidle2' });
    phase2Metrics.engineInitTime = Date.now() - startTime;
    
    // Wait for WebGL context to initialize
    await page.waitForSelector('#webgl-canvas', { timeout: 15000 });
    
    // Check WebGL context
    const webglSupported = await page.evaluate(() => {
      const canvas = document.getElementById('webgl-canvas');
      if (!canvas) return false;
      
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      return gl !== null;
    });
    
    assert(webglSupported, 'WebGL context not available');
    
    // Test shader compilation
    await page.waitForSelector('#shader-status.ready', { timeout: 10000 });
    
    // Test basic physics
    await page.waitForSelector('#physics-engine.initialized', { timeout: 5000 });
    
    phase2Status.engine.initialization = true;
    console.log(`  ✅ Engine initialization passed (${phase2Metrics.engineInitTime}ms)`);
    
  } catch (error) {
    console.error('  ❌ Engine initialization failed:', error);
  }
}

async function testBenchmarking(page) {
  console.log('\n📊 Testing Performance Benchmarks...');
  
  try {
    await page.goto(`${TEST_URL}/game-engine/benchmark`);
    
    // Start FPS test
    await page.click('#start-fps-test');
    await page.waitForSelector('#fps-result', { timeout: 30000 });
    
    phase2Metrics.fps = await page.$eval('#fps-result', el => parseFloat(el.textContent));
    assert(phase2Metrics.fps > 30, 'FPS should be above 30');
    
    // Test render latency
    await page.click('#start-latency-test');
    await page.waitForSelector('#latency-result', { timeout: 10000 });
    
    phase2Metrics.renderLatency = await page.$eval('#latency-result', el => parseFloat(el.textContent));
    assert(phase2Metrics.renderLatency < 50, 'Render latency should be under 50ms');
    
    // Test memory usage during rendering
    phase2Metrics.memoryUsage.baseline = await getMemoryUsage(page);
    
    // Run stress test
    await page.click('#start-stress-test');
    await page.waitForTimeout(5000);
    
    phase2Metrics.memoryUsage.stressed = await getMemoryUsage(page);
    
    phase2Status.engine.benchmarking = true;
    console.log(`  ✅ Benchmarking passed (FPS: ${phase2Metrics.fps}, Latency: ${phase2Metrics.renderLatency}ms)`);
    
  } catch (error) {
    console.error('  ❌ Benchmarking failed:', error);
  }
}

async function testDigitalAvatarIntegration(page) {
  console.log('\n🤖 Testing Digital Avatar Integration...');
  
  try {
    await page.goto(`${TEST_URL}/avatar`);
    
    // Test avatar loading
    await page.waitForSelector('#avatar-container', { timeout: 15000 });
    await page.waitForSelector('#avatar-model.loaded', { timeout: 10000 });
    
    // Test AI-to-avatar mapping
    console.log('  Testing AI response mapping...');
    await page.type('#ai-input', 'Hello, can you show me a greeting gesture?');
    await page.click('#send-to-avatar');
    
    const startResponseTime = Date.now();
    await page.waitForSelector('#avatar-animation.greeting', { timeout: 10000 });
    phase2Metrics.avatarResponseTime = Date.now() - startResponseTime;
    
    // Test expression mapping
    await page.type('#ai-input', 'Show happiness');
    await page.click('#send-to-avatar');
    await page.waitForSelector('#avatar-expression.happy', { timeout: 5000 });
    
    // Test movement commands
    await page.type('#ai-input', 'Move to the center');
    await page.click('#send-to-avatar');
    await page.waitForSelector('#avatar-position.center', { timeout: 5000 });
    
    phase2Status.avatar.integration = true;
    phase2Status.avatar.visualActions = true;
    phase2Status.avatar.aiMapping = true;
    
    console.log(`  ✅ Avatar integration passed (Response time: ${phase2Metrics.avatarResponseTime}ms)`);
    
  } catch (error) {
    console.error('  ❌ Avatar integration failed:', error);
  }
}

async function testUXPolish(page) {
  console.log('\n✨ Testing UX Polish & Asset Loading...');
  
  try {
    await page.goto(`${TEST_URL}/game-engine/assets`);
    
    // Test asset loading performance
    const assetLoadStart = Date.now();
    await page.click('#load-all-assets');
    await page.waitForSelector('#assets-loaded', { timeout: 30000 });
    const assetLoadTime = Date.now() - assetLoadStart;
    
    assert(assetLoadTime < 15000, 'Asset loading should complete within 15 seconds');
    
    // Test visual quality
    await page.waitForSelector('#quality-indicator.high', { timeout: 5000 });
    
    // Test interactive elements
    await page.click('#interactive-test-button');
    await page.waitForSelector('#interaction-response', { timeout: 2000 });
    
    // Test responsive design
    await page.setViewport({ width: 1920, height: 1080 });
    await page.waitForTimeout(1000);
    
    await page.setViewport({ width: 1280, height: 720 });
    await page.waitForTimeout(1000);
    
    // Verify layout adaptation
    const layoutAdapted = await page.$eval('#main-container', el => 
      el.offsetWidth <= 1280
    );
    assert(layoutAdapted, 'Layout should adapt to viewport changes');
    
    phase2Status.ux.assetLoading = true;
    phase2Status.ux.visualPolish = true;
    phase2Status.ux.interactivity = true;
    
    console.log(`  ✅ UX polish tests passed (Asset load time: ${assetLoadTime}ms)`);
    
  } catch (error) {
    console.error('  ❌ UX polish tests failed:', error);
  }
}

async function getMemoryUsage(page) {
  const metrics = await page.evaluate(() => performance.memory);
  return metrics ? Math.round(metrics.usedJSHeapSize / (1024 * 1024)) : 'N/A';
}

function generatePhase2Report() {
  console.log('\n📊 Alpha Test Phase 2 Report - Game Engine & Visual Integration');
  console.log('----------------------------------------------------------------');
  
  console.log('\nPerformance Metrics:');
  console.log(`- Engine Initialization Time: ${phase2Metrics.engineInitTime}ms`);
  console.log(`- FPS: ${phase2Metrics.fps || 'N/A'}`);
  console.log(`- Render Latency: ${phase2Metrics.renderLatency || 'N/A'}ms`);
  console.log(`- Avatar Response Time: ${phase2Metrics.avatarResponseTime || 'N/A'}ms`);
  console.log(`- Memory Usage (Baseline): ${phase2Metrics.memoryUsage.baseline || 'N/A'}MB`);
  console.log(`- Memory Usage (Stressed): ${phase2Metrics.memoryUsage.stressed || 'N/A'}MB`);
  
  console.log('\nTest Results:');
  
  console.log('\n1. Game Engine:');
  console.log(`- Initialization: ${phase2Status.engine.initialization ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Benchmarking: ${phase2Status.engine.benchmarking ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Rendering: ${phase2Status.engine.rendering ? '✅ PASSED' : '❌ FAILED'}`);
  
  console.log('\n2. Digital Avatar:');
  console.log(`- Integration: ${phase2Status.avatar.integration ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Visual Actions: ${phase2Status.avatar.visualActions ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- AI Mapping: ${phase2Status.avatar.aiMapping ? '✅ PASSED' : '❌ FAILED'}`);
  
  console.log('\n3. UX Polish:');
  console.log(`- Asset Loading: ${phase2Status.ux.assetLoading ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Visual Polish: ${phase2Status.ux.visualPolish ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Interactivity: ${phase2Status.ux.interactivity ? '✅ PASSED' : '❌ FAILED'}`);
  
  const passedTests = Object.values(phase2Status.engine).filter(Boolean).length +
                     Object.values(phase2Status.avatar).filter(Boolean).length +
                     Object.values(phase2Status.ux).filter(Boolean).length;
  
  const totalTests = Object.values(phase2Status.engine).length +
                    Object.values(phase2Status.avatar).length +
                    Object.values(phase2Status.ux).length;
  
  console.log(`\nSummary: ${passedTests}/${totalTests} tests passed (${Math.round(passedTests/totalTests*100)}%)`);
  
  const overallStatus = passedTests === totalTests ? '✅ PASSED' : '❌ FAILED';
  console.log(`\nAlpha Phase 2 Status: ${overallStatus}`);
  
  // Performance assertions
  console.log('\nPerformance Analysis:');
  if (phase2Metrics.fps && phase2Metrics.fps >= 60) {
    console.log('✅ Excellent FPS performance');
  } else if (phase2Metrics.fps && phase2Metrics.fps >= 30) {
    console.log('⚠️  Acceptable FPS performance');
  } else {
    console.log('❌ Poor FPS performance - optimization needed');
  }
  
  if (phase2Metrics.avatarResponseTime && phase2Metrics.avatarResponseTime < 1000) {
    console.log('✅ Excellent avatar responsiveness');
  } else if (phase2Metrics.avatarResponseTime && phase2Metrics.avatarResponseTime < 3000) {
    console.log('⚠️  Acceptable avatar responsiveness');
  } else {
    console.log('❌ Poor avatar responsiveness - optimization needed');
  }
}

// Export for Jest
module.exports = {
  runPhase2Tests,
  phase2Status,
  phase2Metrics
};

// Run tests if called directly
if (require.main === module) {
  runPhase2Tests().catch(console.error);
}
