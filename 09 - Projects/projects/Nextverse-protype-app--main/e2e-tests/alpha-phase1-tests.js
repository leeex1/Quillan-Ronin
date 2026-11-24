// Alpha Phase 1 E2E Tests
// Tests the Core Components & AI Tuning as specified in the Alpha Test Plan

const puppeteer = require('puppeteer');
const assert = require('assert');

const TEST_URL = 'http://localhost:3000';

// Test metrics data
const metrics = {
  startupTime: 0,
  responseTime: {},
  memoryUsage: {},
  cpuUsage: {}
};

// Test statuses
const testStatus = {
  quantum: {
    qubitManagement: false,
    quantumGates: false,
    errorCorrection: false
  },
  vm: {
    taskScheduling: false,
    memoryManagement: false,
    performanceBenchmarks: false
  },
  ai: {
    coreFunctionality: false,
    quantumFeatures: false,
    performance: false
  }
};

async function runTests() {
  console.log('🧪 Starting Alpha Phase 1 E2E Tests');
  
  // Launch headless browser
  const browser = await puppeteer.launch({
    headless: false, // Set to true for production testing
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  console.log('📊 Measuring startup performance...');
  
  const startTime = Date.now();
  await page.goto(TEST_URL, { waitUntil: 'networkidle2' });
  metrics.startupTime = Date.now() - startTime;
  
  console.log(`✅ Application loaded in ${metrics.startupTime}ms`);
  
  // Run test suites
  await testQuantumCore(page);
  await testVirtualMachineLayer(page);
  await testAIAssistant(page);
  
  // Generate final report
  generateReport();
  
  await browser.close();
}

async function testQuantumCore(page) {
  console.log('\n🧪 Testing Quantum Core...');
  
  // 1.1 Qubit Management
  console.log('  Testing Qubit Management...');
  try {
    await page.goto(`${TEST_URL}/dashboard/quantum`);
    await page.waitForSelector('#qubit-controls');
    
    // Initialize qubits
    await page.click('#initialize-qubits');
    await page.type('#qubit-count', '100');
    await page.click('#confirm-initialization');
    
    // Verify superposition
    await page.waitForSelector('#qubit-status-superposition.active');
    
    // Test entanglement
    await page.click('#create-entanglement');
    await page.waitForSelector('#entanglement-verified');
    
    // Check decoherence rates
    const decoherenceRate = await page.$eval('#decoherence-rate', el => parseFloat(el.textContent));
    assert(decoherenceRate < 0.1, 'Decoherence rate is too high');
    
    testStatus.quantum.qubitManagement = true;
    console.log('  ✅ Qubit Management tests passed');
  } catch (error) {
    console.error('  ❌ Qubit Management tests failed:', error);
  }
  
  // 1.2 Quantum Gates
  console.log('  Testing Quantum Gates...');
  try {
    await page.goto(`${TEST_URL}/dashboard/quantum/gates`);
    
    // Test single-qubit gates
    const singleGates = ['X', 'Y', 'Z', 'H', 'S', 'T'];
    for (const gate of singleGates) {
      await page.click(`#gate-${gate}`);
      await page.waitForSelector(`#gate-${gate}-applied`);
    }
    
    // Test two-qubit gates
    const twoQubitGates = ['CNOT', 'SWAP', 'CZ'];
    for (const gate of twoQubitGates) {
      await page.click(`#gate-${gate}`);
      await page.waitForSelector(`#gate-${gate}-applied`);
    }
    
    // Verify gate fidelities
    const fidelity = await page.$eval('#gate-fidelity', el => parseFloat(el.textContent));
    assert(fidelity > 99.9, 'Gate fidelity is below target');
    
    testStatus.quantum.quantumGates = true;
    console.log('  ✅ Quantum Gates tests passed');
  } catch (error) {
    console.error('  ❌ Quantum Gates tests failed:', error);
  }
  
  // 1.3 Error Correction
  console.log('  Testing Error Correction...');
  try {
    await page.goto(`${TEST_URL}/dashboard/quantum/error-correction`);
    
    // Test surface code
    await page.click('#enable-surface-code');
    await page.waitForSelector('#surface-code-active');
    
    // Verify logical qubit protection
    await page.click('#introduce-error');
    await page.waitForSelector('#error-detected');
    await page.waitForSelector('#error-corrected', { timeout: 5000 });
    
    testStatus.quantum.errorCorrection = true;
    console.log('  ✅ Error Correction tests passed');
  } catch (error) {
    console.error('  ❌ Error Correction tests failed:', error);
  }
}

async function testVirtualMachineLayer(page) {
  console.log('\n🧪 Testing Virtual Machine Layer...');
  
  // 2.1 Task Scheduling
  console.log('  Testing Task Scheduling...');
  try {
    await page.goto(`${TEST_URL}/dashboard/vm/scheduler`);
    
    // Test quantum-classical hybrid scheduling
    await page.click('#create-hybrid-task');
    await page.waitForSelector('#task-scheduled');
    await page.waitForSelector('#task-completed');
    
    testStatus.vm.taskScheduling = true;
    console.log('  ✅ Task Scheduling tests passed');
  } catch (error) {
    console.error('  ❌ Task Scheduling tests failed:', error);
  }
  
  // 2.2 Memory Management
  console.log('  Testing Memory Management...');
  try {
    await page.goto(`${TEST_URL}/dashboard/vm/memory`);
    
    // Test memory allocation
    await page.click('#allocate-memory');
    await page.type('#memory-size', '100');
    await page.click('#confirm-allocation');
    await page.waitForSelector('#memory-allocated');
    
    // Test garbage collection
    await page.click('#run-gc');
    await page.waitForSelector('#gc-completed');
    
    testStatus.vm.memoryManagement = true;
    console.log('  ✅ Memory Management tests passed');
  } catch (error) {
    console.error('  ❌ Memory Management tests failed:', error);
  }
  
  // 2.3 Performance Benchmarks
  console.log('  Testing Performance Benchmarks...');
  try {
    await page.goto(`${TEST_URL}/dashboard/vm/benchmarks`);
    
    // Run Quantum Volume estimation
    await page.click('#run-quantum-volume');
    await page.waitForSelector('#benchmark-completed');
    
    const qVolume = await page.$eval('#quantum-volume', el => parseInt(el.textContent));
    assert(qVolume > 0, 'Quantum Volume should be positive');
    
    testStatus.vm.performanceBenchmarks = true;
    console.log('  ✅ Performance Benchmarks tests passed');
  } catch (error) {
    console.error('  ❌ Performance Benchmarks tests failed:', error);
  }
}

async function testAIAssistant(page) {
  console.log('\n🧪 Testing AI Assistant Integration...');
  
  // 3.1 Core Functionality
  console.log('  Testing Core Functionality...');
  try {
    await page.goto(`${TEST_URL}/assistant`);
    
    // Test chat interface
    await page.waitForSelector('#chat-interface');
    await page.type('#chat-input', 'Hello, can you help me understand quantum computing?');
    await page.click('#send-message');
    
    // Wait for response
    const startResponseTime = Date.now();
    await page.waitForSelector('.assistant-message');
    metrics.responseTime.simple = Date.now() - startResponseTime;
    
    testStatus.ai.coreFunctionality = true;
    console.log(`  ✅ AI Core Functionality tests passed (response time: ${metrics.responseTime.simple}ms)`);
  } catch (error) {
    console.error('  ❌ AI Core Functionality tests failed:', error);
  }
  
  // 3.2 Quantum-Enhanced Features
  console.log('  Testing Quantum-Enhanced Features...');
  try {
    await page.type('#chat-input', 'Show me a visualization of a quantum circuit');
    await page.click('#send-message');
    
    // Wait for visualization
    await page.waitForSelector('#quantum-visualization');
    
    testStatus.ai.quantumFeatures = true;
    console.log('  ✅ Quantum-Enhanced Features tests passed');
  } catch (error) {
    console.error('  ❌ Quantum-Enhanced Features tests failed:', error);
  }
  
  // 3.3 Performance Testing
  console.log('  Testing Performance...');
  try {
    // Check memory usage
    metrics.memoryUsage.baseline = await getMemoryUsage(page);
    
    // Test concurrent operations
    await page.evaluate(() => {
      for (let i = 0; i < 5; i++) {
        fetch('/api/quantum/simulate');
      }
    });
    
    // Wait and check memory again
    await page.waitForTimeout(2000);
    metrics.memoryUsage.afterLoad = await getMemoryUsage(page);
    
    testStatus.ai.performance = true;
    console.log('  ✅ AI Performance tests passed');
  } catch (error) {
    console.error('  ❌ AI Performance tests failed:', error);
  }
}

async function getMemoryUsage(page) {
  const metrics = await page.evaluate(() => performance.memory);
  return metrics ? metrics.usedJSHeapSize / (1024 * 1024) : 'N/A';
}

function generateReport() {
  console.log('\n📊 Alpha Test Phase 1 Report');
  console.log('---------------------------');
  
  console.log('\nPerformance Metrics:');
  console.log(`- Startup Time: ${metrics.startupTime}ms`);
  console.log(`- Simple Query Response Time: ${metrics.responseTime.simple || 'N/A'}ms`);
  console.log(`- Memory Usage (Baseline): ${metrics.memoryUsage.baseline || 'N/A'}MB`);
  console.log(`- Memory Usage (After Load): ${metrics.memoryUsage.afterLoad || 'N/A'}MB`);
  
  console.log('\nTest Results:');
  
  console.log('\n1. Quantum Core:');
  console.log(`- Qubit Management: ${testStatus.quantum.qubitManagement ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Quantum Gates: ${testStatus.quantum.quantumGates ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Error Correction: ${testStatus.quantum.errorCorrection ? '✅ PASSED' : '❌ FAILED'}`);
  
  console.log('\n2. Virtual Machine Layer:');
  console.log(`- Task Scheduling: ${testStatus.vm.taskScheduling ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Memory Management: ${testStatus.vm.memoryManagement ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Performance Benchmarks: ${testStatus.vm.performanceBenchmarks ? '✅ PASSED' : '❌ FAILED'}`);
  
  console.log('\n3. AI Assistant Integration:');
  console.log(`- Core Functionality: ${testStatus.ai.coreFunctionality ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Quantum-Enhanced Features: ${testStatus.ai.quantumFeatures ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`- Performance: ${testStatus.ai.performance ? '✅ PASSED' : '❌ FAILED'}`);
  
  const passedTests = Object.values(testStatus.quantum).filter(Boolean).length +
                     Object.values(testStatus.vm).filter(Boolean).length +
                     Object.values(testStatus.ai).filter(Boolean).length;
  
  const totalTests = Object.values(testStatus.quantum).length +
                    Object.values(testStatus.vm).length +
                    Object.values(testStatus.ai).length;
  
  console.log(`\nSummary: ${passedTests}/${totalTests} tests passed (${Math.round(passedTests/totalTests*100)}%)`);
  
  const overallStatus = passedTests === totalTests ? '✅ PASSED' : '❌ FAILED';
  console.log(`\nAlpha Phase 1 Status: ${overallStatus}`);
}

// Jest test wrapper
describe('Alpha Phase 1 Tests - Core Components & AI Tuning', () => {
  test('Run all Phase 1 tests', async () => {
    await runTests();
    
    // Assert all major components passed
    expect(testStatus.quantum.qubitManagement).toBe(true);
    expect(testStatus.vm.taskScheduling).toBe(true);
    expect(testStatus.ai.coreFunctionality).toBe(true);
  }, 120000); // 2 minute timeout
});

// Export for direct execution
module.exports = {
  runTests,
  testStatus,
  metrics
};

// Run tests if called directly
if (require.main === module) {
  runTests().catch(console.error);
}
