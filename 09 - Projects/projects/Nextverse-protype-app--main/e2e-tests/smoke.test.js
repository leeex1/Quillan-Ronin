const puppeteer = require('puppeteer');
require('dotenv').config();

describe('NextVerse Platform Smoke Test', () => {
  let browser;
  let page;
  
  beforeAll(async () => {
    browser = await puppeteer.launch({
      headless: process.env.HEADLESS === 'true',
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  test('Application loads successfully', async () => {
    const response = await page.goto(process.env.APP_URL || 'http://localhost:3000', {
      waitUntil: 'networkidle2',
      timeout: 30000,
    });
    
    // Take a screenshot for visual verification
    await page.screenshot({
      path: `${process.env.SCREENSHOT_DIR || '.'}/app-loaded.png`,
      fullPage: true,
    });
    
    // Check response status
    expect(response.status()).toBe(200);
    
    // Verify page title
    const title = await page.title();
    expect(title).toContain('NextVerse');
    
    // Check that main app elements are present
    const appContainer = await page.$('#app');
    expect(appContainer).toBeTruthy();
  }, 60000);
  
  test('Dashboard is accessible', async () => {
    await page.goto(`${process.env.APP_URL || 'http://localhost:3000'}/dashboard`, {
      waitUntil: 'networkidle2',
    });
    
    await page.screenshot({
      path: `${process.env.SCREENSHOT_DIR || '.'}/dashboard.png`,
      fullPage: true,
    });
    
    // Check for dashboard elements
    const dashboardTitle = await page.$eval('h1', el => el.textContent);
    expect(dashboardTitle).toContain('Dashboard');
  }, 30000);
  
  test('AI Assistant responds to basic query', async () => {
    await page.goto(`${process.env.APP_URL || 'http://localhost:3000'}/assistant`, {
      waitUntil: 'networkidle2',
    });
    
    // Find and click the chat input
    await page.waitForSelector('#chat-input');
    await page.type('#chat-input', 'Hello, are you online?');
    
    // Click send button
    await page.click('#send-message');
    
    // Wait for response
    await page.waitForSelector('.assistant-message', { timeout: 15000 });
    
    // Capture the response
    const response = await page.$eval('.assistant-message', el => el.textContent);
    expect(response.length).toBeGreaterThan(0);
    
    await page.screenshot({
      path: `${process.env.SCREENSHOT_DIR || '.'}/assistant-response.png`,
      fullPage: true,
    });
  }, 30000);
});
