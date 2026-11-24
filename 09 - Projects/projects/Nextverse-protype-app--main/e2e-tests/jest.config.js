module.exports = {
  testEnvironment: 'node',
  testTimeout: 60000,
  verbose: true,
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'reports',
      outputName: 'junit.xml',
    }],
    ['jest-html-reporter', {
      pageTitle: 'NextVerse E2E Test Report',
      outputPath: 'reports/test-report.html',
      includeConsoleLog: true,
      includeFailureMsg: true,
    }],
  ],
  setupFilesAfterEnv: ['./jest.setup.js'],
};
