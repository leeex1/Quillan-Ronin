#!/bin/bash

# E2E Test Environment Setup for NextVerse Alpha Testing
# This script prepares all necessary dependencies and tools for E2E testing

echo "🛠️  Setting up NextVerse E2E Test Environment..."

# Create the test directory structure if it doesn't exist
mkdir -p /workspaces/new-app/e2e-tests/results
mkdir -p /workspaces/new-app/e2e-tests/reports
mkdir -p /workspaces/new-app/e2e-tests/screenshots

# Install test dependencies
echo "📦 Installing test dependencies..."
cd /workspaces/new-app/e2e-tests
npm install

# Verify NextVerse installation
echo "🔍 Checking NextVerse platform..."
cd /workspaces/new-app/nextverse-platform

if [ ! -f "package.json" ]; then
  echo "⚠️  NextVerse platform not found! Please check repository setup."
  exit 1
fi

# Install NextVerse dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "📦 Installing NextVerse dependencies..."
  npm install
fi

# Prepare test data directory
mkdir -p /workspaces/new-app/e2e-tests/test-data

# Set up environment variables for testing
echo "🔧 Setting up test environment variables..."
cat > /workspaces/new-app/e2e-tests/.env <<EOL
# E2E Test Environment Configuration
TEST_ENV=alpha_phase1
APP_URL=http://localhost:3000
HEADLESS=false
SCREENSHOT_DIR=/workspaces/new-app/e2e-tests/screenshots
REPORT_DIR=/workspaces/new-app/e2e-tests/reports
TIMEOUT=30000
EOL

echo "✅ E2E Test Environment setup complete!"
echo ""
echo "📋 Available test commands:"
echo "  - npm test                  # Run all tests"
echo "  - npm run test:phase1       # Run Alpha Phase 1 tests only"
echo "  - npm run test:phase2       # Run Alpha Phase 2 tests only"
echo "  - npm run test:phase3       # Run Alpha Phase 3 tests only"
echo "  - npm run test:phase4       # Run Alpha Phase 4 tests only"
echo "  - npm run test:phase5       # Run Alpha Phase 5 tests only"
echo ""
echo "🚀 To start the application and run tests:"
echo "  1. cd /workspaces/new-app/nextverse-platform"
echo "  2. npm run turbo:start"
echo "  3. cd /workspaces/new-app/e2e-tests"
echo "  4. npm run test:phase1"
