#!/bin/bash

# NextVerse Alpha Testing - Test Runner Script

# Load environment variables
if [ -f ".env" ]; then
    source .env
fi

# Parse command line arguments
PHASE="1"
HEADLESS="false"
GENERATE_REPORT="true"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift ;;
        --headless) HEADLESS="$2"; shift ;;
        --no-report) GENERATE_REPORT="false" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "🧪 Running Alpha Phase $PHASE Tests (Headless: $HEADLESS)"

# Update environment variable for headless mode
sed -i "s/HEADLESS=.*/HEADLESS=$HEADLESS/g" .env

# Function to check if app is running
check_app_running() {
    APP_URL=${APP_URL:-http://localhost:3000}
    echo "🔍 Checking if application is running at $APP_URL..."
    if curl -s --head --request GET "$APP_URL" | grep "200" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if application is running
if ! check_app_running; then
    echo "❌ Application is not running! Please start the NextVerse platform first."
    echo "   You can start it with: cd /workspaces/new-app/nextverse-platform && npm run turbo:start"
    exit 1
fi

# Run tests for specified phase
case $PHASE in
    "smoke")
        echo "🔥 Running Smoke Tests..."
        npm run test:quick
        TEST_EXIT_CODE=$?
        ;;
    "1")
        echo "🧪 Running Alpha Phase 1 Tests..."
        npm run test:phase1
        TEST_EXIT_CODE=$?
        ;;
    "2")
        echo "🧪 Running Alpha Phase 2 Tests..."
        npm run test:phase2
        TEST_EXIT_CODE=$?
        ;;
    "3")
        echo "🧪 Running Alpha Phase 3 Tests..."
        npm run test:phase3
        TEST_EXIT_CODE=$?
        ;;
    "4")
        echo "🧪 Running Alpha Phase 4 Tests..."
        npm run test:phase4
        TEST_EXIT_CODE=$?
        ;;
    "5")
        echo "🧪 Running Alpha Phase 5 Tests..."
        npm run test:phase5
        TEST_EXIT_CODE=$?
        ;;
    "all")
        echo "🧪 Running All Alpha Phase Tests..."
        npm test
        TEST_EXIT_CODE=$?
        ;;
    *)
        echo "❌ Unknown test phase: $PHASE"
        echo "   Available options: smoke, 1, 2, 3, 4, 5, all"
        exit 1
        ;;
esac

# Generate additional reports if enabled
if [ "$GENERATE_REPORT" = "true" ]; then
    echo "📊 Generating detailed reports..."
    npm run report:generate
fi

# Display test results location
echo ""
echo "📝 Test results available at:"
echo "   - HTML Report: $PWD/reports/test-report.html"
echo "   - JUnit XML: $PWD/reports/junit.xml"
echo "   - Screenshots: $PWD/screenshots/"

# Exit with test exit code
exit $TEST_EXIT_CODE
