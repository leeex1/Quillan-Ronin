#!/bin/bash

# NextVerse E2E Testing - Start App and Run Tests

# Default parameters
PHASE="1"
HEADLESS="false"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift ;;
        --headless) HEADLESS="true"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "🚀 Starting NextVerse Platform and Running Alpha Phase $PHASE Tests"

# Start the application in a separate process
echo "🚀 Starting NextVerse Platform..."
cd /workspaces/new-app/nextverse-platform

# Check if the platform is already running
if curl -s --head --request GET "http://localhost:3000" | grep "200" > /dev/null; then
    echo "✅ NextVerse Platform is already running on http://localhost:3000"
else
    echo "🚀 Launching NextVerse Platform..."
    npm run turbo:start &
    APP_PID=$!
    
    # Wait for app to start (max 60 seconds)
    echo "⏳ Waiting for application to start..."
    MAX_WAIT=60
    COUNTER=0
    while ! curl -s --head --request GET "http://localhost:3000" | grep "200" > /dev/null; do
        if [ $COUNTER -ge $MAX_WAIT ]; then
            echo "❌ Application failed to start within $MAX_WAIT seconds"
            kill $APP_PID 2>/dev/null
            exit 1
        fi
        sleep 1
        COUNTER=$((COUNTER+1))
        echo -n "."
    done
    echo ""
    echo "✅ NextVerse Platform successfully started!"
fi

# Run tests
echo "🧪 Running tests..."
cd /workspaces/new-app/e2e-tests
bash run-tests.sh --phase "$PHASE" --headless "$HEADLESS"
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests passed successfully!"
else
    echo "❌ Some tests failed! See the reports for details."
fi

# We don't automatically shut down the app, as the user might want to explore it manually
echo ""
echo "📝 The NextVerse Platform is still running at http://localhost:3000"
echo "   To stop it, you can run: pkill -f 'npm run turbo:start'"

exit $TEST_RESULT
