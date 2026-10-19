#!/usr/bin/env bash
# ==============================================================================
# Quillan-Ronin Cloud Agent & Prebuild Bootstrap Script
# ==============================================================================
set -e

echo "[QUILLAN] Bootstrapping Cloud Agent & MCP Environment..."

# 1. Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "[QUILLAN] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Pre-cache core Python MCP server tools into global environment
echo "[QUILLAN] Pre-installing and caching MCP tool servers..."
uv tool install duckduckgo-mcp-server || true
uv tool install mcp-server-fetch || true
uv tool install mcp-server-git || true

# 3. Install NPM dependencies for ThinkingEngine if needed
if [ -f "mcp/thinking-engine/package.json" ]; then
    echo "[QUILLAN] Installing ThinkingEngine dependencies..."
    (cd mcp/thinking-engine && npm install --silent) || true
fi

# 4. Install repo python dependencies
if [ -f "requirements.txt" ]; then
    echo "[QUILLAN] Installing Python requirements..."
    pip install -r requirements.txt || true
fi

echo "[QUILLAN] Cloud Agent Prebuild configuration complete."
