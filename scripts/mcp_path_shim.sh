#!/usr/bin/env bash
# ==============================================================================
# QUILLAN-RONIN v5.4.0-ONI LINUX / WSL / CONTAINER PATH SHIM
# Translates Windows paths (C:\02_QUILLAN) to Linux/WSL (/mnt/c/02_QUILLAN)
# ==============================================================================

set -eo pipefail

translate_path() {
    local input_path="$1"
    if [[ "$input_path" =~ ^[A-Za-z]:[\\/] ]]; then
        local drive_letter="${input_path:0:1}"
        drive_letter="$(echo "$drive_letter" | tr '[:upper:]' '[:lower:]')"
        local remainder="${input_path:2}"
        remainder="${remainder//\\//}"
        echo "/mnt/${drive_letter}${remainder}"
    else
        echo "$input_path"
    fi
}

# Resolve active repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}/09 - Projects/mcp:${REPO_ROOT}/09 - Projects/projects/oni:${REPO_ROOT}/03 - Training & Model/scripts:${REPO_ROOT}:${PYTHONPATH}"

if [[ $# -gt 0 ]]; then
    CMD=()
    for arg in "$@"; do
        CMD+=("$(translate_path "$arg")")
    done
    exec "${CMD[@]}"
else
    echo "Quillan Path Shim active. REPO_ROOT=${REPO_ROOT}"
fi
