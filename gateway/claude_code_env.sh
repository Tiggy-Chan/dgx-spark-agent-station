#!/usr/bin/env bash
# ==============================================================================
# Helper Script: Configure Claude Code to use the Dual DGX Spark DeepSeek Agent
#
# Usage:
#   source gateway/claude_code_env.sh [local|router] [IP_OR_DOMAIN]
#
# Examples:
#   source gateway/claude_code_env.sh local 192.168.3.90
#   source gateway/claude_code_env.sh router 9r.tiggy.cn
# ==============================================================================

MODE="${1:-local}"
HOST="${2:-192.168.3.90}"

if [ "$MODE" = "router" ]; then
    echo "[INFO] Configuring Claude Code for 9Router Public Gateway..."
    export ANTHROPIC_BASE_URL="https://${HOST}"
    read -rsp "Enter your 9router API Key: " ROUTER_KEY
    echo ""
    export ANTHROPIC_API_KEY="$ROUTER_KEY"
    export ANTHROPIC_MODEL="DeepSeek-V4.1-Flash-1017"
else
    echo "[INFO] Configuring Claude Code for Direct LAN Spark Connection..."
    # Note: Anthropic client requires base URL without trailing /v1
    export ANTHROPIC_BASE_URL="http://${HOST}:8888"
    export ANTHROPIC_API_KEY="sk-spark-local"
    export ANTHROPIC_MODEL="DeepSeek-V4.1-Flash-1017"
fi

echo "[SUCCESS] Environment configured successfully."
echo "  ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
echo "  ANTHROPIC_MODEL:    $ANTHROPIC_MODEL"
echo ""
echo "Run 'claude' to start coding with the 748B cluster."
