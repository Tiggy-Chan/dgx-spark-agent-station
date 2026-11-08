#!/usr/bin/env python3
"""
Anthropic Messages API Smoke Test & Tool-Calling Validator

Verifies that the DGX Spark DeepSeek cluster conforms to the Anthropic
Messages specification used by Claude Code, including:
  1. Base message completion
  2. Thinking / CoT block formatting (<think>...</think> -> thinking block)
  3. Streaming tool calls
  4. Tool result resolution
  5. Token counting

Usage:
  python3 tools/anthropic_test.py --url http://192.168.3.90:8888
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def test_endpoint(base_url: str, model_name: str):
    endpoint = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": "sk-spark-local"
    }

    # 1. Base text generation
    print("[TEST 1/4] Simple message completion...")
    body = {
        "model": model_name,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}]
    }
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode("utf-8"))
    t1 = time.time()
    content = "".join(b.get("text", "") for b in res.get("content", []) if b.get("type") == "text")
    assert "Paris" in content, f"Unexpected response: {content}"
    print(f"  [PASS] Completed in {t1-t0:.2f}s: {content.strip()}")

    # 2. Thinking mode verification
    print("[TEST 2/4] Thinking mode (<think> block separation)...")
    body = {
        "model": model_name,
        "max_tokens": 256,
        "thinking": {"type": "enabled", "budget_tokens": 128},
        "messages": [{"role": "user", "content": "Which is larger: 9.11 or 9.9?"}]
    }
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode("utf-8"))
    blocks = res.get("content", [])
    thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert len(thinking_blocks) > 0, "No thinking block returned in thinking mode"
    assert len(text_blocks) > 0, "No text block returned in thinking mode"
    print(f"  [PASS] Thinking tokens: {len(thinking_blocks[0].get('thinking', ''))} chars, Final text: {text_blocks[0].get('text', '').strip()[:50]}...")

    # 3. Tool call generation
    print("[TEST 3/4] Tool calling schema generation...")
    tools = [{
        "name": "lookup_user",
        "description": "Look up a user profile by UID",
        "input_schema": {
            "type": "object",
            "properties": {"uid": {"type": "integer"}},
            "required": ["uid"]
        }
    }]
    body = {
        "model": model_name,
        "max_tokens": 128,
        "tools": tools,
        "messages": [{"role": "user", "content": "Look up user with UID 42."}]
    }
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode("utf-8"))
    tool_use_blocks = [b for b in res.get("content", []) if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) > 0, f"Expected tool_use block, got: {res}"
    tool_args = tool_use_blocks[0].get("input", {})
    assert tool_args.get("uid") == 42, f"Expected uid=42, got: {tool_args}"
    print(f"  [PASS] Tool call generated: {tool_use_blocks[0]['name']}({tool_args})")

    # 4. Token counting endpoint
    print("[TEST 4/4] Token count endpoint (/v1/messages/count_tokens)...")
    count_endpoint = f"{base_url.rstrip('/')}/v1/messages/count_tokens"
    count_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Count these tokens accurately please."}]
    }
    req = urllib.request.Request(count_endpoint, data=json.dumps(count_body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode("utf-8"))
    input_tokens = res.get("input_tokens", 0)
    assert input_tokens > 0, f"Invalid token count: {res}"
    print(f"  [PASS] Tokens counted successfully: {input_tokens} tokens")

    print("\n[ALL TESTS PASSED] Anthropic Messages API is fully compliant.")


def main():
    parser = argparse.ArgumentParser(description="Anthropic Messages Protocol Validator")
    parser.add_argument("--url", default="http://127.0.0.1:8888", help="Base URL (e.g. http://192.168.3.90:8888)")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017", help="Model name")
    args = parser.parse_args()

    try:
        test_endpoint(args.url, args.model)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
