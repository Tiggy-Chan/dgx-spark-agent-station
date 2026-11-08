# Integrating Claude Code with Dual DGX Spark DeepSeek Station

## 1. Overview

Claude Code is an agentic coding assistant that drives terminal execution, file inspection, editing, and git workflows. Unlike standard chat interfaces, coding agents execute multi-turn feedback loops that:
1. Pass large context windows (often 30,000 to 100,000+ tokens containing full repository files and error logs).
2. Generate intermediate reasoning thoughts before invoking structured tool calls.
3. Transmit tool outputs back to the model for iterative debugging.

To run effectively with a 748B reasoning model on private DGX Spark hardware, specific engine governance policies must be applied.

---

## 2. Environment Configuration

### 2.1 Direct LAN Access (Lowest Latency)
When developing on the same local network as the DGX Spark head node:

```bash
export ANTHROPIC_BASE_URL="http://192.168.3.90:8888"
export ANTHROPIC_API_KEY="sk-spark-local"
export ANTHROPIC_MODEL="DeepSeek-V4.1-Flash-1017"

claude
```

### 2.2 Remote Gateway Access (9Router / Public Reverse Proxy)
When developing remotely through a secured gateway:

```bash
export ANTHROPIC_BASE_URL="https://9r.yourdomain.com"
export ANTHROPIC_API_KEY="your-9router-api-key"
export ANTHROPIC_MODEL="DeepSeek-V4.1-Flash-1017"

claude
```

---

## 3. Critical Agent Governance Settings

### 3.1 Preventing Reasoning Loops (`THINKING_BUDGET`)
DeepSeek-V4.1-Flash utilizes reasoning templates. When `REASONING_EFFORT=xhigh` or `max` is requested without budget boundaries, the model can enter self-critique loops on complex prompts, generating 10,000 to 30,000 tokens of internal reasoning before emitting any code or tool call.

* **Recommended Configuration**:
  ```bash
  REASONING_EFFORT="high"
  THINKING_BUDGET=2048
  ```
* **Effect**: Limits reasoning to 2,048 tokens. If the model exceeds this limit, the engine automatically injects the `</think>` closing delimiter, forcing the model to transition immediately into concrete code output or tool invocation.

### 3.2 Multi-Turn History Pruning (`DROP_REASONING_HISTORY=1`)
In agentic workflows, retaining intermediate `<think>` thoughts across 10-20 turns degrades performance:
* Re-reading obsolete thought chains consumes context space.
* Causes the model to over-analyze its previous reasoning rather than current compiler output.

Setting `DROP_REASONING_HISTORY=1` instructs the engine to strip `reasoning_content` from prior assistant turns while preserving the final code, explanations, and tool interactions.

---

## 4. Verification

Run the included protocol validator to ensure the cluster satisfies Claude Code's requirements:

```bash
python3 tools/anthropic_test.py --url http://192.168.3.90:8888
```

Expected output:
```text
[TEST 1/4] Simple message completion...
  [PASS] Completed in 0.22s: Paris
[TEST 2/4] Thinking mode (<think> block separation)...
  [PASS] Thinking tokens: 88 chars, Final text: 9.9 is greater...
[TEST 3/4] Tool calling schema generation...
  [PASS] Tool call generated: lookup_user({'uid': 42})
[TEST 4/4] Token count endpoint (/v1/messages/count_tokens)...
  [PASS] Tokens counted successfully: 17 tokens

[ALL TESTS PASSED] Anthropic Messages API is fully compliant.
```
