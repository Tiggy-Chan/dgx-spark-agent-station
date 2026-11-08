# DGX Spark Agent Station

Turnkey deployment, inference optimization, and agent governance toolkit for serving **DeepSeek-V4.1-Flash-1017 (748B MoE, Uncensored)** on **Dual NVIDIA DGX Spark (2x GB10, 256GB Unified Memory)** clusters.

Optimized as a private local backend for **Claude Code**, **Cursor**, and high-throughput agentic workflows.

---

## 1. Key Performance Highlights

Measured on dual NVIDIA DGX Spark workstations (2x GB10, 400 Gbps RoCE v2 interconnect, 1,048,576 token context window, EXL3 2.9 bpw quantization):

| Workload Scenario | Decode Speed | Wall Speed | First Token (TTFT) | Speculative Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Code Generation** (Thinking OFF) | **104.1 tok/s** | 102.3 tok/s | ~172 ms | 77.8% |
| **Complex Reasoning** (Thinking ON) | **78.7 tok/s** | 77.2 tok/s | ~269 ms | 66.3% |
| **Prose & Docs** (Thinking OFF) | **60.5 tok/s** | 60.1 tok/s | ~128 ms | 59.3% |
| **2-Stream Concurrency** | **97.1 tok/s aggregate** | 48.5 tok/s / stream | ~210 ms | 68.2% |
| **4-Stream Concurrency** | **126.1 tok/s aggregate** | 31.5 tok/s / stream | ~295 ms | 62.4% |

### Multi-Turn Prefix Cache Performance
In agentic coding loops (e.g. Claude Code reading, planning, editing):
* **Cold Initial Turn** (13,409 tokens): 6.4s prefill time.
* **Cached Subsequent Turn** (14,013 tokens, 12,288 cached): **1.2s response time (5.3x acceleration)**.

---

## 2. Core Capabilities

* **Native SM121 Kernel Architecture**: Bypasses datacenter SM100 constraints using custom Zig 0.17.0 and CUDA 13.0 kernels compiled specifically for the GB10 Grace Blackwell Superchip.
* **Dual-Rail 400 Gbps RoCE v2 Interconnect**: Automatic multi-interface binding (`rocep1s0f0` + `roceP2p1s0f0`) with per-node GID discrepancy resolution (e.g. GID 4 on Node 0 vs GID 3 on Node 1).
* **Anthropic Messages Protocol Native**: Direct support for `/v1/messages`, streaming tool calls, `<think>` block extraction, and `/v1/messages/count_tokens`.
* **Agent Reasoning Governance**:
  * `THINKING_BUDGET`: Caps internal chain-of-thought tokens (e.g., 2,048 tokens) to prevent models from entering infinite self-doubt loops on complex prompts.
  * `DROP_REASONING_HISTORY`: Strips obsolete reasoning traces from prior turns, preventing multi-turn context bloat while preserving final code and tool outputs.
* **Two-Tier Memory Multipliers**:
  * `KV_SHARD=1`: Shards KV latents across both physical machines, halving per-node memory footprint.
  * `SPILL_GIB=64`: Asynchronously commits cached conversation blocks to local NVMe SSDs, surviving eviction and restarts.
* **Turnkey Remote Gateway Sync**: Automated database provisioning for 9Router, seamlessly routing Claude Code default aliases (`claude-3-7-sonnet-*`, `my-model`, `DeepSeek-V4.1-Flash-1017`) directly to Spark.

---

## 3. Architecture Overview

```
+------------------------------------+          +------------------------------------+
|       DGX Spark Node 0 (Head)      |          |      DGX Spark Node 1 (Worker)     |
|  GB10 (SM121) + 128GB Unified RAM  |          |  GB10 (SM121) + 128GB Unified RAM  |
|                                    |          |                                    |
|   +----------------------------+   |          |   +----------------------------+   |
|   | TensorFold Native (Rank 0) |   |          |   |  tf-dsv41-lanes (Rank 1)   |   |
|   | 104 GiB Weights + 1M Ctx   |   |          |   | 104 GiB Weights + KV Shard |   |
|   +----------------------------+   |          |   +----------------------------+   |
|                 |                  |          |                 |                  |
|        ConnectX-7 Rail 0           |<========>|        ConnectX-7 Rail 0           |
|        ConnectX-7 Rail 1           |<========>|        ConnectX-7 Rail 1           |
|         (200 Gbps RoCE v2)         |  400Gbps |         (200 Gbps RoCE v2)         |
+------------------------------------+          +------------------------------------+
                  |
             Port 8888 (Anthropic / OpenAI API)
                  |
     +------------+------------+
     |                         |
[ Direct LAN ]         [ 9Router / Proxy ]
     |                         |
     +------------+------------+
                  |
            Claude Code CLI
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full engineering specifications.

---

## 4. Quick Start

### Step 1: Pre-flight Diagnostic
Run the pre-flight inspector on Node 0 (Head) to audit memory, disk space, and interconnect health:
```bash
python3 bootstrap/preflight.py --worker admin@192.168.100.11
```

### Step 2: Configure Environment
Copy the production template and adapt IP addresses:
```bash
cp config/local.sh.example scripts/local.sh
```

Key recommended settings for coding agents:
```bash
CONTEXT=1048576               # 1M token context window
PARALLEL=4                   # 4 concurrent streams
REASONING_EFFORT="high"       # Balanced reasoning depth
THINKING_BUDGET=2048          # Cap reasoning tokens to avoid loops
DROP_REASONING_HISTORY=1      # Prune historical reasoning in multi-turn
KV_SHARD=1                    # Shard KV cache across dual nodes
SPILL_GIB=64                  # 64 GiB NVMe cache persistence per node
```

### Step 3: Automated Preparation & Staging
Launch the automated preparation pipeline:
```bash
./scripts/prepare.sh
```
The script will:
1. Verify dual-rail ConnectX-7 link status and GID tables.
2. Build the optimized SM121 container image using Zig 0.17.0 and CUDA 13.0.
3. Distribute the image to Node 1 via internal network.
4. Download the 196 GiB EXL3 2.9bpw base weights and 203 GB Engram condition tables.
5. Apply the un-censoring matrix overlay (`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`).
6. Generate binary rank caches (`rank0of2.bin`, `rank1of2.bin`).

### Step 4: Launch Cluster
```bash
./start.sh
```
The OpenAI and Anthropic-compatible endpoints will be live at `http://0.0.0.0:8888`.

---

## 5. Connecting Claude Code

### Direct LAN Connection
In your development terminal:
```bash
source gateway/claude_code_env.sh local 192.168.3.90
claude
```

### Remote Gateway Connection (via 9Router)
1. Synchronize model routes and aliases on your router server:
   ```bash
   python3 gateway/router_sync.py --db /opt/9router/data/db/data.sqlite --upstream http://127.0.0.1:8002/v1
   ```
2. Export your credentials and launch:
   ```bash
   source gateway/claude_code_env.sh router 9r.yourdomain.com
   claude
   ```

See [docs/CLAUDE_CODE_GUIDE.md](docs/CLAUDE_CODE_GUIDE.md) for detailed tool-calling patterns and debugging.

---

## 6. Verification and Profiling Tools

### Benchmark Suite
Run single-stream, thinking, prose, and concurrency tests:
```bash
python3 tools/bench_tps.py --url http://192.168.3.90:8888/v1
```

### Anthropic API Protocol Compliance
Validate tool-calling schemas, streaming blocks, and token counting:
```bash
python3 tools/anthropic_test.py --url http://192.168.3.90:8888
```

### Real-time Cluster Monitor
Watch prompt cache hit rates, live decode TPS, and NVMe spill usage in real time:
```bash
python3 tools/cache_monitor.py --url http://192.168.3.90:8888
```

---

## 7. Troubleshooting

Detailed solutions for common production issues are documented in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md):
* RoCE v2 GID index mismatch between nodes
* Worker Docker socket permission errors
* Base image pull timeouts in restricted network environments
* 9Router `model_not_found` resolution
* Reasoning loop runaway mitigation

---

## 8. License & Attribution

* Licensed under the [MIT License](LICENSE).
* Author: **Tiggy-Chan** (`tiggychan233@gmail.com`).
* Built upon research and tools from the open-source community: DeepSeek AI, TensorFold (ashhart, bertholomus), MiaAI-Lab, and drowzeys.
