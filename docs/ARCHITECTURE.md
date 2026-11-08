# Dual DGX Spark (2x GB10) DeepSeek-V4.1-Flash Architecture

## 1. Hardware Substrate and Physical Boundaries

The NVIDIA DGX Spark is powered by the GB10 Grace Blackwell Superchip (SM121 compute architecture) with 128 GB of LPDDR5x unified memory per node. When two units are paired into a distributed workstation cluster, the hardware topology presents two critical operational boundaries:

### 1.1 Memory Capacity vs. Footprint
* Total raw unified memory: 256 GB (128 GB per node).
* OS and base daemon footprint: ~12-16 GiB per node.
* Maximum usable memory for the inference engine: ~220-234 GiB.
* DeepSeek-V4.1-Flash-1017 total parameters: 748B-763B (552B base MoE + 196B Engram condition memory table + 14B DSpark speculative draft blocks).
* Weight footprint under EXL3 2.9 bpw: ~214 GiB.
* Memory headroom remaining for KV cache: ~15-20 GiB across both machines.

### 1.2 Interconnect Topology
* There is no physical inter-chassis NVLink between separate DGX Spark desktop enclosures.
* Node-to-node communication is driven entirely across two ConnectX-7 200GbE QSFP ports operating in point-to-point RoCE v2 (RDMA over Converged Ethernet) mode.
* Each node presents two RoCE interfaces (`rocep1s0f0` and `roceP2p1s0f0`) mapping to two independent PCIe Gen5 x4 physical paths, yielding a dual-rail 400 Gbps aggregate link with effective single-direction bandwidth of ~46-48 GB/s.

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
         [ Claude Code / 9Router ]
```

---

## 2. Why Stock Inference Frameworks Fail on Dual Sparks

| Engine | Root Failure Mechanism |
| :--- | :--- |
| **DeepInfer (Official)** | CUDA kernels hardcoded for data-center Blackwell SM100; compilation fails on SM121. Designed for 64-node disaggregated Prefill-Decode clusters, not 2-node TP=2. Lacks EXL3 trellis quantization support. |
| **vLLM (Mainline)** | Mamba-Attention hybrid layers crash on aarch64 CUDA 13.0 with block-alignment assertions. FP4 Marlin fallback lacks kernel support for 196GB Engram disk-mapped tables. |
| **TensorRT-LLM** | Requires building engine plans for specific TP configurations in FP8; 748B parameters in FP8 exceed the 256GB unified physical memory ceiling. |

### The TensorFold Solution
The cluster utilizes TensorFold's native engine compiled with Zig 0.17.0 and CUDA 13.0 for SM121:
* Hand-tuned CUTLASS and FlashInfer kernels targeting SM121.
* Native EXL3 (2.9 bpw) trellis dequantization directly inside the GEMM epilogue.
* Direct RoCE RDMA exchange across the two ConnectX-7 rails bypassing host networking stacks.
* DSpark speculative multi-token validation implemented in C/Zig without Python runtime overhead.

---

## 3. Two-Tier Memory and Cache Architecture

```
                    [ Incoming Request ]
                             |
                   Check Prompt Prefix
                             |
             +---------------+---------------+
             |                               |
       [ Cache Miss ]                  [ Cache Hit ]
             |                               |
     Run Full Prefill              Resume from Position N
    (60-100 tok/s ingest)           (Skip ~90% Compute)
             |                               |
             +---------------+---------------+
                             |
              Generate KV Cache Latents
                             |
             +---------------+---------------+
             |                               |
       [ Tier 1: RAM ]               [ Tier 2: NVMe ]
     KV Shard Pool (2.1M)            Spill v2 Storage
     1.0 GiB per node                64 GiB per node SSD
   (Owner-only allocation)         (Eviction / Resume buffer)
```

1. **KV Sharding (`KV_SHARD=1`)**:
   Instead of replicating full conversation state on both nodes, each node stores only half of the conversation's compressed KV tensor latents. The missing blocks are fetched on-demand across the 400 Gbps RoCE fabric during decode rounds. This halves per-node KV RAM consumption.

2. **NVMe Spill v2 (`SPILL_GIB=64`)**:
   When prompt states are evicted from memory due to LRU pressure or multi-session concurrency, they are committed to local NVMe SSD storage in 2,048-token blocks. Subsequent turns read the cache from NVMe in milliseconds rather than recomputing the full prefill pass.
