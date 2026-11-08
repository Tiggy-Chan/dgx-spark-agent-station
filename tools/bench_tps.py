#!/usr/bin/env python3
"""
Multi-Scenario Benchmark & Concurrency Profiler for DGX Spark Clusters

Measures:
  1. Single-stream Decode TPS & Wall TPS (Code, Prose, Reasoning)
  2. Time-to-First-Token (TTFT)
  3. DSpark / Speculative Decoding Acceptance Rate
  4. Concurrent Load Scaling (1, 2, 4 streams)

Usage:
  python3 tools/bench_tps.py --url http://192.168.3.90:8888/v1
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

SCENARIOS = [
    {
        "name": "Code Generation (Thinking OFF)",
        "prompt": "Implement a high-performance thread-safe concurrent LRU Cache in modern C++20 with atomic operations and detailed memory-order justification.",
        "max_tokens": 1024,
        "thinking": False
    },
    {
        "name": "Complex Reasoning (Thinking ON)",
        "prompt": "A river flows at speed u. A boat moves upstream with speed v relative to water. A buoy falls into the river at t=0. 15 minutes later, the boat turns around. When does the boat catch the buoy? Prove rigorously in both river-flow and ground frames.",
        "max_tokens": 1024,
        "thinking": True
    },
    {
        "name": "Prose & Documentation (Thinking OFF)",
        "prompt": "Write an architectural overview describing the memory hierarchy and inter-GPU communication protocols in modern unified-memory AI supercomputers.",
        "max_tokens": 1024,
        "thinking": False
    }
]


def test_single_stream(base_url: str, scenario: dict, model_name: str):
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": scenario["prompt"]}],
        "max_tokens": scenario["max_tokens"],
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": scenario["thinking"]}
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()

    usage = res.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    wall_tps = completion_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

    tf = res.get("tensorfold", {})
    engine_tps = tf.get("tokens_per_second", wall_tps)
    ttft_ms = tf.get("time_to_first_token", 0) * 1000

    spec = res.get("speculative", {})
    acc_rate = spec.get("acceptance_rate", 0) * 100

    print(f"[{scenario['name']}]")
    print(f"  - Tokens: {completion_tokens} in {t1 - t0:.2f}s")
    print(f"  - Wall TPS: {wall_tps:.1f} tok/s | Engine Decode TPS: {engine_tps:.1f} tok/s")
    print(f"  - TTFT: {ttft_ms:.1f} ms | Speculative Acceptance: {acc_rate:.1f}%\n")


def test_concurrency(base_url: str, concurrency: int, model_name: str):
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    results: list[dict | None] = [None] * concurrency

    prompts = [
        "Explain QuickSort algorithm and write an optimized implementation.",
        "Describe the raft consensus protocol leader election in detail.",
        "Implement an async TCP echo server using Python asyncio.",
        "Explain the differences between optimistic and pessimistic locking."
    ]

    def worker(idx):
        p = prompts[idx % len(prompts)]
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 512,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                results[idx] = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            results[idx] = {"error": str(e)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    valid_results = [r for r in results if r and "usage" in r]
    total_tokens = sum(r["usage"]["completion_tokens"] for r in valid_results)
    agg_tps = total_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

    print(f"=== Concurrency Sweep ({concurrency} Parallel Streams) ===")
    print(f"  - Total Generated: {total_tokens} tokens across {len(valid_results)}/{concurrency} streams")
    print(f"  - Total Duration:  {t1 - t0:.2f}s")
    print(f"  - Aggregate Cluster TPS: {agg_tps:.1f} tok/s (avg {agg_tps/concurrency:.1f} tok/s per stream)\n")


def main():
    parser = argparse.ArgumentParser(description="DGX Spark Performance Profiler")
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1", help="Base URL of OpenAI-compatible API")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017", help="Model name")
    args = parser.parse_args()

    print(f"Running Performance Benchmark against {args.url} [{args.model}]\n")
    for s in SCENARIOS:
        try:
            test_single_stream(args.url, s, args.model)
        except Exception as e:
            print(f"Error testing {s['name']}: {e}\n")

    test_concurrency(args.url, 2, args.model)
    test_concurrency(args.url, 4, args.model)


if __name__ == "__main__":
    main()
