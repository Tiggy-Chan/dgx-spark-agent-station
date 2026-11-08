#!/usr/bin/env python3
"""
Real-time Prompt Cache, KV Shard & NVMe Spill Monitor for DGX Spark

Polls the /health endpoint of the TensorFold engine to display:
  - Cache hit rate and resumed tokens
  - Live decode TPS and active connections
  - Memory footprint per rank
  - NVMe Spill v2 written blocks and disk space

Usage:
  python3 tools/cache_monitor.py --url http://192.168.3.90:8888
"""

import argparse
import json
import os
import sys
import time
import urllib.request


def monitor(url: str, interval: float):
    endpoint = f"{url.rstrip('/')}/health"

    print(f"Monitoring TensorFold cache and cluster health on {endpoint} (Ctrl+C to stop)...")
    print(f"{'Time':8s} | {'Conns':5s} | {'Live TPS':8s} | {'Cached/Total (Hit%)':22s} | {'Spill Used':10s} | {'Head RAM Free':13s}")
    print("-" * 80)

    prev_cached = 0
    prev_total = 0

    while True:
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "cache_monitor/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))

            now = time.strftime("%H:%M:%S")
            live = data.get("live", {})
            conns = live.get("connections", 0)
            tps = live.get("decode_tokens_per_second", 0.0)

            pc = data.get("prompt_cache", {})
            cached = pc.get("cached_tokens", 0)
            computed = pc.get("computed_tokens", 0)
            total = cached + computed
            hit_pct = (cached / total * 100) if total > 0 else 0.0

            spill = pc.get("spill", [])
            spill_mb = 0
            if spill and isinstance(spill, list):
                spill_mb = sum(s.get("stored_bytes", 0) for s in spill) / (1024 * 1024)

            # Format memory
            mem = data.get("memory", {})
            active_gib = mem.get("active", 0) / (1024 ** 3)

            print(f"{now:8s} | {conns:5d} | {tps:8.1f} | {cached:7d}/{total:7d} ({hit_pct:4.1f}%) | {spill_mb:7.1f} MB | {active_gib:8.1f} GiB active")

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} | Error fetching /health: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Real-time Cluster Health & Cache Monitor")
    parser.add_argument("--url", default="http://127.0.0.1:8888", help="Base URL of the cluster")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()

    monitor(args.url, args.interval)


if __name__ == "__main__":
    main()
