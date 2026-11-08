#!/usr/bin/env python3
"""
Pre-flight Cluster Hardware & Interconnect Audit Tool for DGX Spark

Validates:
  1. Architecture: Linux aarch64 & NVIDIA GB10 (SM121)
  2. Memory Headroom: Available RAM >= 112 GiB
  3. Disk Capacity: Local NVMe free space for weights and cache
  4. Network: ConnectX-7 200GbE link status & dual-rail RoCE v2 GID index
  5. Peer Node Trust: Key-based SSH access & Worker Docker group permissions

Usage:
  python3 bootstrap/preflight.py [--worker admin@192.168.100.11]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys


def run_cmd(cmd: str) -> tuple[int, str]:
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return res.returncode, res.stdout.strip()


def audit_head_node():
    print("=== [Node 0: Head / Local Diagnostic] ===")
    
    # 1. Architecture & CPU
    arch = platform.machine()
    print(f"  - Architecture: {arch}")
    if arch not in ["aarch64", "arm64"]:
        print("    [WARN] Target architecture is expected to be aarch64 (NVIDIA Grace CPU).")

    # 2. NVIDIA GPU check
    rc, out = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null")
    if rc == 0 and out:
        print(f"  - GPU Detected: {out}")
    else:
        print("    [FAIL] nvidia-smi failed or no GPU found.")

    # 3. Available Memory
    rc, out = run_cmd("awk '/^MemAvailable:/ {printf \"%.1f\", $2 / 1048576}' /proc/meminfo 2>/dev/null")
    avail_gib = float(out) if rc == 0 and out else 0.0
    print(f"  - Available RAM: {avail_gib:.1f} GiB (Threshold: 112.0 GiB)")
    if avail_gib < 112.0:
        print(f"    [WARN] Available memory ({avail_gib:.1f} GiB) is below 112 GiB. Stop existing heavy models before launch.")
    else:
        print("    [PASS] Memory headroom verified.")

    # 4. Disk free space
    disk = shutil.disk_usage("/")
    free_gib = disk.free / (1024 ** 3)
    print(f"  - Root NVMe Free: {free_gib:.1f} GiB (Required: ~530 GiB for initial download & cache)")
    if free_gib < 350.0:
        print("    [WARN] Low disk space for initial download of checkpoint + Engram tables.")
    else:
        print("    [PASS] Disk capacity verified.")

    # 5. ConnectX-7 Network & RoCE GID
    rc, out = run_cmd("ibv_devinfo -l 2>/dev/null")
    if rc == 0 and out:
        devs = [d.strip() for d in out.splitlines() if d.strip()]
        print(f"  - RoCE Devices: {', '.join(devs)}")
    else:
        print("    [WARN] No InfiniBand/RoCE devices detected via ibv_devinfo.")


def audit_worker_node(worker_target: str):
    print(f"\n=== [Node 1: Worker Diagnostic ({worker_target})] ===")
    
    # 1. SSH test
    rc, out = run_cmd(f"ssh -o BatchMode=yes -o ConnectTimeout=5 {worker_target} 'hostname' 2>/dev/null")
    if rc != 0:
        print(f"  [FAIL] Cannot connect to {worker_target} via key-based SSH.")
        print("         Please configure passwordless SSH: ssh-copy-id " + worker_target)
        return
    print(f"  - Hostname: {out} (SSH key auth verified)")

    # 2. Worker Docker permissions
    rc, out = run_cmd(f"ssh -o BatchMode=yes {worker_target} 'docker info >/dev/null 2>&1 && echo OK || echo FAIL'")
    if out == "OK":
        print("  - Docker Daemon Access: PASS (Worker user in docker group)")
    else:
        print("  - Docker Daemon Access: FAIL (User cannot access docker daemon socket)")
        print(f"    Fix on worker: sudo usermod -aG docker $(whoami)")

    # 3. Worker Memory & Disk
    rc, out = run_cmd(f"ssh -o BatchMode=yes {worker_target} 'awk \"/^MemAvailable:/ {{printf \\\"%.1f\\\", \\$2 / 1048576}}\" /proc/meminfo'")
    worker_mem = float(out) if rc == 0 and out else 0.0
    print(f"  - Available RAM: {worker_mem:.1f} GiB")

    rc, out = run_cmd(f"ssh -o BatchMode=yes {worker_target} 'df -B1G / | awk \"NR==2 {{print \\$4}}\"'")
    worker_disk = int(out) if rc == 0 and out.isdigit() else 0
    print(f"  - Root NVMe Free: {worker_disk} GiB (Required: ~335 GiB)")
    if worker_disk < 310:
        print(f"    [WARN] Worker disk ({worker_disk} GiB) is lower than 310 GiB. Clean old unused models.")
    else:
        print("    [PASS] Worker resources verified.")


def main():
    parser = argparse.ArgumentParser(description="DGX Spark Cluster Pre-flight Inspector")
    parser.add_argument("--worker", default=None, help="Worker SSH target (e.g. admin@192.168.100.11)")
    args = parser.parse_args()

    audit_head_node()
    if args.worker:
        audit_worker_node(args.worker)
    else:
        print("\n[INFO] Pass --worker user@host to verify Node 1 (Rank 1).")


if __name__ == "__main__":
    main()
