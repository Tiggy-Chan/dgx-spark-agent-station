#!/usr/bin/env python3
"""
9Router Database Synchronizer & Model Alias Provisioner

Automates the registration and alias mapping of the Dual DGX Spark
DeepSeek-V4.1-Flash (748B) cluster into 9Router SQLite databases.

Solves the common issue where Claude Code defaults to "claude-3-7-sonnet-*"
or "claude-3-5-sonnet-*" and 9Router returns "model_not_found" or "No active credentials".

Usage:
  python3 gateway/router_sync.py --db /opt/9router/data/db/data.sqlite --upstream http://127.0.0.1:8002/v1
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid

TARGET_COMBOS = [
    "DeepSeek-V4.1-Flash-1017",
    "my-model",
    "my-model/DeepSeek-V4.1-Flash-1017",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
]

DEFAULT_UPSTREAM_TARGET = "openai-compatible-chat/my-model"
NODE_ID = "openai-compatible-chat-spark-qwen38"


def sync_database(db_path: str, upstream_url: str):
    if not os.path.exists(db_path):
        print(f"Error: 9Router database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    backup_path = f"{db_path}.bak_{int(time.time())}"
    print(f"[BACKUP] Creating backup at: {backup_path}")
    shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Ensure providerConnections has credentials for the providerNode
    print("[STEP 1] Auditing provider connection bindings...")
    row = cursor.execute("SELECT id FROM providerConnections WHERE provider=?", (NODE_ID,)).fetchone()
    if not row:
        src = cursor.execute("SELECT authType, name, email, priority, data FROM providerConnections WHERE name='my-model' OR provider='openai-compatible-chat' LIMIT 1").fetchone()
        if src:
            cursor.execute(
                "INSERT INTO providerConnections (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (str(uuid.uuid4()), NODE_ID, src[0], src[1], src[2], src[3], src[4], now, now)
            )
            print(f"[OK] Registered active provider credential for '{NODE_ID}'")
        else:
            dummy_data = json.dumps({
                "apiKey": "sk-spark-local",
                "defaultModel": "my-model",
                "testStatus": "active",
                "providerSpecificData": {
                    "prefix": "my-model",
                    "apiType": "chat",
                    "baseUrl": upstream_url,
                    "nodeName": "my-model"
                }
            })
            cursor.execute(
                "INSERT INTO providerConnections (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt) "
                "VALUES (?, ?, 'api-key', 'spark-cluster', 'spark@local', 1, 1, ?, ?, ?)",
                (str(uuid.uuid4()), NODE_ID, dummy_data, now, now)
            )
            print(f"[OK] Created fresh provider connection for '{NODE_ID}' -> {upstream_url}")
    else:
        print(f"[INFO] Provider connection for '{NODE_ID}' already exists.")

    # 2. Provision model combos and aliases
    print("[STEP 2] Mapping model combos and Claude Code aliases...")
    mapped_target = json.dumps([DEFAULT_UPSTREAM_TARGET])
    for combo_name in TARGET_COMBOS:
        existing = cursor.execute("SELECT id FROM combos WHERE name=?", (combo_name,)).fetchone()
        if existing:
            cursor.execute("UPDATE combos SET models=?, updatedAt=? WHERE id=?", (mapped_target, now, existing[0]))
            print(f"  - Updated alias: {combo_name} -> {DEFAULT_UPSTREAM_TARGET}")
        else:
            cursor.execute(
                "INSERT INTO combos (id, name, kind, models, createdAt, updatedAt) VALUES (?, ?, 'fallback', ?, ?, ?)",
                (str(uuid.uuid4()), combo_name, mapped_target, now, now)
            )
            print(f"  - Created alias: {combo_name} -> {DEFAULT_UPSTREAM_TARGET}")

    conn.commit()
    conn.close()
    print("[SUCCESS] Database synchronization completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Synchronize 9Router DB with DGX Spark cluster endpoints")
    parser.add_argument("--db", default="/opt/9router/data/db/data.sqlite", help="Path to 9router sqlite DB")
    parser.add_argument("--upstream", default="http://127.0.0.1:8002/v1", help="Upstream URL of Spark tunnel")
    args = parser.parse_args()

    sync_database(args.db, args.upstream)


if __name__ == "__main__":
    main()
