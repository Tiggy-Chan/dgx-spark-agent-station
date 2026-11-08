#!/usr/bin/env python3
"""
Streaming In-Memory Proxy Bridge & Reverse Tunnel Helper

Provides a temporary HTTP/HTTPS CONNECT forwarder that allows DGX Spark
nodes in restricted network environments to pull NVIDIA NGC base images
(nvcr.io) and GitHub submodule trees without writing static credentials or
proxy files to disk.

Usage:
  python3 bootstrap/proxy_tunnel.py --port 18888
"""

import argparse
import select
import socket
import sys
import threading
from urllib.parse import urlparse


def handle_client(client_sock: socket.socket):
    try:
        req = client_sock.recv(4096)
        if not req:
            return
        first_line = req.split(b"\r\n")[0].decode(errors="ignore")
        parts = first_line.split(" ")
        if len(parts) < 2:
            return

        method, url = parts[0], parts[1]

        if method == "CONNECT":
            # HTTPS Tunneling
            host, port = url.split(":")
            server_sock = socket.create_connection((host, int(port)), timeout=15)
            client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = [client_sock, server_sock]
            while True:
                r, _, _ = select.select(sockets, [], [], 120)
                if not r:
                    break
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        return
                    out = server_sock if s is client_sock else client_sock
                    out.sendall(data)
        elif url.startswith("http://"):
            # Plain HTTP Proxying
            u = urlparse(url)
            server_sock = socket.create_connection((u.hostname, u.port or 80), timeout=15)
            server_sock.sendall(req)
            sockets = [client_sock, server_sock]
            while True:
                r, _, _ = select.select(sockets, [], [], 120)
                if not r:
                    break
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        return
                    out = server_sock if s is client_sock else client_sock
                    out.sendall(data)
    except Exception:
        pass
    finally:
        client_sock.close()


def run_server(bind_ip: str, port: int):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_ip, port))
    server.listen(256)
    print(f"[INFO] Streaming proxy bridge listening on {bind_ip}:{port}")
    try:
        while True:
            cs, _ = server.accept()
            threading.Thread(target=handle_client, args=(cs,), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[INFO] Proxy server stopped.")
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser(description="In-Memory Streaming Proxy Bridge")
    parser.add_argument("--bind", default="127.0.0.1", help="Binding IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18888, help="Listening port (default: 18888)")
    args = parser.parse_args()

    run_server(args.bind, args.port)


if __name__ == "__main__":
    main()
