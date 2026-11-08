# Troubleshooting & Known Pitfalls on Dual DGX Spark

## 1. RoCE v2 GID Index Disparity

### Problem
During distributed collective communication setup, the model hangs indefinitely or logs:
```text
ibv_modify_qp failed with 22 Invalid argument ... remote GID ::
```

### Cause
On DGX Spark clusters, IPv4 RoCE v2 is frequently registered at different index numbers in the GID table across different nodes. For example:
* Node 0 (`spark-5d42`): RoCE v2 on `rocep1s0f0` resides at **GID Index 4**.
* Node 1 (`fusionxparkgb10-0d46`): RoCE v2 resides at **GID Index 3**.

Hardcoding a uniform `NCCL_IB_GID_INDEX=3` across both nodes causes silent QP state transition failure on Node 0.

### Resolution
Query the actual GID table on both nodes:
```bash
show_gids | grep -E "roce|enp"
```
Ensure `scripts/local.sh` or the launch scripts configure GID indices individually per node:
```bash
# Node 0
TF_RDMA_GID_INDEX=4
NCCL_IB_GID_INDEX=4

# Node 1
TF_RDMA_GID_INDEX=3
NCCL_IB_GID_INDEX=3
```

---

## 2. Worker Docker Socket Permission Denied

### Problem
Running `./scripts/prepare.sh` or `./start.sh` fails with:
```text
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock
```

### Cause
The remote user on the worker node (Rank 1) does not belong to the local `docker` group.

### Resolution
Log into Node 1 with root privilege and add the execution user to the docker group:
```bash
sudo usermod -aG docker admin
# Verify
sudo -u admin docker ps
```

---

## 3. Base Image Pull Failures in Restricted Networks

### Problem
Docker fails to pull the official NVIDIA NGC base image (`nvcr.io/nvidia/pytorch:26.07-py3`), logging timeout errors:
```text
Error response from daemon: Get "https://nvcr.io/v2/": context deadline exceeded
```

### Resolution
Use the streaming in-memory proxy bridge included in `bootstrap/proxy_tunnel.py`:
1. Start the proxy bridge on a node with egress connectivity:
   ```bash
   python3 bootstrap/proxy_tunnel.py --port 18888
   ```
2. Configure Docker daemon drop-in proxy on the DGX Spark:
   ```bash
   sudo mkdir -p /etc/systemd/system/docker.service.d
   cat << 'EOF' | sudo tee /etc/systemd/system/docker.service.d/proxy.conf
   [Service]
   Environment="HTTP_PROXY=http://127.0.0.1:18888"
   Environment="HTTPS_PROXY=http://127.0.0.1:18888"
   Environment="NO_PROXY=localhost,127.0.0.1,192.168.0.0/16"
   EOF
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   ```
3. Remove the proxy drop-in once the image build is completed.

---

## 4. 9Router Returns "model_not_found" (404)

### Problem
Calls to `https://9r.domain.com/v1/chat/completions` or `/v1/messages` fail with:
```json
{"error": {"message": "No active credentials for provider: openrouter", "code": "model_not_found"}}
```

### Cause
9Router does not have an explicit route mapping for `DeepSeek-V4.1-Flash-1017` or Claude Code defaults (`claude-3-7-sonnet-*`), routing the request to external OpenRouter or Anthropic channels instead.

### Resolution
Run the database synchronizer on your router server:
```bash
python3 gateway/router_sync.py --db /opt/9router/data/db/data.sqlite --upstream http://127.0.0.1:8002/v1
```

---

## 5. Reasoning Loop Runaway (32,000 Token Exhaustion)

### Problem
The model generates for minutes on end, eventually returning an empty response with all tokens consumed inside `reasoning_content`:
```text
[tensorfold] warning: a reply reached max_tokens while still thinking, so its content is empty
```

### Resolution
Add a strict thinking budget in `scripts/local.sh`:
```bash
REASONING_EFFORT="high"
THINKING_BUDGET=2048
```
This limits internal thinking tokens to 2,048, automatically cutting off excessive self-reflection and proceeding directly to answer generation.
