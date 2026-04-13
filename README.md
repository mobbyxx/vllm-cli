# vllm-cli

Ollama-style CLI for managing [vLLM](https://github.com/vllm-project/vllm) Docker containers — with smart GPU memory estimation and OOM prevention, built for NVIDIA DGX Spark and standard GPU servers.

```
vllm-cli run meta-llama/Llama-3.1-8B-Instruct
```

---

## Features

- **Simple UX** — `run`, `pull`, `stop`, `ps`, `list`, `rm`, `show` commands like Ollama
- **Memory Guard** — estimates VRAM before loading a model, warns if it won't fit
- **DGX Spark support** — detects Blackwell unified memory via NVML fallback to `/proc/meminfo`
- **TTY-aware output** — rich spinner/table UI in interactive mode, clean text in pipes/scripts
- **Pure Go** — no `exec.Command("docker")`, uses the Docker SDK directly
- **Multi-model** — auto-assigns ports 8000–8099, multiple models run simultaneously

---

## Requirements

- Linux (amd64 or arm64)
- [Docker](https://docs.docker.com/engine/install/) with NVIDIA Container Toolkit
- NVIDIA GPU (or DGX Spark unified memory)
- Docker daemon running

Verify your setup:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Installation

### Option A: Download pre-built binary

**amd64 (standard Linux / x86_64):**
```bash
curl -L https://github.com/mobbyxx/vllm-cli/releases/latest/download/vllm-cli-linux-amd64 -o vllm-cli
chmod +x vllm-cli
sudo mv vllm-cli /usr/local/bin/vllm-cli
```

**arm64 (DGX Spark / Grace Blackwell):**
```bash
curl -L https://github.com/mobbyxx/vllm-cli/releases/latest/download/vllm-cli-linux-arm64 -o vllm-cli
chmod +x vllm-cli
sudo mv vllm-cli /usr/local/bin/vllm-cli
```

### Option B: Build from source

Requires [Go 1.21+](https://go.dev/dl/).

```bash
git clone https://github.com/mobbyxx/vllm-cli.git
cd vllm-cli

# Standard build (with NVML GPU detection)
go build -o vllm-cli .

# No-CGO build (GPU detection via /proc/meminfo only)
CGO_ENABLED=0 go build -o vllm-cli .

sudo mv vllm-cli /usr/local/bin/vllm-cli
```

For **arm64 (DGX Spark / Grace Blackwell)**, cross-compile on your build machine:

```bash
GOOS=linux GOARCH=arm64 go build -o vllm-cli .
sudo mv vllm-cli /usr/local/bin/vllm-cli
```

### Verify installation

```bash
vllm-cli --version
# vllm-cli version v0.1.0-dev

vllm-cli --help
```

---

## Usage

### Run a model

```bash
vllm-cli run meta-llama/Llama-3.1-8B-Instruct
```

The `run` command:
1. Checks available GPU memory
2. Pulls the vLLM Docker image if not present
3. Starts the container on an available port (8000–8099)
4. Polls `/health` until the server is ready
5. Prints the API endpoint

```
Model:   meta-llama/Llama-3.1-8B-Instruct
Port:    8000
API:     http://localhost:8000/v1
Docs:    http://localhost:8000/docs
```

### Flags for `run`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | auto | Force a specific port |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |
| `--max-model-len` | — | Override context length |
| `--quantization` | — | `awq`, `gptq`, `fp8`, etc. |
| `--dtype` | — | `float16`, `bfloat16`, `float32` |
| `--tensor-parallel-size` | — | Number of GPUs |
| `--force` | — | Start even if memory guard warns |

### Download a model (without running)

```bash
vllm-cli pull meta-llama/Llama-3.1-8B-Instruct
```

### List downloaded models

```bash
vllm-cli list   # or: vllm-cli ls
```

### Show running containers

```bash
vllm-cli ps
```

```
NAME                                      PORT   STATUS   UPTIME
meta-llama/Llama-3.1-8B-Instruct         8000   running  12m
mistralai/Mistral-7B-Instruct-v0.3        8001   running  3m
```

### Model details

```bash
vllm-cli show meta-llama/Llama-3.1-8B-Instruct
```

```
Architecture
  Parameters:   8.0B
  Hidden Size:  4096
  Layers:       32
  ...

Memory Estimate
  Weights:      14.2 GB
  KV Cache:     2.0 GB
  Overhead:     3.2 GB
  Total:        19.4 GB

Status
  Cached:  yes (14.2 GB)
  Running: yes (port 8000, uptime 12m)
```

### Stop a model

```bash
vllm-cli stop meta-llama/Llama-3.1-8B-Instruct
vllm-cli stop --all
```

### Remove a model

```bash
# Remove container only
vllm-cli rm meta-llama/Llama-3.1-8B-Instruct

# Remove container + cached model data
vllm-cli rm --model-data meta-llama/Llama-3.1-8B-Instruct

# Remove everything
vllm-cli rm --all --model-data
```

---

## Configuration

Config file: `~/.config/vllm-cli/config.yaml` (auto-created on first run)

```yaml
docker_image: vllm/vllm-openai:latest
port_range_start: 8000
port_range_end: 8099
hf_cache_path: ""          # default: ~/.cache/huggingface
gpu_memory_ceiling: 0      # 0 = no ceiling
default_gpu_utilization: 0.9
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for gated models |
| `VLLM_CLI_DOCKER_IMAGE` | Override Docker image |
| `VLLM_CLI_PORT_START` | Override port range start |

---

## Use as OpenAI API

Once running, models are accessible via the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Global Flags

| Flag | Description |
|------|-------------|
| `--verbose`, `-v` | Enable verbose output |
| `--config` | Custom config file path |
| `--no-color` | Disable colored output |

---

## License

MIT
