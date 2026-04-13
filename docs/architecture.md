# vllm-cli Architecture

This document walks through the internal design of vllm-cli for developers who want to understand or extend the codebase.

---

## Overview

```
  User
    |
    | vllm-cli run meta-llama/Llama-3.1-8B-Instruct
    v
+----------+
|  cmd/    |   Cobra subcommands
+----------+
     |
     +----------------+------------------+------------------+
     |                |                  |                  |
     v                v                  v                  v
+----------+  +---------------+  +----------+  +-------------------+
| docker/  |  | huggingface/  |  |  gpu/    |  |    memory/        |
| Docker   |  |  HF Hub API   |  |  NVML    |  | Estimate + Guard  |
|   SDK    |  |  cache scan   |  | /proc    |  |                   |
+----------+  +---------------+  +----------+  +-------------------+
     |                |
     v                v
 Docker daemon    api.huggingface.co
 (UNIX socket)   (HTTPS)
```

The CLI has no server process, no database, and no background daemon. Every invocation is a short-lived process that talks directly to the Docker daemon (via the Docker SDK Go client) and optionally to the HuggingFace API.

---

## Package dependency graph

```
main
 └── cmd
      ├── internal/config
      ├── internal/errors
      ├── internal/types
      ├── internal/docker
      │    └── internal/errors
      │    └── internal/types
      ├── internal/gpu
      │    └── internal/types
      ├── internal/huggingface
      │    └── internal/errors
      │    └── internal/types
      ├── internal/memory
      │    └── internal/huggingface
      │    └── internal/types
      └── internal/tui
           ├── internal/errors
           └── internal/tui/styles
           └── internal/tui/components
```

`internal/types` is the only package with zero internal dependencies — it's the shared vocabulary for all other packages. Nothing in `internal/` imports from `cmd/`.

---

## `cmd/` layer

### Command registration

Every subcommand self-registers with `rootCmd` via Go's `init()` mechanism:

```go
// cmd/run.go
func init() {
    rootCmd.AddCommand(runCmd)
    runCmd.Flags().BoolVar(&runForce, "force", false, "Skip memory guard check")
    // ...
}
```

Because `main.go` imports `cmd`, all `init()` functions run before `cmd.Execute()` is called. No central registry or switch statement is needed.

### Error surfacing

The pattern used in every subcommand:

1. Call `tui.PrintError(err)` to write to stderr with the correct format.
2. Return `nil` from `RunE` so Cobra doesn't also print the error.
3. Return the actual error only when you want the process to exit non-zero (rare — currently only used on argument-parse failures).

`cmd.Execute()` in `root.go` catches any returned errors and calls `os.Exit(1)`:

```go
func Execute() {
    if err := rootCmd.Execute(); err != nil {
        var cliErr *clierrors.CLIError
        if stderrors.As(err, &cliErr) {
            os.Exit(1)
        }
        fmt.Fprintf(os.Stderr, "Error: %s\n", err)
        os.Exit(1)
    }
}
```

---

## `internal/types`

The three shared structs that move through the whole system:

### `ModelRef`

```go
type ModelRef struct {
    Owner string  // "meta-llama"
    Name  string  // "Llama-3.1-8B-Instruct"
}
```

- `String()` returns `"meta-llama/Llama-3.1-8B-Instruct"` — used in HF API URLs and user output.
- `Slug()` returns `"meta-llama--Llama-3.1-8B-Instruct"` — used in Docker container names and the local HF cache directory name.
- `ParseModelRef(s)` requires exactly one `/`. Bare names like `"gpt2"` return an error.

### `ContainerInfo`

```go
type ContainerInfo struct {
    ID        string
    ModelRef  ModelRef
    Port      int
    Status    string     // "starting", "running", "stopping"
    CreatedAt time.Time
}
```

Populated by `docker.ListManaged()` from Docker labels.

### `SystemGPUInfo` and `GPUInfo`

```go
type GPUInfo struct {
    Index         int
    Name          string
    MemoryTotalMB int
    MemoryUsedMB  int
    MemoryFreeMB  int
    Architecture  int   // NVML arch enum (10 = Blackwell)
    IsUnified     bool  // true for DGX Spark
}

type SystemGPUInfo struct {
    GPUs          []GPUInfo
    TotalMemoryMB int
    FreeMemoryMB  int
}
```

`IsUnified` changes how the memory guard calculates available memory and triggers an extra OOM warning for unified-memory systems.

---

## `internal/errors`

### `CLIError`

```go
type CLIError struct {
    Message string
    Hint    string
    Err     error   // wrapped cause, for errors.Is/As
}
```

`Error()` formats as:

```
Error: model "gpt2" not found on HuggingFace
Hint: Check the model ID at https://huggingface.co/gpt2
```

### Sentinel constructors

These cover the common failure modes:

| Function | Trigger |
|----------|---------|
| `ErrModelNotFound(ref)` | HF API 404 |
| `ErrGatedModel(ref)` | HF API 401/403 |
| `ErrDockerNotRunning()` | Docker ping fails |
| `ErrPortExhausted()` | 8000-8099 all in use |
| `ErrMemoryInsufficient(needed, avail)` | Guard check fails |

### `FormatError`

`FormatError(err)` is used by `tui.PrintError` to produce the canonical text. If `err` is already a `CLIError`, it's printed as-is. Otherwise it wraps the raw error with `Hint: Use --verbose for details`.

---

## `internal/config`

### Config file location

Config lives at `$XDG_CONFIG_HOME/vllm-cli/config.yaml`, which defaults to `~/.config/vllm-cli/config.yaml`. The `ConfigDir()` function respects `XDG_CONFIG_HOME` if set.

### Load sequence

1. Start with `DefaultConfig()` (hardcoded sensible values).
2. Apply env-variable overrides (`VLLM_CLI_DOCKER_IMAGE`, `VLLM_CLI_PORT_START`).
3. Read the YAML file. If it doesn't exist, write it with defaults and return.
4. Unmarshal YAML on top of the existing struct (merges cleanly).
5. Re-apply env overrides so env always wins over the file.

### Defaults

| Key | Default |
|-----|---------|
| `docker_image` | `vllm/vllm-openai:latest` |
| `port_range_start` | `8000` |
| `port_range_end` | `8099` |
| `hf_cache_path` | `~/.cache/huggingface/hub` |
| `gpu_memory_ceiling_gb` | `0` (disabled) |
| `default_gpu_utilization` | `0.9` |

---

## `internal/docker`

### Client

`NewClient()` creates a Docker SDK client from environment (`DOCKER_HOST`, `DOCKER_TLS_VERIFY`, etc.), negotiates the API version, and pings the daemon. Any failure returns `ErrDockerNotRunning()`.

### Container lifecycle

```
CreateAndStart(opts)
    |
    +-- ContainerCreate (name: vllm-{slug})
    |     Config: image, cmd, env, labels, exposed ports
    |     HostConfig: runtime=nvidia, --gpus all, IpcMode=host,
    |                 ShmSize=16GB, port bindings, HF cache mount
    +-- ContainerStart
    |
    v
ContainerInfo{Status: "starting"}
    |
    v  (in goroutine)
WaitForHealthy(port, 300s)
    |   polls GET http://localhost:{port}/health every 2s
    v
ContainerInfo{Status: "running"}
    |
    v  (on stop)
Stop(containerID)
    |
    +-- ContainerStop (30s timeout)
    +-- ContainerRemove
```

### Container labels

Three labels are written on every container:

```
managed-by=vllm-cli
vllm-cli.model=owner/name
vllm-cli.port=8000
```

`ListManaged()` filters by `managed-by=vllm-cli` so it only sees containers created by vllm-cli.

### Port binding

The container always exposes its internal port 8000. The host port (8000-8099) is chosen by `PortManager.FindAvailablePort()`, which:

1. Queries Docker labels to find ports already claimed by managed containers.
2. Calls `net.Listen("tcp", ":PORT")` to verify the port is actually free on the host.
3. Returns the first free port, or `ErrPortExhausted()` if the whole range is taken.

### Docker configuration details

| Setting | Value | Why |
|---------|-------|-----|
| `runtime` | `nvidia` | NVIDIA Container Toolkit requirement |
| `DeviceRequests` | `Count: -1, Capabilities: [["gpu"]]` | Equivalent to `--gpus all` |
| `IpcMode` | `host` | vLLM requires host IPC for shared memory |
| `ShmSize` | 16 GB | PyTorch DataLoader workers need large `/dev/shm` |
| HF cache mount | `~/.cache/huggingface/hub` -> `/root/.cache/huggingface` | Shares models between host and container |

---

## `internal/gpu`

### Build-tag split

Two files provide the `Detect()` function — exactly one is compiled per build:

| File | Build tag | Behaviour |
|------|-----------|-----------|
| `detector_cgo.go` | `//go:build cgo` | Calls NVML via `go-nvml` |
| `detector_nocgo.go` | `//go:build !cgo` | Returns `ErrNoGPU` immediately |

Note: `go-nvml` uses `dlopen` at runtime to load `libnvidia-ml.so`. It does not use `import "C"`, so it's not a true CGO dependency in the usual sense. `CGO_ENABLED=0` builds simply omit the NVML path entirely.

### NVML detection flow (cgo build)

```
nvml.Init()
  |-- FAIL -> return ErrNoGPU
  v
nvml.DeviceGetCount()
  |-- count == 0 -> return ErrNoGPU
  v
for each device:
    DeviceGetName()
    DeviceGetArchitecture()
    DeviceGetMemoryInfo()
      |-- SUCCESS   -> discrete GPU, use NVML memory values
      |-- ERROR_NOT_SUPPORTED
      |     |
      |     +-> IsUnified = true
      |         ReadProcMeminfo() -> use system memory as GPU memory
      v
return SystemGPUInfo
```

### Blackwell / DGX Spark unified memory

On DGX Spark, `DeviceGetArchitecture` returns `10` (defined as `BlackwellArchitecture`). More importantly, `DeviceGetMemoryInfo` returns `ERROR_NOT_SUPPORTED` because the GPU and CPU share the same physical memory pool. The fallback to `/proc/meminfo` gives a practical approximation of total and available memory.

When `GPUInfo.IsUnified == true`, the memory guard and `run.go` both emit an extra warning: OOM on unified memory can crash the entire system, not just the container.

---

## `internal/huggingface`

### Client

`NewClient()` reads `HF_TOKEN` from environment. All requests include `Authorization: Bearer {token}` when the token is present. The client retries up to 3 times on HTTP 429 with exponential backoff (5s, 10s, 15s).

### `GetModelInfo`

Calls `GET https://huggingface.co/api/models/{owner}/{name}`. Returns:

- `ModelInfo` on 200
- `ErrModelNotFound` on 404
- `ErrGatedModel` on 401 or 403 (HF returns 401/403 for non-existent models when no token is provided — not 404)

### `GetModelConfig`

Fetches `https://huggingface.co/{owner}/{name}/resolve/main/config.json` directly (not the API endpoint). This is the architecture file that feeds the memory estimator.

### `ModelConfig` and the GPT-2 legacy field problem

Standard Llama/Mistral models use `hidden_size`, `num_hidden_layers`, `num_attention_heads`. GPT-2 and older models use `n_embd`, `n_layer`, `n_head`. The custom `UnmarshalJSON` handles both by unmarshalling into an alias struct with extra fields, then copying the legacy values into the standard fields if the standard fields came out as zero.

### Cache layout

HuggingFace stores models at:

```
~/.cache/huggingface/hub/
  models--{owner}--{name}/
    snapshots/
      {commit-hash}/
        config.json
        model.safetensors
        tokenizer.json
        ...
```

`ScanCache()` reads this layout to populate `CachedModel` entries. `dirSize()` follows symlinks because HF uses a content-addressable blob store with symlinks from the snapshot directory to a `blobs/` directory.

---

## `internal/memory`

### `Estimate(cfg *ModelConfig)`

Returns a `MemoryEstimate` broken down into:

```
WeightsGB  = NumParameters * bytesPerParam(dtype, quantBits) / 1e9

KVCacheGB  = 2                    // K and V
           * num_hidden_layers
           * num_key_value_heads   // falls back to num_attention_heads if zero
           * head_dim              // hidden_size / num_attention_heads
           * min(max_position_embeddings, 8192)  // capped at 8K
           * 2                    // bytes per fp16 element
           / 1e9

OverheadGB = (WeightsGB + KVCacheGB) * 0.20 + 0.5

TotalGB    = WeightsGB + KVCacheGB + OverheadGB
```

Bytes per parameter by dtype:

| dtype | bytes |
|-------|-------|
| `float32` | 4 |
| `float16` / `bfloat16` | 2 |
| `int8` | 1 |
| `int4` | 0.5 |

Unknown dtype defaults to `float16`.

### `Check(estimate, gpuInfo, gpuMemCeiling)`

Determines available memory using this priority order:

1. `gpuMemCeiling > 0` (from config): use that value directly.
2. `gpuInfo == nil`: assume safe (can't check, proceed anyway).
3. `gpuInfo.GPUs[0].IsUnified`: use `gpuInfo.FreeMemoryMB` (system free memory).
4. Discrete GPUs: use the minimum `MemoryFreeMB` across all GPUs (the bottleneck GPU for tensor-parallel workloads).

Returns `GuardResult{Safe: bool, NeededGB, AvailableGB, Message}`.

---

## `internal/tui`

### TTY detection

```go
func IsTTY() bool {
    return term.IsTerminal(int(os.Stdout.Fd()))
}
```

This uses `golang.org/x/term`. When stdout is a terminal, rich output (Bubble Tea spinners, lipgloss colours, unicode symbols) is used. When piped (e.g., in CI or scripts), plain text goes to stdout and errors go to stderr without ANSI codes.

### Output functions

| Function | TTY | Non-TTY |
|----------|-----|---------|
| `PrintSuccess(msg)` | green `✓ msg` | `✓ msg` |
| `PrintWarning(msg)` | yellow `⚠ msg` | `⚠ msg` |
| `PrintError(err)` | red formatted to stderr | plain text to stderr |

### Spinner and the buffered channel pattern

When a long operation runs in a goroutine alongside a Bubble Tea spinner, the channel must be buffered with size 2:

```go
done := make(chan error, 2)  // buffer = 2, not 1

go func() {
    done <- expensiveOperation()  // send #1
}()

spinnerGoroutine := func() {
    result := <-done   // receive #1
    done <- result     // send #2 (re-queue for the caller to read)
    p.Quit()
}
go spinnerGoroutine()
p.Run()  // blocks until Quit()

err := <-done  // receive #2
```

A buffer of 1 deadlocks: the re-send blocks because the slot is already occupied when the caller hasn't consumed yet. Buffer 2 ensures the re-send never blocks.

### `internal/tui/styles`

Seven pre-built lipgloss styles:

| Name | Appearance |
|------|-----------|
| `Title` | bold, blue (ANSI 12) |
| `Success` | green (ANSI 10) |
| `Warning` | yellow (ANSI 11) |
| `Error` | red (ANSI 9) |
| `Muted` | gray (ANSI 8) |
| `Bold` | bold |
| `Header` | bold + underline |

### `internal/tui/components`

Three reusable UI components:

- **SpinnerModel**: a Bubble Tea model wrapping `charmbracelet/bubbles/spinner`. Used by `run` and `pull` when in TTY mode.
- **RenderTable(headers, rows, isTTY)**: column-aligned table with optional lipgloss header styling. Used by `ps` and `list`.
- **RenderMemoryBar(usedGB, totalGB, isTTY)**: fills a 20-character bar with `█` / `░` blocks, coloured green/yellow/red at 80%/90% thresholds.

---

## Request flow walkthrough

Full trace of `vllm-cli run meta-llama/Llama-3.1-8B-Instruct`:

```
1. main.go: cmd.Execute()
   |
2. cmd/root.go: rootCmd.Execute() -> dispatches to runCmd.RunE
   |
3. cmd/run.go: runRun("meta-llama/Llama-3.1-8B-Instruct")
   |
4. types.ParseModelRef("meta-llama/Llama-3.1-8B-Instruct")
   |  -> ModelRef{Owner: "meta-llama", Name: "Llama-3.1-8B-Instruct"}
   |
5. config.Load()
   |  -> reads ~/.config/vllm-cli/config.yaml
   |  -> applies VLLM_CLI_DOCKER_IMAGE / VLLM_CLI_PORT_START env overrides
   |
6. docker.NewClient()
   |  -> SDK client from env, ping daemon
   |
7. dockerClient.GetContainer("meta-llama--Llama-3.1-8B-Instruct")
   |  -> ListManaged filtered by label, check slug match
   |  -> nil (not already running)
   |
8. Memory guard (skipped if --force):
   |
   |  a. huggingface.NewClient().GetModelConfig(ref)
   |     -> GET https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json
   |     -> ModelConfig{HiddenSize: 4096, NumHiddenLayers: 32, ...}
   |
   |  b. memory.Estimate(modelCfg)
   |     -> WeightsGB = 8e9 * 2 / 1e9 = 16 GB (fp16)
   |     -> KVCacheGB = 2*32*8*128*8192*2 / 1e9 ≈ 2.1 GB
   |     -> OverheadGB = (16+2.1)*0.20 + 0.5 ≈ 4.1 GB
   |     -> TotalGB ≈ 22.2 GB
   |
   |  c. gpu.Detect()
   |     -> nvml.Init() -> OK
   |     -> DeviceGetMemoryInfo -> Free: 24576 MB (24 GB)
   |     -> SystemGPUInfo{FreeMemoryMB: 24576}
   |
   |  d. memory.Check(estimate, gpuInfo, 0)
   |     -> 22.2 GB < 24 GB -> Safe = true
   |     -> tui.PrintSuccess("Memory check passed (22.2GB needed, 24.0GB available)")
   |
9. dockerClient.EnsureImage("vllm/vllm-openai:latest")
   |  -> ImageInspectWithRaw -> image exists locally -> return nil channel
   |
10. docker.NewPortManager(8000, 8099).FindAvailablePort(dockerClient)
    |  -> ListManaged -> no containers -> usedPorts = {}
    |  -> net.Listen(":8000") -> free
    |  -> port = 8000
    |
11. dockerClient.CreateAndStart(ContainerOpts{...})
    |  -> ContainerCreate "vllm-meta-llama--Llama-3.1-8B-Instruct"
    |     labels: managed-by=vllm-cli, vllm-cli.model=meta-llama/..., vllm-cli.port=8000
    |     cmd: ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--gpu-memory-utilization", "0.90"]
    |     runtime=nvidia, gpus all, IpcMode=host, ShmSize=16GB
    |  -> ContainerStart
    |
12. tui.IsTTY() -> true (interactive terminal)
    |
13. Goroutine: docker.WaitForHealthy(8000, 300s)
    |  -> polls GET http://localhost:8000/health every 2s
    |
14. Bubble Tea spinner: "Waiting for meta-llama/Llama-3.1-8B-Instruct..."
    |  (displays while goroutine polls)
    |
15. /health returns 200 (model loaded, ~60-120s later)
    |  -> healthDone <- nil
    |  -> spinner goroutine re-queues nil, calls p.Quit()
    |  -> spinner exits
    |
16. select reads nil from healthDone
    |
17. tui.PrintSuccess("Model meta-llama/Llama-3.1-8B-Instruct is ready!")
    fmt.Printf("  OpenAI-compatible API: http://localhost:8000/v1\n")
    fmt.Printf("  Health endpoint:       http://localhost:8000/health\n")
    |
18. return nil -> os.Exit(0)
```
