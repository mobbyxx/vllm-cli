# Contributing to vllm-cli

Thanks for your interest. This guide covers everything you need to build, extend, and send changes back to the project.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Go | 1.21 | [go.dev/dl](https://go.dev/dl/) |
| Docker | 24+ | With NVIDIA Container Toolkit |
| NVIDIA driver | any | `nvidia-smi` must work |
| Git | 2.x | For cloning and tagging |

Verify your Docker GPU setup before doing anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Setup

```bash
git clone https://github.com/mobbyxx/vllm-cli.git
cd vllm-cli

# Standard build (NVML-backed GPU detection)
go build -o vllm-cli .

# No-CGO build (GPU detection always returns ErrNoGPU)
CGO_ENABLED=0 go build -o vllm-cli .

# Vet
go vet ./...

# Smoke test: should print version and exit 0
./vllm-cli --version
```

---

## Project structure

```
vllm-cli/
├── main.go                          # Entry point — calls cmd.Execute()
├── cmd/                             # Cobra subcommands (one file per command)
│   ├── root.go                      # rootCmd, Execute(), global flags
│   ├── run.go                       # `vllm-cli run`
│   ├── pull.go                      # `vllm-cli pull`
│   ├── stop.go                      # `vllm-cli stop`
│   ├── list.go                      # `vllm-cli list` / `ls`
│   ├── ps.go                        # `vllm-cli ps`
│   ├── rm.go                        # `vllm-cli rm`
│   └── show.go                      # `vllm-cli show`
└── internal/
    ├── config/
    │   └── config.go                # XDG config path, YAML load, env overrides
    ├── errors/
    │   └── errors.go                # CLIError type, sentinel constructors
    ├── types/
    │   ├── model.go                 # ModelRef (Owner/Name, Slug())
    │   ├── container.go             # ContainerInfo
    │   └── gpu.go                   # GPUInfo, SystemGPUInfo
    ├── docker/
    │   ├── client.go                # Docker SDK wrapper, NewClient()
    │   ├── container.go             # CreateAndStart, Stop, ListManaged, GetContainer
    │   ├── health.go                # WaitForHealthy, IsHealthy
    │   ├── image.go                 # EnsureImage (pull with progress channel)
    │   └── ports.go                 # PortManager, FindAvailablePort
    ├── gpu/
    │   ├── detector_cgo.go          # Detect() via NVML (build tag: cgo)
    │   ├── detector_nocgo.go        # Detect() stub returning ErrNoGPU (!cgo)
    │   └── procmem.go               # ReadProcMeminfo() for Blackwell fallback
    ├── huggingface/
    │   ├── client.go                # GetModelInfo, GetModelConfig, retry logic
    │   ├── types.go                 # ModelInfo, ModelConfig (with GPT-2 aliases)
    │   └── cache.go                 # ScanCache, FindCachedModel, RemoveCachedModel
    ├── memory/
    │   ├── estimator.go             # Estimate() — weights + KV cache + overhead
    │   └── guard.go                 # Check() — safe/unsafe verdict
    └── tui/
        ├── output.go                # IsTTY(), PrintError/Success/Warning
        ├── styles/
        │   └── theme.go             # lipgloss colour palette
        └── components/
            ├── spinner.go           # Bubble Tea spinner model
            ├── table.go             # RenderTable (TTY-aware)
            └── membar.go            # RenderMemoryBar
```

---

## How to add a new command

Say you want to add `vllm-cli logs` that tails a running container's logs.

**1. Create `cmd/logs.go`**

```go
package cmd

import (
    "github.com/spf13/cobra"

    "github.com/user/vllm-cli/internal/docker"
    clierrors "github.com/user/vllm-cli/internal/errors"
    "github.com/user/vllm-cli/internal/tui"
    "github.com/user/vllm-cli/internal/types"
)

var logsCmd = &cobra.Command{
    Use:   "logs <model>",
    Short: "Tail logs for a running model container",
    Args:  cobra.ExactArgs(1),
    RunE: func(cmd *cobra.Command, args []string) error {
        return runLogs(args[0])
    },
}

func init() {
    rootCmd.AddCommand(logsCmd)
    // logsCmd.Flags().IntVarP(&logsTail, "tail", "n", 50, "Number of lines to show")
}

func runLogs(modelArg string) error {
    ref, err := types.ParseModelRef(modelArg)
    if err != nil {
        cliErr := clierrors.NewCLIError(
            err.Error(),
            `Model must be in "owner/name" format, e.g. mistralai/Mistral-7B-v0.1`,
            err,
        )
        tui.PrintError(cliErr)
        return cliErr
    }

    dockerClient, err := docker.NewClient()
    if err != nil {
        tui.PrintError(err)
        return nil
    }
    defer dockerClient.Close()

    // ... your logic here ...
    return nil
}
```

**2. The `init()` function registers the command automatically.** No changes to `root.go` or `main.go` are needed.

**3. Conventions to follow:**

- Parse the model argument with `types.ParseModelRef(modelArg)` — bare names like `gpt2` are invalid.
- Wrap errors with `clierrors.NewCLIError(message, hint, underlyingErr)` so users see `Error: ...\nHint: ...`.
- Print errors via `tui.PrintError(err)` — it handles TTY vs plain-text automatically.
- Use `tui.IsTTY()` to branch between interactive (Bubble Tea) and scriptable (plain text) paths.
- Return `nil` after calling `tui.PrintError()` so Cobra doesn't print its own redundant error.
- Only return a non-nil error when you want the process to exit with code 1 (e.g., after parsing failures).

---

## How to extend an existing internal package

### Adding a field to Config

1. Add the field to `Config` in `internal/config/config.go`:

```go
type Config struct {
    // ... existing fields ...
    MyNewField string `yaml:"my_new_field"`
}
```

2. Set a sensible default in `DefaultConfig()`:

```go
func DefaultConfig() *Config {
    return &Config{
        // ... existing defaults ...
        MyNewField: "default-value",
    }
}
```

3. Add an env-variable override in `Load()` if needed:

```go
if v := os.Getenv("VLLM_CLI_MY_NEW_FIELD"); v != "" {
    cfg.MyNewField = v
}
```

The config file (`~/.config/vllm-cli/config.yaml`) is auto-created with defaults on first run.

### Adding a new error type

Add a constructor to `internal/errors/errors.go`:

```go
func ErrContainerCrashed(ref types.ModelRef) *CLIError {
    return &CLIError{
        Message: fmt.Sprintf("Container for %q exited unexpectedly", ref.String()),
        Hint:    fmt.Sprintf("Run 'docker logs vllm-%s' to see why", ref.Slug()),
    }
}
```

Then call it wherever appropriate:

```go
tui.PrintError(clierrors.ErrContainerCrashed(ref))
```

---

## Key conventions

### Container naming

Every container is named `vllm-{owner}--{name}`, built from `ModelRef.Slug()`:

```
meta-llama/Llama-3.1-8B-Instruct  ->  vllm-meta-llama--Llama-3.1-8B-Instruct
```

### Docker labels

All managed containers carry these labels so `ListManaged()` can filter them:

| Label | Value |
|-------|-------|
| `managed-by` | `vllm-cli` |
| `vllm-cli.model` | `owner/name` |
| `vllm-cli.port` | port number as string |

### Error format

`CLIError.Error()` returns:

```
Error: <human-readable message>
Hint: <actionable suggestion>
```

Always give a hint. Users copy-paste the exact text into issues.

### TTY detection

```go
isTTY := tui.IsTTY()  // wraps term.IsTerminal(int(os.Stdout.Fd()))
```

Use this at the top of every `runXxx()` function that produces interactive output. Branch on it to choose between Bubble Tea components and plain `fmt.Println`.

### Spinner channel buffer

The buffered channel pattern in `run.go` and `pull.go` is intentional:

```go
done := make(chan error, 2)  // buffer of 2, not 1
```

The goroutine sends once, then the forwarder reads and re-sends. A buffer of 1 would deadlock the second send. Don't change this to an unbuffered channel.

### Model reference parsing

`types.ParseModelRef` requires exactly one slash:

```go
// Good
types.ParseModelRef("meta-llama/Llama-3.1-8B-Instruct")

// Fails with a CLIError
types.ParseModelRef("gpt2")
```

### HuggingFace authentication

The HF API returns 401 or 403 for non-existent models when `HF_TOKEN` is not set — not 404. Both status codes are mapped to `ErrGatedModel`.

---

## Building for different architectures

```bash
# Linux amd64 (standard GPU servers)
GOOS=linux GOARCH=amd64 go build -o vllm-cli-linux-amd64 .

# Linux arm64 (DGX Spark / Grace Blackwell)
GOOS=linux GOARCH=arm64 go build -o vllm-cli-linux-arm64 .

# No-CGO (GPU detection disabled — for CI or cross-compile without libc)
CGO_ENABLED=0 go build -o vllm-cli-nocgo .
```

The CGO-enabled build links against `libdl` at runtime to load NVML. The `!cgo` variant skips GPU detection entirely and always returns `ErrNoGPU`, which causes the memory guard to be skipped.

---

## Running tests

```bash
go test ./...
```

There are no integration tests requiring a live Docker daemon or GPU yet. Unit tests cover the memory estimator and config loader. If you add a new package, add at least a basic test file.

---

## Creating a release

```bash
git tag v0.2.0
git push origin v0.2.0
```

Tag format is `vMAJOR.MINOR.PATCH`. The GitHub Actions workflow picks up the tag, builds binaries for `linux/amd64` and `linux/arm64`, and publishes them as a GitHub Release.

If you need to build release binaries locally:

```bash
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o dist/vllm-cli-linux-amd64 .
GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o dist/vllm-cli-linux-arm64 .
```

The `-ldflags="-s -w"` strips debug info and reduces binary size.
