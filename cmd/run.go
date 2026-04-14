package cmd

import (
	"context"
	stderrors "errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/gpu"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/memory"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/components"
	"github.com/user/vllm-cli/internal/types"
)

var (
	runForce              bool
	runPort               int
	runGPUMemUtil         float64
	runQuantization       string
	runDtype              string
	runMaxModelLen        int
	runTensorParallelSize int
	runImage              string
)

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Start a vLLM model server",
	Long: `Start a vLLM model server in a Docker container.

The model argument must be in "owner/name" format, e.g.:
  vllm-cli run mistralai/Mistral-7B-Instruct-v0.1

Set HF_TOKEN environment variable for gated models.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runRun(args[0])
	},
}

func init() {
	rootCmd.AddCommand(runCmd)

	runCmd.Flags().BoolVar(&runForce, "force", false, "Skip memory guard check")
	runCmd.Flags().IntVar(&runPort, "port", 0, "Port to serve on (default: auto-assign)")
	runCmd.Flags().Float64Var(&runGPUMemUtil, "gpu-memory-utilization", 0.9, "GPU memory utilization fraction (0.0-1.0)")
	runCmd.Flags().StringVar(&runQuantization, "quantization", "", "Quantization method (awq, gptq, or empty)")
	runCmd.Flags().StringVar(&runDtype, "dtype", "", "Data type (float16, bfloat16, auto)")
	runCmd.Flags().IntVar(&runMaxModelLen, "max-model-len", 0, "Maximum sequence length (context window)")
	runCmd.Flags().IntVar(&runTensorParallelSize, "tensor-parallel-size", 1, "Number of GPUs for tensor parallelism")
	runCmd.Flags().StringVar(&runImage, "image", "", "Override the vLLM Docker image (default: from config)")
}

func runRun(modelArg string) error {
	isTTY := tui.IsTTY()

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

	cfg, err := config.Load()
	if err != nil {
		tui.PrintError(clierrors.NewCLIError("failed to load config", "Check your config file", err))
		return nil
	}

	if runImage != "" {
		cfg.DockerImage = runImage
	}

	dockerClient, err := docker.NewClient()
	if err != nil {
		tui.PrintError(err)
		return nil
	}
	defer dockerClient.Close()

	existing, err := dockerClient.GetContainer(ref.Slug())
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			"failed to check for running containers",
			"Ensure Docker daemon is running",
			err,
		))
		return nil
	}
	if existing != nil && existing.Status == "running" {
		fmt.Printf("Model already running on port %d\n", existing.Port)
		return nil
	}

	if !runForce {
		hfClient := huggingface.NewClient()
		modelCfg, cfgErr := hfClient.GetModelConfig(ref)
		if cfgErr != nil {
			tui.PrintWarning(fmt.Sprintf("could not fetch model config (%s), skipping memory check", cfgErr.Error()))
		} else {
			estimate := memory.Estimate(modelCfg, nil)

			gpuInfo, gpuErr := gpu.Detect()
			if gpuErr != nil && stderrors.Is(gpuErr, gpu.ErrNoGPU) {
				tui.PrintWarning("No GPU detected, skipping memory check")
			} else if gpuErr != nil {
				tui.PrintWarning(fmt.Sprintf("could not detect GPU (%s), skipping memory check", gpuErr.Error()))
			} else {
				guardResult := memory.Check(estimate, gpuInfo, cfg.GPUMemoryCeiling)
				if !guardResult.Safe {
					msg := fmt.Sprintf("model may not fit in GPU memory (%.1fGB needed, %.1fGB available)\nHint: Use --force to skip this check, or stop other models to free memory",
						guardResult.NeededGB, guardResult.AvailableGB)
					if len(gpuInfo.GPUs) > 0 && gpuInfo.GPUs[0].IsUnified {
						tui.PrintWarning("This system uses unified memory — OOM may crash the entire system")
					}
					tui.PrintError(clierrors.NewCLIError(
						fmt.Sprintf("model may not fit in GPU memory (%.1fGB needed, %.1fGB available)", guardResult.NeededGB, guardResult.AvailableGB),
						"Use --force to skip this check, or stop other models to free memory",
						nil,
					))
					_ = msg
					return nil
				}
				tui.PrintSuccess(fmt.Sprintf("Memory check passed (%.1fGB needed, %.1fGB available)", guardResult.NeededGB, guardResult.AvailableGB))
			}
		}
	}

	progressCh, err := dockerClient.EnsureImage(cfg.DockerImage)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			"failed to check Docker image",
			"Check your internet connection and Docker daemon",
			err,
		))
		return nil
	}
	if progressCh != nil {
		fmt.Printf("Pulling Docker image %s...\n", cfg.DockerImage)
		for event := range progressCh {
			if event.Error != "" {
				tui.PrintError(clierrors.NewCLIError(
					fmt.Sprintf("Docker image pull failed: %s", event.Error),
					"Check your internet connection",
					nil,
				))
				return nil
			}
			if verbose && event.Status != "" {
				fmt.Printf("  %s %s\n", event.Status, event.Progress)
			}
		}
		fmt.Printf("Docker image ready.\n")
	}

	port := runPort
	if port == 0 {
		portMgr := docker.NewPortManager(cfg.PortRangeStart, cfg.PortRangeEnd)
		port, err = portMgr.FindAvailablePort(dockerClient)
		if err != nil {
			tui.PrintError(err)
			return nil
		}
	}

	extraArgs := []string{}
	if runMaxModelLen > 0 {
		extraArgs = append(extraArgs, "--max-model-len", fmt.Sprintf("%d", runMaxModelLen))
	}
	if runQuantization != "" {
		extraArgs = append(extraArgs, "--quantization", runQuantization)
	}
	if runDtype != "" {
		extraArgs = append(extraArgs, "--dtype", runDtype)
	}
	if runTensorParallelSize > 1 {
		extraArgs = append(extraArgs, "--tensor-parallel-size", fmt.Sprintf("%d", runTensorParallelSize))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	opts := docker.ContainerOpts{
		ModelRef:    ref,
		Port:        port,
		HFCachePath: cfg.HFCachePath,
		HFToken:     huggingface.ResolveToken(),
		DockerImage: cfg.DockerImage,
		ExtraArgs:   extraArgs,
		GPUMemUtil:  runGPUMemUtil,
	}

	containerInfo, err := dockerClient.CreateAndStart(opts)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			"failed to start container",
			"Ensure Docker is running and you have GPU support",
			err,
		))
		return nil
	}

	if ctx.Err() != nil {
		_ = dockerClient.Stop(containerInfo.ID)
		fmt.Println("\nAborted.")
		return nil
	}

	fmt.Printf("Starting %s on port %d...\n", ref.String(), port)

	healthCh := docker.WaitForHealthyAsync(port, 300*time.Second)
	logCh := make(chan string, 64)
	_ = dockerClient.StreamLogs(ctx, containerInfo.ID, logCh)

	var startupErr error

	if isTTY {
		model := components.NewStartupModel(ref.String(), logCh, healthCh)
		p := tea.NewProgram(model)

		go func() {
			<-ctx.Done()
			p.Quit()
		}()

		finalModel, _ := p.Run()
		final := finalModel.(components.StartupModel)

		if final.Quitted() || ctx.Err() != nil {
			fmt.Println("\nStopping...")
			_ = dockerClient.Stop(containerInfo.ID)
			fmt.Println("Stopped.")
			return nil
		}

		startupErr = final.Err()
	} else {
		for {
			select {
			case line, ok := <-logCh:
				if ok {
					fmt.Println(line)
				}
			case healthErr := <-healthCh:
				startupErr = healthErr
				goto done
			case <-ctx.Done():
				fmt.Println("\nStopping...")
				_ = dockerClient.Stop(containerInfo.ID)
				fmt.Println("Stopped.")
				return nil
			}
		}
	done:
	}

	if startupErr != nil {
		_ = dockerClient.Stop(containerInfo.ID)
		tui.PrintError(clierrors.NewCLIError(
			"model failed to start within 300s",
			fmt.Sprintf("Check logs with 'docker logs vllm-%s'", ref.Slug()),
			nil,
		))
		return nil
	}

	tui.PrintSuccess(fmt.Sprintf("Model %s is ready!", ref.String()))
	fmt.Printf("  OpenAI-compatible API: http://localhost:%d/v1\n", port)
	fmt.Printf("  Health endpoint:       http://localhost:%d/health\n", port)
	return nil
}
