package cmd

import (
	"context"
	stderrors "errors"
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/components"
	"github.com/user/vllm-cli/internal/types"
)

var pullImage string

var pullCmd = &cobra.Command{
	Use:   "pull <model>",
	Short: "Download a model from HuggingFace",
	Long: `Download a model from HuggingFace to the local cache.

The model argument must be in "owner/name" format, e.g.:
  vllm-cli pull mistralai/Mistral-7B-Instruct-v0.1

Set HF_TOKEN environment variable for gated models.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runPull(args[0])
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
	pullCmd.Flags().StringVar(&pullImage, "image", "", "Override the vLLM Docker image (default: from config)")
}

func runPull(modelArg string) error {
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

	if pullImage != "" {
		cfg.DockerImage = pullImage
	}

	cached, err := huggingface.FindCachedModel(cfg.HFCachePath, ref)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			fmt.Sprintf("checking local cache: %s", err.Error()),
			fmt.Sprintf("Ensure %s is accessible", cfg.HFCachePath),
			err,
		))
		return nil
	}
	if cached != nil && len(cached.Snapshots) > 0 {
		fmt.Printf("Model already downloaded (%s)\n", huggingface.FormatSize(cached.SizeBytes))
		return nil
	}

	hfClient := huggingface.NewClient()
	modelInfo, err := hfClient.GetModelInfo(ref)
	if err != nil {
		var cliErr *clierrors.CLIError
		if stderrors.As(err, &cliErr) {
			if huggingface.ResolveToken() == "" {
				tui.PrintError(clierrors.NewCLIError(
					fmt.Sprintf("model %q not found on HuggingFace (or requires authentication)", ref.String()),
					fmt.Sprintf("Set HF_TOKEN if this is a gated model, or verify the model ID at https://huggingface.co/%s", ref.String()),
					err,
				))
			} else {
				tui.PrintError(cliErr)
			}
		} else {
			tui.PrintError(clierrors.NewCLIError(
				fmt.Sprintf("model %q not found on HuggingFace", ref.String()),
				fmt.Sprintf("Check the model ID at https://huggingface.co/%s", ref.String()),
				err,
			))
		}
		return nil
	}

	dockerClient, err := docker.NewClient()
	if err != nil {
		tui.PrintError(err)
		return nil
	}
	defer dockerClient.Close()

	fmt.Printf("Checking Docker image %s...\n", cfg.DockerImage)
	progressCh, err := dockerClient.EnsureImage(cfg.DockerImage)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			"failed to pull Docker image",
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

	hfToken := huggingface.ResolveToken()
	opts := docker.ContainerOpts{
		ModelRef:    ref,
		Port:        0,
		HFCachePath: cfg.HFCachePath,
		HFToken:     hfToken,
		DockerImage: cfg.DockerImage,
		GPUMemUtil:  cfg.DefaultGPUUtilization,
	}

	containerInfo, err := dockerClient.CreateAndStart(opts)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			"failed to start download container",
			"Ensure Docker is running and you have GPU support",
			err,
		))
		return nil
	}

	totalSize := modelInfo.TotalSize()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dlProgressCh := huggingface.WatchDownloadProgress(ctx, cfg.HFCachePath, ref, totalSize, 2*time.Second)

	doneCh := make(chan error, 1)
	go func() {
		doneCh <- waitForContainerExit(dockerClient, ref.Slug())
		close(doneCh)
	}()

	var downloadErr error

	if isTTY {
		model := components.NewDownloadModel(ref.String(), totalSize, dlProgressCh, doneCh)
		p := tea.NewProgram(model)
		finalModel, _ := p.Run()
		final := finalModel.(components.DownloadModel)
		downloadErr = final.Err()
	} else {
		fmt.Printf("Downloading %s...\n", ref.String())
		lastPct := -1
		for {
			select {
			case prog, ok := <-dlProgressCh:
				if ok && prog.Total > 0 {
					pct := int(float64(prog.Downloaded) / float64(prog.Total) * 100)
					if pct != lastPct && pct%5 == 0 {
						fmt.Printf("  %d%% — %s / %s\n", pct,
							huggingface.FormatSize(prog.Downloaded),
							huggingface.FormatSize(prog.Total))
						lastPct = pct
					}
				}
			case err := <-doneCh:
				downloadErr = err
				goto downloadDone
			}
		}
	downloadDone:
	}

	cancel()
	_ = dockerClient.Remove(containerInfo.ID)

	if downloadErr != nil {
		tui.PrintError(clierrors.NewCLIError(
			"waiting for download container",
			"Check Docker logs for details",
			downloadErr,
		))
		return nil
	}

	pulled, err := huggingface.FindCachedModel(cfg.HFCachePath, ref)
	if err != nil || pulled == nil {
		tui.PrintError(clierrors.NewCLIError(
			fmt.Sprintf("model %s not found in cache after download", ref.String()),
			"The download may have failed silently",
			nil,
		))
		return nil
	}

	tui.PrintSuccess(fmt.Sprintf("Successfully pulled %s (%s)", ref.String(), huggingface.FormatSize(pulled.SizeBytes)))
	return nil
}

func waitForContainerExit(dockerClient *docker.Client, modelSlug string) error {
	for {
		time.Sleep(2 * time.Second)
		ctr, err := dockerClient.GetContainer(modelSlug)
		if err != nil {
			return err
		}
		if ctr == nil {
			return nil
		}
		if ctr.Status == "exited" || ctr.Status == "dead" {
			return nil
		}
	}
}
