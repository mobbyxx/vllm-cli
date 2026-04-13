package cmd

import (
	stderrors "errors"
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/memory"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/styles"
	"github.com/user/vllm-cli/internal/types"
)

var showCmd = &cobra.Command{
	Use:   "show <model>",
	Short: "Display comprehensive model information",
	Long: `Display comprehensive model information including architecture, memory estimate, and running status.

The model argument must be in "owner/name" format, e.g.:
  vllm-cli show openai-community/gpt2
  vllm-cli show meta-llama/Llama-3.1-8B-Instruct`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runShow(args[0])
	},
}

func init() {
	rootCmd.AddCommand(showCmd)
}

func sectionHeader(isTTY bool, label string) string {
	if isTTY {
		return styles.Title.Render(label)
	}
	return label
}

func runShow(modelArg string) error {
	ref, err := types.ParseModelRef(modelArg)
	if err != nil {
		cliErr := clierrors.NewCLIError(
			err.Error(),
			`Model must be in "owner/name" format, e.g. openai-community/gpt2`,
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

	isTTY := tui.IsTTY()
	hfClient := huggingface.NewClient()

	_, err = hfClient.GetModelInfo(ref)
	if err != nil {
		var cliErr *clierrors.CLIError
		if stderrors.As(err, &cliErr) {
			newErr := clierrors.NewCLIError(
				fmt.Sprintf("model %q not found on HuggingFace", ref.String()),
				fmt.Sprintf("Check the model ID at https://huggingface.co/%s", ref.String()),
				err,
			)
			tui.PrintError(newErr)
			return newErr
		}
		tui.PrintError(err)
		return err
	}

	fmt.Printf("Model: %s\n", ref.String())

	modelCfg, cfgErr := hfClient.GetModelConfig(ref)

	if cfgErr != nil {
		fmt.Printf("\n%s\n  (unavailable: %s)\n", sectionHeader(isTTY, "Architecture:"), cfgErr.Error())
	} else {
		paramCount := huggingface.EstimateParameterCount(modelCfg)

		quantization := "none"
		if modelCfg.QuantizationConfig != nil && modelCfg.QuantizationConfig.QuantMethod != "" {
			quantization = modelCfg.QuantizationConfig.QuantMethod
		}

		dtype := modelCfg.TorchDtype
		if dtype == "" {
			dtype = "float16"
		}

		fmt.Printf("\n%s\n", sectionHeader(isTTY, "Architecture:"))
		if paramCount > 0 {
			fmt.Printf("  Parameters:      %s\n", formatParams(paramCount))
		}
		if modelCfg.HiddenSize > 0 {
			fmt.Printf("  Hidden Size:     %d\n", modelCfg.HiddenSize)
		}
		if modelCfg.NumHiddenLayers > 0 {
			fmt.Printf("  Layers:          %d\n", modelCfg.NumHiddenLayers)
		}
		if modelCfg.NumAttentionHeads > 0 {
			fmt.Printf("  Attention Heads: %d\n", modelCfg.NumAttentionHeads)
		}
		if modelCfg.MaxPositionEmbeddings > 0 {
			fmt.Printf("  Context Length:  %d\n", modelCfg.MaxPositionEmbeddings)
		}
		fmt.Printf("  Dtype:           %s\n", dtype)
		fmt.Printf("  Quantization:    %s\n", quantization)

		memEst := memory.Estimate(modelCfg)
		fmt.Printf("\n%s\n", sectionHeader(isTTY, "Memory Estimate:"))
		fmt.Printf("  Weights:         %.1f GB\n", memEst.WeightsGB)
		fmt.Printf("  KV Cache:        %.1f GB\n", memEst.KVCacheGB)
		fmt.Printf("  Overhead:        %.1f GB\n", memEst.OverheadGB)
		fmt.Printf("  Total:           ~%.1f GB\n", memEst.TotalGB)
	}

	fmt.Printf("\n%s\n", sectionHeader(isTTY, "Status:"))

	cached, err := huggingface.FindCachedModel(cfg.HFCachePath, ref)
	if err != nil || cached == nil || len(cached.Snapshots) == 0 {
		fmt.Printf("  Cached:          no\n")
	} else {
		fmt.Printf("  Cached:          yes (%s)\n", huggingface.FormatSize(cached.SizeBytes))
	}

	dockerClient, dockerErr := docker.NewClient()
	if dockerErr != nil {
		fmt.Printf("  Running:         unknown (Docker unavailable)\n")
	} else {
		defer dockerClient.Close()
		ctr, err := dockerClient.GetContainer(ref.Slug())
		if err != nil || ctr == nil || ctr.Status != "running" {
			fmt.Printf("  Running:         no\n")
		} else {
			fmt.Printf("  Running:         yes (port %d)\n", ctr.Port)
			fmt.Printf("  API:             http://localhost:%d/v1\n", ctr.Port)
			if !ctr.CreatedAt.IsZero() {
				uptime := time.Since(ctr.CreatedAt).Truncate(time.Second)
				fmt.Printf("  Uptime:          %s\n", uptime)
			}
		}
	}

	return nil
}

func formatParams(n int64) string {
	if n >= 1_000_000_000 {
		return fmt.Sprintf("%.1fB", float64(n)/1_000_000_000)
	}
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	}
	return fmt.Sprintf("%d", n)
}
