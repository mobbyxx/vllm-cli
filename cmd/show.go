package cmd

import (
	stderrors "errors"
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/gpu"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/memory"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/styles"
	"github.com/user/vllm-cli/internal/types"
)

var showQuant bool

var showCmd = &cobra.Command{
	Use:   "show <model>",
	Short: "Display comprehensive model information",
	Long: `Display comprehensive model information including architecture, memory estimate, and running status.

The model argument must be in "owner/name" format, e.g.:
  vllm-cli show openai-community/gpt2
  vllm-cli show meta-llama/Llama-3.1-8B-Instruct

Use --quant to see VRAM estimates for different quantization levels.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runShow(args[0])
	},
}

func init() {
	showCmd.Flags().BoolVar(&showQuant, "quant", false, "show VRAM estimates for different quantization levels")
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
		paramCount, activeParamCount := huggingface.EstimateParameterCount(modelCfg)

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
			if activeParamCount > 0 && activeParamCount < paramCount {
				fmt.Printf("  Parameters:      %s total, %s active\n", formatParams(paramCount), formatParams(activeParamCount))
			} else {
				fmt.Printf("  Parameters:      %s\n", formatParams(paramCount))
			}
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

		estimateOpts := &memory.EstimateOpts{GPUMemUtil: 0.9}

		gpuInfo, gpuErr := gpu.Detect()
		if gpuErr == nil && gpuInfo != nil && len(gpuInfo.GPUs) > 0 {
			if gpuInfo.GPUs[0].IsUnified {
				estimateOpts.GPUTotalGB = float64(gpuInfo.TotalMemoryMB) / 1024.0
			} else {
				estimateOpts.GPUTotalGB = float64(gpuInfo.GPUs[0].MemoryTotalMB) / 1024.0
			}
		}

		memEst := memory.Estimate(modelCfg, estimateOpts)
		fmt.Printf("\n%s\n", sectionHeader(isTTY, "Memory Estimate:"))
		fmt.Printf("  Weights:         %.1f GB\n", memEst.WeightsGB)
		fmt.Printf("  KV Cache:        %.1f GB  (%dk context)\n", memEst.KVCacheGB, memEst.EffSeqLen/1024)
		fmt.Printf("  CUDA Overhead:   %.1f GB\n", memEst.CUDAOverheadGB)
		fmt.Printf("  ─────────────────────────\n")
		fmt.Printf("  Total:           %.1f GB\n", memEst.TotalGB)

		if showQuant {
			printQuantTable(isTTY, memEst.ParameterCount, memEst.Dtype, memEst)
		}

		if memEst.GPUTotalGB > 0 {
			fmt.Printf("\n%s\n", sectionHeader(isTTY, fmt.Sprintf("GPU Budget (%.0f%% of %.0f GB):", memEst.GPUMemUtil*100, memEst.GPUTotalGB)))
			fmt.Printf("  Usable:          %.1f GB\n", memEst.UsableGB)
			fmt.Printf("  Weights:        -%.1f GB\n", memEst.WeightsGB)
			fmt.Printf("  CUDA Overhead:  -%.1f GB\n", memEst.CUDAOverheadGB)
			fmt.Printf("  ─────────────────────────\n")
			fmt.Printf("  KV Cache Budget: %.1f GB\n", memEst.KVCacheMaxGB)
			if memEst.MaxTokens > 0 {
				fmt.Printf("  Max Context:     %dk tokens\n", memEst.MaxTokens/1024)
			}
			if memEst.MaxSeqLen > 0 && memEst.MaxTokens < memEst.MaxSeqLen {
				fmt.Printf("  Model Supports:  %dk tokens (limited by GPU memory)\n", memEst.MaxSeqLen/1024)
			}
		}
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

func printQuantTable(isTTY bool, paramCount int64, currentDtype string, memEst *memory.MemoryEstimate) {
	type quantLevel struct {
		label string
		bpp   float64
		dtype string
	}

	levels := []quantLevel{
		{"FP32", 4.0, "float32"},
		{"FP16/BF16", 2.0, "float16"},
		{"INT8 (W8A8)", 1.0, "int8"},
		{"INT4 (W4A16)", 0.5, "int4"},
	}

	kvCacheGB := memEst.KVCacheGB
	cudaGB := memEst.CUDAOverheadGB

	fmt.Printf("\n%s\n", sectionHeader(isTTY, "Quantization Estimates:"))

	for _, q := range levels {
		weightsGB := float64(paramCount) * q.bpp / 1e9
		totalGB := weightsGB + kvCacheGB + cudaGB

		marker := "  "
		switch currentDtype {
		case "float32":
			if q.dtype == "float32" {
				marker = "← "
			}
		case "float16", "bfloat16":
			if q.dtype == "float16" {
				marker = "← "
			}
		case "int8":
			if q.dtype == "int8" {
				marker = "← "
			}
		case "int4":
			if q.dtype == "int4" {
				marker = "← "
			}
		}

		fmt.Printf("  %-14s %7.1f GB  (weights %6.1f GB + KV %4.1f GB + CUDA %3.1f GB) %s\n",
			q.label, totalGB, weightsGB, kvCacheGB, cudaGB, marker)
	}
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
