package cmd

import (
	"fmt"
	"sort"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/components"
)

var listCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List downloaded models",
	Long:    `List all models downloaded to the local HuggingFace cache.`,
	Args:    cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runList()
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
}

func runList() error {
	cfg, err := config.Load()
	if err != nil {
		tui.PrintError(clierrors.NewCLIError("failed to load config", "Check your config file", err))
		return nil
	}

	models, err := huggingface.ScanCache(cfg.HFCachePath)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			fmt.Sprintf("failed to scan cache: %s", err.Error()),
			fmt.Sprintf("Ensure %s is accessible", cfg.HFCachePath),
			err,
		))
		return nil
	}

	if len(models) == 0 {
		fmt.Println("No models downloaded. Use 'vllm-cli pull <model>' to download.")
		return nil
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].Ref.String() < models[j].Ref.String()
	})

	isTTY := tui.IsTTY()

	headers := []string{"NAME", "SIZE"}
	rows := make([][]string, len(models))
	for i, m := range models {
		rows[i] = []string{m.Ref.String(), huggingface.FormatSize(m.SizeBytes)}
	}

	fmt.Print(components.RenderTable(headers, rows, isTTY))
	return nil
}
