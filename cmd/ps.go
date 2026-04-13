package cmd

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/tui/components"
)

var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List running vLLM model containers",
	Long:  `Show all vLLM model containers currently managed by vllm-cli.`,
	Args:  cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runPs()
	},
}

func init() {
	rootCmd.AddCommand(psCmd)
}

func runPs() error {
	dockerClient, err := docker.NewClient()
	if err != nil {
		tui.PrintError(err)
		return nil
	}
	defer dockerClient.Close()

	containers, err := dockerClient.ListManaged()
	if err != nil {
		tui.PrintError(clierrors.NewCLIError("failed to list containers", "Ensure Docker daemon is running", err))
		return nil
	}

	if len(containers) == 0 {
		fmt.Println("No models running. Use 'vllm-cli run <model>' to start.")
		return nil
	}

	isTTY := tui.IsTTY()

	headers := []string{"NAME", "PORT", "STATUS", "UPTIME"}
	rows := make([][]string, len(containers))
	for i, ctr := range containers {
		status := "starting"
		if docker.IsHealthy(ctr.Port) {
			status = "running"
		}
		uptime := formatUptime(time.Since(ctr.CreatedAt))
		rows[i] = []string{ctr.ModelRef.String(), fmt.Sprintf("%d", ctr.Port), status, uptime}
	}

	fmt.Print(components.RenderTable(headers, rows, isTTY))
	return nil
}

func formatUptime(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm %ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh %dm", int(d.Hours()), int(d.Minutes())%60)
}
