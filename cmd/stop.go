package cmd

import (
	stderrors "errors"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/types"
)

var stopAll bool

var stopCmd = &cobra.Command{
	Use:   "stop [<model>]",
	Short: "Stop a running vLLM model container",
	Long: `Stop a running vLLM model container.

You can stop a specific model:
  vllm-cli stop meta-llama/Llama-3.1-8B-Instruct

Or stop all running models:
  vllm-cli stop --all`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runStop(args)
	},
}

func init() {
	stopCmd.Flags().BoolVarP(&stopAll, "all", "a", false, "Stop all running models")
	rootCmd.AddCommand(stopCmd)
}

func runStop(args []string) error {
	dockerClient, err := docker.NewClient()
	if err != nil {
		tui.PrintError(err)
		return nil
	}
	defer dockerClient.Close()

	if stopAll {
		return stopAllModels(dockerClient)
	}

	if len(args) != 1 {
		cliErr := clierrors.NewCLIError(
			"model name required (or use --all)",
			"Run 'vllm-cli stop --help' for usage",
			nil,
		)
		tui.PrintError(cliErr)
		return cliErr
	}

	return stopSingleModel(dockerClient, args[0])
}

func stopAllModels(dockerClient *docker.Client) error {
	containers, err := dockerClient.ListManaged()
	if err != nil {
		tui.PrintError(err)
		return nil
	}

	if len(containers) == 0 {
		fmt.Println("No models running.")
		return nil
	}

	for _, ctr := range containers {
		if err := dockerClient.Stop(ctr.ID); err != nil {
			var cliErr *clierrors.CLIError
			if stderrors.As(err, &cliErr) {
				tui.PrintError(cliErr)
			} else {
				tui.PrintError(clierrors.NewCLIError(
					fmt.Sprintf("failed to stop model %s", ctr.ModelRef.String()),
					"Check Docker logs for details",
					err,
				))
			}
			return nil
		}
		tui.PrintSuccess(fmt.Sprintf("Stopped %s", ctr.ModelRef.String()))
	}
	return nil
}

func stopSingleModel(dockerClient *docker.Client, modelArg string) error {
	ref, err := types.ParseModelRef(modelArg)
	if err != nil {
		tui.PrintError(clierrors.NewCLIError(
			err.Error(),
			`Model must be in "owner/name" format, e.g. meta-llama/Llama-3.1-8B-Instruct`,
			err,
		))
		return nil
	}

	container, err := dockerClient.GetContainer(ref.Slug())
	if err != nil {
		tui.PrintError(err)
		return nil
	}

	if container == nil {
		fmt.Printf("Model %s is not running.\n", ref.String())
		return nil
	}

	if err := dockerClient.Stop(container.ID); err != nil {
		var cliErr *clierrors.CLIError
		if stderrors.As(err, &cliErr) {
			tui.PrintError(cliErr)
		} else {
			tui.PrintError(clierrors.NewCLIError(
				fmt.Sprintf("failed to stop model %s", ref.String()),
				"Check Docker logs for details",
				err,
			))
		}
		return nil
	}

	tui.PrintSuccess(fmt.Sprintf("Stopped %s (was on port %d)", ref.String(), container.Port))
	return nil
}
