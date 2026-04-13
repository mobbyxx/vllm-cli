package cmd

import (
	"bufio"
	stderrors "errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/user/vllm-cli/internal/config"
	"github.com/user/vllm-cli/internal/docker"
	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/tui"
	"github.com/user/vllm-cli/internal/types"
)

var (
	rmModelData bool
	rmAll       bool
)

var rmCmd = &cobra.Command{
	Use:   "rm [<model>]",
	Short: "Remove a vLLM model container and optionally cached data",
	Long: `Remove a vLLM model container and optionally its cached model data.

Remove a specific model:
  vllm-cli rm meta-llama/Llama-3.1-8B-Instruct

Remove a model and its cached data:
  vllm-cli rm --model-data meta-llama/Llama-3.1-8B-Instruct

Remove all managed containers:
  vllm-cli rm --all

Remove all managed containers and their cached data:
  vllm-cli rm --all --model-data`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runRm(args)
	},
}

func init() {
	rmCmd.Flags().BoolVarP(&rmModelData, "model-data", "d", false, "Also remove cached model files from HuggingFace cache")
	rmCmd.Flags().BoolVarP(&rmAll, "all", "a", false, "Remove all managed containers")
	rootCmd.AddCommand(rmCmd)
}

func runRm(args []string) error {
	cfg, err := config.Load()
	if err != nil {
		tui.PrintError(clierrors.NewCLIError("failed to load config", "Check your config file", err))
		return nil
	}

	dockerClient, err := docker.NewClient()
	if err != nil {
		tui.PrintError(err)
		return nil
	}
	defer dockerClient.Close()

	if rmAll {
		return rmAllModels(dockerClient, cfg)
	}

	if len(args) != 1 {
		cliErr := clierrors.NewCLIError(
			"model name required (or use --all)",
			"Run 'vllm-cli rm --help' for usage",
			nil,
		)
		tui.PrintError(cliErr)
		return cliErr
	}

	return rmSingleModel(dockerClient, cfg, args[0])
}

func rmSingleModel(dockerClient *docker.Client, cfg *config.Config, modelArg string) error {
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

	var cachedModel *huggingface.CachedModel
	if rmModelData {
		cachedModel, err = huggingface.FindCachedModel(cfg.HFCachePath, ref)
		if err != nil {
			tui.PrintError(clierrors.NewCLIError(
				"failed to scan cache",
				"Check HF cache path in config",
				err,
			))
			return nil
		}
	}

	if container == nil && (cachedModel == nil || !rmModelData) {
		cached, scanErr := huggingface.FindCachedModel(cfg.HFCachePath, ref)
		if scanErr == nil && cached != nil {
			tui.PrintError(clierrors.NewCLIError(
				fmt.Sprintf("model %s not found as a running container", ref.String()),
				"Use --model-data to also remove cached data, or just the model is not running",
				nil,
			))
		} else {
			tui.PrintError(clierrors.NewCLIError(
				fmt.Sprintf("model %s not found", ref.String()),
				"Use 'vllm-cli ps' to list running models or check the model name",
				nil,
			))
		}
		return nil
	}

	if container != nil {
		fmt.Printf("Stopping %s...\n", ref.String())
		if stopErr := dockerClient.Stop(container.ID); stopErr != nil {
			var cliErr *clierrors.CLIError
			if stderrors.As(stopErr, &cliErr) {
				tui.PrintError(cliErr)
			} else {
				tui.PrintError(clierrors.NewCLIError(
					fmt.Sprintf("failed to stop container for %s", ref.String()),
					"Check Docker logs for details",
					stopErr,
				))
			}
			return nil
		}
	}

	if rmModelData && cachedModel != nil {
		size := huggingface.FormatSize(cachedModel.SizeBytes)
		if tui.IsTTY() {
			tui.PrintWarning(fmt.Sprintf("This will permanently delete %s of cached model data for %s", size, ref.String()))
			fmt.Printf("Remove cached data for %s (%s)? [y/N] ", ref.String(), size)
			reader := bufio.NewReader(os.Stdin)
			answer, readErr := reader.ReadString('\n')
			if readErr != nil {
				tui.PrintError(clierrors.NewCLIError("failed to read input", "", readErr))
				return nil
			}
			answer = strings.TrimSpace(strings.ToLower(answer))
			if answer == "y" || answer == "yes" {
				if removeErr := huggingface.RemoveCachedModel(cfg.HFCachePath, ref); removeErr != nil {
					tui.PrintError(clierrors.NewCLIError(
						"failed to remove cached data",
						"Check permissions on HF cache directory",
						removeErr,
					))
					return nil
				}
				tui.PrintSuccess(fmt.Sprintf("Removed %s and cached data (%s)", ref.String(), size))
			} else {
				if container != nil {
					tui.PrintSuccess(fmt.Sprintf("Removed %s", ref.String()))
				} else {
					fmt.Println("Aborted.")
				}
			}
		} else {
			if container != nil {
				tui.PrintSuccess(fmt.Sprintf("Removed %s", ref.String()))
			}
			tui.PrintWarning(fmt.Sprintf("Skipping cache removal for %s: not running in interactive terminal", ref.String()))
		}
	} else if container != nil {
		tui.PrintSuccess(fmt.Sprintf("Removed %s", ref.String()))
	}

	return nil
}

func rmAllModels(dockerClient *docker.Client, cfg *config.Config) error {
	containers, err := dockerClient.ListManaged()
	if err != nil {
		tui.PrintError(err)
		return nil
	}

	if len(containers) == 0 && !rmModelData {
		fmt.Println("No models running.")
		return nil
	}

	for _, ctr := range containers {
		ref := ctr.ModelRef
		fmt.Printf("Stopping %s...\n", ref.String())
		if stopErr := dockerClient.Stop(ctr.ID); stopErr != nil {
			var cliErr *clierrors.CLIError
			if stderrors.As(stopErr, &cliErr) {
				tui.PrintError(cliErr)
			} else {
				tui.PrintError(clierrors.NewCLIError(
					fmt.Sprintf("failed to stop container for %s", ref.String()),
					"Check Docker logs for details",
					stopErr,
				))
			}
			return nil
		}
		tui.PrintSuccess(fmt.Sprintf("Removed %s", ref.String()))

		if rmModelData {
			cachedModel, scanErr := huggingface.FindCachedModel(cfg.HFCachePath, ref)
			if scanErr != nil {
				tui.PrintWarning(fmt.Sprintf("failed to scan cache for %s: %v", ref.String(), scanErr))
				continue
			}
			if cachedModel == nil {
				continue
			}
			size := huggingface.FormatSize(cachedModel.SizeBytes)
			if tui.IsTTY() {
				fmt.Printf("Remove cached data for %s (%s)? [y/N] ", ref.String(), size)
				reader := bufio.NewReader(os.Stdin)
				answer, readErr := reader.ReadString('\n')
				if readErr != nil {
					tui.PrintError(clierrors.NewCLIError("failed to read input", "", readErr))
					return nil
				}
				answer = strings.TrimSpace(strings.ToLower(answer))
				if answer == "y" || answer == "yes" {
					if removeErr := huggingface.RemoveCachedModel(cfg.HFCachePath, ref); removeErr != nil {
						tui.PrintError(clierrors.NewCLIError(
							fmt.Sprintf("failed to remove cached data for %s", ref.String()),
							"Check permissions on HF cache directory",
							removeErr,
						))
					} else {
						tui.PrintSuccess(fmt.Sprintf("Removed cached data for %s (%s)", ref.String(), size))
					}
				}
			} else {
				tui.PrintWarning(fmt.Sprintf("Skipping cache removal for %s: not running in interactive terminal", ref.String()))
			}
		}
	}

	return nil
}
