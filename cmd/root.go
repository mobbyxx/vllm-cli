// Package cmd implements the vllm-cli command-line interface using Cobra.
// Each subcommand (run, pull, stop, list, ps, rm, show) lives in its own file
// and self-registers with rootCmd via an init() function. The Execute() function
// is the single entry point called from main.go.
package cmd

import (
	stderrors "errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	clierrors "github.com/user/vllm-cli/internal/errors"
)

const Version = "v0.1.0-dev"

var verbose bool
var configPath string
var noColor bool

var rootCmd = &cobra.Command{
	Use:           "vllm-cli",
	Short:         "Manage vLLM containers with Ollama-like simplicity",
	Long:          "vllm-cli provides an intuitive command-line interface for managing vLLM Docker containers.\n\nModel Commands:\n  pull\t\tDownload a model from HuggingFace\n  run\t\tStart a vLLM container\n  stop\t\tStop running models\n  list/ls\tList downloaded models\n  ps\t\tShow running containers\n  rm\t\tRemove models and containers\n  show\t\tDisplay model information",
	Version:       Version,
	SilenceUsage:  true,
	SilenceErrors: true,
}

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

func init() {
	rootCmd.PersistentFlags().BoolVar(&verbose, "verbose", false, "Enable verbose output")
	rootCmd.PersistentFlags().StringVar(&configPath, "config", "", "config file path (default: ~/.config/vllm-cli/config.yaml)")
	rootCmd.PersistentFlags().BoolVar(&noColor, "no-color", false, "Disable colored output")
	rootCmd.InitDefaultVersionFlag()
}
