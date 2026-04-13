package errors

import (
	stderrors "errors"
	"fmt"

	"github.com/user/vllm-cli/internal/types"
)

// CLIError is a user-facing error with an optional hint.
type CLIError struct {
	Message string
	Hint    string
	Err     error
}

// Error formats the error as "Error: <message>\nHint: <hint>".
func (e *CLIError) Error() string {
	if e.Hint != "" {
		return fmt.Sprintf("Error: %s\nHint: %s", e.Message, e.Hint)
	}
	return fmt.Sprintf("Error: %s", e.Message)
}

// Unwrap returns the wrapped error for errors.Is/As compatibility.
func (e *CLIError) Unwrap() error {
	return e.Err
}

// NewCLIError creates a new CLIError.
func NewCLIError(msg, hint string, err error) *CLIError {
	return &CLIError{Message: msg, Hint: hint, Err: err}
}

// FormatError formats any error for display.
// If err is a CLIError, it is formatted as-is. Otherwise it wraps with a hint.
func FormatError(err error) string {
	if err == nil {
		return ""
	}
	var cliErr *CLIError
	if stderrors.As(err, &cliErr) {
		return cliErr.Error()
	}
	return fmt.Sprintf("Error: %s\nHint: Use --verbose for details", err.Error())
}

// ErrModelNotFound returns a CLIError for a model not found on HuggingFace.
func ErrModelNotFound(ref types.ModelRef) *CLIError {
	return &CLIError{
		Message: fmt.Sprintf("Model %q not found on HuggingFace", ref.String()),
		Hint:    fmt.Sprintf("Check the model ID at https://huggingface.co/%s", ref.String()),
	}
}

// ErrDockerNotRunning returns a CLIError when Docker daemon is not running.
func ErrDockerNotRunning() *CLIError {
	return &CLIError{
		Message: "Docker daemon is not running",
		Hint:    "Start Docker with 'systemctl start docker' or 'sudo dockerd'",
	}
}

// ErrGatedModel returns a CLIError for a gated model requiring authentication.
func ErrGatedModel(ref types.ModelRef) *CLIError {
	return &CLIError{
		Message: fmt.Sprintf("Access denied to model %q (gated model)", ref.String()),
		Hint:    fmt.Sprintf("Set HF_TOKEN environment variable and accept the license at https://huggingface.co/%s", ref.String()),
	}
}

// ErrPortExhausted returns a CLIError when no ports are available.
func ErrPortExhausted() *CLIError {
	return &CLIError{
		Message: "No ports available in range 8000-8099",
		Hint:    "Stop some models with 'vllm-cli stop <model>' to free up ports",
	}
}

// ErrMemoryInsufficient returns a CLIError when model needs more memory than available.
func ErrMemoryInsufficient(neededGB, availableGB float64) *CLIError {
	return &CLIError{
		Message: fmt.Sprintf("Model needs ~%.1fGB but only %.1fGB available", neededGB, availableGB),
		Hint:    "Use --force to override this check, or stop other models to free memory",
	}
}
