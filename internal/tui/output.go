package tui

import (
	"fmt"
	"os"

	"golang.org/x/term"

	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/tui/styles"
)

func IsTTY() bool {
	return term.IsTerminal(int(os.Stdout.Fd()))
}

func PrintStyled(s string) {
	if IsTTY() {
		fmt.Println(styles.Bold.Render(s))
	} else {
		fmt.Println(s)
	}
}

func PrintSuccess(msg string) {
	if IsTTY() {
		fmt.Println(styles.Success.Render("✓ " + msg))
	} else {
		fmt.Println("✓ " + msg)
	}
}

func PrintWarning(msg string) {
	if IsTTY() {
		fmt.Println(styles.Warning.Render("⚠ " + msg))
	} else {
		fmt.Println("⚠ " + msg)
	}
}

func PrintError(err error) {
	if err == nil {
		return
	}
	msg := clierrors.FormatError(err)
	if IsTTY() {
		fmt.Fprintln(os.Stderr, styles.Error.Render(msg))
	} else {
		fmt.Fprintln(os.Stderr, msg)
	}
}
