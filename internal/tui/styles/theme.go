// Package styles defines the lipgloss style palette used across all vllm-cli
// terminal output: Title, Success, Warning, Error, Muted, Bold, and Header.
package styles

import "github.com/charmbracelet/lipgloss"

var (
	// Title renders text in bold blue.
	Title = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))

	// Success renders text in green.
	Success = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))

	// Warning renders text in yellow.
	Warning = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))

	// Error renders text in red.
	Error = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))

	// Muted renders text in gray.
	Muted = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))

	// Bold renders text in bold.
	Bold = lipgloss.NewStyle().Bold(true)

	// Header renders text in bold and underline.
	Header = lipgloss.NewStyle().Bold(true).Underline(true)
)
