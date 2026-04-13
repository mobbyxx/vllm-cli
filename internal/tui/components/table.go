package components

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/user/vllm-cli/internal/tui/styles"
)

func RenderTable(headers []string, rows [][]string, isTTY bool) string {
	if len(headers) == 0 {
		return ""
	}

	colWidths := make([]int, len(headers))
	for i, h := range headers {
		colWidths[i] = len(h)
	}
	for _, row := range rows {
		for i, cell := range row {
			if i < len(colWidths) && len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	var sb strings.Builder

	if isTTY {
		headerCells := make([]string, len(headers))
		for i, h := range headers {
			padded := padRight(h, colWidths[i])
			headerCells[i] = styles.Header.Render(padded)
		}
		sb.WriteString(strings.Join(headerCells, "  "))
		sb.WriteString("\n")

		for _, row := range rows {
			cells := make([]string, len(headers))
			for i := range headers {
				cell := ""
				if i < len(row) {
					cell = row[i]
				}
				cells[i] = lipgloss.NewStyle().Render(padRight(cell, colWidths[i]))
			}
			sb.WriteString(strings.Join(cells, "  "))
			sb.WriteString("\n")
		}
	} else {
		for _, row := range rows {
			cells := make([]string, len(headers))
			for i := range headers {
				cell := ""
				if i < len(row) {
					cell = row[i]
				}
				cells[i] = padRight(cell, colWidths[i])
			}
			sb.WriteString(strings.Join(cells, "  "))
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func padRight(s string, width int) string {
	if len(s) >= width {
		return s
	}
	return s + strings.Repeat(" ", width-len(s))
}
