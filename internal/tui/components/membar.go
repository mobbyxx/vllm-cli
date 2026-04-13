package components

import (
	"fmt"
	"strings"

	"github.com/user/vllm-cli/internal/tui/styles"
)

const (
	barWidth      = 20
	warnThreshold = 0.80
	errThreshold  = 0.90
)

func RenderMemoryBar(usedGB, totalGB float64, isTTY bool) string {
	if totalGB <= 0 {
		totalGB = 1
	}

	pct := usedGB / totalGB
	if pct > 1.0 {
		pct = 1.0
	}

	filled := int(pct * float64(barWidth))
	empty := barWidth - filled

	bar := strings.Repeat("█", filled) + strings.Repeat("░", empty)
	label := fmt.Sprintf(" %.1f/%.1f GB (%.0f%%)", usedGB, totalGB, pct*100)

	if isTTY {
		var barStyled string
		switch {
		case pct >= errThreshold:
			barStyled = styles.Error.Render("[" + bar + "]")
		case pct >= warnThreshold:
			barStyled = styles.Warning.Render("[" + bar + "]")
		default:
			barStyled = styles.Success.Render("[" + bar + "]")
		}
		return barStyled + label
	}

	return "[" + bar + "]" + label
}
