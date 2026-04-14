package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/user/vllm-cli/internal/tui/styles"
)

const maxVisibleLogLines = 8

type LogLineMsg string
type HealthyMsg struct{}
type HealthFailedMsg struct{ Err error }

type StartupModel struct {
	spinner   spinner.Model
	modelName string
	startTime time.Time
	logLines  []string
	healthy   bool
	err       error
	quitted   bool
	logCh     <-chan string
	healthCh  <-chan error
}

func NewStartupModel(modelName string, logCh <-chan string, healthCh <-chan error) StartupModel {
	s := spinner.New()
	s.Spinner = spinner.Dot
	return StartupModel{
		spinner:   s,
		modelName: modelName,
		startTime: time.Now(),
		logLines:  []string{},
		logCh:     logCh,
		healthCh:  healthCh,
	}
}

func (m StartupModel) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		m.waitForLogLine(),
		m.waitForHealth(),
	)
}

func (m StartupModel) waitForLogLine() tea.Cmd {
	return func() tea.Msg {
		line, ok := <-m.logCh
		if !ok {
			return nil
		}
		return LogLineMsg(line)
	}
}

func (m StartupModel) waitForHealth() tea.Cmd {
	return func() tea.Msg {
		err, ok := <-m.healthCh
		if !ok {
			return nil
		}
		if err != nil {
			return HealthFailedMsg{Err: err}
		}
		return HealthyMsg{}
	}
}

func (m StartupModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" || msg.String() == "q" {
			m.quitted = true
			return m, tea.Quit
		}

	case LogLineMsg:
		m.logLines = append(m.logLines, string(msg))
		if len(m.logLines) > maxVisibleLogLines {
			m.logLines = m.logLines[len(m.logLines)-maxVisibleLogLines:]
		}
		return m, m.waitForLogLine()

	case HealthyMsg:
		m.healthy = true
		return m, tea.Quit

	case HealthFailedMsg:
		m.err = msg.Err
		return m, tea.Quit

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m StartupModel) View() string {
	elapsed := time.Since(m.startTime).Round(time.Second)

	var b strings.Builder

	header := fmt.Sprintf("%s Loading %s (%s)",
		m.spinner.View(),
		styles.Bold.Render(m.modelName),
		styles.Muted.Render(elapsed.String()),
	)
	b.WriteString(header)
	b.WriteString("\n")

	if len(m.logLines) > 0 {
		dimLine := lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
		b.WriteString("\n")
		for _, line := range m.logLines {
			truncated := line
			if len(truncated) > 120 {
				truncated = truncated[:117] + "..."
			}
			b.WriteString("  ")
			b.WriteString(dimLine.Render(truncated))
			b.WriteString("\n")
		}
	}

	return b.String()
}

func (m StartupModel) Healthy() bool { return m.healthy }
func (m StartupModel) Err() error    { return m.err }
func (m StartupModel) Quitted() bool { return m.quitted }
