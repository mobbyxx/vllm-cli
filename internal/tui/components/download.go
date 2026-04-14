package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/user/vllm-cli/internal/huggingface"
	"github.com/user/vllm-cli/internal/tui/styles"
)

type downloadProgressMsg huggingface.DownloadProgress
type downloadDoneMsg struct{ err error }

type DownloadModel struct {
	spinner    spinner.Model
	progress   progress.Model
	modelName  string
	startTime  time.Time
	downloaded int64
	total      int64
	done       bool
	err        error
	progressCh <-chan huggingface.DownloadProgress
	doneCh     <-chan error
}

func NewDownloadModel(modelName string, totalSize int64, progressCh <-chan huggingface.DownloadProgress, doneCh <-chan error) DownloadModel {
	s := spinner.New()
	s.Spinner = spinner.Dot

	p := progress.New(
		progress.WithDefaultGradient(),
		progress.WithWidth(40),
		progress.WithoutPercentage(),
	)

	return DownloadModel{
		spinner:    s,
		progress:   p,
		modelName:  modelName,
		startTime:  time.Now(),
		total:      totalSize,
		progressCh: progressCh,
		doneCh:     doneCh,
	}
}

func (m DownloadModel) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		m.waitForProgress(),
		m.waitForDone(),
	)
}

func (m DownloadModel) waitForProgress() tea.Cmd {
	return func() tea.Msg {
		p, ok := <-m.progressCh
		if !ok {
			return nil
		}
		return downloadProgressMsg(p)
	}
}

func (m DownloadModel) waitForDone() tea.Cmd {
	return func() tea.Msg {
		err, ok := <-m.doneCh
		if !ok {
			return downloadDoneMsg{}
		}
		return downloadDoneMsg{err: err}
	}
}

func (m DownloadModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" || msg.String() == "q" {
			return m, tea.Quit
		}

	case downloadProgressMsg:
		m.downloaded = msg.Downloaded
		if msg.Total > 0 {
			m.total = msg.Total
		}
		return m, m.waitForProgress()

	case downloadDoneMsg:
		m.done = true
		m.err = msg.err
		return m, tea.Quit

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case progress.FrameMsg:
		mdl, cmd := m.progress.Update(msg)
		m.progress = mdl.(progress.Model)
		return m, cmd
	}

	return m, nil
}

func (m DownloadModel) View() string {
	elapsed := time.Since(m.startTime).Round(time.Second)

	var b strings.Builder

	header := fmt.Sprintf("%s Downloading %s (%s)",
		m.spinner.View(),
		styles.Bold.Render(m.modelName),
		styles.Muted.Render(elapsed.String()),
	)
	b.WriteString(header)
	b.WriteString("\n\n")

	if m.total > 0 {
		fraction := float64(m.downloaded) / float64(m.total)
		if fraction > 1.0 {
			fraction = 1.0
		}
		pct := int(fraction * 100)

		b.WriteString("  ")
		b.WriteString(m.progress.ViewAs(fraction))
		b.WriteString(fmt.Sprintf(" %d%%", pct))
		b.WriteString("\n")

		b.WriteString(fmt.Sprintf("  %s / %s",
			styles.Bold.Render(huggingface.FormatSize(m.downloaded)),
			styles.Muted.Render(huggingface.FormatSize(m.total)),
		))

		if elapsed.Seconds() > 2 && m.downloaded > 0 {
			speed := float64(m.downloaded) / elapsed.Seconds()
			remaining := float64(m.total-m.downloaded) / speed
			if remaining > 0 {
				b.WriteString(fmt.Sprintf("  %s remaining",
					styles.Muted.Render(time.Duration(remaining*float64(time.Second)).Round(time.Second).String()),
				))
			}
		}
		b.WriteString("\n")
	} else {
		b.WriteString(fmt.Sprintf("  %s downloaded",
			styles.Bold.Render(huggingface.FormatSize(m.downloaded)),
		))
		b.WriteString("\n")
	}

	return b.String()
}

func (m DownloadModel) Done() bool { return m.done }
func (m DownloadModel) Err() error { return m.err }
