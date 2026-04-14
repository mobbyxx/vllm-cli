package docker

import (
	"bufio"
	"context"
	"io"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"

	clierrors "github.com/user/vllm-cli/internal/errors"
)

type Client struct {
	cli *client.Client
}

func NewClient() (*Client, error) {
	cli, err := client.NewClientWithOpts(
		client.FromEnv,
		client.WithAPIVersionNegotiation(),
	)
	if err != nil {
		return nil, clierrors.ErrDockerNotRunning()
	}

	ctx := context.Background()
	if _, err := cli.Ping(ctx); err != nil {
		cli.Close()
		return nil, clierrors.ErrDockerNotRunning()
	}

	return &Client{cli: cli}, nil
}

func (c *Client) Close() {
	if c.cli != nil {
		c.cli.Close()
	}
}

// StreamLogs demuxes the Docker multiplexed log stream and sends each line
// to the channel. The 8-byte Docker header per frame is non-trivial to handle
// correctly — stdcopy.StdCopy does this for us.
func (c *Client) StreamLogs(ctx context.Context, containerID string, lines chan<- string) error {
	reader, err := c.cli.ContainerLogs(ctx, containerID, dockercontainer.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
		Timestamps: false,
	})
	if err != nil {
		close(lines)
		return err
	}

	go func() {
		defer close(lines)
		defer reader.Close()

		pr, pw := io.Pipe()
		go func() {
			defer pw.Close()
			_, _ = stdcopy.StdCopy(pw, pw, reader)
		}()

		scanner := bufio.NewScanner(pr)
		scanner.Buffer(make([]byte, 64*1024), 256*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			select {
			case lines <- line:
			case <-ctx.Done():
				return
			}
		}
	}()

	return nil
}
