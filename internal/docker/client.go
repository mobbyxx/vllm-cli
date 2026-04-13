// Package docker provides a high-level wrapper around the Docker SDK for
// managing vLLM containers. It handles container creation, lifecycle (start,
// stop, remove), image pulls, port management, and health polling.
package docker

import (
	"context"

	"github.com/docker/docker/client"

	clierrors "github.com/user/vllm-cli/internal/errors"
)

// Client wraps the Docker SDK client.
type Client struct {
	cli *client.Client
}

// NewClient creates a Docker client and verifies connectivity by pinging the daemon.
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

// Close releases Docker client resources.
func (c *Client) Close() {
	if c.cli != nil {
		c.cli.Close()
	}
}
