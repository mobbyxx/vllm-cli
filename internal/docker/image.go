package docker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/docker/docker/api/types/image"
)

// PullProgress represents a Docker image pull progress event.
type PullProgress struct {
	Status   string `json:"status"`
	Progress string `json:"progress"`
	ID       string `json:"id"`
	Error    string `json:"error"`
}

// EnsureImage checks if a Docker image exists locally, and pulls it if not.
// Returns a channel of progress events during pull, or nil channel if already present.
func (c *Client) EnsureImage(imageName string) (<-chan PullProgress, error) {
	ctx := context.Background()

	// Check if image exists locally
	_, _, err := c.cli.ImageInspectWithRaw(ctx, imageName)
	if err == nil {
		// Image already present
		return nil, nil
	}

	// Pull the image
	reader, err := c.cli.ImagePull(ctx, imageName, image.PullOptions{})
	if err != nil {
		return nil, fmt.Errorf("pulling image %s: %w", imageName, err)
	}

	// Stream progress events through a channel
	ch := make(chan PullProgress, 10)
	go func() {
		defer close(ch)
		defer reader.Close()

		decoder := json.NewDecoder(reader)
		for {
			var event PullProgress
			if err := decoder.Decode(&event); err != nil {
				if err != io.EOF {
					ch <- PullProgress{Error: err.Error()}
				}
				return
			}
			ch <- event
		}
	}()

	return ch, nil
}
