package docker

import (
	"context"
	"fmt"
	"strconv"
	"time"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/go-connections/nat"

	"github.com/user/vllm-cli/internal/types"
)

const (
	labelManagedBy      = "managed-by"
	labelManagedByValue = "vllm-cli"
	labelModel          = "vllm-cli.model"
	labelPort           = "vllm-cli.port"
)

// ContainerOpts holds options for creating a vLLM container.
type ContainerOpts struct {
	ModelRef    types.ModelRef
	Port        int
	HFCachePath string
	HFToken     string
	DockerImage string
	ExtraArgs   []string
	GPUMemUtil  float64
	Force       bool
}

// containerNameFor returns the Docker container name for a model.
func containerNameFor(ref types.ModelRef) string {
	return "vllm-" + ref.Slug()
}

// CreateAndStart creates and starts a vLLM container with the given options.
func (c *Client) CreateAndStart(opts ContainerOpts) (*types.ContainerInfo, error) {
	ctx := context.Background()

	containerName := containerNameFor(opts.ModelRef)

	// Build command
	cmd := []string{
		"--model", opts.ModelRef.String(),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", opts.GPUMemUtil),
	}
	cmd = append(cmd, opts.ExtraArgs...)

	// Environment variables
	env := []string{}
	if opts.HFToken != "" {
		env = append(env, "HF_TOKEN="+opts.HFToken)
	}

	// Labels
	labels := map[string]string{
		labelManagedBy: labelManagedByValue,
		labelModel:     opts.ModelRef.String(),
		labelPort:      strconv.Itoa(opts.Port),
	}

	// Port binding: host port → container port 8000
	portSet := nat.PortSet{"8000/tcp": struct{}{}}
	portBinding := nat.PortMap{
		"8000/tcp": []nat.PortBinding{
			{HostIP: "0.0.0.0", HostPort: strconv.Itoa(opts.Port)},
		},
	}

	// Container config
	containerConfig := &dockercontainer.Config{
		Image:        opts.DockerImage,
		Cmd:          cmd,
		Env:          env,
		Labels:       labels,
		ExposedPorts: portSet,
	}

	// 16GB ShmSize
	const shmSize = int64(16) * 1024 * 1024 * 1024

	// Host config with GPU, IPC host, ShmSize, port bindings, volume
	hostConfig := &dockercontainer.HostConfig{
		Runtime: "nvidia",
		Resources: dockercontainer.Resources{
			DeviceRequests: []dockercontainer.DeviceRequest{
				{
					Count:        -1,
					Capabilities: [][]string{{"gpu"}},
				},
			},
		},
		IpcMode:      "host",
		ShmSize:      shmSize,
		PortBindings: portBinding,
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: opts.HFCachePath,
				Target: "/root/.cache/huggingface",
			},
		},
	}

	resp, err := c.cli.ContainerCreate(ctx, containerConfig, hostConfig, &network.NetworkingConfig{}, nil, containerName)
	if err != nil {
		return nil, fmt.Errorf("creating container: %w", err)
	}

	if err := c.cli.ContainerStart(ctx, resp.ID, dockercontainer.StartOptions{}); err != nil {
		// Clean up the created container if start fails
		_ = c.cli.ContainerRemove(ctx, resp.ID, dockercontainer.RemoveOptions{Force: true})
		return nil, fmt.Errorf("starting container: %w", err)
	}

	return &types.ContainerInfo{
		ID:        resp.ID,
		ModelRef:  opts.ModelRef,
		Port:      opts.Port,
		Status:    "starting",
		CreatedAt: time.Now(),
	}, nil
}

// Stop stops a container gracefully (30s timeout) then removes it.
func (c *Client) Stop(containerID string) error {
	ctx := context.Background()
	timeout := 30
	if err := c.cli.ContainerStop(ctx, containerID, dockercontainer.StopOptions{Timeout: &timeout}); err != nil {
		return fmt.Errorf("stopping container: %w", err)
	}
	if err := c.cli.ContainerRemove(ctx, containerID, dockercontainer.RemoveOptions{}); err != nil {
		return fmt.Errorf("removing container after stop: %w", err)
	}
	return nil
}

// Remove force-removes a container.
func (c *Client) Remove(containerID string) error {
	ctx := context.Background()
	return c.cli.ContainerRemove(ctx, containerID, dockercontainer.RemoveOptions{Force: true})
}

// ListManaged returns all vllm-cli managed containers.
func (c *Client) ListManaged() ([]types.ContainerInfo, error) {
	ctx := context.Background()

	f := filters.NewArgs()
	f.Add("label", labelManagedBy+"="+labelManagedByValue)

	containers, err := c.cli.ContainerList(ctx, dockercontainer.ListOptions{
		All:     true,
		Filters: f,
	})
	if err != nil {
		return nil, fmt.Errorf("listing containers: %w", err)
	}

	result := make([]types.ContainerInfo, 0, len(containers))
	for _, ctr := range containers {
		modelStr := ctr.Labels[labelModel]
		portStr := ctr.Labels[labelPort]

		ref, err := types.ParseModelRef(modelStr)
		if err != nil {
			continue // skip malformed containers
		}

		port, _ := strconv.Atoi(portStr)

		result = append(result, types.ContainerInfo{
			ID:       ctr.ID,
			ModelRef: ref,
			Port:     port,
			Status:   ctr.State,
		})
	}
	return result, nil
}

// GetContainer finds a vllm-cli container by model slug.
func (c *Client) GetContainer(modelSlug string) (*types.ContainerInfo, error) {
	containers, err := c.ListManaged()
	if err != nil {
		return nil, err
	}
	for _, ctr := range containers {
		if ctr.ModelRef.Slug() == modelSlug {
			return &ctr, nil
		}
	}
	return nil, nil
}
