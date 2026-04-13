package docker

import (
	"fmt"
	"net"

	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/types"
)

// PortManager manages port assignment for vllm-cli containers.
type PortManager struct {
	RangeStart int // default 8000
	RangeEnd   int // default 8099
}

// NewPortManager creates a PortManager with the given port range.
func NewPortManager(rangeStart, rangeEnd int) *PortManager {
	return &PortManager{
		RangeStart: rangeStart,
		RangeEnd:   rangeEnd,
	}
}

// FindAvailablePort finds the first available port in the manager's range.
// It checks Docker labels first to see which ports are in use, then verifies
// the candidate port is actually free on the host.
func (pm *PortManager) FindAvailablePort(docker *Client) (int, error) {
	// Get ports already in use by managed containers
	usedPorts := make(map[int]bool)
	containers, err := docker.ListManaged()
	if err != nil {
		return 0, fmt.Errorf("listing managed containers: %w", err)
	}
	for _, ctr := range containers {
		if ctr.Port > 0 {
			usedPorts[ctr.Port] = true
		}
	}

	// Find first free port
	for port := pm.RangeStart; port <= pm.RangeEnd; port++ {
		if usedPorts[port] {
			continue
		}
		if IsPortFree(port) {
			return port, nil
		}
	}

	return 0, clierrors.ErrPortExhausted()
}

// IsPortFree checks if a TCP port is free on localhost.
func IsPortFree(port int) bool {
	addr := fmt.Sprintf(":%d", port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return false
	}
	ln.Close()
	return true
}

// GetUsedPorts returns a map of port → ModelRef for all managed containers.
func GetUsedPorts(docker *Client) (map[int]types.ModelRef, error) {
	containers, err := docker.ListManaged()
	if err != nil {
		return nil, fmt.Errorf("listing managed containers: %w", err)
	}

	portMap := make(map[int]types.ModelRef, len(containers))
	for _, ctr := range containers {
		if ctr.Port > 0 {
			portMap[ctr.Port] = ctr.ModelRef
		}
	}
	return portMap, nil
}
