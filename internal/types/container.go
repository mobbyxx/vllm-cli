package types

import "time"

// ContainerInfo holds information about a running vllm-cli managed container.
type ContainerInfo struct {
	ID        string
	ModelRef  ModelRef
	Port      int
	Status    string // "starting", "running", "stopping"
	CreatedAt time.Time
}
