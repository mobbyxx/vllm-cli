package types

// GPUInfo holds information about a single GPU.
type GPUInfo struct {
	Index         int
	Name          string
	MemoryTotalMB int
	MemoryUsedMB  int
	MemoryFreeMB  int
	Architecture  int  // NVML arch enum value
	IsUnified     bool // true for DGX Spark unified memory
}

// SystemGPUInfo aggregates GPU information across all GPUs.
type SystemGPUInfo struct {
	GPUs          []GPUInfo
	TotalMemoryMB int // sum across GPUs, or system memory for unified
	FreeMemoryMB  int
}
