//go:build cgo

// Package gpu provides GPU detection and memory reporting via NVML.
// This file is compiled when CGO is enabled (the default). The !cgo variant
// in detector_nocgo.go is compiled for CGO_ENABLED=0 builds.
package gpu

import (
	"fmt"

	"github.com/NVIDIA/go-nvml/pkg/nvml"

	"github.com/user/vllm-cli/internal/types"
)

// BlackwellArchitecture is the NVML architecture enum for DGX Spark (unified memory).
const BlackwellArchitecture = 10

// ErrNoGPU is returned when no GPU is available (NVML not found or no devices).
var ErrNoGPU = fmt.Errorf("no GPU detected")

// Detect queries GPU information via NVML.
// Returns SystemGPUInfo on success, or ErrNoGPU if no GPU is available.
// Gracefully handles NVML unavailability (compiles and runs on non-GPU systems).
func Detect() (*types.SystemGPUInfo, error) {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		return nil, ErrNoGPU
	}
	defer nvml.Shutdown()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("getting GPU count: %s", nvml.ErrorString(ret))
	}
	if count == 0 {
		return nil, ErrNoGPU
	}

	sysInfo := &types.SystemGPUInfo{
		GPUs: make([]types.GPUInfo, 0, count),
	}

	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			continue
		}

		gpuInfo := types.GPUInfo{Index: i}

		// Get GPU name
		name, ret := nvml.DeviceGetName(device)
		if ret == nvml.SUCCESS {
			gpuInfo.Name = name
		}

		// Get architecture to detect unified memory (DGX Spark = Blackwell = 10)
		arch, ret := nvml.DeviceGetArchitecture(device)
		if ret == nvml.SUCCESS {
			gpuInfo.Architecture = int(arch)
		}

		// Try to get memory info
		memInfo, ret := nvml.DeviceGetMemoryInfo(device)
		if ret == nvml.SUCCESS {
			// Discrete GPU: use NVML memory values
			gpuInfo.MemoryTotalMB = int(memInfo.Total / 1024 / 1024)
			gpuInfo.MemoryUsedMB = int(memInfo.Used / 1024 / 1024)
			gpuInfo.MemoryFreeMB = int(memInfo.Free / 1024 / 1024)
			gpuInfo.IsUnified = false
		} else if ret == nvml.ERROR_NOT_SUPPORTED {
			// Unified memory (DGX Spark): DeviceGetMemoryInfo not supported
			// Fall back to /proc/meminfo for system memory
			gpuInfo.IsUnified = true
			totalKB, availableKB, procErr := ReadProcMeminfo()
			if procErr == nil {
				gpuInfo.MemoryTotalMB = int(totalKB / 1024)
				gpuInfo.MemoryFreeMB = int(availableKB / 1024)
				gpuInfo.MemoryUsedMB = gpuInfo.MemoryTotalMB - gpuInfo.MemoryFreeMB
			}
		}
		// Other errors: record GPU with zero memory (still add to list)

		sysInfo.GPUs = append(sysInfo.GPUs, gpuInfo)
		sysInfo.TotalMemoryMB += gpuInfo.MemoryTotalMB
		sysInfo.FreeMemoryMB += gpuInfo.MemoryFreeMB
	}

	if len(sysInfo.GPUs) == 0 {
		return nil, ErrNoGPU
	}

	return sysInfo, nil
}
