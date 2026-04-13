//go:build !cgo

// Package gpu provides GPU detection when CGO is disabled.
// This file shells out to nvidia-smi instead of using NVML directly,
// so CGO_ENABLED=0 release binaries can still detect GPUs.
package gpu

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/user/vllm-cli/internal/types"
)

// BlackwellArchitecture is the NVML architecture enum for DGX Spark (unified memory).
const BlackwellArchitecture = 10

// ErrNoGPU is returned when no GPU is available.
var ErrNoGPU = fmt.Errorf("no GPU detected")

// Detect queries GPU information by shelling out to nvidia-smi.
// Returns SystemGPUInfo on success, or ErrNoGPU if nvidia-smi is not available or reports no GPUs.
func Detect() (*types.SystemGPUInfo, error) {
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.free,memory.used",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, ErrNoGPU
	}

	lines := strings.TrimSpace(string(out))
	if lines == "" {
		return nil, ErrNoGPU
	}

	sysInfo := &types.SystemGPUInfo{}

	for _, line := range strings.Split(lines, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		fields := strings.SplitN(line, ", ", 5)
		if len(fields) < 5 {
			continue
		}

		idx, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
		name := strings.TrimSpace(fields[1])
		memTotalStr := strings.TrimSpace(fields[2])
		memFreeStr := strings.TrimSpace(fields[3])
		memUsedStr := strings.TrimSpace(fields[4])

		gpuInfo := types.GPUInfo{
			Index: idx,
			Name:  name,
		}

		memTotal, totalErr := strconv.Atoi(memTotalStr)
		memFree, freeErr := strconv.Atoi(memFreeStr)
		memUsed, usedErr := strconv.Atoi(memUsedStr)

		if totalErr != nil || freeErr != nil || usedErr != nil {
			// nvidia-smi returns "[Not Supported]" for unified memory (e.g. DGX Spark).
			// Fall back to /proc/meminfo.
			gpuInfo.IsUnified = true
			totalKB, availableKB, procErr := ReadProcMeminfo()
			if procErr == nil {
				gpuInfo.MemoryTotalMB = int(totalKB / 1024)
				gpuInfo.MemoryFreeMB = int(availableKB / 1024)
				gpuInfo.MemoryUsedMB = gpuInfo.MemoryTotalMB - gpuInfo.MemoryFreeMB
			}
		} else {
			gpuInfo.MemoryTotalMB = memTotal
			gpuInfo.MemoryFreeMB = memFree
			gpuInfo.MemoryUsedMB = memUsed
		}

		sysInfo.GPUs = append(sysInfo.GPUs, gpuInfo)
		sysInfo.TotalMemoryMB += gpuInfo.MemoryTotalMB
		sysInfo.FreeMemoryMB += gpuInfo.MemoryFreeMB
	}

	if len(sysInfo.GPUs) == 0 {
		return nil, ErrNoGPU
	}

	return sysInfo, nil
}
