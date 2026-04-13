package memory

import (
	"fmt"

	"github.com/user/vllm-cli/internal/types"
)

// GuardResult holds the result of a memory guard check.
type GuardResult struct {
	Safe        bool
	NeededGB    float64
	AvailableGB float64
	Message     string // human-readable verdict
}

// Check determines whether the model can be safely loaded given available GPU memory.
// gpuMemCeiling > 0 overrides the detected available memory (from config).
func Check(estimate *MemoryEstimate, gpuInfo *types.SystemGPUInfo, gpuMemCeiling int) *GuardResult {
	result := &GuardResult{
		NeededGB: estimate.TotalGB,
	}

	// Determine available memory
	if gpuMemCeiling > 0 {
		// Manual ceiling from config
		result.AvailableGB = float64(gpuMemCeiling)
	} else if gpuInfo == nil || len(gpuInfo.GPUs) == 0 {
		// No GPU info — assume safe (can't check)
		result.Safe = true
		result.AvailableGB = 0
		result.Message = fmt.Sprintf("GPU info unavailable, proceeding (need ~%.1fGB)", estimate.TotalGB)
		return result
	} else if gpuInfo.GPUs[0].IsUnified {
		// Unified memory (DGX Spark): use system free memory
		result.AvailableGB = float64(gpuInfo.FreeMemoryMB) / 1024.0
	} else {
		// Discrete GPUs with --gpus all: use minimum free across all GPUs
		minFree := gpuInfo.GPUs[0].MemoryFreeMB
		for _, g := range gpuInfo.GPUs[1:] {
			if g.MemoryFreeMB < minFree {
				minFree = g.MemoryFreeMB
			}
		}
		result.AvailableGB = float64(minFree) / 1024.0
	}

	// Compare
	if estimate.TotalGB <= result.AvailableGB {
		result.Safe = true
		result.Message = fmt.Sprintf("Model fits (%.1fGB needed, %.1fGB available)", estimate.TotalGB, result.AvailableGB)
	} else {
		result.Safe = false
		result.Message = fmt.Sprintf("Model may not fit (%.1fGB needed, only %.1fGB available)", estimate.TotalGB, result.AvailableGB)
	}

	// Extra warning for unified memory
	if gpuInfo != nil && len(gpuInfo.GPUs) > 0 && gpuInfo.GPUs[0].IsUnified && !result.Safe {
		result.Message += "\nWarning: Running out of unified memory can crash the entire system"
	}

	return result
}
