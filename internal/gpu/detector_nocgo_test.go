//go:build !cgo

package gpu

import (
	"testing"
)

func TestDetect(t *testing.T) {
	info, err := Detect()
	if err != nil {
		t.Skipf("nvidia-smi not available or no GPU: %v", err)
	}
	if len(info.GPUs) == 0 {
		t.Fatal("expected at least one GPU")
	}
	for _, g := range info.GPUs {
		t.Logf("GPU %d: %s, Total=%dMB, Free=%dMB, Used=%dMB, Unified=%v",
			g.Index, g.Name, g.MemoryTotalMB, g.MemoryFreeMB, g.MemoryUsedMB, g.IsUnified)
	}
	t.Logf("System: TotalMemory=%dMB, FreeMemory=%dMB", info.TotalMemoryMB, info.FreeMemoryMB)

	gpu := info.GPUs[0]
	if gpu.Name == "" {
		t.Error("expected non-empty GPU name")
	}
	if gpu.MemoryTotalMB == 0 {
		t.Error("expected non-zero total memory")
	}
}
