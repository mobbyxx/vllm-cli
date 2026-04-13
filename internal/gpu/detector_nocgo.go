//go:build !cgo

package gpu

import (
	"fmt"

	"github.com/user/vllm-cli/internal/types"
)

const BlackwellArchitecture = 10

var ErrNoGPU = fmt.Errorf("no GPU detected")

func Detect() (*types.SystemGPUInfo, error) {
	return nil, ErrNoGPU
}
