// Package memory estimates GPU memory requirements for a given model and
// enforces a pre-flight memory guard before container creation. It covers
// weights, KV cache, and CUDA overhead for both discrete and unified GPUs.
package memory

import (
	"github.com/user/vllm-cli/internal/huggingface"
)

// MemoryEstimate holds the breakdown of estimated GPU memory for a model.
type MemoryEstimate struct {
	ParameterCount       int64
	ActiveParameterCount int64
	WeightsGB            float64
	KVCacheGB            float64
	CUDAOverheadGB       float64
	TotalGB              float64

	KVCachePerTokenBytes float64

	GPUTotalGB   float64
	GPUMemUtil   float64
	UsableGB     float64
	KVCacheMaxGB float64
	MaxTokens    int
	MaxSeqLen    int
	EffSeqLen    int

	Dtype       string
	IsQuantized bool
	QuantMethod string
	IsMoE       bool
	IsMLA       bool
}

// bytesPerParam returns bytes per parameter for the given dtype.
func bytesPerParam(dtype string, quantBits int) float64 {
	switch dtype {
	case "float32":
		return 4.0
	case "float16", "bfloat16":
		return 2.0
	case "int8":
		return 1.0
	case "int4":
		return 0.5
	default:
		// Check quantization bits directly
		switch quantBits {
		case 4:
			return 0.5
		case 8:
			return 1.0
		}
		// Default to float16 if unknown
		return 2.0
	}
}

// EstimateOpts controls GPU-aware memory estimation.
type EstimateOpts struct {
	GPUTotalGB float64 // total GPU memory in GB (0 = unknown)
	GPUMemUtil float64 // vLLM --gpu-memory-utilization (default 0.9)
}

// Estimate calculates the memory required to load a model.
// When GPU info is provided via opts, it computes a realistic breakdown
// matching vLLM's actual allocation strategy:
//
//	weights → CUDA overhead → remaining budget for KV cache.
func Estimate(cfg *huggingface.ModelConfig, opts *EstimateOpts) *MemoryEstimate {
	est := &MemoryEstimate{}

	est.Dtype = cfg.TorchDtype
	if est.Dtype == "" {
		est.Dtype = "float16"
	}

	quantBits := 0
	if cfg.QuantizationConfig != nil {
		est.IsQuantized = true
		est.QuantMethod = cfg.QuantizationConfig.QuantMethod
		quantBits = cfg.QuantizationConfig.Bits
		switch quantBits {
		case 4:
			est.Dtype = "int4"
		case 8:
			est.Dtype = "int8"
		}
	}

	est.ParameterCount = cfg.NumParameters
	est.ActiveParameterCount = cfg.NumParameters
	if est.ParameterCount == 0 {
		est.ParameterCount, est.ActiveParameterCount = huggingface.EstimateParameterCount(cfg)
	}
	est.IsMoE = cfg.NRoutedExperts > 0
	est.IsMLA = cfg.KVLoraRank > 0

	bpp := bytesPerParam(est.Dtype, quantBits)
	est.WeightsGB = float64(est.ParameterCount) * bpp / 1e9

	// CUDA context + activation memory (empirical, ~500 MB)
	est.CUDAOverheadGB = 0.5

	numLayers := cfg.NumHiddenLayers

	est.MaxSeqLen = cfg.MaxPositionEmbeddings
	if est.MaxSeqLen == 0 {
		est.MaxSeqLen = 2048
	}

	if cfg.KVLoraRank > 0 {
		// MLA: vLLM caches compressed latent (kv_lora_rank + qk_rope_head_dim) per layer
		elementsPerToken := float64(cfg.KVLoraRank + cfg.QKRopeHeadDim)
		est.KVCachePerTokenBytes = float64(numLayers) * elementsPerToken * 2.0
	} else {
		numKVHeads := cfg.NumKeyValueHeads
		if numKVHeads == 0 {
			numKVHeads = cfg.NumAttentionHeads
		}
		numHeads := cfg.NumAttentionHeads
		headDim := 0
		if numHeads > 0 {
			headDim = cfg.HeadDim
			if headDim == 0 {
				headDim = cfg.HiddenSize / numHeads
			}
		}
		// Standard: 2(K+V) × layers × kv_heads × head_dim × 2 bytes (fp16)
		if numLayers > 0 && numKVHeads > 0 && headDim > 0 {
			est.KVCachePerTokenBytes = 2.0 * float64(numLayers) * float64(numKVHeads) * float64(headDim) * 2.0
		}
	}

	gpuMemUtil := 0.9
	if opts != nil && opts.GPUMemUtil > 0 {
		gpuMemUtil = opts.GPUMemUtil
	}
	est.GPUMemUtil = gpuMemUtil

	if opts != nil && opts.GPUTotalGB > 0 {
		est.GPUTotalGB = opts.GPUTotalGB
		est.UsableGB = est.GPUTotalGB * gpuMemUtil
		est.KVCacheMaxGB = est.UsableGB - est.WeightsGB - est.CUDAOverheadGB
		if est.KVCacheMaxGB < 0 {
			est.KVCacheMaxGB = 0
		}

		if est.KVCachePerTokenBytes > 0 {
			est.MaxTokens = int(est.KVCacheMaxGB * 1e9 / est.KVCachePerTokenBytes)
		}
		if est.MaxTokens > est.MaxSeqLen {
			est.MaxTokens = est.MaxSeqLen
		}
		est.EffSeqLen = est.MaxTokens

		est.KVCacheGB = float64(est.EffSeqLen) * est.KVCachePerTokenBytes / 1e9
		est.TotalGB = est.WeightsGB + est.KVCacheGB + est.CUDAOverheadGB
	} else {
		est.EffSeqLen = est.MaxSeqLen
		est.KVCacheGB = float64(est.EffSeqLen) * est.KVCachePerTokenBytes / 1e9
		est.TotalGB = est.WeightsGB + est.KVCacheGB + est.CUDAOverheadGB
	}

	return est
}
