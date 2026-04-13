package memory

import (
	"github.com/user/vllm-cli/internal/huggingface"
)

// MemoryEstimate holds the breakdown of estimated GPU memory for a model.
type MemoryEstimate struct {
	ParameterCount int64   // total params
	WeightsGB      float64 // params × bytes_per_param / 1e9
	KVCacheGB      float64 // estimated KV cache
	OverheadGB     float64 // ~20% + 0.5GB CUDA context
	TotalGB        float64 // sum of all above
	Dtype          string  // "float16", "bfloat16", "int4", "int8", "float32"
	IsQuantized    bool
	QuantMethod    string // "awq", "gptq", ""
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

// Estimate calculates the memory required to load a model.
func Estimate(cfg *huggingface.ModelConfig) *MemoryEstimate {
	est := &MemoryEstimate{}

	// Determine dtype and quantization
	est.Dtype = cfg.TorchDtype
	if est.Dtype == "" {
		est.Dtype = "float16" // default assumption
	}

	quantBits := 0
	if cfg.QuantizationConfig != nil {
		est.IsQuantized = true
		est.QuantMethod = cfg.QuantizationConfig.QuantMethod
		quantBits = cfg.QuantizationConfig.Bits
		// Map quantization to dtype string
		switch quantBits {
		case 4:
			est.Dtype = "int4"
		case 8:
			est.Dtype = "int8"
		}
	}

	// Parameter count
	est.ParameterCount = cfg.NumParameters
	if est.ParameterCount == 0 {
		est.ParameterCount = huggingface.EstimateParameterCount(cfg)
	}

	// Weights memory
	bpp := bytesPerParam(est.Dtype, quantBits)
	est.WeightsGB = float64(est.ParameterCount) * bpp / 1e9

	// KV Cache estimate
	// Formula: 2 × num_layers × num_kv_heads × head_dim × max_seq_len × bytes_per_element / 1e9
	// head_dim = hidden_size / num_attention_heads
	numLayers := cfg.NumHiddenLayers
	numKVHeads := cfg.NumKeyValueHeads
	if numKVHeads == 0 {
		numKVHeads = cfg.NumAttentionHeads // fallback to MHA
	}
	hiddenSize := cfg.HiddenSize
	numHeads := cfg.NumAttentionHeads
	maxSeqLen := cfg.MaxPositionEmbeddings
	if maxSeqLen == 0 {
		maxSeqLen = 2048 // conservative default
	}
	if maxSeqLen > 8192 {
		maxSeqLen = 8192 // cap at 8K for estimation purposes
	}

	kvCacheBytes := float64(0)
	if numLayers > 0 && numKVHeads > 0 && hiddenSize > 0 && numHeads > 0 {
		headDim := hiddenSize / numHeads
		// 2 = K + V, 2 bytes per element (fp16)
		kvCacheBytes = float64(2) * float64(numLayers) * float64(numKVHeads) * float64(headDim) * float64(maxSeqLen) * 2.0
	}
	est.KVCacheGB = kvCacheBytes / 1e9

	// Overhead: 20% of (weights + kv_cache) + 0.5GB CUDA context
	est.OverheadGB = (est.WeightsGB+est.KVCacheGB)*0.20 + 0.5

	// Total
	est.TotalGB = est.WeightsGB + est.KVCacheGB + est.OverheadGB

	return est
}
