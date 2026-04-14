package huggingface

import "encoding/json"

// ModelSibling represents a single file in the model repository.
type ModelSibling struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size"`
}

// ModelInfo holds metadata from the HuggingFace API for a model.
type ModelInfo struct {
	ID          string         `json:"id"`
	PipelineTag string         `json:"pipeline_tag"`
	LibraryName string         `json:"library_name"`
	Gated       interface{}    `json:"gated"` // can be bool or string
	Downloads   int            `json:"downloads"`
	Siblings    []ModelSibling `json:"siblings"`
}

// TotalSize returns the sum of all file sizes in the model repository.
func (m *ModelInfo) TotalSize() int64 {
	var total int64
	for _, s := range m.Siblings {
		total += s.Size
	}
	return total
}

// IsGated returns true if the model requires authentication.
func (m *ModelInfo) IsGated() bool {
	if m.Gated == nil {
		return false
	}
	switch v := m.Gated.(type) {
	case bool:
		return v
	case string:
		return v != "" && v != "false"
	}
	return false
}

// QuantConfig holds quantization configuration from config.json.
type QuantConfig struct {
	QuantMethod string `json:"quant_method"` // "awq", "gptq"
	Bits        int    `json:"bits"`         // 4, 8
}

// ModelConfig holds the architecture parameters from config.json.
// It handles standard HuggingFace naming, legacy GPT-2 naming (n_embd, n_layer, n_head),
// and multimodal models where parameters live in text_config.
type ModelConfig struct {
	NumParameters         int64        `json:"num_parameters"`
	HiddenSize            int          `json:"hidden_size"`
	NumHiddenLayers       int          `json:"num_hidden_layers"`
	NumAttentionHeads     int          `json:"num_attention_heads"`
	NumKeyValueHeads      int          `json:"num_key_value_heads"`
	HeadDim               int          `json:"head_dim"`
	IntermediateSize      int          `json:"intermediate_size"`
	VocabSize             int          `json:"vocab_size"`
	MaxPositionEmbeddings int          `json:"max_position_embeddings"`
	TorchDtype            string       `json:"torch_dtype"`
	QuantizationConfig    *QuantConfig `json:"quantization_config"`

	// MoE (Mixture of Experts) — DeepSeek-V2/V3, GLM-5.1, Mixtral
	NRoutedExperts     int `json:"n_routed_experts"`
	NSharedExperts     int `json:"n_shared_experts"`
	MoeIntermSize      int `json:"moe_intermediate_size"`
	NumExpertsPerTok   int `json:"num_experts_per_tok"`
	MoeLayerFreq       int `json:"moe_layer_freq"`
	FirstKDenseReplace int `json:"first_k_dense_replace"`

	// MLA (Multi-Latent Attention) — DeepSeek-V2/V3, GLM-5.1
	KVLoraRank    int `json:"kv_lora_rank"`
	QLoraRank     int `json:"q_lora_rank"`
	QKNopeHeadDim int `json:"qk_nope_head_dim"`
	QKRopeHeadDim int `json:"qk_rope_head_dim"`
	VHeadDim      int `json:"v_head_dim"`
}

// UnmarshalJSON implements custom JSON unmarshalling to handle standard HuggingFace naming,
// legacy GPT-2 field names, multimodal text_config fallback, and dtype alias.
func (c *ModelConfig) UnmarshalJSON(data []byte) error {
	type Alias ModelConfig
	aux := &struct {
		*Alias
		NEmbd      int          `json:"n_embd"`
		NLayer     int          `json:"n_layer"`
		NHead      int          `json:"n_head"`
		NCtx       int          `json:"n_ctx"`
		Dtype      string       `json:"dtype"`
		TextConfig *ModelConfig `json:"text_config"`
	}{
		Alias: (*Alias)(c),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	// GPT-2 aliases
	if c.HiddenSize == 0 && aux.NEmbd != 0 {
		c.HiddenSize = aux.NEmbd
	}
	if c.NumHiddenLayers == 0 && aux.NLayer != 0 {
		c.NumHiddenLayers = aux.NLayer
	}
	if c.NumAttentionHeads == 0 && aux.NHead != 0 {
		c.NumAttentionHeads = aux.NHead
	}
	if c.MaxPositionEmbeddings == 0 && aux.NCtx != 0 {
		c.MaxPositionEmbeddings = aux.NCtx
	}

	// "dtype" alias for "torch_dtype" (used by Gemma 4 and others)
	if c.TorchDtype == "" && aux.Dtype != "" {
		c.TorchDtype = aux.Dtype
	}

	// Multimodal fallback: pull text architecture from text_config
	if aux.TextConfig != nil && c.HiddenSize == 0 {
		tc := aux.TextConfig
		c.HiddenSize = tc.HiddenSize
		c.NumHiddenLayers = tc.NumHiddenLayers
		c.NumAttentionHeads = tc.NumAttentionHeads
		c.NumKeyValueHeads = tc.NumKeyValueHeads
		c.HeadDim = tc.HeadDim
		c.IntermediateSize = tc.IntermediateSize
		c.MaxPositionEmbeddings = tc.MaxPositionEmbeddings
		if tc.VocabSize != 0 {
			c.VocabSize = tc.VocabSize
		}
		if tc.TorchDtype != "" {
			c.TorchDtype = tc.TorchDtype
		}
		c.NRoutedExperts = tc.NRoutedExperts
		c.NSharedExperts = tc.NSharedExperts
		c.MoeIntermSize = tc.MoeIntermSize
		c.NumExpertsPerTok = tc.NumExpertsPerTok
		c.MoeLayerFreq = tc.MoeLayerFreq
		c.FirstKDenseReplace = tc.FirstKDenseReplace
		c.KVLoraRank = tc.KVLoraRank
		c.QLoraRank = tc.QLoraRank
		c.QKNopeHeadDim = tc.QKNopeHeadDim
		c.QKRopeHeadDim = tc.QKRopeHeadDim
		c.VHeadDim = tc.VHeadDim
	}

	return nil
}
