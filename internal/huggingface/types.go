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
// It handles both standard HuggingFace naming (hidden_size, num_hidden_layers, num_attention_heads)
// and legacy GPT-2 naming (n_embd, n_layer, n_head).
type ModelConfig struct {
	NumParameters         int64        `json:"num_parameters"`
	HiddenSize            int          `json:"hidden_size"`
	NumHiddenLayers       int          `json:"num_hidden_layers"`
	NumAttentionHeads     int          `json:"num_attention_heads"`
	NumKeyValueHeads      int          `json:"num_key_value_heads"`
	IntermediateSize      int          `json:"intermediate_size"`
	VocabSize             int          `json:"vocab_size"`
	MaxPositionEmbeddings int          `json:"max_position_embeddings"`
	TorchDtype            string       `json:"torch_dtype"`
	QuantizationConfig    *QuantConfig `json:"quantization_config"`
}

// UnmarshalJSON implements custom JSON unmarshalling to handle both standard
// HuggingFace naming conventions and legacy GPT-2 style field names.
func (c *ModelConfig) UnmarshalJSON(data []byte) error {
	// Use an alias to avoid infinite recursion
	type Alias ModelConfig
	aux := &struct {
		*Alias
		// GPT-2 style aliases
		NEmbd  int `json:"n_embd"`
		NLayer int `json:"n_layer"`
		NHead  int `json:"n_head"`
		NCtx   int `json:"n_ctx"`
	}{
		Alias: (*Alias)(c),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	// Populate standard fields from GPT-2 aliases if standard fields are zero
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
	return nil
}
