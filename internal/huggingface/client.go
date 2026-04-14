// Package huggingface provides a client for the HuggingFace Hub API and local
// cache scanning. It fetches model metadata, config.json architecture parameters,
// and manages the ~/.cache/huggingface/hub directory layout.
package huggingface

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	clierrors "github.com/user/vllm-cli/internal/errors"
	"github.com/user/vllm-cli/internal/types"
)

const (
	hfBaseURL    = "https://huggingface.co"
	hfAPIBaseURL = "https://huggingface.co/api"
	hfTimeout    = 30 * time.Second
)

// Client is a HuggingFace API client.
type Client struct {
	http    *http.Client
	hfToken string
}

// NewClient creates a new HuggingFace client, reading HF_TOKEN from environment.
func NewClient() *Client {
	return &Client{
		http:    &http.Client{Timeout: hfTimeout},
		hfToken: ResolveToken(),
	}
}

// newRequest creates an HTTP request with optional authorization header.
func (c *Client) newRequest(method, url string) (*http.Request, error) {
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		return nil, err
	}
	if c.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.hfToken)
	}
	return req, nil
}

// doWithRetry executes an HTTP request with basic retry on 429.
func (c *Client) doWithRetry(req *http.Request) (*http.Response, error) {
	for attempt := 0; attempt < 3; attempt++ {
		resp, err := c.http.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusTooManyRequests {
			resp.Body.Close()
			time.Sleep(time.Duration(attempt+1) * 5 * time.Second)
			// Rebuild the request (body already consumed)
			newReq, newErr := c.newRequest(req.Method, req.URL.String())
			if newErr != nil {
				return nil, newErr
			}
			req = newReq
			continue
		}
		return resp, nil
	}
	return nil, fmt.Errorf("rate limited after 3 attempts")
}

// GetModelInfo fetches model metadata from the HuggingFace API.
func (c *Client) GetModelInfo(ref types.ModelRef) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/models/%s/%s", hfAPIBaseURL, ref.Owner, ref.Name)
	req, err := c.newRequest("GET", url)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.doWithRetry(req)
	if err != nil {
		return nil, fmt.Errorf("fetching model info: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusNotFound:
		return nil, clierrors.ErrModelNotFound(ref)
	case http.StatusForbidden, http.StatusUnauthorized:
		return nil, clierrors.ErrGatedModel(ref)
	case http.StatusOK:
		// OK
	default:
		body, _ := io.ReadAll(resp.Body)
		return nil, clierrors.NewCLIError(
			fmt.Sprintf("HuggingFace API returned %d", resp.StatusCode),
			"Check your internet connection or try again later",
			fmt.Errorf("status %d: %s", resp.StatusCode, string(body)),
		)
	}

	var info ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decoding model info: %w", err)
	}
	return &info, nil
}

// GetModelConfig fetches and parses config.json from the HuggingFace repository.
func (c *Client) GetModelConfig(ref types.ModelRef) (*ModelConfig, error) {
	url := fmt.Sprintf("%s/%s/%s/resolve/main/config.json", hfBaseURL, ref.Owner, ref.Name)
	req, err := c.newRequest("GET", url)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.doWithRetry(req)
	if err != nil {
		return nil, fmt.Errorf("fetching config.json: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusNotFound:
		return nil, clierrors.NewCLIError(
			fmt.Sprintf("config.json not found for model %s", ref.String()),
			"The model may not have a config.json file",
			nil,
		)
	case http.StatusForbidden, http.StatusUnauthorized:
		return nil, clierrors.ErrGatedModel(ref)
	case http.StatusOK:
		// OK
	default:
		return nil, clierrors.NewCLIError(
			fmt.Sprintf("HuggingFace returned %d fetching config.json", resp.StatusCode),
			"Check your internet connection",
			nil,
		)
	}

	var cfg ModelConfig
	if err := json.NewDecoder(resp.Body).Decode(&cfg); err != nil {
		return nil, fmt.Errorf("decoding config.json: %w", err)
	}
	return &cfg, nil
}

// EstimateParameterCount estimates the number of parameters from config.json
// if NumParameters is not set directly.
// Returns (totalParams, activeParams) where activeParams < totalParams for MoE models.
func EstimateParameterCount(cfg *ModelConfig) (total int64, active int64) {
	if cfg.NumParameters > 0 {
		return cfg.NumParameters, cfg.NumParameters
	}

	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return 0, 0
	}

	h := int64(cfg.HiddenSize)
	nLayers := int64(cfg.NumHiddenLayers)

	// Embedding layer: vocab_size × hidden_size
	embedding := int64(cfg.VocabSize) * h
	total += embedding
	active += embedding

	// --- Attention ---
	numHeads := cfg.NumAttentionHeads
	if numHeads == 0 {
		numHeads = 1
	}

	var attnPerLayer int64

	if cfg.KVLoraRank > 0 {
		// MLA (Multi-Latent Attention): DeepSeek-V2/V3, GLM-5.1
		kvRank := int64(cfg.KVLoraRank)
		rope := int64(cfg.QKRopeHeadDim)
		nope := int64(cfg.QKNopeHeadDim)
		vDim := int64(cfg.VHeadDim)
		nh := int64(numHeads)
		qkHeadDim := nope + rope

		if cfg.QLoraRank > 0 {
			// Case A (DeepSeek-V2/V3): fused down-proj + separate up-projs with LayerNorms
			qRank := int64(cfg.QLoraRank)
			attnPerLayer = h*(qRank+kvRank+rope) + // fused_qkv_a_proj
				qRank + // q_a_layernorm
				qRank*nh*qkHeadDim + // q_b_proj
				kvRank + // kv_a_layernorm
				kvRank*nh*(nope+vDim) + // kv_b_proj
				nh*vDim*h // o_proj
		} else {
			// Case B (GLM-5.1): direct Q projection, compressed KV
			attnPerLayer = h*nh*qkHeadDim + // q_proj
				h*(kvRank+rope) + // kv_a_proj_with_mqa
				kvRank + // kv_a_layernorm
				kvRank*nh*(nope+vDim) + // kv_b_proj
				nh*vDim*h // o_proj
		}
	} else {
		// Standard GQA/MHA attention
		numKVHeads := cfg.NumKeyValueHeads
		if numKVHeads == 0 {
			numKVHeads = numHeads
		}
		headDim := int64(cfg.HeadDim)
		if headDim == 0 {
			headDim = h / int64(numHeads)
		}
		kvDim := int64(numKVHeads) * headDim
		qDim := int64(numHeads) * headDim
		attnPerLayer = h*qDim + h*kvDim + h*kvDim + h*qDim
	}

	// Attention is the same in every layer (not affected by MoE)
	total += nLayers * attnPerLayer
	active += nLayers * attnPerLayer

	// --- MLP ---
	if cfg.NRoutedExperts > 0 && cfg.MoeIntermSize > 0 {
		// MoE model: some layers are dense, the rest are MoE
		denseLayers := int64(cfg.FirstKDenseReplace) // e.g. 3 for GLM-5.1
		moeFreq := cfg.MoeLayerFreq
		if moeFreq == 0 {
			moeFreq = 1
		}

		// Count MoE vs dense layers
		var moeLayers, nonMoeLayers int64
		for i := int64(0); i < nLayers; i++ {
			if i < denseLayers {
				nonMoeLayers++
			} else if moeFreq == 1 || (i-denseLayers)%int64(moeFreq) == 0 {
				moeLayers++
			} else {
				nonMoeLayers++
			}
		}

		// Dense MLP params: 3 × h × intermediate_size (SwiGLU: gate + up + down)
		denseMLP := int64(3) * h * int64(cfg.IntermediateSize)

		// Per-expert MLP: 3 × h × moe_intermediate_size
		expertMLP := int64(3) * h * int64(cfg.MoeIntermSize)

		// Shared expert MLP: same structure as dense but using moe_intermediate_size
		sharedMLP := int64(0)
		if cfg.NSharedExperts > 0 {
			sharedMLP = int64(cfg.NSharedExperts) * 3 * h * int64(cfg.MoeIntermSize)
		}

		// Router/gate: hidden_size × n_routed_experts per MoE layer
		routerParams := h * int64(cfg.NRoutedExperts)

		// Total MLP params
		totalMoeMLP := moeLayers * (int64(cfg.NRoutedExperts)*expertMLP + sharedMLP + routerParams)
		totalDenseMLP := nonMoeLayers * denseMLP
		total += totalMoeMLP + totalDenseMLP

		// Active MLP: only top-k experts fire per token
		topK := cfg.NumExpertsPerTok
		if topK == 0 {
			topK = 1
		}
		activeMoeMLP := moeLayers * (int64(topK)*expertMLP + sharedMLP + routerParams)
		active += activeMoeMLP + totalDenseMLP
	} else if cfg.IntermediateSize > 0 {
		// Standard dense MLP
		mlpParams := nLayers * 3 * h * int64(cfg.IntermediateSize)
		total += mlpParams
		active += mlpParams
	}

	// LM head: vocab_size × hidden_size
	lmHead := int64(cfg.VocabSize) * h
	total += lmHead
	active += lmHead

	return total, active
}
