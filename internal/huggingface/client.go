package huggingface

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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
		hfToken: os.Getenv("HF_TOKEN"),
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
func EstimateParameterCount(cfg *ModelConfig) int64 {
	if cfg.NumParameters > 0 {
		return cfg.NumParameters
	}

	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return 0
	}

	var params int64

	// Embedding layer: vocab_size × hidden_size
	params += int64(cfg.VocabSize) * int64(cfg.HiddenSize)

	// Attention per layer: 4 × hidden_size²
	// (Q, K, V, O projections — simplified, doesn't account for GQA exactly)
	attentionParams := int64(cfg.NumHiddenLayers) * 4 * int64(cfg.HiddenSize) * int64(cfg.HiddenSize)
	params += attentionParams

	// MLP per layer (SwiGLU): 3 × hidden_size × intermediate_size
	if cfg.IntermediateSize > 0 {
		mlpParams := int64(cfg.NumHiddenLayers) * 3 * int64(cfg.HiddenSize) * int64(cfg.IntermediateSize)
		params += mlpParams
	}

	// LM head: vocab_size × hidden_size
	params += int64(cfg.VocabSize) * int64(cfg.HiddenSize)

	return params
}
