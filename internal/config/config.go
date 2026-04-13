package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"gopkg.in/yaml.v3"
)

// Config holds all vllm-cli configuration.
type Config struct {
	DockerImage           string  `yaml:"docker_image"`
	PortRangeStart        int     `yaml:"port_range_start"`
	PortRangeEnd          int     `yaml:"port_range_end"`
	HFCachePath           string  `yaml:"hf_cache_path"`
	GPUMemoryCeiling      int     `yaml:"gpu_memory_ceiling_gb"`
	DefaultGPUUtilization float64 `yaml:"default_gpu_utilization"`
	Verbose               bool    `yaml:"-"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	home, _ := os.UserHomeDir()
	return &Config{
		DockerImage:           "vllm/vllm-openai:latest",
		PortRangeStart:        8000,
		PortRangeEnd:          8099,
		HFCachePath:           filepath.Join(home, ".cache", "huggingface", "hub"),
		GPUMemoryCeiling:      0,
		DefaultGPUUtilization: 0.9,
	}
}

// ConfigDir returns the XDG config directory for vllm-cli.
func ConfigDir() string {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "vllm-cli")
	}
	home, err := os.UserConfigDir()
	if err != nil {
		home, _ = os.UserHomeDir()
		return filepath.Join(home, ".config", "vllm-cli")
	}
	return filepath.Join(home, "vllm-cli")
}

// Load reads the config file, creating it with defaults if it doesn't exist.
// Environment variable overrides are applied after loading.
func Load() (*Config, error) {
	cfg := DefaultConfig()

	// Check env overrides
	if img := os.Getenv("VLLM_CLI_DOCKER_IMAGE"); img != "" {
		cfg.DockerImage = img
	}
	if portStart := os.Getenv("VLLM_CLI_PORT_START"); portStart != "" {
		if v, err := strconv.Atoi(portStart); err == nil && v > 0 {
			cfg.PortRangeStart = v
		}
	}

	cfgDir := ConfigDir()
	cfgFile := filepath.Join(cfgDir, "config.yaml")

	data, err := os.ReadFile(cfgFile)
	if os.IsNotExist(err) {
		// Create default config file
		if mkErr := os.MkdirAll(cfgDir, 0755); mkErr != nil {
			return cfg, nil // Return defaults even if we can't create dir
		}
		if writeData, marshalErr := yaml.Marshal(cfg); marshalErr == nil {
			_ = os.WriteFile(cfgFile, writeData, 0644)
		}
		return cfg, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading config file %s: %w", cfgFile, err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config file %s: %w", cfgFile, err)
	}

	// Re-apply env overrides (override file values)
	if img := os.Getenv("VLLM_CLI_DOCKER_IMAGE"); img != "" {
		cfg.DockerImage = img
	}
	if portStart := os.Getenv("VLLM_CLI_PORT_START"); portStart != "" {
		if v, err := strconv.Atoi(portStart); err == nil && v > 0 {
			cfg.PortRangeStart = v
		}
	}

	return cfg, nil
}
