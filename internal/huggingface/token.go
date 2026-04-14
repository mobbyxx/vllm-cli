package huggingface

import (
	"os"
	"path/filepath"
	"strings"
)

// ResolveToken returns the HuggingFace API token by checking:
//  1. HF_TOKEN environment variable
//  2. Token file at $HF_HOME/token (or ~/.cache/huggingface/token)
//
// Returns empty string if no token is found.
func ResolveToken() string {
	if token := os.Getenv("HF_TOKEN"); token != "" {
		return token
	}
	return readTokenFile()
}

func tokenFilePath() string {
	if hfHome := os.Getenv("HF_HOME"); hfHome != "" {
		return filepath.Join(hfHome, "token")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".cache", "huggingface", "token")
}

func readTokenFile() string {
	path := tokenFilePath()
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}
