package huggingface

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveToken_EnvVar(t *testing.T) {
	t.Setenv("HF_TOKEN", "env-token-123")
	token := ResolveToken()
	if token != "env-token-123" {
		t.Errorf("expected env-token-123, got %q", token)
	}
}

func TestResolveToken_FileFallback(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	tmpDir := t.TempDir()
	t.Setenv("HF_HOME", tmpDir)

	tokenPath := filepath.Join(tmpDir, "token")
	if err := os.WriteFile(tokenPath, []byte("file-token-456\n"), 0600); err != nil {
		t.Fatal(err)
	}

	token := ResolveToken()
	if token != "file-token-456" {
		t.Errorf("expected file-token-456, got %q", token)
	}
}

func TestResolveToken_EnvTakesPrecedence(t *testing.T) {
	t.Setenv("HF_TOKEN", "env-wins")
	tmpDir := t.TempDir()
	t.Setenv("HF_HOME", tmpDir)
	if err := os.WriteFile(filepath.Join(tmpDir, "token"), []byte("file-token"), 0600); err != nil {
		t.Fatal(err)
	}

	token := ResolveToken()
	if token != "env-wins" {
		t.Errorf("expected env-wins, got %q", token)
	}
}

func TestResolveToken_NoTokenAnywhere(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_HOME", t.TempDir())

	token := ResolveToken()
	if token != "" {
		t.Errorf("expected empty, got %q", token)
	}
}

func TestResolveToken_WhitespaceTrimmed(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	tmpDir := t.TempDir()
	t.Setenv("HF_HOME", tmpDir)
	if err := os.WriteFile(filepath.Join(tmpDir, "token"), []byte("  tok-123  \n"), 0600); err != nil {
		t.Fatal(err)
	}

	token := ResolveToken()
	if token != "tok-123" {
		t.Errorf("expected tok-123, got %q", token)
	}
}
