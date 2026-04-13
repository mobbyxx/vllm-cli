package huggingface

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/user/vllm-cli/internal/types"
)

// CachedModel represents a model found in the HuggingFace cache.
type CachedModel struct {
	Ref       types.ModelRef
	SizeBytes int64
	Snapshots []string // commit hashes
	CachePath string   // full path to model dir
	HasConfig bool     // config.json present in any snapshot
}

// ScanCache scans the HuggingFace cache directory for downloaded models.
// The cache layout is: {cachePath}/models--{org}--{name}/snapshots/{hash}/
func ScanCache(cachePath string) ([]CachedModel, error) {
	entries, err := os.ReadDir(cachePath)
	if err != nil {
		if os.IsNotExist(err) {
			return []CachedModel{}, nil
		}
		return nil, fmt.Errorf("reading cache dir %s: %w", cachePath, err)
	}

	var models []CachedModel
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		// HF cache dirs start with "models--"
		if !strings.HasPrefix(name, "models--") {
			continue
		}

		// Parse org and model name from directory name
		// Format: models--{org}--{model}
		// Note: org and model names cannot contain "--", but we split on first two "--"
		parts := strings.SplitN(strings.TrimPrefix(name, "models--"), "--", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			continue
		}

		ref := types.ModelRef{Owner: parts[0], Name: parts[1]}
		modelDir := filepath.Join(cachePath, name)

		cachedModel := CachedModel{
			Ref:       ref,
			CachePath: modelDir,
			Snapshots: []string{},
		}

		// Scan snapshots directory
		snapshotsDir := filepath.Join(modelDir, "snapshots")
		snapEntries, err := os.ReadDir(snapshotsDir)
		if err == nil {
			for _, snap := range snapEntries {
				if !snap.IsDir() {
					continue
				}
				snapHash := snap.Name()
				cachedModel.Snapshots = append(cachedModel.Snapshots, snapHash)

				// Check for config.json
				snapDir := filepath.Join(snapshotsDir, snapHash)
				if _, err := os.Stat(filepath.Join(snapDir, "config.json")); err == nil {
					cachedModel.HasConfig = true
				}

				// Calculate size (follow symlinks to blobs)
				size, _ := dirSize(snapDir)
				cachedModel.SizeBytes += size
			}
		}

		models = append(models, cachedModel)
	}

	if models == nil {
		return []CachedModel{}, nil
	}
	return models, nil
}

// dirSize calculates the total size of all files in a directory, following symlinks.
func dirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}
		// Follow symlinks to get real file info
		if info.Mode()&os.ModeSymlink != 0 {
			realInfo, err := os.Stat(filePath)
			if err == nil {
				info = realInfo
			}
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}

// FindCachedModel finds a specific model in the cache.
func FindCachedModel(cachePath string, ref types.ModelRef) (*CachedModel, error) {
	models, err := ScanCache(cachePath)
	if err != nil {
		return nil, err
	}
	for _, m := range models {
		if m.Ref.Owner == ref.Owner && m.Ref.Name == ref.Name {
			return &m, nil
		}
	}
	return nil, nil
}

// RemoveCachedModel deletes a model from the cache.
func RemoveCachedModel(cachePath string, ref types.ModelRef) error {
	// Find the model directory
	dirName := "models--" + ref.Owner + "--" + ref.Name
	modelDir := filepath.Join(cachePath, dirName)

	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		return fmt.Errorf("model %s not found in cache", ref.String())
	}

	if err := os.RemoveAll(modelDir); err != nil {
		return fmt.Errorf("removing model directory: %w", err)
	}
	return nil
}

// FormatSize returns a human-readable file size string.
func FormatSize(bytes int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}
