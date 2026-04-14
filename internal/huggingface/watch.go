package huggingface

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"github.com/user/vllm-cli/internal/types"
)

type DownloadProgress struct {
	Downloaded int64
	Total      int64
}

func WatchDownloadProgress(ctx context.Context, cachePath string, ref types.ModelRef, totalSize int64, interval time.Duration) <-chan DownloadProgress {
	ch := make(chan DownloadProgress, 8)

	modelDir := filepath.Join(cachePath, "models--"+ref.Owner+"--"+ref.Name)

	go func() {
		defer close(ch)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				size := cacheCurrentSize(modelDir)
				select {
				case ch <- DownloadProgress{Downloaded: size, Total: totalSize}:
				default:
				}
				if totalSize > 0 && size >= totalSize {
					return
				}
			}
		}
	}()

	return ch
}

func cacheCurrentSize(modelDir string) int64 {
	var size int64
	_ = filepath.Walk(modelDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.Mode()&os.ModeSymlink != 0 {
			realInfo, err := os.Stat(path)
			if err == nil {
				info = realInfo
			}
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}
