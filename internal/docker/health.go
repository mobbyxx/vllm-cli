package docker

import (
	"fmt"
	"net/http"
	"time"
)

// WaitForHealthy polls the vLLM health endpoint until it returns 200 or times out.
// Poll interval: 2s. Default timeout: 300s.
func WaitForHealthy(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	url := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 5 * time.Second}

	start := time.Now()
	for {
		select {
		case <-ticker.C:
			elapsed := time.Since(start).Round(time.Second)
			fmt.Printf("Waiting for model to load... (%s elapsed)\r", elapsed)

			if IsHealthy(port) {
				fmt.Println() // clear the \r line
				return nil
			}

			if time.Now().After(deadline) {
				return fmt.Errorf("model did not become healthy after %s", timeout)
			}
		}
		_ = client
		_ = url
	}
}

// IsHealthy returns true if the vLLM health endpoint returns 200.
func IsHealthy(port int) bool {
	url := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
