package docker

import (
	"fmt"
	"net/http"
	"time"
)

func WaitForHealthy(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if IsHealthy(port) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("model did not become healthy after %s", timeout)
		}
	}
	return nil
}

func WaitForHealthyAsync(port int, timeout time.Duration) <-chan error {
	ch := make(chan error, 1)
	go func() {
		ch <- WaitForHealthy(port, timeout)
		close(ch)
	}()
	return ch
}

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
