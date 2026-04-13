package huggingface

import (
	"fmt"
	"os"
	"path/filepath"
)

func testScan() {
	// Create mock cache structure
	tmpDir, _ := os.MkdirTemp("", "test-hf-cache")
	defer os.RemoveAll(tmpDir)

	snapDir := filepath.Join(tmpDir, "models--test-org--test-model", "snapshots", "abc123")
	os.MkdirAll(snapDir, 0755)
	os.WriteFile(filepath.Join(snapDir, "config.json"), []byte(`{"hidden_size": 768}`), 0644)
	os.WriteFile(filepath.Join(snapDir, "pytorch_model.bin"), make([]byte, 1024*1024), 0644) // 1MB

	models, err := ScanCache(tmpDir)
	if err != nil { fmt.Println("ERROR:", err); os.Exit(1) }
	if len(models) != 1 { fmt.Printf("FAIL: expected 1 model, got %d\n", len(models)); os.Exit(1) }
	m := models[0]
	if m.Ref.Owner != "test-org" { fmt.Println("FAIL: wrong owner"); os.Exit(1) }
	if m.Ref.Name != "test-model" { fmt.Println("FAIL: wrong name"); os.Exit(1) }
	if !m.HasConfig { fmt.Println("FAIL: should have config"); os.Exit(1) }
	fmt.Printf("Found: %s/%s, Size: %d, HasConfig: %v\n", m.Ref.Owner, m.Ref.Name, m.SizeBytes, m.HasConfig)
	fmt.Println("ALL PASS")
}

func testEmpty() {
	tmpDir, _ := os.MkdirTemp("", "test-hf-cache-empty")
	defer os.RemoveAll(tmpDir)

	models, err := ScanCache(tmpDir)
	if err != nil { fmt.Println("ERROR:", err); os.Exit(1) }
	if models == nil { fmt.Println("FAIL: models is nil (should be empty slice)"); os.Exit(1) }
	if len(models) != 0 { fmt.Printf("FAIL: expected 0 models, got %d\n", len(models)); os.Exit(1) }
	fmt.Println("Empty cache: OK")
	fmt.Println("ALL PASS")
}

func testFormatSize() {
	cases := []struct{ bytes int64; expected string }{
		{1073741824, "1.0 GB"},
		{524288000, "500.0 MB"},
		{1024, "1.0 KB"},
	}
	for _, tc := range cases {
		result := FormatSize(tc.bytes)
		fmt.Printf("FormatSize(%d) = %q (expected %q)\n", tc.bytes, result, tc.expected)
		if result != tc.expected {
			fmt.Printf("FAIL: got %q want %q\n", result, tc.expected)
			os.Exit(1)
		}
	}
	fmt.Println("ALL PASS")
}
