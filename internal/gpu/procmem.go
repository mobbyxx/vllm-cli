package gpu

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ReadProcMeminfo reads total and available memory from /proc/meminfo.
// Returns values in kilobytes.
func ReadProcMeminfo() (totalKB, availableKB int64, err error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, fmt.Errorf("opening /proc/meminfo: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			totalKB, err = parseMemLine(line)
			if err != nil {
				return 0, 0, fmt.Errorf("parsing MemTotal: %w", err)
			}
		} else if strings.HasPrefix(line, "MemAvailable:") {
			availableKB, err = parseMemLine(line)
			if err != nil {
				return 0, 0, fmt.Errorf("parsing MemAvailable: %w", err)
			}
		}
		if totalKB > 0 && availableKB > 0 {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, fmt.Errorf("reading /proc/meminfo: %w", err)
	}
	if totalKB == 0 {
		return 0, 0, fmt.Errorf("MemTotal not found in /proc/meminfo")
	}
	return totalKB, availableKB, nil
}

// parseMemLine parses a /proc/meminfo line like "MemTotal:  16384000 kB"
// and returns the value in kilobytes.
func parseMemLine(line string) (int64, error) {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return 0, fmt.Errorf("unexpected line format: %q", line)
	}
	val, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing value %q: %w", fields[1], err)
	}
	return val, nil
}
