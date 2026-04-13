package types

import (
	"fmt"
	"strings"
)

// ModelRef identifies a HuggingFace model by owner and name.
type ModelRef struct {
	Owner string // e.g. "meta-llama"
	Name  string // e.g. "Llama-3.1-8B-Instruct"
}

// ParseModelRef parses a HuggingFace model ID string like "owner/name".
func ParseModelRef(s string) (ModelRef, error) {
	parts := strings.Split(s, "/")
	if len(parts) != 2 {
		return ModelRef{}, fmt.Errorf("invalid model reference %q: must be in format 'owner/name'", s)
	}
	if parts[0] == "" || parts[1] == "" {
		return ModelRef{}, fmt.Errorf("invalid model reference %q: owner and name must be non-empty", s)
	}
	return ModelRef{Owner: parts[0], Name: parts[1]}, nil
}

// String returns the HuggingFace model ID: "owner/name".
func (m ModelRef) String() string {
	return m.Owner + "/" + m.Name
}

// Slug returns a Docker/filesystem-safe name: "owner--name".
func (m ModelRef) Slug() string {
	return m.Owner + "--" + m.Name
}
