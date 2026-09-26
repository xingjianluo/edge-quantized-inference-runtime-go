package testutil

import (
	"encoding/json"
	"os"
)

func writeFile(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

func jsonMarshalIndent(v any) ([]byte, error) {
	return json.MarshalIndent(v, "", "  ")
}
