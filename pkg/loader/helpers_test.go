package loader_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

func rewriteManifest(t *testing.T, dir string, m *model.Manifest) {
	t.Helper()
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, packageio.ManifestName), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func tamperByte(t *testing.T, path string, idx int) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if idx >= len(raw) {
		t.Fatalf("cannot tamper byte %d of %d-byte file", idx, len(raw))
	}
	raw[idx] ^= 0xff
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendByte(name, dir string, base []byte, b byte) error {
	raw := append(append([]byte{}, base...), b)
	return os.WriteFile(filepath.Join(dir, name), raw, 0o644)
}
