// Package testutil builds deterministic local model packages on disk for
// tests. It is intentionally small: callers declare the manifest layers and
// supply integer quantized weights.
package testutil

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// Bundle is a constructed package directory plus its manifest.
type Bundle struct {
	Dir      string
	Manifest *model.Manifest
}

// WriteOptions controls package construction.
type WriteOptions struct {
	// CorruptWeights / CorruptBias flip a blob byte after checksum
	// generation to simulate on-device package corruption.
	CorruptWeights bool
	CorruptBias    bool
	// SkipChecksums leaves checksum fields blank.
	SkipChecksums bool
	// ExtraWeightBytes appends trailing bytes to weights.bin.
	ExtraWeightBytes int
}

// ManifestOptions describes a simple single-hidden-layer classifier.
type ManifestOptions struct {
	ModelID         string
	ModelVersion    string
	InputPrecision  model.Precision
	WeightPrecision model.Precision
	InputDim        int
	Hidden          int
	Outputs         int
	// Weight values: w0[hidden][input], w1[outputs][hidden].
	W0         [][]int
	W1         [][]int
	B0         []float64
	B1         []float64
	Scales     [2]float64 // layer scale, per layer
	ZeroPoints [2]int
	InputScale float64
	InputZero  int
}

// WriteTwoLayer writes a two-layer ReLU classifier into
// t.TempDir()/<subdir> and returns the bundle.
func WriteTwoLayer(t *testing.T, subdir string, o ManifestOptions) Bundle {
	t.Helper()
	labels := make([]string, o.Outputs)
	for i := range labels {
		labels[i] = labelName(i)
	}
	l0 := model.LayerSpec{
		Name: "fc0",
		Weights: model.TensorSpec{
			Name:      "fc0/w",
			Precision: o.WeightPrecision,
			Rows:      o.Hidden,
			Cols:      o.InputDim,
			ZeroPoint: o.ZeroPoints[0],
			Scale:     o.Scales[0],
		},
		Bias: o.B0,
	}
	l1 := model.LayerSpec{
		Name: "fc1",
		Weights: model.TensorSpec{
			Name:      "fc1/w",
			Precision: model.PrecisionINT8,
			Rows:      o.Outputs,
			Cols:      o.Hidden,
			ZeroPoint: o.ZeroPoints[1],
			Scale:     o.Scales[1],
		},
		Bias: o.B1,
	}
	m := &model.Manifest{
		FormatVersion:  model.FormatV1,
		ModelID:        o.ModelID,
		ModelVersion:   o.ModelVersion,
		InputPrecision: o.InputPrecision,
		InputZeroPoint: o.InputZero,
		InputScale:     o.InputScale,
		InputDim:       o.InputDim,
		OutputLabels:   labels,
		Layers:         []model.LayerSpec{l0, l1},
		CreatedAt:      time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC),
	}
	q0 := flatten(o.W0)
	q1 := flatten(o.W1)
	weights, err := packageio.EncodeWeights(m.Layers, [][]int{q0, q1})
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	bias, err := packageio.EncodeBias(m.Layers)
	if err != nil {
		t.Fatalf("encode bias: %v", err)
	}
	dir := filepath.Join(t.TempDir(), subdir)
	if err := packageio.WritePackage(dir, m, weights, bias); err != nil {
		t.Fatalf("write package: %v", err)
	}
	return Bundle{Dir: dir, Manifest: m}
}

// WriteRaw writes an arbitrary manifest and raw blobs to disk, optionally
// tampering with them after checksum generation. Checksums in m are filled in
// by WritePackage unless opts.SkipChecksums.
func WriteRaw(t *testing.T, subdir string, m *model.Manifest, weights, bias []byte, opts WriteOptions) Bundle {
	t.Helper()
	dir := filepath.Join(t.TempDir(), subdir)
	if err := packageio.WritePackage(dir, m, weights, bias); err != nil {
		t.Fatalf("write package: %v", err)
	}
	if opts.SkipChecksums {
		m.WeightsSHA256 = ""
		m.BiasSHA256 = ""
	}
	if opts.ExtraWeightBytes > 0 {
		weights = append(weights, make([]byte, opts.ExtraWeightBytes)...)
	}
	if opts.CorruptWeights && len(weights) > 0 {
		weights[0] ^= 0xff
	}
	if opts.CorruptBias && len(bias) > 0 {
		bias[0] ^= 0xff
	}
	if opts.CorruptWeights || opts.CorruptBias || opts.ExtraWeightBytes > 0 || opts.SkipChecksums {
		if err := writeBlobsAndManifest(t, dir, m, weights, bias, opts.SkipChecksums); err != nil {
			t.Fatalf("rewrite package: %v", err)
		}
	}
	return Bundle{Dir: dir, Manifest: m}
}

func writeBlobsAndManifest(t *testing.T, dir string, m *model.Manifest, weights, bias []byte, skipChecksum bool) error {
	if err := writeFile(filepath.Join(dir, packageio.WeightsName), weights); err != nil {
		return err
	}
	if err := writeFile(filepath.Join(dir, packageio.BiasName), bias); err != nil {
		return err
	}
	if skipChecksum {
		m.WeightsSHA256 = ""
		m.BiasSHA256 = ""
	}
	raw, err := jsonMarshalIndent(m)
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(dir, packageio.ManifestName), raw)
}

func flatten(m [][]int) []int {
	var out []int
	for _, r := range m {
		out = append(out, r...)
	}
	return out
}

func labelName(i int) string {
	const names = "abcdefghijklmnopqrstuvwxyz"
	if i < len(names) {
		return string(names[i])
	}
	return "label"
}
