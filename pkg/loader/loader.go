// Package loader validates and loads local quantized model packages into
// immutable, fully materialized model.LoadedModel values.
//
// A package is accepted only if every check below passes; there is no partial
// load and no silent precision fallback:
//
//  1. the directory and model.json exist,
//  2. the manifest format version is supported,
//  3. required manifest fields are present and well-formed,
//  4. every weight/input precision is supported,
//  5. every quantized tensor carries valid quantization parameters
//     (scale > 0, zero point inside the representable integer range),
//  6. all dimensions line up (input dim -> layer chain -> output labels),
//  7. the weight and bias blobs pass sha256 verification, contain exactly
//     the declared bytes, and all biases are finite and match the manifest,
//  8. the model fits the device memory budget, when one is configured.
package loader

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// metadataOverhead is a fixed per-model estimate for the manifest and runtime
// bookkeeping in device memory accounting.
const metadataOverhead int64 = 4096

// Options controls load-time resource accounting.
type Options struct {
	// DeviceMemoryBytes is the device memory budget the loaded model must
	// fit in. Zero means no device-level check at load time.
	DeviceMemoryBytes int64
}

// Load reads, authenticates and validates the package in dir and returns a
// fully loaded model.
func Load(dir string, opts Options) (*model.LoadedModel, error) {
	if !packageio.Exists(dir) {
		return nil, merrors.New(merrors.ReasonPackageNotFound,
			"no model package at %q", dir)
	}
	m, err := packageio.ReadManifest(dir)
	if err != nil {
		return nil, err
	}
	if err := validateManifest(m); err != nil {
		return nil, err
	}

	weightBlob, err := packageio.ReadWeights(dir)
	if err != nil {
		return nil, err
	}
	biasBlob, err := packageio.ReadBias(dir)
	if err != nil {
		return nil, err
	}
	if err := verifyChecksum("weights", weightBlob, m.WeightsSHA256); err != nil {
		return nil, err
	}
	if err := verifyChecksum("bias", biasBlob, m.BiasSHA256); err != nil {
		return nil, err
	}

	loaded, err := unpack(m, weightBlob, biasBlob)
	if err != nil {
		return nil, err
	}

	if opts.DeviceMemoryBytes > 0 {
		if need := ModelMemoryBytes(loaded); need > opts.DeviceMemoryBytes {
			return nil, merrors.New(merrors.ReasonDeviceBudgetExceeded,
				"model %q needs %d device bytes, budget is %d",
				m.ModelID, need, opts.DeviceMemoryBytes)
		}
	}
	return loaded, nil
}

func validateManifest(m *model.Manifest) error {
	// Version is checked first: an unknown format must never be interpreted
	// under this runtime's field semantics.
	if m.FormatVersion != model.FormatV1 {
		return merrors.New(merrors.ReasonUnsupportedVersion,
			"format version %q is not supported (want %q)",
			m.FormatVersion, model.FormatV1)
	}
	if strings.TrimSpace(m.ModelID) == "" {
		return merrors.New(merrors.ReasonInvalidManifest, "model_id is required")
	}
	if strings.TrimSpace(m.ModelVersion) == "" {
		return merrors.New(merrors.ReasonInvalidManifest, "model_version is required")
	}
	if m.InputDim <= 0 {
		return merrors.New(merrors.ReasonInvalidManifest,
			"input_dim must be positive, got %d", m.InputDim)
	}
	if len(m.Layers) == 0 {
		return merrors.New(merrors.ReasonInvalidManifest,
			"model %q declares no layers", m.ModelID)
	}
	if len(m.OutputLabels) == 0 {
		return merrors.New(merrors.ReasonInvalidManifest,
			"model %q declares no output labels", m.ModelID)
	}
	if !m.InputPrecision.IsSupported() {
		return merrors.New(merrors.ReasonUnsupportedPrecision,
			"input precision %q is not supported by %s",
			m.InputPrecision, model.FormatV1)
	}
	if err := validateQuant("input", m.InputPrecision, m.InputZeroPoint, m.InputScale); err != nil {
		return err
	}

	seen := map[string]bool{}
	for li := range m.Layers {
		l := &m.Layers[li]
		if strings.TrimSpace(l.Name) == "" {
			return merrors.New(merrors.ReasonInvalidManifest,
				"layer #%d has no name", li)
		}
		if seen[l.Name] {
			return merrors.New(merrors.ReasonInvalidManifest,
				"duplicate layer name %q", l.Name)
		}
		seen[l.Name] = true
		w := l.Weights
		w.Name = strings.TrimSpace(w.Name)
		if !w.Precision.IsSupported() {
			return merrors.New(merrors.ReasonUnsupportedPrecision,
				"layer %q uses unsupported precision %q", l.Name, w.Precision)
		}
		if w.Rows <= 0 || w.Cols <= 0 {
			return merrors.New(merrors.ReasonDimensionMismatch,
				"layer %q weight matrix must have positive dimensions, got %dx%d",
				l.Name, w.Rows, w.Cols)
		}
		if err := validateQuant("layer "+l.Name, w.Precision, w.ZeroPoint, w.Scale); err != nil {
			return err
		}
		if len(l.Bias) != w.Rows {
			return merrors.New(merrors.ReasonDimensionMismatch,
				"layer %q: bias length %d != output dim %d",
				l.Name, len(l.Bias), w.Rows)
		}
	}

	// Chain dimensions: layer 0 consumes InputDim; each later layer must
	// consume the previous layer's output dim.
	if m.Layers[0].Weights.Cols != m.InputDim {
		return merrors.New(merrors.ReasonDimensionMismatch,
			"layer %q input dim %d != model input dim %d",
			m.Layers[0].Name, m.Layers[0].Weights.Cols, m.InputDim)
	}
	for i := 1; i < len(m.Layers); i++ {
		prev := m.Layers[i-1].Weights.Rows
		cur := m.Layers[i].Weights.Cols
		if cur != prev {
			return merrors.New(merrors.ReasonDimensionMismatch,
				"layer %q input dim %d != previous layer %q output dim %d",
				m.Layers[i].Name, cur, m.Layers[i-1].Name, prev)
		}
	}
	lastRows := m.Layers[len(m.Layers)-1].Weights.Rows
	if len(m.OutputLabels) != lastRows {
		return merrors.New(merrors.ReasonDimensionMismatch,
			"output labels %d != final layer output dim %d",
			len(m.OutputLabels), lastRows)
	}
	return nil
}

func validateQuant(where string, p model.Precision, zeroPoint int, scale float64) error {
	min, max := quantRange(p)
	if scale <= 0 {
		return merrors.New(merrors.ReasonMissingQuantization,
			"%s: non-positive or missing quant scale %v", where, scale)
	}
	if zeroPoint < min || zeroPoint > max {
		return merrors.New(merrors.ReasonMissingQuantization,
			"%s: zero point %d outside %s range [%d,%d]",
			where, zeroPoint, p, min, max)
	}
	return nil
}

func quantRange(p model.Precision) (min, max int) {
	switch p {
	case model.PrecisionINT8:
		return -128, 127
	case model.PrecisionINT4:
		return -8, 7
	default:
		return 0, 0
	}
}

func verifyChecksum(kind string, blob []byte, wantHex string) error {
	if strings.TrimSpace(wantHex) == "" {
		return merrors.New(merrors.ReasonWeightCorrupt,
			"manifest declares no %s checksum", kind)
	}
	sum := sha256.Sum256(blob)
	if got := hex.EncodeToString(sum[:]); !strings.EqualFold(got, wantHex) {
		return merrors.New(merrors.ReasonWeightCorrupt,
			"%s checksum mismatch: manifest says %s, blob has %s",
			kind, wantHex, got)
	}
	return nil
}

func unpack(m *model.Manifest, weightBlob, biasBlob []byte) (*model.LoadedModel, error) {
	var wantWeightBytes int
	for li := range m.Layers {
		wantWeightBytes += m.Layers[li].Weights.PackedSize()
	}
	if len(weightBlob) != wantWeightBytes {
		return nil, merrors.New(merrors.ReasonWeightCorrupt,
			"weights blob is %d bytes, manifest geometry requires %d",
			len(weightBlob), wantWeightBytes)
	}

	biases, err := packageio.DecodeBias(m.Layers, biasBlob)
	if err != nil {
		return nil, err
	}

	layers := make([]model.Layer, len(m.Layers))
	off := 0
	maxWidth := m.InputDim
	for li := range m.Layers {
		size := m.Layers[li].Weights.PackedSize()
		chunk := weightBlob[off : off+size]
		off += size

		// For an odd number of int4 elements the unused high nibble of the
		// final byte must be zero; otherwise the blob holds an undocumented
		// extra quantized value.
		spec := m.Layers[li].Weights
		if spec.Precision == model.PrecisionINT4 && (spec.Rows*spec.Cols)%2 == 1 {
			if chunk[len(chunk)-1]&0xf0 != 0 {
				return nil, merrors.New(merrors.ReasonWeightCorrupt,
					"layer %q: non-zero padding nibble in weights",
					m.Layers[li].Name)
			}
		}

		// Blob biases must match the manifest-declared biases bit for bit; a
		// disagreement means the package was tampered with or truncated.
		for i, b := range biases[li] {
			if b != m.Layers[li].Bias[i] {
				return nil, merrors.New(merrors.ReasonWeightCorrupt,
					"layer %q bias[%d]: blob value %v != manifest value %v",
					m.Layers[li].Name, i, b, m.Layers[li].Bias[i])
			}
		}

		// Defensive copy: callers must not be able to mutate package bytes
		// through the returned model.
		wCopy := make([]byte, len(chunk))
		copy(wCopy, chunk)
		layers[li] = model.Layer{
			Spec:   m.Layers[li],
			Weight: wCopy,
			Bias:   biases[li],
		}
		if spec.Rows > maxWidth {
			maxWidth = spec.Rows
		}
	}

	biasBytes := 0
	for li := range m.Layers {
		biasBytes += m.Layers[li].Weights.Rows * 8
	}
	return &model.LoadedModel{
		Manifest:         *m,
		Layers:           layers,
		WeightBytes:      wantWeightBytes,
		ActivationsBytes: (m.InputDim + 2*maxWidth) * 8,
	}, nil
}

// ModelMemoryBytes reports the steady-state device memory required by a fully
// loaded model: packed weights, float64 biases, per-inference activation
// buffers and a fixed bookkeeping estimate.
func ModelMemoryBytes(m *model.LoadedModel) int64 {
	if m == nil {
		return 0
	}
	biasBytes := 0
	for li := range m.Layers {
		biasBytes += m.Layers[li].Spec.Weights.Rows * 8
	}
	return int64(m.WeightBytes) + int64(biasBytes) +
		int64(m.ActivationsBytes) + metadataOverhead
}
