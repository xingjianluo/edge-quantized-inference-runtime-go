// Package packageio reads and writes local model package directories.
//
// A v1 package directory contains:
//
//	model.json - JSON encoded model.Manifest
//	weights.bin - all layer weight matrices concatenated in layer order,
//	             packed according to each layer's declared precision
//	bias.bin    - all layer biases concatenated in layer order, float64
//	             little-endian, with no padding
package packageio

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// ManifestPath / blob file names within a package directory.
const (
	ManifestName = "model.json"
	WeightsName  = "weights.bin"
	BiasName     = "bias.bin"
)

// Exists reports whether dir looks like a package directory (it contains a
// model.json file).
func Exists(dir string) bool {
	info, err := os.Stat(filepath.Join(dir, ManifestName))
	return err == nil && !info.IsDir()
}

// WritePackage writes a complete model package into dir, creating the
// directory if needed. weights and bias are authenticated: the manifest's
// sha256 fields are filled in before model.json is written.
func WritePackage(dir string, m *model.Manifest, weights, bias []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return merrors.New(merrors.ReasonPackageNotFound,
			"create package directory %q: %v", dir, err)
	}
	wSum := sha256.Sum256(weights)
	bSum := sha256.Sum256(bias)
	m.WeightsSHA256 = hex.EncodeToString(wSum[:])
	m.BiasSHA256 = hex.EncodeToString(bSum[:])

	manifestBytes, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return merrors.New(merrors.ReasonInvalidManifest,
			"encode manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, WeightsName), weights, 0o644); err != nil {
		return merrors.New(merrors.ReasonPackageNotFound, "write weights: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, BiasName), bias, 0o644); err != nil {
		return merrors.New(merrors.ReasonPackageNotFound, "write bias: %v", err)
	}
	// Manifest is written last and references the blob checksums, so a
	// directory missing model.json is never mistaken for a valid package.
	if err := os.WriteFile(filepath.Join(dir, ManifestName), manifestBytes, 0o644); err != nil {
		return merrors.New(merrors.ReasonPackageNotFound, "write manifest: %v", err)
	}
	return nil
}

// ReadManifest reads and parses model.json from dir.
func ReadManifest(dir string) (*model.Manifest, error) {
	path := filepath.Join(dir, ManifestName)
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, merrors.New(merrors.ReasonPackageNotFound,
				"no model.json in %q", dir)
		}
		return nil, merrors.New(merrors.ReasonPackageNotFound,
			"read manifest from %q: %v", dir, err)
	}
	var m model.Manifest
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, merrors.New(merrors.ReasonInvalidManifest,
			"parse %s: %v", path, err)
	}
	return &m, nil
}

// ReadWeights reads the packed weight blob from dir.
func ReadWeights(dir string) ([]byte, error) {
	return readBlob(dir, WeightsName)
}

// ReadBias reads the concatenated bias blob from dir.
func ReadBias(dir string) ([]byte, error) {
	return readBlob(dir, BiasName)
}

func readBlob(dir, name string) ([]byte, error) {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, merrors.New(merrors.ReasonWeightCorrupt,
				"package in %q is missing %s", dir, name)
		}
		return nil, merrors.New(merrors.ReasonWeightCorrupt,
			"read %s from %q: %v", name, dir, err)
	}
	return raw, nil
}

// EncodeWeights packs one flat row-major quantized matrix per layer into a
// single weight blob in layer order. q[l] must contain rows*cols quantized
// values in the legal range for the layer's precision; any violation is a
// weight_corrupt error rather than silent clipping.
func EncodeWeights(layers []model.LayerSpec, q [][]int) ([]byte, error) {
	if len(q) != len(layers) {
		return nil, merrors.New(merrors.ReasonWeightCorrupt,
			"got %d weight matrices for %d layers", len(q), len(layers))
	}
	var buf []byte
	for li := range layers {
		spec := layers[li].Weights
		n := spec.Rows * spec.Cols
		if len(q[li]) != n {
			return nil, merrors.New(merrors.ReasonDimensionMismatch,
				"layer %q: got %d quantized weights, want %d (%dx%d)",
				layers[li].Name, len(q[li]), n, spec.Rows, spec.Cols)
		}
		switch spec.Precision {
		case model.PrecisionINT8:
			for i, v := range q[li] {
				if v < math.MinInt8 || v > math.MaxInt8 {
					return nil, merrors.New(merrors.ReasonWeightCorrupt,
						"layer %q weight[%d]=%d outside int8 range",
						layers[li].Name, i, v)
				}
				buf = append(buf, byte(int8(v)))
			}
		case model.PrecisionINT4:
			for i := 0; i < n; i += 2 {
				lo := q[li][i]
				if lo < -8 || lo > 7 {
					return nil, merrors.New(merrors.ReasonWeightCorrupt,
						"layer %q weight[%d]=%d outside int4 range",
						layers[li].Name, i, lo)
				}
				packed := byte(lo) & 0x0f
				if i+1 < n {
					hi := q[li][i+1]
					if hi < -8 || hi > 7 {
						return nil, merrors.New(merrors.ReasonWeightCorrupt,
							"layer %q weight[%d]=%d outside int4 range",
							layers[li].Name, i+1, hi)
					}
					packed |= (byte(hi) & 0x0f) << 4
				}
				buf = append(buf, packed)
			}
		default:
			return nil, merrors.New(merrors.ReasonUnsupportedPrecision,
				"layer %q: cannot encode precision %q",
				layers[li].Name, spec.Precision)
		}
	}
	return buf, nil
}

// EncodeBias concatenates layer biases into the bias blob format (float64
// little-endian, no padding).
func EncodeBias(layers []model.LayerSpec) ([]byte, error) {
	var buf []byte
	tmp := make([]byte, 8)
	for li := range layers {
		if len(layers[li].Bias) != layers[li].Weights.Rows {
			return nil, merrors.New(merrors.ReasonDimensionMismatch,
				"layer %q: got %d biases, want %d",
				layers[li].Name, len(layers[li].Bias), layers[li].Weights.Rows)
		}
		for i, b := range layers[li].Bias {
			if math.IsNaN(b) || math.IsInf(b, 0) {
				return nil, merrors.New(merrors.ReasonWeightCorrupt,
					"layer %q bias[%d] is not finite", layers[li].Name, i)
			}
			binary.LittleEndian.PutUint64(tmp, math.Float64bits(b))
			buf = append(buf, tmp...)
		}
	}
	return buf, nil
}

// DecodeBias parses a bias blob and splits it into per-layer bias vectors of
// the declared widths.
func DecodeBias(layers []model.LayerSpec, blob []byte) ([][]float64, error) {
	out := make([][]float64, len(layers))
	pos := 0
	for li := range layers {
		n := layers[li].Weights.Rows
		if pos+8*n > len(blob) {
			return nil, merrors.New(merrors.ReasonWeightCorrupt,
				"bias blob truncated at layer %q: have %d bytes, need %d",
				layers[li].Name, len(blob)-pos, 8*n)
		}
		row := make([]float64, n)
		for i := range row {
			bits := binary.LittleEndian.Uint64(blob[pos:])
			v := math.Float64frombits(bits)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, merrors.New(merrors.ReasonWeightCorrupt,
					"layer %q bias[%d] is not finite", layers[li].Name, i)
			}
			row[i] = v
			pos += 8
		}
		out[li] = row
	}
	if pos != len(blob) {
		return nil, merrors.New(merrors.ReasonWeightCorrupt,
			"bias blob has %d trailing unexpected bytes", len(blob)-pos)
	}
	return out, nil
}
