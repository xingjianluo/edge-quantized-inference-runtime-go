// Package model defines the on-disk and in-memory representation of a local
// quantized model package used by the edge inference runtime.
package model

import "time"

// Precision is the storage precision of a quantized tensor.
type Precision string

const (
	// PrecisionINT8 stores each weight as one signed 8-bit value.
	PrecisionINT8 Precision = "int8"
	// PrecisionINT4 stores two signed 4-bit values per byte (low nibble
	// first, then high nibble), each in the range [-8, 7].
	PrecisionINT4 Precision = "int4"
)

// FormatVersion is the model package manifest format version.
type FormatVersion string

const (
	// FormatV1 is the only supported package format: a JSON manifest named
	// model.json plus two concatenated blobs weights.bin and bias.bin.
	FormatV1 FormatVersion = "edge-qmodel/v1"
)

// TensorSpec describes the shape and quantized storage of one parameter tensor.
type TensorSpec struct {
	Name      string    `json:"name"`
	Precision Precision `json:"precision"`
	Rows      int       `json:"rows"`
	Cols      int       `json:"cols"`
	// ZeroPoint is the integer zero point used for affine quantization.
	ZeroPoint int `json:"zero_point"`
	// Scale is the positive multiplicative scale used for affine
	// quantization: real = (q - zero_point) * scale.
	Scale float64 `json:"scale"`
}

// LayerSpec describes one dense feed-forward layer.
type LayerSpec struct {
	Name    string     `json:"name"`
	Weights TensorSpec `json:"weights"`
	Bias    []float64  `json:"bias"`
}

// Manifest is the parsed model.json descriptor of a model package.
type Manifest struct {
	FormatVersion FormatVersion `json:"format_version"`
	ModelID       string        `json:"model_id"`
	ModelVersion  string        `json:"model_version"`
	// InputPrecision declares the precision contract applied to user inputs
	// (input quantization is affine with input_zero_point/input_scale).
	InputPrecision Precision   `json:"input_precision"`
	InputZeroPoint int         `json:"input_zero_point"`
	InputScale     float64     `json:"input_scale"`
	InputDim       int         `json:"input_dim"`
	OutputLabels   []string    `json:"output_labels"`
	Layers         []LayerSpec `json:"layers"`
	// WeightsSHA256 / BiasSHA256 authenticate the concatenated weight and
	// bias blobs (hex encoded sha256).
	WeightsSHA256 string    `json:"weights_sha256"`
	BiasSHA256    string    `json:"bias_sha256"`
	CreatedAt     time.Time `json:"created_at"`
}

// Layer is a validated, loaded layer with raw quantized weights attached.
type Layer struct {
	Spec   LayerSpec
	Weight []byte // packed quantized weights in row-major order
	Bias   []float64
}

// LoadedModel is a fully validated model ready for inference. All layers are
// present; a LoadedModel is immutable after the loader returns it.
type LoadedModel struct {
	Manifest Manifest
	Layers   []Layer
	// WeightBytes is the total packed weight blob size.
	WeightBytes int
	// ActivationsBytes is the steady-state size of per-inference activation
	// buffers, used for device resource accounting.
	ActivationsBytes int
}

// IsSupported reports whether p is a precision the runtime can execute.
func (p Precision) IsSupported() bool { return p == PrecisionINT8 || p == PrecisionINT4 }

// PackedSize returns the on-disk byte length of a weight matrix of rows*cols
// quantized elements with this precision. It returns 0 for an unsupported
// precision or non-positive shape.
func (s TensorSpec) PackedSize() int {
	n := s.Rows * s.Cols
	if n <= 0 {
		return 0
	}
	switch s.Precision {
	case PrecisionINT8:
		return n
	case PrecisionINT4:
		return (n + 1) / 2
	default:
		return 0
	}
}
