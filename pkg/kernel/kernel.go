// Package kernel implements deterministic dequantization and feed-forward
// inference for loaded quantized models.
//
// Arithmetic is performed in float64 with a fixed summation order, so the same
// (model, input) pair produces bit-identical logits on every call, regardless
// of session, goroutine or batch size. Batch inference computes each row with
// exactly the same code path and order as single inference.
package kernel

import (
	"math"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// Prediction is the fully explained result of one inference.
type Prediction struct {
	ModelID      string
	ModelVersion string
	// Precision is the precision actually used for arithmetic (the model's
	// weight precision). It is always reported so callers can verify they
	// never received results from an unexpected precision.
	Precision  model.Precision
	LabelIndex int
	Label      string
	Logits     []float64
	Probs      []float64
	Confidence float64
}

// Infer runs one deterministic forward pass.
func Infer(m *model.LoadedModel, input []float64) (Prediction, error) {
	if err := validateInput(m, input); err != nil {
		return Prediction{}, err
	}

	act := make([]float64, m.Manifest.InputDim)
	copy(act, input)

	for li := range m.Layers {
		act = forwardLayer(m, li, act)
	}
	logits := act
	probs := softmax(logits)
	idx := argmax(logits)

	return Prediction{
		ModelID:      m.Manifest.ModelID,
		ModelVersion: m.Manifest.ModelVersion,
		Precision:    modelWeightPrecision(m),
		LabelIndex:   idx,
		Label:        m.Manifest.OutputLabels[idx],
		Logits:       logits,
		Probs:        probs,
		Confidence:   probs[idx],
	}, nil
}

// InferBatch runs a batch of independent forward passes in input order. It
// first validates every row, then computes rows strictly in index order using
// Infer. No per-batch buffers are shared between rows.
func InferBatch(m *model.LoadedModel, inputs [][]float64) ([]Prediction, error) {
	if len(inputs) == 0 {
		return nil, merrors.New(merrors.ReasonInputShape, "empty inference batch")
	}
	for i, in := range inputs {
		if err := validateInput(m, in); err != nil {
			return nil, merrors.New(merrors.ReasonInputShape,
				"batch row %d invalid: %v", i, err)
		}
	}
	out := make([]Prediction, len(inputs))
	for i, in := range inputs {
		p, err := Infer(m, in)
		if err != nil {
			return nil, err
		}
		out[i] = p
	}
	return out, nil
}

func validateInput(m *model.LoadedModel, input []float64) error {
	if len(input) != m.Manifest.InputDim {
		return merrors.New(merrors.ReasonInputShape,
			"input has %d features, model %q expects %d",
			len(input), m.Manifest.ModelID, m.Manifest.InputDim)
	}
	for i, v := range input {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return merrors.New(merrors.ReasonInputValue,
				"input[%d] is not finite (%v)", i, v)
		}
	}
	return nil
}

// forwardLayer computes: out_r = bias_r + sum_c dequant(w[r,c]) * act_c for
// every output row r, with hidden activations passed through ReLU. The input
// is quantized per the manifest and weights dequantized per layer, simulating
// true on-device integer execution while remaining fully deterministic.
func forwardLayer(m *model.LoadedModel, layerIdx int, act []float64) []float64 {
	layer := &m.Layers[layerIdx]
	spec := layer.Spec.Weights
	rows, cols := spec.Rows, spec.Cols
	out := make([]float64, rows)

	// Quantize activations into the input precision (input precision for
	// layer 0, previous layer's weight precision otherwise), with clamping
	// to the representable range.
	qMin, qMax := quantRange(layerIdx, m)
	prevScale, prevZero := inputQuant(m)
	if layerIdx > 0 {
		ps := m.Layers[layerIdx-1].Spec.Weights
		prevScale, prevZero = ps.Scale, ps.ZeroPoint
	}
	qAct := make([]int, cols)
	for c := 0; c < cols; c++ {
		q := math.Round(act[c]/prevScale) + float64(prevZero)
		if q < float64(qMin) {
			q = float64(qMin)
		}
		if q > float64(qMax) {
			q = float64(qMax)
		}
		qAct[c] = int(q)
	}

	for r := 0; r < rows; r++ {
		acc := layer.Bias[r]
		for c := 0; c < cols; c++ {
			acc += dequantWeight(layer, r*cols+c, spec) *
				(float64(qAct[c]-prevZero) * prevScale)
		}
		out[r] = acc
	}

	// ReLU on every layer except the final one, whose rows are logits.
	if layerIdx < len(m.Layers)-1 {
		for r := range out {
			if out[r] < 0 {
				out[r] = 0
			}
		}
	}
	return out
}

func inputQuant(m *model.LoadedModel) (scale float64, zero int) {
	return m.Manifest.InputScale, m.Manifest.InputZeroPoint
}

func quantRange(layerIdx int, m *model.LoadedModel) (min, max int) {
	p := m.Manifest.InputPrecision
	if layerIdx > 0 {
		p = m.Layers[layerIdx-1].Spec.Weights.Precision
	}
	switch p {
	case model.PrecisionINT8:
		return -128, 127
	default:
		return -8, 7
	}
}

// dequantWeight reads quantized element idx and returns
// (q - zero_point) * scale.
func dequantWeight(layer *model.Layer, idx int, spec model.TensorSpec) float64 {
	var q int
	switch spec.Precision {
	case model.PrecisionINT8:
		q = int(int8(layer.Weight[idx]))
	case model.PrecisionINT4:
		b := layer.Weight[idx/2]
		var raw byte
		if idx%2 == 0 {
			raw = b & 0x0f
		} else {
			raw = (b >> 4) & 0x0f
		}
		if raw > 7 { // sign-extend 4-bit two's complement
			q = int(raw) - 16
		} else {
			q = int(raw)
		}
	}
	return float64(q-spec.ZeroPoint) * spec.Scale
}

// softmax computes probabilities in output index order with a numerically
// stable shift; ties in logits do not affect its deterministic result.
func softmax(x []float64) []float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	p := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		p[i] = math.Exp(v - max)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// argmax returns the smallest index holding the maximum value, so repeated or
// tied runs are deterministic.
func argmax(x []float64) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func modelWeightPrecision(m *model.LoadedModel) model.Precision {
	return m.Layers[0].Spec.Weights.Precision
}
