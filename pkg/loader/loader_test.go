package loader_test

import (
	"path/filepath"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/loader"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// validOpts builds a small int8 2->2->2 classifier whose second output logit
// dominates for positive inputs.
func validOpts() testutil.ManifestOptions {
	return testutil.ManifestOptions{
		ModelID:         "classifier-int8",
		ModelVersion:    "1.0.0",
		InputPrecision:  model.PrecisionINT8,
		WeightPrecision: model.PrecisionINT8,
		InputDim:        2,
		Hidden:          2,
		Outputs:         2,
		W0:              [][]int{{10, 0}, {0, 10}},
		W1:              [][]int{{10, 0}, {0, 10}},
		B0:              []float64{0, 0},
		B1:              []float64{0, 0},
		Scales:          [2]float64{0.1, 0.1},
		ZeroPoints:      [2]int{0, 0},
		InputScale:      0.1,
		InputZero:       0,
	}
}

func TestLoad_ValidINT8(t *testing.T) {
	b := testutil.WriteTwoLayer(t, "ok-int8", validOpts())
	m, err := loader.Load(b.Dir, loader.Options{})
	if err != nil {
		t.Fatalf("expected valid package to load, got %v", err)
	}
	if got := m.Manifest.ModelID; got != "classifier-int8" {
		t.Fatalf("model id = %q", got)
	}
	if len(m.Layers) != 2 {
		t.Fatalf("want 2 fully materialized layers, got %d (partial load forbidden)", len(m.Layers))
	}
	t.Logf("判定依据: 模型 %s 两层权重字节=%d 均已加载，精度=%s，尺寸=%dx%d/%dx%d",
		m.Manifest.ModelID, m.WeightBytes,
		m.Layers[0].Spec.Weights.Precision,
		m.Layers[0].Spec.Weights.Rows, m.Layers[0].Spec.Weights.Cols,
		m.Layers[1].Spec.Weights.Rows, m.Layers[1].Spec.Weights.Cols)
}

func TestLoad_ValidINT4(t *testing.T) {
	o := validOpts()
	o.ModelID = "classifier-int4"
	o.WeightPrecision = model.PrecisionINT4
	o.W0 = [][]int{{7, 0}, {0, 7}}
	b := testutil.WriteTwoLayer(t, "ok-int4", o)
	m, err := loader.Load(b.Dir, loader.Options{})
	if err != nil {
		t.Fatalf("int4 weights are supported by %s: %v", model.FormatV1, err)
	}
	p := m.Layers[0].Spec.Weights.Precision
	if p != model.PrecisionINT4 {
		t.Fatalf("精度必须按声明保留为 int4，实际 %q（不得静默降级）", p)
	}
	t.Logf("判定依据: 输入精度 %s + 权重精度 %s 是受支持组合；2x2 int4 权重占用 %d 字节（每字节两个 nibble）",
		m.Manifest.InputPrecision, p, m.Layers[0].Spec.Weights.PackedSize())
}

func TestLoad_RejectReasons(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(m *model.Manifest)
		opts   func(o *testutil.ManifestOptions)
		tamper func(dir string)
		want   merrors.Reason
	}{
		{
			name: "unsupported version",
			mutate: func(m *model.Manifest) {
				m.FormatVersion = "edge-qmodel/v9"
			},
			want: merrors.ReasonUnsupportedVersion,
		},
		{
			name: "unsupported weight precision fp16",
			opts: func(o *testutil.ManifestOptions) {
				o.ModelID = "fp16-model"
			},
			mutate: func(m *model.Manifest) {
				m.Layers[0].Weights.Precision = "fp16"
			},
			want: merrors.ReasonUnsupportedPrecision,
		},
		{
			name: "unsupported input precision",
			mutate: func(m *model.Manifest) {
				m.InputPrecision = "int2"
			},
			want: merrors.ReasonUnsupportedPrecision,
		},
		{
			name: "missing weight scale",
			mutate: func(m *model.Manifest) {
				m.Layers[0].Weights.Scale = 0
			},
			want: merrors.ReasonMissingQuantization,
		},
		{
			name: "missing input scale",
			mutate: func(m *model.Manifest) {
				m.InputScale = 0
			},
			want: merrors.ReasonMissingQuantization,
		},
		{
			name: "zero point out of range int8",
			mutate: func(m *model.Manifest) {
				m.Layers[0].Weights.ZeroPoint = 200
			},
			want: merrors.ReasonMissingQuantization,
		},
		{
			name: "layer chain dimension mismatch",
			mutate: func(m *model.Manifest) {
				// fc1 expects hidden=2 inputs; claim 3 columns.
				m.Layers[1].Weights.Cols = 3
			},
			want: merrors.ReasonDimensionMismatch,
		},
		{
			name: "input dim mismatch",
			mutate: func(m *model.Manifest) {
				m.InputDim = 3
			},
			want: merrors.ReasonDimensionMismatch,
		},
		{
			name: "output labels mismatch",
			mutate: func(m *model.Manifest) {
				m.OutputLabels = []string{"only-one"}
			},
			want: merrors.ReasonDimensionMismatch,
		},
		{
			name: "bias length mismatch",
			mutate: func(m *model.Manifest) {
				m.Layers[0].Bias = []float64{1}
			},
			want: merrors.ReasonDimensionMismatch,
		},
		{
			name:   "corrupt weights fail checksum",
			tamper: func(dir string) { tamperByte(t, filepath.Join(dir, packageio.WeightsName), 0) },
			want:   merrors.ReasonWeightCorrupt,
		},
		{
			name:   "corrupt bias fails checksum",
			tamper: func(dir string) { tamperByte(t, filepath.Join(dir, packageio.BiasName), 0) },
			want:   merrors.ReasonWeightCorrupt,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			o := validOpts()
			if tc.opts != nil {
				tc.opts(&o)
			}
			b := testutil.WriteTwoLayer(t, "case", o)
			if tc.mutate != nil {
				tc.mutate(b.Manifest)
				rewriteManifest(t, b.Dir, b.Manifest)
			}
			if tc.tamper != nil {
				tc.tamper(b.Dir)
			}
			_, err := loader.Load(b.Dir, loader.Options{})
			if err == nil {
				t.Fatalf("输入: 用例=%q；期望加载阶段以原因 %q 拒绝，却加载成功", tc.name, tc.want)
			}
			got := merrors.ReasonOf(err)
			if got != tc.want {
				t.Fatalf("输入: 用例=%q；判定依据: 错误=%v，期望原因=%q 实际=%q（原因必须可区分）",
					tc.name, err, tc.want, got)
			}
			t.Logf("输入: 用例=%q；判定依据: 加载被拒绝，原因=%q（%v）——不发生部分加载或精度降级",
				tc.name, got, err)
		})
	}
}

func TestLoad_DeviceBudgetRejectedAtLoad(t *testing.T) {
	b := testutil.WriteTwoLayer(t, "budget", validOpts())
	// First learn the true footprint, then demand a budget one byte smaller.
	full, err := loader.Load(b.Dir, loader.Options{})
	if err != nil {
		t.Fatal(err)
	}
	need := loader.ModelMemoryBytes(full)

	_, err = loader.Load(b.Dir, loader.Options{DeviceMemoryBytes: need - 1})
	if got := merrors.ReasonOf(err); got != merrors.ReasonDeviceBudgetExceeded {
		t.Fatalf("输入: 预算=%d 需要=%d；期望原因 %q，实际 %q (%v)",
			need-1, need, merrors.ReasonDeviceBudgetExceeded, got, err)
	}
	t.Logf("输入: 设备预算=%d 模型需求=%d；判定依据: 预算不足在加载阶段拒绝（device_budget_exceeded）",
		need-1, need)

	m, err := loader.Load(b.Dir, loader.Options{DeviceMemoryBytes: need})
	if err != nil {
		t.Fatalf("预算恰好等于需求时应成功: %v", err)
	}
	t.Logf("判定依据: 预算=%d == 需求=%d，加载成功，占用字节=%d", need, need, loader.ModelMemoryBytes(m))
}

func TestLoad_MissingDirectory(t *testing.T) {
	_, err := loader.Load(filepath.Join(t.TempDir(), "does-not-exist"), loader.Options{})
	if got := merrors.ReasonOf(err); got != merrors.ReasonPackageNotFound {
		t.Fatalf("want package_not_found, got %q (%v)", got, err)
	}
}
