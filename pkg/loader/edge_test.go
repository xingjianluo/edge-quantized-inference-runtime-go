package loader_test

import (
	"testing"
	"time"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/packageio"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/loader"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// TestLoad_INT4OddShapeAndTrailingBytes covers the int4 nibble-packing edge:
// a layer with an odd element count carries one padding nibble, and a blob
// with unexplained extra bytes must be rejected as corrupt.
func TestLoad_INT4OddShapeAndTrailingBytes(t *testing.T) {
	// fc0: 3 outputs x 1 input = 3 int4 elements -> 2 bytes (one padding nibble)
	// fc1: 2 outputs x 3 inputs (int8) -> 6 bytes.
	layers := []model.LayerSpec{
		{
			Name: "fc0",
			Weights: model.TensorSpec{
				Name: "fc0/w", Precision: model.PrecisionINT4,
				Rows: 3, Cols: 1, ZeroPoint: 0, Scale: 0.5,
			},
			Bias: []float64{0, 0, 0},
		},
		{
			Name: "fc1",
			Weights: model.TensorSpec{
				Name: "fc1/w", Precision: model.PrecisionINT8,
				Rows: 2, Cols: 3, ZeroPoint: 0, Scale: 0.5,
			},
			Bias: []float64{0, 0},
		},
	}
	m := &model.Manifest{
		FormatVersion:  model.FormatV1,
		ModelID:        "odd-int4",
		ModelVersion:   "2.0.0",
		InputPrecision: model.PrecisionINT8,
		InputZeroPoint: 0,
		InputScale:     1,
		InputDim:       1,
		OutputLabels:   []string{"x", "y"},
		Layers:         layers,
		CreatedAt:      time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC),
	}
	q := [][]int{{-8, -1, 7}, {1, 2, 3, 4, 5, 6}}
	weights, err := packageio.EncodeWeights(layers, q)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights) != 2+6 {
		t.Fatalf("3 个 int4 应打包为 2 字节，权重总长应为 8，实际 %d", len(weights))
	}
	bias, err := packageio.EncodeBias(layers)
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	if err := packageio.WritePackage(dir, m, weights, bias); err != nil {
		t.Fatal(err)
	}
	loaded, err := loader.Load(dir, loader.Options{})
	if err != nil {
		t.Fatalf("奇数 int4 元素 + 零填充 nibble 应正常加载: %v", err)
	}
	t.Logf("判定依据: int4 层 3x1 打包为 %d 字节且末字节高 nibble=0，加载成功，层 0 权重字节=%v",
		2, loaded.Layers[0].Weight)

	// Append an unexplained byte: geometry no longer matches -> weight_corrupt.
	if err := appendByte(packageio.WeightsName, dir, weights, 0xAA); err != nil {
		t.Fatal(err)
	}
	rewriteManifest(t, dir, m) // checksums still describe the 8-byte blob
	// Rebuild blob file with trailing byte but keep manifest checksum stale.
	_, err = loader.Load(dir, loader.Options{})
	if got := merrors.ReasonOf(err); got != merrors.ReasonWeightCorrupt {
		t.Fatalf("权重 blob 多出 1 字节必须 weight_corrupt，实际 %q (%v)", got, err)
	}
	t.Logf("判定依据: 权重 9 字节 != 清单几何所需 8 字节，加载以 weight_corrupt 拒绝（%v）", err)
}

// TestLoad_MissingChecksums covers the "checksum field blank" rejection path.
func TestLoad_MissingChecksums(t *testing.T) {
	b := testutil.WriteTwoLayer(t, "no-checksum", validOpts())
	b.Manifest.WeightsSHA256 = ""
	rewriteManifest(t, b.Dir, b.Manifest)
	_, err := loader.Load(b.Dir, loader.Options{})
	if got := merrors.ReasonOf(err); got != merrors.ReasonWeightCorrupt {
		t.Fatalf("缺失权重校验和必须 weight_corrupt，实际 %q (%v)", got, err)
	}
	t.Logf("判定依据: 清单中 weights_sha256 为空，拒绝原因 weight_corrupt（%v）", err)
}
