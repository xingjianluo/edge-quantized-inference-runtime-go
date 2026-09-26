package kernel_test

import (
	"math"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/kernel"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/loader"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

func loadModel(t *testing.T, b testutil.Bundle) *model.LoadedModel {
	t.Helper()
	m, err := loader.Load(b.Dir, loader.Options{})
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	return m
}

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

func TestInfer_DeterministicClasses(t *testing.T) {
	m := loadModel(t, testutil.WriteTwoLayer(t, "kernel-int8", validOpts()))

	inputs := [][]float64{{1, 0}, {0, 1}, {0.9, 0.1}, {0.1, 0.9}}
	wantLabels := []string{"a", "b", "a", "b"}
	for i, in := range inputs {
		p, err := kernel.Infer(m, in)
		if err != nil {
			t.Fatalf("输入=%v: %v", in, err)
		}
		t.Logf("输入=%v 判定依据: logits=%v probs=%.6f 选中=%s(下标%d) 精度=%s 期望=%s",
			in, roundAll(p.Logits), p.Probs, p.Label, p.LabelIndex, p.Precision, wantLabels[i])
		if p.Label != wantLabels[i] {
			t.Fatalf("输入=%v 得到 %s，期望 %s", in, p.Label, wantLabels[i])
		}
		if p.Precision != model.PrecisionINT8 {
			t.Fatalf("结果必须如实报告实际权重精度 int8，实际 %q", p.Precision)
		}
	}
}

func TestInfer_BatchEqualsOneByOne(t *testing.T) {
	m := loadModel(t, testutil.WriteTwoLayer(t, "batch-consistency", validOpts()))
	inputs := [][]float64{{1, 0}, {0, 1}, {0.7, 0.3}, {0.2, 0.8}, {0.5, 0.5}}

	batch, err := kernel.InferBatch(m, inputs)
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	for i, in := range inputs {
		single, err := kernel.Infer(m, in)
		if err != nil {
			t.Fatalf("single row %d: %v", i, err)
		}
		if !samePrediction(single, batch[i]) {
			t.Fatalf("输入=%v: 批量结果 %+v 与逐条结果 %+v 不一致", in, batch[i], single)
		}
		t.Logf("输入=%v 判定依据: 批量与逐条 label=%s logits=%v confidence=%.12f 完全一致",
			in, single.Label, roundAll(single.Logits), single.Confidence)
	}
}

func TestInfer_RepeatableAcrossReloads(t *testing.T) {
	b := testutil.WriteTwoLayer(t, "repeatable", validOpts())
	in := []float64{0.67, 0.33}

	m1 := loadModel(t, b)
	m2 := loadModel(t, b)
	p1, _ := kernel.Infer(m1, in)
	p2, _ := kernel.Infer(m2, in)
	// Repeated calls on the same model.
	p3, _ := kernel.Infer(m1, in)
	if !samePrediction(p1, p2) || !samePrediction(p1, p3) {
		t.Fatalf("重复执行/重新加载结果必须一致: %+v vs %+v vs %+v", p1, p2, p3)
	}
	t.Logf("输入=%v 判定依据: 三次执行（含重新加载）label=%s logits=%v confidence=%.15f 位级一致",
		in, p1.Label, roundAll(p1.Logits), p1.Confidence)
}

func TestInfer_INT4RunsAndReportsPrecision(t *testing.T) {
	o := validOpts()
	o.ModelID = "classifier-int4"
	o.WeightPrecision = model.PrecisionINT4
	o.W0 = [][]int{{7, 0}, {0, 7}}
	m := loadModel(t, testutil.WriteTwoLayer(t, "kernel-int4", o))

	p, err := kernel.Infer(m, []float64{1, 0})
	if err != nil {
		t.Fatalf("int4 推理失败: %v", err)
	}
	if p.Precision != model.PrecisionINT4 {
		t.Fatalf("int4 模型的结果必须报告 int4，实际 %q（不得返回错误精度数值）", p.Precision)
	}
	t.Logf("输入=[1 0] 判定依据: int4 模型输出 label=%s logits=%v 精度=%s",
		p.Label, roundAll(p.Logits), p.Precision)
}

func TestInfer_BadInputRejectedImmediately(t *testing.T) {
	m := loadModel(t, testutil.WriteTwoLayer(t, "bad-input", validOpts()))

	if _, err := kernel.Infer(m, []float64{1}); merrors.ReasonOf(err) != merrors.ReasonInputShape {
		t.Fatalf("输入=[1] 维度错误必须立即以 input_shape 拒绝，实际 %v", err)
	}
	t.Logf("输入=[1] 判定依据: 期望维度=2 实际维度=1，推理开始前拒绝（input_shape）")

	if _, err := kernel.Infer(m, []float64{math.NaN(), 0}); merrors.ReasonOf(err) != merrors.ReasonInputValue {
		t.Fatalf("NaN 输入必须以 input_value 拒绝，实际 %v", err)
	}
	t.Logf("输入=[NaN 0] 判定依据: 非有限值在推理开始前拒绝（input_value），不会产生伪数值")

	if _, err := kernel.InferBatch(m, [][]float64{{1, 0}, {0}}); merrors.ReasonOf(err) != merrors.ReasonInputShape {
		t.Fatalf("批量中第二行维度错误必须在任何计算前拒绝，实际 %v", err)
	}
	t.Logf("输入批量第2行=[0] 判定依据: 整批先校验，任一非法则不执行任何推理（input_shape）")
}

func samePrediction(a, b kernel.Prediction) bool {
	if a.Label != b.Label || a.LabelIndex != b.LabelIndex ||
		a.Precision != b.Precision || a.Confidence != b.Confidence ||
		len(a.Logits) != len(b.Logits) || len(a.Probs) != len(b.Probs) {
		return false
	}
	for i := range a.Logits {
		if a.Logits[i] != b.Logits[i] || a.Probs[i] != b.Probs[i] {
			return false
		}
	}
	return true
}

func roundAll(xs []float64) []float64 {
	out := make([]float64, len(xs))
	scale := math.Pow(10, 6)
	for i, v := range xs {
		out[i] = math.Round(v*scale) / scale
	}
	return out
}
