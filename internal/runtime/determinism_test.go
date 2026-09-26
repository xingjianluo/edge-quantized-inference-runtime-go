package runtime

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

var toyInputs = [][]float32{
	{0.1, -0.5},
	{1.25, 0.0},
	{-0.75, 0.5},
	{2.0, -2.0},
}

func TestInfer_DeterministicAndBatchConsistent(t *testing.T) {
	const budget = 1 << 30
	rt := newTestRuntime(t, budget)
	pkg := loadToy(t, rt, "toy")
	view := buildView(pkg)
	one := workspaceBytes(view, 1)
	t.Logf("判定依据: 输入维度=%d 输出维度=%d 单条工作区=%d字节 模型权重=%d字节",
		view.inputDim, view.layers[len(view.layers)-1].outDim, one, pkg.Bytes())

	s1, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	defer s1.Close()
	s2, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("建第二个会话失败: %v", err)
	}
	defer s2.Close()

	// 基准：s1 逐条推理两遍。
	var first, second [][]float32
	for _, in := range toyInputs {
		first = append(first, mustInfer(t, s1, in))
	}
	for _, in := range toyInputs {
		second = append(second, mustInfer(t, s1, in))
	}
	for i := range toyInputs {
		t.Logf("输入=%v 输出=%v", []float32(toyInputs[i]), first[i])
		if !reflect.DeepEqual(first[i], second[i]) {
			t.Fatalf("重复执行不一致: input=%v first=%v second=%v",
				toyInputs[i], first[i], second[i])
		}
	}

	// 批量：必须与逐条逐位一致。
	batchOut, err := s1.InferBatch(context.Background(), toyInputs)
	if err != nil {
		t.Fatalf("批量推理失败: %v", err)
	}
	for i := range toyInputs {
		if !reflect.DeepEqual(batchOut[i], first[i]) {
			t.Fatalf("批量与逐条不一致: input=%v batch=%v single=%v",
				toyInputs[i], batchOut[i], first[i])
		}
	}

	// 另一会话结果必须一致（无可变共享状态）。
	for i, in := range toyInputs {
		got := mustInfer(t, s2, in)
		if !reflect.DeepEqual(got, first[i]) {
			t.Fatalf("跨会话结果不一致: input=%v s2=%v s1=%v", in, got, first[i])
		}
	}
	t.Logf("判定依据: 逐条两遍 + 批量 + 独立会话 三者结果全部 DeepEqual，结论一致可复现")
}

func TestInfer_InputDimMismatch(t *testing.T) {
	rt := newTestRuntime(t, 1<<20)
	loadToy(t, rt, "toy")
	s, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	bad := []float32{0.1, 0.2, 0.3}
	if _, err := s.Infer(context.Background(), bad); !Is(err, ErrInputDimMismatch) {
		t.Fatalf("输入=%v 期望 ErrInputDimMismatch，实际 %v", bad, err)
	}
	// 维度错误必须在批量路径上对每条同样立即报错，不得算到中途。
	if _, err := s.InferBatch(context.Background(), [][]float32{{0, 0}, bad}); !Is(err, ErrInputDimMismatch) {
		t.Fatalf("批量中维度错误应立即报 ErrInputDimMismatch，实际 %v", err)
	}
	t.Logf("判定依据: 输入=%v 长度3!=模型2，errors.Is(ErrInputDimMismatch)=true，批量第2条即拒绝", bad)
}

func TestPrecision_ProducesDistinctResults(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)

	int8Pkg := testutil.LoadToy(t, testutil.DefaultToy())
	uint8Spec := testutil.DefaultToy()
	uint8Spec.Precision = "uint8"
	uint8Spec.LayerZP = 128
	uint8Pkg := testutil.LoadToy(t, uint8Spec)

	if err := rt.LoadModel("m_i8", int8Pkg); err != nil {
		t.Fatal(err)
	}
	if err := rt.LoadModel("m_u8", uint8Pkg); err != nil {
		t.Fatal(err)
	}
	si, _ := rt.NewSession("m_i8", SessionOptions{})
	defer si.Close()
	su, _ := rt.NewSession("m_u8", SessionOptions{})
	defer su.Close()

	in := []float32{0.5, -0.25}
	oi := mustInfer(t, si, in)
	ou := mustInfer(t, su, in)
	t.Logf("判定依据: 相同输入=%v int8输出=%v uint8输出=%v（精度不同，数值可区分）", in, oi, ou)
	if reflect.DeepEqual(oi, ou) {
		t.Fatalf("不同精度模型不应得到完全相同输出: %v vs %v", oi, ou)
	}
	if fmt.Sprint(oi) == "" {
		t.Fatal("空输出")
	}
}
