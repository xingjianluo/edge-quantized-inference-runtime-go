package runtime

import (
	"context"
	"math"
	"testing"
)

// TestInfer_InvalidNonFiniteInputRejected 对 NaN / +Inf / -Inf 这类本身
// 得不到有意义结果的输入，必须以 ErrInvalidInput 明确拒绝，而不是静默饱和
// 成量程端点、产出“看起来正常”的数值。与维度错误 ErrInputDimMismatch 可区分。
func TestInfer_InvalidNonFiniteInputRejected(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	s, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	nan := float32(math.NaN())
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	cases := []struct {
		name  string
		input []float32
	}{
		{"第二个值是NaN", []float32{0.2, nan}},
		{"第一个值是正无穷", []float32{posInf, -0.3}},
		{"第二个值是负无穷", []float32{0.1, negInf}},
		{"两个值都是NaN", []float32{nan, nan}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := s.Infer(context.Background(), tc.input)
			if !Is(err, ErrInvalidInput) {
				t.Fatalf("输入=%v 期望 ErrInvalidInput，实际 out=%v err=%v",
					tc.input, out, err)
			}
			if out != nil {
				t.Fatalf("异常输入不得返回任何结果，实际 %v", out)
			}
			// 批量路径同样拒绝（逐条计算前即报错），不返回部分结果。
			batch, berr := s.InferBatch(context.Background(),
				[][]float32{{0.1, 0.2}, tc.input, {0.3, 0.4}})
			if !Is(berr, ErrInvalidInput) {
				t.Fatalf("批量中异常输入应报 ErrInvalidInput，实际 %v", berr)
			}
			if batch != nil {
				t.Fatalf("异常批次不得返回部分结果，实际 %v", batch)
			}
			t.Logf("场景=%s 输入=%v；判定依据: errors.Is(ErrInvalidInput)=true 且不返回结果",
				tc.name, tc.input)
		})
	}

	// 拒绝异常输入不影响后续正常输入：同一会话紧接着推理正常输入应成功且可复现。
	good := []float32{0.2, -0.3}
	want := mustInfer(t, s, good)
	for i := 0; i < 3; i++ {
		if got := mustInfer(t, s, good); !equalF32s(got, want) {
			t.Fatalf("拒绝异常输入后正常输入结果不稳定: got=%v want=%v", got, want)
		}
	}
	t.Logf("判定依据: 拒绝 NaN/Inf 后同会话正常输入=%v 输出=%v 三次可复现", good, want)

	// 维度错误与值错误必须可区分。
	if _, err := s.Infer(context.Background(), []float32{1, 2, 3}); !Is(err, ErrInputDimMismatch) {
		t.Fatalf("维度错误应仍为 ErrInputDimMismatch，实际 %v", err)
	}
	t.Logf("判定依据: 维度错误=ErrInputDimMismatch，值无意义=ErrInvalidInput，两类原因可区分")
}

func equalF32s(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
