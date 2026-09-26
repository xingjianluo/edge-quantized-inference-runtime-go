package runtime

import (
	"context"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// toyPkg 返回走完整加载校验的合法玩具模型包。
func toyPkg(t *testing.T) *modelpkg.Package {
	t.Helper()
	return testutil.LoadToy(t, testutil.DefaultToy())
}

// newTestRuntime 构造一个给定预算的运行时并在测试结束时关闭。
func newTestRuntime(t *testing.T, budget int64) *Runtime {
	t.Helper()
	rt, err := New(Config{MaxDeviceBytes: budget})
	if err != nil {
		t.Fatalf("创建运行时失败: %v", err)
	}
	t.Cleanup(func() { _ = rt.Close() })
	return rt
}

// loadToy 加载玩具模型到运行时。
func loadToy(t *testing.T, rt *Runtime, name string) *modelpkg.Package {
	t.Helper()
	pkg := toyPkg(t)
	if err := rt.LoadModel(name, pkg); err != nil {
		t.Fatalf("加载模型 %s 失败: %v", name, err)
	}
	return pkg
}

// mustInfer 执行一次推理，失败即终止（用于期望成功的路径）。
func mustInfer(t *testing.T, s *Session, in []float32) []float32 {
	t.Helper()
	out, err := s.Infer(context.Background(), in)
	if err != nil {
		t.Fatalf("推理失败: input=%v err=%v", in, err)
	}
	return out
}
