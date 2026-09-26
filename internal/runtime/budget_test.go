package runtime

import (
	"context"
	"testing"
)

func TestLoadModel_DeviceBudgetRejectsAndKeepsState(t *testing.T) {
	pkg := toyPkg(t)
	// 预算小于模型权重：连模型都放不下。
	rt := newTestRuntime(t, pkg.Bytes()-1)
	if err := rt.LoadModel("toy", pkg); !Is(err, ErrInsufficientResources) {
		t.Fatalf("期望设备预算不足 ErrInsufficientResources，实际 %v", err)
	}
	t.Logf("判定依据: 模型权重=%d字节 > 设备预算=%d字节 -> 拒绝，used=%d（状态不变）",
		pkg.Bytes(), pkg.Bytes()-1, rt.UsedBytes())
	if rt.UsedBytes() != 0 {
		t.Fatalf("被拒绝后设备占用应为0，实际 %d", rt.UsedBytes())
	}
	// 被拒绝的模型必须不可见、不能用于建会话。
	if _, err := rt.NewSession("toy", SessionOptions{}); !Is(err, ErrUnknownModel) {
		t.Fatalf("被拒绝的模型不应可见，实际 %v", err)
	}
}

func TestSession_DeviceAndSessionBudget(t *testing.T) {
	pkg := toyPkg(t)
	// 预算只够模型 + 一条工作区。
	rt := newTestRuntime(t, pkg.Bytes()+workspaceBytes(buildView(pkg), 1))
	loadToy(t, rt, "toy")

	// 第一个会话占用唯一一份工作区后，第二个会话因设备预算失败。
	s1, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("第一个会话应成功: %v", err)
	}
	defer s1.Close()
	if _, err := rt.NewSession("toy", SessionOptions{}); !Is(err, ErrInsufficientResources) {
		t.Fatalf("设备内存耗尽时第二个会话应失败，实际 %v", err)
	}
	t.Logf("判定依据: 设备仅剩1份工作区；第二会话创建 -> ErrInsufficientResources，第一会话仍可用")

	// 第一会话仍正常工作，证明失败只影响被拒绝者。
	out := mustInfer(t, s1, []float32{0.2, -0.3})
	t.Logf("输入=[0.2 -0.3] 幸存会话输出=%v", out)

	// 关闭后预算归还，可再建会话，证明干净释放、无后续加载失败。
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}
	s2, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("会话关闭归还预算后应能再建会话: %v", err)
	}
	defer s2.Close()
}

func TestSession_BatchBudgetRejectsOversize(t *testing.T) {
	pkg := toyPkg(t)
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	view := buildView(pkg)
	one := workspaceBytes(view, 1)

	// 会话预算只允许 2 条。
	s, err := rt.NewSession("toy", SessionOptions{MaxBatchBytes: 2 * one})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	inputs := make([][]float32, 3)
	for i := range inputs {
		inputs[i] = []float32{0.1, 0.1}
	}
	if _, err := s.InferBatch(context.Background(), inputs); !Is(err, ErrSessionBudgetExceeded) {
		t.Fatalf("超会话批次预算应报 ErrSessionBudgetExceeded，实际 %v", err)
	}
	t.Logf("判定依据: 3条工作区=%d > 会话预算=%d -> ErrSessionBudgetExceeded",
		workspaceBytes(view, 3), 2*one)

	// 超预算只影响这一批：同一会话在预算内仍可正常推理。
	ok, err := s.InferBatch(context.Background(), inputs[:2])
	if err != nil || len(ok) != 2 {
		t.Fatalf("预算内批次应成功，got len=%d err=%v", len(ok), err)
	}
}

func TestUnloadModel_RejectsWhenSessionActive(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	s, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := rt.UnloadModel("toy"); err == nil {
		t.Fatal("存在活跃会话时不应允许卸载模型")
	}
	s.Close()
	if err := rt.UnloadModel("toy"); err != nil {
		t.Fatalf("会话关闭后卸载应成功: %v", err)
	}
	if _, err := rt.NewSession("toy", SessionOptions{}); !Is(err, ErrUnknownModel) {
		t.Fatalf("卸载后建会话应 ErrUnknownModel，实际 %v", err)
	}
	t.Logf("判定依据: 活跃会话时卸载被拒；会话关闭后卸载成功且模型不再可见")
}
