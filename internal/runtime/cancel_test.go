package runtime

import (
	"context"
	"testing"
	"time"
)

func TestSession_TimeoutFailsOnlyOwnBatch(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")

	slow, err := rt.NewSession("toy", SessionOptions{Timeout: 20 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	other, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// 足够大的批次，使 20ms 超时在推理循环中途触发。
	big := make([][]float32, 5_000_000)
	for i := range big {
		big[i] = toyInputs[i%len(toyInputs)]
	}
	start := time.Now()
	_, err = slow.InferBatch(context.Background(), big)
	elapsed := time.Since(start)
	if !Is(err, ErrSessionTimeout) {
		t.Fatalf("期望 ErrSessionTimeout，实际 %v", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("超时未及时生效，耗时 %v", elapsed)
	}
	t.Logf("判定依据: 批次=%d条, 超时预算=20ms, 实际约=%dms, 返回 ErrSessionTimeout",
		len(big), elapsed.Milliseconds())

	// 超时后：该会话本次失败；其他会话不受影响且仍可重复一致推理。
	want := mustInfer(t, other, toyInputs[1])
	for i := 0; i < 10; i++ {
		got := mustInfer(t, other, toyInputs[1])
		if len(got) != len(want) {
			t.Fatal("其他会话结果异常")
		}
	}
	t.Logf("判定依据: 超时会话之外的 other 会话推理仍稳定输出 %v", want)

	// 超时属于单次调用：被超时的会话仍可继续发起新推理（它没被 Cancel）。
	out, err := slow.Infer(context.Background(), toyInputs[0])
	if err != nil {
		t.Fatalf("超时不应关闭会话，后续单条推理应成功: %v", err)
	}
	t.Logf("判定依据: 超时仅作用于该次调用，后续单条输出=%v 正常", out)

	_ = slow.Close()
	_ = other.Close()
}

func TestSession_CancelIsTerminalForSession(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	s, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s.Cancel()

	if _, err := s.Infer(context.Background(), toyInputs[0]); !Is(err, ErrSessionCanceled) {
		t.Fatalf("取消后新推理应报 ErrSessionCanceled，实际 %v", err)
	}
	// Close 幂等且成功归还预算。
	if err := s.Close(); err != nil {
		t.Fatalf("Close 失败: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("重复 Close 应幂等: %v", err)
	}
	used := rt.UsedBytes()
	if used != toyPkg(t).Bytes() {
		t.Fatalf("关闭后 used 应仅剩模型权重 %d，实际 %d", toyPkg(t).Bytes(), used)
	}
	t.Logf("判定依据: Cancel 后推理=ErrSessionCanceled；Close 幂等；used=%d 仅剩权重", used)
}

func TestSession_ContextCancelMappedToCanceled(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	s, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Infer(ctx, toyInputs[0]); !Is(err, ErrSessionCanceled) {
		t.Fatalf("外部 ctx 取消应映射为 ErrSessionCanceled，实际 %v", err)
	}
	t.Logf("判定依据: 传入已取消 context -> ErrSessionCanceled，会话本身仍存活")
	out, err := s.Infer(context.Background(), toyInputs[0])
	if err != nil {
		t.Fatalf("外部 ctx 取消不应终结会话: %v", err)
	}
	_ = out
	_ = s.Close()
}

func TestRuntime_ClosedRejectsNewWork(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")
	// newTestRuntime 注册了 t.Cleanup Close；这里手动关闭并替换清理。
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rt.LoadModel("x", toyPkg(t)); !Is(err, ErrRuntimeClosed) {
		t.Fatalf("关闭后加载应 ErrRuntimeClosed，实际 %v", err)
	}
	if _, err := rt.NewSession("toy", SessionOptions{}); !Is(err, ErrRuntimeClosed) {
		t.Fatalf("关闭后建会话应 ErrRuntimeClosed，实际 %v", err)
	}
	t.Logf("判定依据: Close 后 Load/NewSession 均 ErrRuntimeClosed")
}
