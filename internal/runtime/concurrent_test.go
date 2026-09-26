package runtime

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

// TestConcurrentSessions_Isolation 多个会话高并发推理：
// 每个会话独立计算，结果必须与串行基准一致；彼此不串状态。
func TestConcurrentSessions_Isolation(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")

	// 先求串行基准。
	base, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := make([][]float32, len(toyInputs))
	for i, in := range toyInputs {
		want[i] = mustInfer(t, base, in)
	}
	base.Close()

	const nSessions = 16
	const rounds = 25
	var wg sync.WaitGroup
	var mu sync.Mutex
	failures := 0
	sessions := make([]*Session, 0, nSessions)

	for g := 0; g < nSessions; g++ {
		s, err := rt.NewSession("toy", SessionOptions{})
		if err != nil {
			t.Fatalf("会话 %d 创建失败: %v", g, err)
		}
		sessions = append(sessions, s)
		wg.Add(1)
		go func(s *Session, id int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				for i, in := range toyInputs {
					out, err := s.Infer(context.Background(), in)
					if err != nil {
						mu.Lock()
						failures++
						t.Errorf("会话 %d 第 %d 轮推理错误: %v", id, r, err)
						mu.Unlock()
						return
					}
					if !reflect.DeepEqual(out, want[i]) {
						mu.Lock()
						failures++
						t.Errorf("会话 %d 结果被污染: input=%v got=%v want=%v",
							id, in, out, want[i])
						mu.Unlock()
						return
					}
				}
			}
		}(s, g)
	}
	wg.Wait()
	for _, s := range sessions {
		_ = s.Close()
	}
	t.Logf("判定依据: %d 个会话 x %d 轮 x %d 条全部与串行基准 DeepEqual，失败=%d，会话间无共享可变状态",
		nSessions, rounds, len(toyInputs), failures)
	if failures != 0 {
		t.Fatalf("并发下出现 %d 个失败/污染", failures)
	}

	// 所有会话关闭后，预算应回落到仅模型权重；随后可再次加载新模型。
	if got := rt.UsedBytes(); got != toyPkg(t).Bytes() {
		t.Fatalf("全部会话关闭后 used 应仅剩模型权重，got=%d", got)
	}
	if err := rt.LoadModel("toy2", toyPkg(t)); err != nil {
		t.Fatalf("并发会话全部释放后应能继续加载，实际: %v", err)
	}
}

// TestConcurrent_OneSessionCancelDoesNotAffectOthers 一个会话在推理循环中被
// 取消/超预算，其他会话的结果必须保持正确。
func TestConcurrent_OneSessionCancelDoesNotAffectOthers(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadToy(t, rt, "toy")

	good, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	victim, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	victimErr := error(nil)

	go func() {
		defer wg.Done()
		// 大批输入，中途由主 goroutine 取消。
		big := make([][]float32, 100000)
		for i := range big {
			big[i] = []float32{0.1, 0.2}
		}
		_, victimErr = victim.InferBatch(context.Background(), big)
	}()

	goodOK := true
	go func() {
		defer wg.Done()
		want := mustInfer(t, good, toyInputs[0])
		for i := 0; i < 200; i++ {
			out, err := good.Infer(context.Background(), toyInputs[0])
			if err != nil || !reflect.DeepEqual(out, want) {
				goodOK = false
				return
			}
		}
	}()

	victim.Cancel()
	wg.Wait()

	if victimErr == nil || !Is(victimErr, ErrSessionCanceled) {
		t.Fatalf("被取消会话应报 ErrSessionCanceled，实际 %v", victimErr)
	}
	if !goodOK {
		t.Fatal("受害者会话被取消，但其他会话结果异常（隔离失败）")
	}
	t.Logf("判定依据: victim=%v；同设备其他会话 200 次推理全部正确，取消仅影响自身",
		fmt.Sprint(victimErr))

	// 受害者关闭后可正常再建会话，证明释放干净。
	if err := victim.Close(); err != nil {
		t.Fatal(err)
	}
	s3, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("取消并关闭后应能新建会话: %v", err)
	}
	_ = s3.Close()
	_ = good.Close()
}
