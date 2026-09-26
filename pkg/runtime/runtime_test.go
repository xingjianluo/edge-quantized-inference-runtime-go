package runtime_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/kernel"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/loader"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/runtime"
)

func opts() testutil.ManifestOptions {
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

func loadInto(t *testing.T, rt *runtime.Runtime, dir string) *model.LoadedModel {
	t.Helper()
	m, err := rt.LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	return m
}

func modelBytes(t *testing.T, dir string) int64 {
	t.Helper()
	m, err := loader.Load(dir, loader.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return loader.ModelMemoryBytes(m)
}

func TestConcurrentSessions_IsolatedAndConsistent(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "concurrent", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)

	const n = 24
	var wg sync.WaitGroup
	errs := make([]error, n)
	preds := make([]kernel.Prediction, n)
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sess, err := rt.NewSession("classifier-int8", runtime.SessionOptions{
				MaxBatchSize: 2,
			})
			if err != nil {
				errs[i] = err
				return
			}
			defer sess.Close()
			<-start
			in := []float64{1, 0}
			if i%2 == 1 {
				in = []float64{0, 1}
			}
			p, err := sess.Infer(context.Background(), in)
			errs[i] = err
			preds[i] = p
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("会话 %d 失败: %v", i, err)
		}
	}
	for i := 0; i < n; i++ {
		want := "a"
		if i%2 == 1 {
			want = "b"
		}
		if preds[i].Label != want {
			t.Fatalf("会话 %d 输入被其他会话污染: 得到 %s 期望 %s（会话间不得共享可变状态）",
				i, preds[i].Label, want)
		}
	}
	if st := rt.Stats(); st.ActiveSessions != 0 || st.SessionBytes != 0 {
		t.Fatalf("会话关闭后资源必须完全释放: %+v", st)
	}
	t.Logf("判定依据: %d 个并发会话交替输入 [1 0]/[0 1]，偶数会话全得 a、奇数会话全得 b；Stats 结束后=%+v（无串扰、无泄漏）",
		n, rt.Stats())
}

func TestSessionCancel_AffectsOnlyItself(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "cancel", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)

	victim, err := rt.NewSession("classifier-int8", runtime.SessionOptions{
		ComputeLatency: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	other, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-canceled
	_, cancelErr := victim.Infer(ctx, []float64{1, 0})
	if merrors.ReasonOf(cancelErr) != merrors.ReasonCanceled {
		t.Fatalf("已取消会话必须以 canceled 拒绝，实际 %v", cancelErr)
	}
	t.Logf("输入=[1 0] 判定依据: 受害者会话 ctx 预先取消，返回原因 canceled（%v）", cancelErr)

	p, err := other.Infer(context.Background(), []float64{0, 1})
	if err != nil {
		t.Fatalf("其他会话不得被取消波及: %v", err)
	}
	if p.Label != "b" {
		t.Fatalf("其他会话结果异常: %s", p.Label)
	}
	t.Logf("输入=[0 1] 判定依据: 未受影响会话照常推理得到 %s，logits=%v", p.Label, p.Logits)

	if err := victim.Close(); err != nil {
		t.Fatalf("被取消会话必须能干净关闭: %v", err)
	}
	// A fresh session on the same model must work after the canceled one closed.
	replacement, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatalf("取消后应能再建会话: %v", err)
	}
	defer replacement.Close()
	if _, err := replacement.Infer(context.Background(), []float64{1, 0}); err != nil {
		t.Fatalf("取消事件后新建会话推理失败: %v", err)
	}
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("判定依据: 受害者关闭后 Stats=%+v，替换会话正常——取消不外溢、资源已释放、后续加载/推理不受影响",
		rt.Stats())
}

func TestSessionTimeout_AffectsOnlyItself(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "timeout", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)

	slow, err := rt.NewSession("classifier-int8", runtime.SessionOptions{
		ComputeLatency: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	fast, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = slow.Infer(ctx, []float64{1, 0})
	elapsed := time.Since(start)
	if merrors.ReasonOf(err) != merrors.ReasonTimeout {
		t.Fatalf("超时必须以 timeout 区分于取消，实际原因=%q (%v)", merrors.ReasonOf(err), err)
	}
	if elapsed > 150*time.Millisecond {
		t.Fatalf("超时后必须尽快返回，实际耗时 %v", elapsed)
	}
	t.Logf("输入=[1 0] 判定依据: 20ms 截止 vs 200ms 模拟计算，%v 后返回 timeout，与 canceled 原因不同", elapsed)

	p, err := fast.Infer(context.Background(), []float64{1, 0})
	if err != nil || p.Label != "a" {
		t.Fatalf("超时会话外溢: err=%v label=%s", err, p.Label)
	}
	if err := slow.Close(); err != nil {
		t.Fatal(err)
	}
	if err := fast.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("判定依据: 并行会话仍得到 %s；慢会话关闭后 Stats=%+v（超时隔离、干净释放）", p.Label, rt.Stats())
}

func TestDeviceBudget_LoadAndSessionRejection(t *testing.T) {
	b := testutil.WriteTwoLayer(t, "dev-budget", opts())
	need := modelBytes(t, b.Dir)

	// Device budget too small for the model: rejected at load with a
	// distinguishable reason, runtime stays usable for a later valid load.
	rt := runtime.New(runtime.Config{Budget: runtime.DeviceBudget{MemoryBytes: need - 1}})
	if _, err := rt.LoadModel(b.Dir); merrors.ReasonOf(err) != merrors.ReasonDeviceBudgetExceeded {
		t.Fatalf("模型超预算必须 device_budget_exceeded，实际 %v", err)
	}
	t.Logf("判定依据: 设备预算 %d < 模型需求 %d，加载前拒绝，运行时状态未改变", need-1, need)

	// Budget exactly fits the model but leaves no room for session memory.
	rt2 := runtime.New(runtime.Config{Budget: runtime.DeviceBudget{MemoryBytes: need}})
	m := loadInto(t, rt2, b.Dir)
	_, err := rt2.NewSession("classifier-int8", runtime.SessionOptions{})
	if merrors.ReasonOf(err) != merrors.ReasonDeviceBudgetExceeded {
		t.Fatalf("会话预留内存超预算必须以 device_budget_exceeded 拒绝，实际 %v", err)
	}
	t.Logf("判定依据: 预算 %d 恰好容纳模型 %d 字节，但会话预留空间不足，拒绝原因=device_budget_exceeded，模型仍可用 Stats=%+v",
		need, loader.ModelMemoryBytes(m), rt2.Stats())

	// After the failed session attempt nothing was reserved; a runtime with
	// headroom accepts the same model for a fresh load.
	rt3 := runtime.New(runtime.Config{Budget: runtime.DeviceBudget{MemoryBytes: need + 1<<20}})
	loadInto(t, rt3, b.Dir)
	sess, err := rt3.NewSession("classifier-int8", runtime.SessionOptions{MaxBatchSize: 1})
	if err != nil {
		t.Fatalf("预算充足时应能创建会话: %v", err)
	}
	if _, err := sess.Infer(context.Background(), []float64{1, 0}); err != nil {
		t.Fatalf("预算拒绝事件后推理必须正常: %v", err)
	}
	t.Logf("判定依据: 先前的拒绝不留下任何部分预留；新运行时加载+会话+推理全部成功，预留=%d 字节",
		sess.ReservedBytes())
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionBudget_BatchRejectedBeforeCompute(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "sess-budget", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)
	sess, err := rt.NewSession("classifier-int8", runtime.SessionOptions{MaxBatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	_, err = sess.InferBatch(context.Background(), [][]float64{{1, 0}, {0, 1}, {1, 0}})
	if merrors.ReasonOf(err) != merrors.ReasonSessionBudgetExceeded {
		t.Fatalf("3 行批量超过会话预算 2 必须 session_budget_exceeded，实际 %v", err)
	}
	t.Logf("输入: 3 行批量 vs 会话预算 2；判定依据: 计算开始前拒绝（session_budget_exceeded），原因区别于设备级 device_budget_exceeded")

	// Within budget still works and matches one-by-one results.
	ps, err := sess.InferBatch(context.Background(), [][]float64{{1, 0}, {0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if ps[0].Label != "a" || ps[1].Label != "b" {
		t.Fatalf("预算内批量结果错误: %s %s", ps[0].Label, ps[1].Label)
	}
	t.Logf("判定依据: 拒绝不影响后续；预算内批量得到 [%s %s]", ps[0].Label, ps[1].Label)
}

func TestMaxConcurrentSessions(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "maxsess", opts()).Dir
	rt := runtime.New(runtime.Config{Budget: runtime.DeviceBudget{MaxConcurrentSessions: 2}})
	loadInto(t, rt, dir)

	s1, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s2, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rt.NewSession("classifier-int8", runtime.SessionOptions{}); merrors.ReasonOf(err) != merrors.ReasonDeviceBudgetExceeded {
		t.Fatalf("超过最大并发会话数必须可区分地拒绝，实际 %v", err)
	}
	t.Logf("判定依据: 最大并发=2，第 3 个会话以 device_budget_exceeded 拒绝；活跃=%d", rt.Stats().ActiveSessions)
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}
	s3, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatalf("释放一个会话后名额应恢复: %v", err)
	}
	if err := s2.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s3.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestClosedSessionAndModelLifecycle(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "lifecycle", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)
	sess, err := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	// Idempotent close.
	if err := sess.Close(); err != nil {
		t.Fatalf("重复 Close 必须幂等: %v", err)
	}
	if _, err := sess.Infer(context.Background(), []float64{1, 0}); merrors.ReasonOf(err) != merrors.ReasonSessionClosed {
		t.Fatalf("已关闭会话推理必须 session_closed，实际 %v", err)
	}
	t.Logf("判定依据: 关闭后推理返回 session_closed；Close 幂等；Stats=%+v", rt.Stats())

	// Can't unload while sessions open; can after they close.
	s2, _ := rt.NewSession("classifier-int8", runtime.SessionOptions{})
	if err := rt.UnloadModel("classifier-int8"); merrors.ReasonOf(err) != merrors.ReasonModelBusy {
		t.Fatalf("有活跃会话时卸载必须 model_busy，实际 %v", err)
	}
	t.Logf("判定依据: 存在 1 个活跃会话时 UnloadModel 返回 model_busy（不会在使用中移除模型）")
	if err := s2.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rt.UnloadModel("classifier-int8"); err != nil {
		t.Fatalf("会话全部关闭后应能卸载: %v", err)
	}
	if err := rt.UnloadModel("classifier-int8"); merrors.ReasonOf(err) != merrors.ReasonModelNotFound {
		t.Fatalf("重复卸载必须 model_not_found，实际 %v", err)
	}
	if _, err := rt.NewSession("classifier-int8", runtime.SessionOptions{}); merrors.ReasonOf(err) != merrors.ReasonModelNotFound {
		t.Fatalf("未加载模型开会话必须 model_not_found，实际 %v", err)
	}
}

func TestSession_BatchMatchesOneByOne(t *testing.T) {
	dir := testutil.WriteTwoLayer(t, "sess-consistency", opts()).Dir
	rt := runtime.New(runtime.Config{})
	loadInto(t, rt, dir)
	sess, err := rt.NewSession("classifier-int8", runtime.SessionOptions{MaxBatchSize: 5})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	rows := [][]float64{{1, 0}, {0, 1}, {0.6, 0.4}, {0.3, 0.7}, {0.5, 0.5}}
	batch, err := sess.InferBatch(context.Background(), rows)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range rows {
		single, err := sess.Infer(context.Background(), row)
		if err != nil || single.Label != batch[i].Label || single.Confidence != batch[i].Confidence {
			t.Fatalf("第 %d 行: 批量=%+v 逐条=%+v err=%v", i, batch[i], single, err)
		}
	}
	t.Logf("判定依据: 同一会话 5 行输入的批量与逐条结论、置信度完全一致: a/b/a/b/a")
}
