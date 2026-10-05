package runtime

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// loadSignedToy 把一个带校验和的玩具代际包登记到运行时（首次加载走普通 LoadModel）。
func loadSignedToy(t *testing.T, rt *Runtime, name string, spec testutil.ToySpec) *modelpkg.Package {
	t.Helper()
	spec.WithChecksums = true
	pkg := testutil.LoadToy(t, spec)
	if err := rt.LoadModel(name, pkg); err != nil {
		t.Fatalf("首次加载 %s 失败: %v", name, err)
	}
	return pkg
}

// TestUpdate_HotSwitch_NewSessionsUseNewGen_OldSessionUnaffected
// 核心热切换语义：切换不中断服务；切换前创建的旧会话继续用旧权重跑完，
// 切换后创建的新会话用新权重；切换瞬间不存在模型不可用窗口。
func TestUpdate_HotSwitch_NewSessionsUseNewGen_OldSessionUnaffected(t *testing.T) {
	pkgV1 := testutil.LoadToy(t, testutil.SignedToy())
	rt := newTestRuntime(t, 1<<30)
	if err := rt.LoadModel("toy", pkgV1); err != nil {
		t.Fatal(err)
	}

	// 切换前：创建旧会话并记录“旧权重下”的基准输出（切换后它仍应得到同样结果）。
	oldSess, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	oldRev := oldSess.Revision()
	oldOut := make([][]float32, len(toyInputs))
	for i, in := range toyInputs {
		oldOut[i] = mustInfer(t, oldSess, in)
	}
	active, _ := rt.ActiveModel("toy")
	t.Logf("场景=热切换前; 输入=%v; 当前生效 revision=%s 格式=%s/%s",
		toyInputs, active.Revision[:12], active.FormatVersion, active.Precision)

	// 切换：同版本字符串 v2/int8、结构完全一致，仅权重内容（Seed）不同。
	newSpec := testutil.SignedToy()
	newSpec.Seed = 99
	pkgV2 := testutil.LoadToyForUpdate(t, newSpec)
	if err := rt.UpdateModel("toy", pkgV2); err != nil {
		t.Fatalf("在线切换失败: %v", err)
	}

	// 切换瞬间服务不中断：立刻就能对同名模型建新会话，且用的是新代权重。
	newSess, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("切换后新建会话不应失败（无不可用窗口）: %v", err)
	}
	newRev := newSess.Revision()
	if newRev == oldRev {
		t.Fatalf("新会话必须绑定新代权重，两代内容指纹不应相同: %s", newRev)
	}
	if got, _ := rt.ActiveModel("toy"); got.Revision != newRev {
		t.Fatalf("当前对外生效版本应为新代 %s，实际 %s", newRev, got.Revision)
	}

	// 旧会话在切换后重放同一批输入：必须逐位等于切换前记录的旧权重结果。
	for i, in := range toyInputs {
		got := mustInfer(t, oldSess, in)
		if !reflect.DeepEqual(got, oldOut[i]) {
			t.Fatalf("旧会话结果受切换影响: input=%v 切换前=%v 切换后=%v",
				in, oldOut[i], got)
		}
		if oldSess.Revision() != oldRev {
			t.Fatalf("旧会话绑定的代际不得被切换改变")
		}
	}

	// 新会话结果与“直接加载新版本”拿到的结论一致。
	direct := testutil.LoadToyForUpdate(t, newSpec)
	directView := buildView(direct)
	for _, in := range toyInputs {
		got := mustInfer(t, newSess, in)
		want, err := inferOne(directView, in)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("新会话结果与直接加载新版本不一致: input=%v session=%v direct=%v",
				in, got, want)
		}
	}

	// 同版本同结构、仅权重不同，结果也要可区分（不能悄悄用错版本）。
	distinguishable := false
	for i := range toyInputs {
		if !reflect.DeepEqual(mustInfer(t, newSess, toyInputs[i]), oldOut[i]) {
			distinguishable = true
		}
	}
	if !distinguishable {
		t.Fatal("两代权重不同但全部输出相同，无法从结果区分版本")
	}
	t.Logf("判定依据: 旧会话revision=%s 重放4条全部等于切换前; 新会话revision=%s 与直接加载一致且输出可区分",
		oldRev[:12], newRev[:12])

	_ = oldSess.Close()
	_ = newSess.Close()
}

// TestUpdate_ConcurrentSessionsNeverSeeUnavailability
// 高并发下持续建会话+推理的同时做切换：任何会话都不得遇到“模型不存在/不可用”，
// 每个会话的结果必须等于其绑定代际的直接推理基准。
func TestUpdate_ConcurrentSessionsNeverSeeUnavailability(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadSignedToy(t, rt, "toy", testutil.DefaultToy())
	gen0, _ := rt.ActiveModel("toy")

	// 预先算出两代基准视图。
	view0 := buildView(testutil.LoadToy(t, testutil.DefaultToy()))
	spec1 := testutil.SignedToy()
	spec1.Seed = 40
	pkg1 := testutil.LoadToyForUpdate(t, spec1)
	view1 := buildView(pkg1)

	const clients = 12
	const rounds = 30
	errCh := make(chan error, clients*2)
	done := make(chan struct{})

	// 客户端：不断建“短会话”推理几条再关闭，覆盖切换前后的建会话路径。
	for c := 0; c < clients; c++ {
		go func() {
			for {
				select {
				case <-done:
					errCh <- nil
					return
				default:
				}
				s, err := rt.NewSession("toy", SessionOptions{})
				if err != nil {
					errCh <- fmt.Errorf("切换期间建会话失败（服务中断窗口）: %w", err)
					return
				}
				rev := s.Revision()
				for _, in := range toyInputs {
					out, err := s.Infer(context.Background(), in)
					if err != nil {
						errCh <- fmt.Errorf("切换期间推理失败: %w", err)
						s.Close()
						return
					}
					var want []float32
					if rev == gen0.Revision {
						want, _ = inferOne(view0, in)
					} else {
						want, _ = inferOne(view1, in)
					}
					if !reflect.DeepEqual(out, want) {
						errCh <- fmt.Errorf("revision=%s input=%v got=%v want=%v",
							rev[:8], in, out, want)
						s.Close()
						return
					}
				}
				_ = s.Close()
			}
		}()
	}

	// 中途切换。
	if err := rt.UpdateModel("toy", pkg1); err != nil {
		t.Fatal(err)
	}
	// 让客户端在两代上都跑一会儿。
	for i := 0; i < rounds; i++ {
		s, err := rt.NewSession("toy", SessionOptions{})
		if err != nil {
			t.Fatalf("切换后建会话失败: %v", err)
		}
		_ = s.Close()
	}
	close(done)
	for c := 0; c < clients; c++ {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
	active, _ := rt.ActiveModel("toy")
	t.Logf("判定依据: %d 个客户端在切换全程建会话/推理无 ErrUnknownModel/失败；当前生效 revision=%s",
		clients, active.Revision[:12])
}

// TestUpdate_DifferentShapeGeneration 新一代可以是不同精度/不同输入维度的模型：
// 旧会话仍按旧代维度推理，新会话按新代维度推理，互不影响。
func TestUpdate_DifferentShapeGeneration(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadSignedToy(t, rt, "toy", testutil.DefaultToy())
	probe := []float32{0.5, -0.25}
	oldSess, _ := rt.NewSession("toy", SessionOptions{})
	oldOut := mustInfer(t, oldSess, probe)

	// 新一代：v2/uint8（夹具仍为 2 维输入），换权重种子并带校验和。
	u8 := testutil.SignedToy()
	u8.Precision = modelpkg.UINT8
	u8.LayerZP = 128
	u8.Seed = 7
	pkgU8 := testutil.LoadToyForUpdate(t, u8)
	if err := rt.UpdateModel("toy", pkgU8); err != nil {
		t.Fatal(err)
	}
	newSess, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if newSess.Precision() != modelpkg.UINT8 {
		t.Fatalf("新会话应为 uint8 代，实际 %s", newSess.Precision())
	}
	newOut := mustInfer(t, newSess, probe)
	if reflect.DeepEqual(newOut, oldOut) {
		t.Fatal("不同精度代的输出应可区分")
	}
	// 旧会话依旧按 int8 代给出旧结果。
	if got := mustInfer(t, oldSess, probe); !reflect.DeepEqual(got, oldOut) {
		t.Fatalf("旧会话结果被新代影响: got=%v want=%v", got, oldOut)
	}
	active, _ := rt.ActiveModel("toy")
	t.Logf("判定依据: 跨精度切换 int8->uint8；输入=%v 旧会话=%v 新会话=%v，当前=%s/%s",
		probe,
		oldOut, newOut, active.FormatVersion, active.Precision)

	// 回退后新会话恢复 int8，结果与最初一致。
	if _, err := rt.RollbackModel("toy"); err != nil {
		t.Fatal(err)
	}
	rbSess, _ := rt.NewSession("toy", SessionOptions{})
	if rbSess.Precision() != modelpkg.INT8 {
		t.Fatalf("回退后应为 int8，实际 %s", rbSess.Precision())
	}
	rbOut := mustInfer(t, rbSess, probe)
	if !reflect.DeepEqual(rbOut, oldOut) {
		t.Fatalf("回退后结果应与第一代一致: got=%v want=%v", rbOut, oldOut)
	}
	t.Logf("判定依据: 回退后 precision=int8，输出=%v 与第一代一致", rbOut)
	_ = oldSess.Close()
	_ = newSess.Close()
	_ = rbSess.Close()
}
