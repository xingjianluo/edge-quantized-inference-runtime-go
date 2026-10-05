package runtime

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"testing/fstest"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// 两代“结构相同、权重不同”的玩具模型规格：用于验证在线切换与结果可区分。
func toyGen1() testutil.ToySpec {
	s := testutil.DefaultToy()
	s.ModelVersion = "2026.09-r1"
	s.SeedOffset = 0
	return s
}

func toyGen2() testutil.ToySpec {
	s := testutil.DefaultToy()
	s.ModelVersion = "2026.10-r2"
	s.SeedOffset = 100 // 仅改变权重内容，结构完全一致
	return s
}

// loadSpec 走完整 Load 校验后把指定规格的玩具模型加载进运行时。
func loadSpec(t *testing.T, rt *Runtime, name string, spec testutil.ToySpec) *modelpkg.Package {
	t.Helper()
	pkg := testutil.LoadToy(t, spec)
	if err := rt.LoadModel(name, pkg); err != nil {
		t.Fatalf("加载模型 %s 失败: %v", name, err)
	}
	return pkg
}

// updateSpec 走完整 Load 校验后在线更新到指定规格。
func updateSpec(t *testing.T, rt *Runtime, name string, spec testutil.ToySpec) *modelpkg.Package {
	t.Helper()
	pkg := testutil.LoadToy(t, spec)
	if err := rt.UpdateModel(name, pkg); err != nil {
		t.Fatalf("在线更新 %s 失败: %v", name, err)
	}
	return pkg
}

// inferLogged 推理并打印输入/输出，便于日志核对判定依据。
func inferLogged(t *testing.T, s *Session, in []float32) []float32 {
	t.Helper()
	out, err := s.Infer(context.Background(), in)
	if err != nil {
		t.Fatalf("推理失败: input=%v err=%v", in, err)
	}
	t.Logf("会话(模型=%s 代=%d) 输入=%v -> 输出=%v", s.ModelName(), s.Generation(), in, out)
	return out
}

func equalOut(a, b []float32) bool {
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

// TestUpdateSwitchAndDistinguishable 验证：在线切换后新会话用新版本，
// 且结果与“直接加载该版本”逐位一致；两代结构相同但结果可区分。
func TestUpdateSwitchAndDistinguishable(t *testing.T) {
	input := []float32{0.5, -0.25}
	t.Logf("场景: 在线切换 gen1->gen2；输入=%v；判定: 新会话输出==直接加载gen2的输出，且!=gen1输出", input)

	rt := newTestRuntime(t, 1<<20)
	pkg1 := loadSpec(t, rt, "detector", toyGen1())

	s1, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	out1 := inferLogged(t, s1, input)
	_ = s1.Close()

	pkg2 := updateSpec(t, rt, "detector", toyGen2())
	if pkg1.Fingerprint() == pkg2.Fingerprint() {
		t.Fatalf("两代指纹不应相同: %s", pkg1.Fingerprint())
	}
	t.Logf("gen1 指纹=%s gen2 指纹=%s（可区分）", pkg1.Fingerprint()[:16], pkg2.Fingerprint()[:16])

	s2, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("切换后建会话失败: %v", err)
	}
	defer func() { _ = s2.Close() }()
	if s2.Generation() != 2 {
		t.Fatalf("切换后新会话应绑定第 2 代，实际第 %d 代", s2.Generation())
	}
	out2 := inferLogged(t, s2, input)

	if equalOut(out1, out2) {
		t.Fatalf("不同版本（结构相同、权重不同）输出必须可区分，却相同: %v", out1)
	}

	// 与“直接加载 gen2”的结论逐位一致。
	rt2 := newTestRuntime(t, 1<<20)
	loadSpec(t, rt2, "detector", toyGen2())
	sd, err := rt2.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("直接加载建会话失败: %v", err)
	}
	defer func() { _ = sd.Close() }()
	outDirect := inferLogged(t, sd, input)
	if !equalOut(out2, outDirect) {
		t.Fatalf("更新后会话输出 %v 与直接加载该版本的输出 %v 不一致", out2, outDirect)
	}
	t.Logf("判定通过: 更新后输出==直接加载输出==%v，且与 gen1 输出 %v 不同", out2, out1)

	// 更新后重复跑一致。
	if again := inferLogged(t, s2, input); !equalOut(again, out2) {
		t.Fatalf("更新后重复推理不一致: %v vs %v", again, out2)
	}
}

// TestUpdateOldSessionUnaffected 验证：切换前已存在的会话继续用旧权重跑完，
// 结果与切换前逐位一致；切换后新建的会话用新版本。
func TestUpdateOldSessionUnaffected(t *testing.T) {
	input := []float32{1.0, 0.5}
	t.Logf("场景: 切换期间旧会话结果不受影响；输入=%v；判定: 旧会话切换前后输出逐位一致", input)

	rt := newTestRuntime(t, 1<<20)
	loadSpec(t, rt, "detector", toyGen1())

	oldSess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	before := inferLogged(t, oldSess, input)

	updateSpec(t, rt, "detector", toyGen2())
	t.Logf("已在线切换到 gen2；旧会话绑定代=%d 应继续用 gen1 权重", oldSess.Generation())

	after := inferLogged(t, oldSess, input)
	if !equalOut(before, after) {
		t.Fatalf("旧会话在切换前后结果不一致: 前=%v 后=%v", before, after)
	}

	newSess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("切换后建会话失败: %v", err)
	}
	defer func() { _ = newSess.Close() }()
	outNew := inferLogged(t, newSess, input)
	if equalOut(outNew, after) {
		t.Fatalf("新会话应使用新版本得到不同结果，却与旧会话相同: %v", outNew)
	}
	if err := oldSess.Close(); err != nil {
		t.Fatalf("旧会话关闭失败: %v", err)
	}
	t.Logf("判定通过: 旧会话输出保持 %v，新会话输出 %v", after, outNew)
}

// TestUpdateNoUnavailableWindow 验证：更新/回退瞬间不存在模型不可用窗口，
// 并发新建会话始终成功，且每次结果必属某一代的确定输出。
func TestUpdateNoUnavailableWindow(t *testing.T) {
	input := []float32{0.25, 0.75}
	t.Logf("场景: 更新/回退与并发建会话交错；输入=%v；判定: 无失败，输出∈{gen1输出, gen2输出}", input)

	rt := newTestRuntime(t, 1<<20)
	loadSpec(t, rt, "detector", toyGen1())

	// 基准输出。
	base1 := func() []float32 {
		s, err := rt.NewSession("detector", SessionOptions{})
		if err != nil {
			t.Fatalf("建基准会话失败: %v", err)
		}
		defer func() { _ = s.Close() }()
		return mustInfer(t, s, input)
	}()
	pkg2 := testutil.LoadToy(t, toyGen2())
	if err := rt.UpdateModel("detector", pkg2); err != nil {
		t.Fatalf("预更新失败: %v", err)
	}
	base2 := func() []float32 {
		s, err := rt.NewSession("detector", SessionOptions{})
		if err != nil {
			t.Fatalf("建基准会话失败: %v", err)
		}
		defer func() { _ = s.Close() }()
		return mustInfer(t, s, input)
	}()
	if equalOut(base1, base2) {
		t.Fatalf("两代基准输出应可区分")
	}
	t.Logf("gen1 基准输出=%v gen2 基准输出=%v", base1, base2)

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers*64)
	stop := make(chan struct{})
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				select {
				case <-stop:
					return
				default:
				}
				s, err := rt.NewSession("detector", SessionOptions{})
				if err != nil {
					errs <- fmt.Errorf("worker %d: 更新期间建会话失败: %w", id, err)
					return
				}
				out, err := s.Infer(context.Background(), input)
				if err != nil {
					errs <- fmt.Errorf("worker %d: 推理失败: %w", id, err)
					_ = s.Close()
					return
				}
				if !equalOut(out, base1) && !equalOut(out, base2) {
					errs <- fmt.Errorf("worker %d: 输出 %v 既非 gen1 %v 也非 gen2 %v", id, out, base1, base2)
				}
				_ = s.Close()
			}
		}(w)
	}
	// 主 goroutine 反复来回切换，制造更新/回退与建会话的交错。
	for i := 0; i < 20; i++ {
		if err := rt.UpdateModel("detector", testutil.LoadToy(t, toyGen1())); err != nil {
			t.Fatalf("来回切换更新失败: %v", err)
		}
		if err := rt.UpdateModel("detector", testutil.LoadToy(t, toyGen2())); err != nil {
			t.Fatalf("来回切换更新失败: %v", err)
		}
		if err := rt.RollbackModel("detector"); err != nil {
			t.Fatalf("回退失败: %v", err)
		}
	}
	close(stop)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	t.Logf("判定通过: %d 个并发 worker 在 60 次切换/回退期间建会话与推理全部成功，输出均属于某一代", workers)
}

// TestUpdateInvalidPackageKeepsOld 验证：新包不合规时在 Load 阶段即以可区分
// 原因失败，在线更新不发生，旧版本继续对外服务，设备记账不变。
func TestUpdateInvalidPackageKeepsOld(t *testing.T) {
	input := []float32{0.5, 0.5}
	rt := newTestRuntime(t, 1<<20)
	loadSpec(t, rt, "detector", toyGen1())

	sess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	defer func() { _ = sess.Close() }()
	usedBefore := rt.UsedBytes() // 权重 12 + 会话工作区 20
	baseline := mustInfer(t, sess, input)
	t.Logf("场景: 不合规新包不得替换在役版本；输入=%v；gen1 基准输出=%v", input, baseline)

	// 篡改权重内容（声明了校验和）：Load 必须拒绝。
	tamper := func(spec testutil.ToySpec) (fstest.MapFS, string) {
		spec.WithChecksum = true
		ffsys, dir := testutil.ToyFS(spec)
		mfs := ffsys.(fstest.MapFS)
		f := mfs[dir+"/fc1.bin"]
		buf := append([]byte(nil), f.Data...)
		buf[0] ^= 0xFF // 内容被改动
		mfs[dir+"/fc1.bin"] = &fstest.MapFile{Data: buf}
		return mfs, dir
	}

	cases := []struct {
		name string
		load func() error
		want error
	}{
		{"版本×精度组合不支持(v1+uint8)", func() error {
			spec := toyGen2()
			spec.Version = modelpkg.V1
			spec.Precision = modelpkg.UINT8
			ffsys, dir := testutil.ToyFS(spec)
			_, err := modelpkg.Load(ffsys, dir)
			return err
		}, modelpkg.ErrVersionPrecisionMismatch},
		{"量化参数缺失(scale=0)", func() error {
			spec := toyGen2()
			spec.LayerScale = [2]float64{0, 0.05}
			ffsys, dir := testutil.ToyFS(spec)
			_, err := modelpkg.Load(ffsys, dir)
			return err
		}, modelpkg.ErrMissingQuantParams},
		{"权重内容被改动(校验和不符)", func() error {
			mfs, dir := tamper(toyGen2())
			_, err := modelpkg.Load(mfs, dir)
			return err
		}, modelpkg.ErrWeightCorrupted},
	}
	for _, c := range cases {
		err := c.load()
		if !errors.Is(err, c.want) {
			t.Fatalf("场景[%s]: 期望错误 %v，实际 %v", c.name, c.want, err)
		}
		t.Logf("场景[%s]: 新包被 Load 拒绝，原因=%v（可区分）", c.name, err)
		// 在役版本不受任何影响。
		if got := rt.UsedBytes(); got != usedBefore {
			t.Fatalf("场景[%s]: 失败后设备记账应不变，%d -> %d", c.name, usedBefore, got)
		}
		av, err := rt.ActiveVersion("detector")
		if err != nil || av.Generation != 1 {
			t.Fatalf("场景[%s]: 在役版本应仍为第 1 代: %+v err=%v", c.name, av, err)
		}
		if out := mustInfer(t, sess, input); !equalOut(out, baseline) {
			t.Fatalf("场景[%s]: 旧版本输出被污染: %v != %v", c.name, out, baseline)
		}
	}
	t.Logf("判定通过: 所有不合规包均未影响在役 gen1，输出保持 %v，记账保持 %d 字节", baseline, usedBefore)
}

// TestUpdateInsufficientSpaceKeepsOld 验证：设备剩余空间不足以同时容纳新旧
// 两套权重时更新整体失败（ErrInsufficientResources），旧版本继续服务；
// 释放空间后同一更新可以成功，证明失败原因确为空间。
func TestUpdateInsufficientSpaceKeepsOld(t *testing.T) {
	input := []float32{0.5, -0.5}
	// 预算 = gen1 权重(12) + 一个会话工作区(20) = 32，容不下第二套权重。
	rt := newTestRuntime(t, 32)
	loadSpec(t, rt, "detector", toyGen1())
	sess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	baseline := mustInfer(t, sess, input)
	t.Logf("场景: 空间不足时更新整体失败；输入=%v；gen1 基准输出=%v；预算=32 已用=%d", input, baseline, rt.UsedBytes())

	pkg2 := testutil.LoadToy(t, toyGen2())
	err = rt.UpdateModel("detector", pkg2)
	if !errors.Is(err, ErrInsufficientResources) {
		t.Fatalf("期望 ErrInsufficientResources，实际 %v", err)
	}
	t.Logf("更新被拒: %v（与包不合规可区分）", err)

	if got := rt.UsedBytes(); got != 32 {
		t.Fatalf("更新失败后记账应不变，实际 %d", got)
	}
	av, _ := rt.ActiveVersion("detector")
	if av.Generation != 1 {
		t.Fatalf("在役版本应仍为第 1 代，实际 %+v", av)
	}
	if out := mustInfer(t, sess, input); !equalOut(out, baseline) {
		t.Fatalf("旧版本服务受影响: %v != %v", out, baseline)
	}
	t.Logf("旧版本继续正常服务，输出=%v", baseline)

	// 释放会话工作区后，同一更新成功：证明失败确因空间而非包或状态。
	if err := sess.Close(); err != nil {
		t.Fatalf("关闭会话失败: %v", err)
	}
	if err := rt.UpdateModel("detector", pkg2); err != nil {
		t.Fatalf("释放空间后更新应成功: %v", err)
	}
	av, _ = rt.ActiveVersion("detector")
	if av.Generation != 2 {
		t.Fatalf("释放空间后应切换到第 2 代，实际 %+v", av)
	}
	t.Logf("判定通过: 释放空间后更新成功，当前在役=%+v", av)
}

// TestRollbackAndVersionQuery 验证：一键回退、版本查询、无回退目标时的
// 可区分错误，以及回退后结果与旧版一致。
func TestRollbackAndVersionQuery(t *testing.T) {
	input := []float32{-0.75, 0.25}
	t.Logf("场景: 更新后一键回退；输入=%v；判定: 回退后新会话输出==gen1输出", input)

	rt := newTestRuntime(t, 1<<20)
	loadSpec(t, rt, "detector", toyGen1())

	// 未发生过更新：没有回退目标。
	if err := rt.RollbackModel("detector"); !errors.Is(err, ErrNoRollbackTarget) {
		t.Fatalf("无回退目标时应返回 ErrNoRollbackTarget，实际 %v", err)
	}
	t.Logf("无回退目标时被拒绝: %v", ErrNoRollbackTarget)

	s1, _ := rt.NewSession("detector", SessionOptions{})
	out1 := inferLogged(t, s1, input)
	_ = s1.Close()

	updateSpec(t, rt, "detector", toyGen2())

	// 版本查询：active=gen2，standby=gen1 可回退。
	av, err := rt.ActiveVersion("detector")
	if err != nil {
		t.Fatalf("查询在役版本失败: %v", err)
	}
	if av.Generation != 2 || av.Role != RoleActive || av.ModelVersion != "2026.10-r2" {
		t.Fatalf("在役版本信息不符: %+v", av)
	}
	versions, err := rt.ModelVersions("detector")
	if err != nil {
		t.Fatalf("查询版本列表失败: %v", err)
	}
	t.Logf("驻留版本: %+v", versions)
	if len(versions) != 2 || versions[0].Role != RoleStandby || versions[1].Role != RoleActive {
		t.Fatalf("版本列表应为 [gen1=standby, gen2=active]，实际 %+v", versions)
	}

	// 一键回退。
	if err := rt.RollbackModel("detector"); err != nil {
		t.Fatalf("回退失败: %v", err)
	}
	av, _ = rt.ActiveVersion("detector")
	if av.Generation != 1 || av.ModelVersion != "2026.09-r1" {
		t.Fatalf("回退后应在第 1 代，实际 %+v", av)
	}
	sr, _ := rt.NewSession("detector", SessionOptions{})
	defer func() { _ = sr.Close() }()
	outRollback := inferLogged(t, sr, input)
	if !equalOut(outRollback, out1) {
		t.Fatalf("回退后输出 %v 应与 gen1 输出 %v 一致", outRollback, out1)
	}
	t.Logf("判定通过: 回退后输出==gen1输出==%v", out1)

	// 再次回退（前滚到 gen2）：角色互换。
	if err := rt.RollbackModel("detector"); err != nil {
		t.Fatalf("再次回退失败: %v", err)
	}
	av, _ = rt.ActiveVersion("detector")
	if av.Generation != 2 {
		t.Fatalf("再次回退后应在第 2 代，实际 %+v", av)
	}
	t.Logf("再次回退（前滚）后 在役=%+v", av)
}

// TestRetiredGenerationReclaimed 验证：退役代（比回退目标更旧的版本）在其
// 旧会话全部结束后自动回收权重，回收后设备可继续加载与建会话。
func TestRetiredGenerationReclaimed(t *testing.T) {
	input := []float32{0.1, 0.2}
	// 预算 56 = 三代权重(36) + 一个会话工作区(20)。
	rt := newTestRuntime(t, 56)
	loadSpec(t, rt, "detector", toyGen1())
	oldSess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	t.Logf("场景: gen1->gen2->gen3 两次更新后 gen1 退役；输入=%v；判定: 旧会话结束后 gen1 权重自动回收", input)

	updateSpec(t, rt, "detector", toyGen2()) // gen1 -> standby
	gen3 := toyGen2()
	gen3.ModelVersion = "2026.10-r3"
	gen3.SeedOffset = 200
	updateSpec(t, rt, "detector", gen3) // gen1 退役(draining, 仍有 oldSess)，gen2 -> standby

	usedWithRetired := rt.UsedBytes()
	// 12*3 权重 + 20 工作区 = 56。
	if usedWithRetired != 56 {
		t.Fatalf("三代共存期间记账应为 56，实际 %d", usedWithRetired)
	}
	versions, _ := rt.ModelVersions("detector")
	t.Logf("三代驻留: %+v（记账 %d 字节）", versions, usedWithRetired)
	var sawDraining bool
	for _, v := range versions {
		if v.Role == RoleDraining && v.Generation == 1 && v.ActiveSessions == 1 {
			sawDraining = true
		}
	}
	if !sawDraining {
		t.Fatalf("gen1 应处于 draining 且有 1 个旧会话: %+v", versions)
	}

	// 旧会话结果仍是 gen1 的，然后关闭触发回收。
	outOld := inferLogged(t, oldSess, input)
	if err := oldSess.Close(); err != nil {
		t.Fatalf("关闭旧会话失败: %v", err)
	}
	usedAfter := rt.UsedBytes()
	t.Logf("旧会话结束后: 记账 %d -> %d（回收 gen1 权重 12 字节）", usedWithRetired, usedAfter)
	if usedAfter != 24 { // gen2 + gen3 权重
		t.Fatalf("回收后记账应为 24，实际 %d", usedAfter)
	}
	versions, _ = rt.ModelVersions("detector")
	if len(versions) != 2 {
		t.Fatalf("回收后应只剩 gen2/gen3 两代，实际 %+v", versions)
	}

	// 回收后设备可继续建会话、继续加载新模型。
	s, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("回收后建会话失败: %v", err)
	}
	_ = s.Close()
	loadSpec(t, rt, "other", toyGen1())
	t.Logf("判定通过: gen1 已自动回收，设备可继续建会话并加载新模型（gen1 旧会话最后输出=%v）", outOld)
}

// TestAbnormalInputRejected 验证：NaN/±Inf 输入被明确拒绝（ErrInputNotFinite），
// 与维度错误可区分，不会产出看似正常实则无意义的数值。
func TestAbnormalInputRejected(t *testing.T) {
	rt := newTestRuntime(t, 1<<20)
	loadSpec(t, rt, "detector", toyGen1())
	sess, err := rt.NewSession("detector", SessionOptions{})
	if err != nil {
		t.Fatalf("建会话失败: %v", err)
	}
	defer func() { _ = sess.Close() }()

	cases := []struct {
		name  string
		input []float32
		want  error
	}{
		{"含NaN", []float32{float32(math.NaN()), 0.5}, ErrInputNotFinite},
		{"含+Inf", []float32{0.5, float32(math.Inf(1))}, ErrInputNotFinite},
		{"含-Inf", []float32{float32(math.Inf(-1)), 0.5}, ErrInputNotFinite},
		{"维度不符", []float32{0.5}, ErrInputDimMismatch},
	}
	for _, c := range cases {
		_, err := sess.Infer(context.Background(), c.input)
		if !errors.Is(err, c.want) {
			t.Fatalf("场景[%s]: 输入=%v 期望 %v，实际 %v", c.name, c.input, c.want, err)
		}
		t.Logf("场景[%s]: 输入=%v 被拒绝，原因=%v", c.name, c.input, err)
	}
	// 批量中含异常输入同样整批拒绝。
	_, err = sess.InferBatch(context.Background(), [][]float32{{0.5, 0.5}, {float32(math.NaN()), 0}})
	if !errors.Is(err, ErrInputNotFinite) {
		t.Fatalf("批量含 NaN 应整批拒绝，实际 %v", err)
	}
	t.Logf("判定通过: 异常输入均被拒绝且原因可区分，会话仍可正常使用")
	if out := mustInfer(t, sess, []float32{0.5, 0.5}); len(out) != 2 {
		t.Fatalf("拒绝异常输入后会话应仍正常，输出=%v", out)
	}
}
