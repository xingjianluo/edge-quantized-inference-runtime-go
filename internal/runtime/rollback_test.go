package runtime

import (
	"errors"
	"reflect"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// TestRollback_RestoresPreviousVersion 一键回退：
// 回退后新会话重新使用上一代权重；结果与该版本直接加载一致；
// 没有更早版本时返回 ErrNoRollbackTarget。
func TestRollback_RestoresPreviousVersion(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadSignedToy(t, rt, "toy", testutil.DefaultToy())
	r0, _ := rt.ActiveModel("toy")
	out0 := make([][]float32, len(toyInputs))
	s0, _ := rt.NewSession("toy", SessionOptions{})
	for i, in := range toyInputs {
		out0[i] = mustInfer(t, s0, in)
	}
	_ = s0.Close()

	// 升到第二代。
	spec1 := testutil.SignedToy()
	spec1.Seed = 33
	pkg1 := testutil.LoadToyForUpdate(t, spec1)
	if err := rt.UpdateModel("toy", pkg1); err != nil {
		t.Fatal(err)
	}
	r1, _ := rt.ActiveModel("toy")
	s1, _ := rt.NewSession("toy", SessionOptions{})
	out1 := make([][]float32, len(toyInputs))
	for i, in := range toyInputs {
		out1[i] = mustInfer(t, s1, in)
	}
	_ = s1.Close()
	if reflect.DeepEqual(out0, out1) {
		t.Fatal("两代输出不应完全相同")
	}
	t.Logf("场景=更新成功; gen0=%s gen1=%s, 两代输出可区分", r0.Revision[:12], r1.Revision[:12])

	// 一键回退。
	rb, err := rt.RollbackModel("toy")
	if err != nil {
		t.Fatalf("回退失败: %v", err)
	}
	if rb.Revision != r0.Revision {
		t.Fatalf("回退后应生效 gen0 %s，实际 %s", r0.Revision[:12], rb.Revision[:12])
	}
	active, _ := rt.ActiveModel("toy")
	if active.Revision != r0.Revision {
		t.Fatalf("ActiveModel 应指向回退后的 gen0")
	}
	sb, _ := rt.NewSession("toy", SessionOptions{})
	for i, in := range toyInputs {
		if got := mustInfer(t, sb, in); !reflect.DeepEqual(got, out0[i]) {
			t.Fatalf("回退后新会话结果不等于旧版本基准: input=%v got=%v want=%v",
				in, got, out0[i])
		}
	}
	t.Logf("判定依据: 回退后 revision=%s，4 条输入结果全部等于回退前记录的 gen0 基准",
		active.Revision[:12])
	_ = sb.Close()

	// 已在第一代，再回退必须明确失败。
	if _, err := rt.RollbackModel("toy"); !errors.Is(err, ErrNoRollbackTarget) {
		t.Fatalf("第一代再回退应为 ErrNoRollbackTarget，实际 %v", err)
	}
	t.Logf("判定依据: 第一代再回退 -> ErrNoRollbackTarget")

	// 版本列表：gen1 已被顶出但无旧会话，已立即回收，栈内仅剩 gen0（可回退为空）。
	versions, _ := rt.ModelVersions("toy")
	if len(versions) != 1 || !versions[0].Active || versions[0].Revision != r0.Revision {
		t.Fatalf("回退且无旧会话后应仅剩 gen0，实际 %+v", versions)
	}
}

// TestModelVersions_ListsActiveAndRollbackTargets 查询当前版本与可回退版本：
// 任意时刻只保留紧邻上一代作为回退目标；更早的代若已无会话则立即回收、不再列出。
func TestModelVersions_ListsActiveAndRollbackTargets(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	loadSignedToy(t, rt, "toy", testutil.DefaultToy())
	r0, _ := rt.ActiveModel("toy")
	spec1 := testutil.SignedToy()
	spec1.Seed = 55
	pkg1 := testutil.LoadToyForUpdate(t, spec1)
	if err := rt.UpdateModel("toy", pkg1); err != nil {
		t.Fatal(err)
	}
	r1, _ := rt.ActiveModel("toy")

	// 第二次更新：gen0 超出“保留一代回退”窗口，无旧会话应被立即回收。
	spec2 := testutil.SignedToy()
	spec2.Seed = 77
	if err := rt.UpdateModel("toy", testutil.LoadToyForUpdate(t, spec2)); err != nil {
		t.Fatal(err)
	}
	r2, _ := rt.ActiveModel("toy")
	versions, err := rt.ModelVersions("toy")
	if err != nil {
		t.Fatal(err)
	}
	if len(versions) != 2 {
		t.Fatalf("应只剩 gen1(可回退)+gen2(生效)，实际 %d: %+v", len(versions), versions)
	}
	if versions[0].Revision != r1.Revision || versions[0].Active {
		t.Fatalf("栈底应为可回退的 gen1 且 Active=false，实际 %+v", versions[0])
	}
	if versions[1].Revision != r2.Revision || !versions[1].Active {
		t.Fatalf("栈顶应为当前生效的 gen2，实际 %+v", versions[1])
	}
	t.Logf("场景=连续两次更新; gen0=%s 已超回退窗口且无会话被自动回收:", r0.Revision[:12])
	for i, v := range versions {
		t.Logf("  [%d] revision=%s %s/%s 权重=%d字节 active=%v 可回退=%v",
			i, v.Revision[:12], v.FormatVersion, v.Precision, v.WeightBytes,
			v.Active, !v.Active && !v.Retiring)
	}

	// 回退一次：gen1 生效，gen2 被顶出立即回收；再回退 -> 无目标。
	if _, err := rt.RollbackModel("toy"); err != nil {
		t.Fatal(err)
	}
	if a, _ := rt.ActiveModel("toy"); a.Revision != r1.Revision {
		t.Fatalf("回退后应生效 gen1，实际 %s", a.Revision[:12])
	}
	if _, err := rt.RollbackModel("toy"); !errors.Is(err, ErrNoRollbackTarget) {
		t.Fatalf("gen0 已被回收，再回退应为 ErrNoRollbackTarget，实际 %v", err)
	}
	t.Logf("判定依据: 回退后生效 gen1；gen0 早已回收，二次回退 ErrNoRollbackTarget")

	if _, err := rt.ActiveModel("nope"); !errors.Is(err, ErrUnknownModel) {
		t.Fatalf("查询不存在模型应 ErrUnknownModel，实际 %v", err)
	}
}

// TestUpdate_EarlierGenRetainedUntilOldSessionsDone
// 超出回退窗口的更早代若仍被旧会话使用，不会被强行回收，而是标记 Retiring，
// 等最后一个旧会话结束后自动回收；期间记账包含它的权重。
func TestUpdate_EarlierGenRetainedUntilOldSessionsDone(t *testing.T) {
	pkg0 := testutil.LoadToy(t, testutil.SignedToy())
	w := pkg0.Bytes()
	rt := newTestRuntime(t, 1<<30)
	if err := rt.LoadModel("toy", pkg0); err != nil {
		t.Fatal(err)
	}
	sess0, _ := rt.NewSession("toy", SessionOptions{})
	base0 := mustInfer(t, sess0, toyInputs[0])

	// gen0 -> gen1：gen0 仍是回退目标，sess0 继续跑。
	spec1 := testutil.SignedToy()
	spec1.Seed = 55
	if err := rt.UpdateModel("toy", testutil.LoadToyForUpdate(t, spec1)); err != nil {
		t.Fatal(err)
	}
	// gen1 -> gen2：gen0 超出回退窗口，但 sess0 还在，gen0 进入 Retiring 暂驻。
	spec2 := testutil.SignedToy()
	spec2.Seed = 77
	if err := rt.UpdateModel("toy", testutil.LoadToyForUpdate(t, spec2)); err != nil {
		t.Fatal(err)
	}
	if got, want := rt.UsedBytes(), 3*w+sess0.reserve; got != want {
		t.Fatalf("三代权重+1工作区应同时记账=%d，实际=%d", want, got)
	}
	versions, _ := rt.ModelVersions("toy")
	var retiring *VersionInfo
	for i := range versions {
		if versions[i].Retiring {
			retiring = &versions[i]
		}
	}
	if retiring == nil || retiring.Revision != pkg0.Revision() || retiring.ActiveSessions != 1 {
		t.Fatalf("gen0 应以 Retiring 且 1 个旧会话列出，实际 %+v", versions)
	}
	t.Logf("判定依据: gen0 超窗但有旧会话，Retiring=true 权重继续记账，used=%d", rt.UsedBytes())

	// 旧会话结果仍等于 gen0 基准，不受两次切换影响。
	if got := mustInfer(t, sess0, toyInputs[0]); !reflect.DeepEqual(got, base0) {
		t.Fatalf("跨两次切换的旧会话结果被污染: got=%v want=%v", got, base0)
	}

	// 关闭最后一个 gen0 会话：gen0 自动回收，used 减少一份权重，版本列表不再有它。
	if err := sess0.Close(); err != nil {
		t.Fatal(err)
	}
	if got, want := rt.UsedBytes(), int64(2)*w; got != want {
		t.Fatalf("gen0 回收后应为两代权重=%d，实际=%d", want, got)
	}
	for _, v := range mustVersions(t, rt, "toy") {
		if v.Revision == pkg0.Revision() {
			t.Fatalf("gen0 回收后不应再列出: %+v", v)
		}
	}
	t.Logf("判定依据: gen0 最后一个旧会话结束 -> 自动回收，used=%d，版本列表无残留",
		rt.UsedBytes())
}

func mustVersions(t *testing.T, rt *Runtime, name string) []VersionInfo {
	t.Helper()
	vs, err := rt.ModelVersions(name)
	if err != nil {
		t.Fatal(err)
	}
	return vs
}

// TestUpdateRollback_CoexistAccountingAndAutoReclaim
// 新旧共存期间资源如实记账；旧代在最后一个旧会话结束时自动回收；
// 回收后设备可继续加载、继续建会话，无残留占用。
func TestUpdateRollback_CoexistAccountingAndAutoReclaim(t *testing.T) {
	pkg0 := testutil.LoadToy(t, testutil.SignedToy())
	w := pkg0.Bytes()
	// 预算精确到：两代权重 + 一个工作区，制造共存压力。
	rt := newTestRuntime(t, 2*w+workspaceBytes(buildView(pkg0), 1))
	if err := rt.LoadModel("toy", pkg0); err != nil {
		t.Fatal(err)
	}
	oldSess, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	usedBefore := rt.UsedBytes() // 旧权重 + 1 工作区

	spec1 := testutil.SignedToy()
	spec1.Seed = 66
	pkg1 := testutil.LoadToyForUpdate(t, spec1)
	if err := rt.UpdateModel("toy", pkg1); err != nil {
		t.Fatalf("共存预算应足够（旧权重+新权重+1工作区）: %v", err)
	}
	if got, want := rt.UsedBytes(), usedBefore+w; got != want {
		t.Fatalf("新旧共存记账应=旧占用+新权重 %d，实际 %d", want, got)
	}
	t.Logf("判定依据: 切换后新旧两代同时驻留，used %d -> %d（+%d 新权重），旧会话仍在跑",
		usedBefore, rt.UsedBytes(), w)

	// 旧会话还在时，空间已满，新会话应因资源不足失败，但旧会话推理照常。
	if _, err := rt.NewSession("toy", SessionOptions{}); !errors.Is(err, ErrInsufficientResources) {
		t.Fatalf("共存占满预算时新会话应 ErrInsufficientResources，实际 %v", err)
	}
	out := mustInfer(t, oldSess, toyInputs[0])
	t.Logf("判定依据: 预算占满只拒绝新建会话；旧会话输入=%v 仍输出=%v", toyInputs[0], out)

	// 旧会话关闭：它不是最后驻留的旧代——gen0 仍是“可回退版本”，
	// 权重继续保留在栈内（可一键回退），但不再有会话绑定。
	if err := oldSess.Close(); err != nil {
		t.Fatal(err)
	}
	if got, want := rt.UsedBytes(), 2*w; got != want {
		t.Fatalf("旧会话结束后 used 应为两代权重（旧代仍可回退）%d，实际 %d", want, got)
	}
	versions, _ := rt.ModelVersions("toy")
	for _, v := range versions {
		if v.Retiring {
			t.Fatalf("旧代仍是回退目标，不应出现 Retiring 代: %+v", v)
		}
	}
	t.Logf("判定依据: 旧会话结束后无 Retiring 代；gen0 作为回退目标继续驻留，used=%d（两代权重）",
		rt.UsedBytes())

	// 空间腾出一个工作区：可以继续建会话，且使用更新后的版本。
	ns, err := rt.NewSession("toy", SessionOptions{})
	if err != nil {
		t.Fatalf("旧会话释放工作区后应能继续建会话: %v", err)
	}
	if ns.Revision() != pkg1.Revision() {
		t.Fatal("新会话应使用更新后的版本")
	}
	_ = ns.Close()

	// 回退顶出当前代、且无旧会话时：立即归还被顶出代的字节，旧代重新生效。
	before := rt.UsedBytes()
	if _, err := rt.RollbackModel("toy"); err != nil {
		t.Fatal(err)
	}
	if rt.UsedBytes() != before-w {
		t.Fatalf("回退顶出无会话代应立即回收 %d 字节，实际 used=%d", w, rt.UsedBytes())
	}
	t.Logf("判定依据: 回退后被顶出代立即归还，used %d -> %d，无残留占用",
		before, rt.UsedBytes())
}
