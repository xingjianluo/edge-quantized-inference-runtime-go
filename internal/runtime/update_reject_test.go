package runtime

import (
	"errors"
	"io/fs"
	"reflect"
	"testing"
	"testing/fstest"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// TestUpdateFromFS_InvalidPackagesRejected_OldVersionKept
// 各类不合规新包在切换前就被拒绝，原因可区分；原版本继续正常对外服务，
// 设备记账不变、无残留。
func TestUpdateFromFS_InvalidPackagesRejected_OldVersionKept(t *testing.T) {
	cases := []struct {
		name    string
		spec    testutil.ToySpec
		tamper  bool
		wantErr error
	}{
		{"v1+uint8组合不支持", func() testutil.ToySpec {
			s := testutil.SignedToy()
			s.Version = modelpkg.V1
			s.Precision = modelpkg.UINT8
			s.LayerZP = 128
			return s
		}(), false, modelpkg.ErrVersionPrecisionMismatch},
		{"量化参数缺失", func() testutil.ToySpec {
			s := testutil.SignedToy()
			s.LayerScale[0] = 0
			return s
		}(), false, modelpkg.ErrMissingQuantParams},
		{"权重内容被改动(校验和不符)", func() testutil.ToySpec {
			return testutil.SignedToy()
		}(), true, modelpkg.ErrChecksumMismatch},
		{"更新包未声明校验和", func() testutil.ToySpec {
			s := testutil.DefaultToy() // WithChecksums=false
			return s
		}(), false, modelpkg.ErrChecksumRequired},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pkg0 := testutil.LoadToy(t, testutil.SignedToy())
			rt := newTestRuntime(t, 1<<30)
			if err := rt.LoadModel("toy", pkg0); err != nil {
				t.Fatal(err)
			}
			sess, _ := rt.NewSession("toy", SessionOptions{})
			wantOut := make([][]float32, len(toyInputs))
			for i, in := range toyInputs {
				wantOut[i] = mustInfer(t, sess, in)
			}
			usedBefore := rt.UsedBytes()
			revBefore := sess.Revision()

			ffsys, dir := testutil.ToyFS(tc.spec)
			if tc.tamper {
				ffsys = tamperWeightFile(t, ffsys, dir+"/fc1.bin")
			}
			_, err := rt.UpdateModelFromFS("toy", ffsys, dir)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("输入场景=%s；期望可区分原因 %v，实际 %v", tc.name, tc.wantErr, err)
			}
			t.Logf("场景=%s；判定依据: errors.Is(err,%v)=true，明细: %v", tc.name, tc.wantErr, err)

			// 旧版本必须继续服务：生效版本不变、记账不变、旧会话结果不变。
			active, _ := rt.ActiveModel("toy")
			if active.Revision != pkg0.Revision() {
				t.Fatal("更新失败后当前对外版本不应改变")
			}
			if rt.UsedBytes() != usedBefore {
				t.Fatalf("更新失败后记账应保持 %d，实际 %d（有残留）", usedBefore, rt.UsedBytes())
			}
			for i, in := range toyInputs {
				if got := mustInfer(t, sess, in); !reflect.DeepEqual(got, wantOut[i]) {
					t.Fatalf("更新失败后旧会话结果被改变: input=%v", in)
				}
			}
			if sess.Revision() != revBefore {
				t.Fatal("旧会话代际不应改变")
			}
			// 切换后新建的会话仍能成功且仍是旧权重（不存在短暂不可用）。
			s2, err := rt.NewSession("toy", SessionOptions{})
			if err != nil {
				t.Fatalf("更新失败期间/之后模型仍应可建会话: %v", err)
			}
			if s2.Revision() != pkg0.Revision() {
				t.Fatal("更新失败后新会话必须仍使用旧权重")
			}
			t.Logf("判定依据: 当前版本=%s 未变、used=%d 无残留、新旧会话结果均为旧版本",
				active.Revision[:12], rt.UsedBytes())
			_ = s2.Close()
			_ = sess.Close()
		})
	}
}

// tamperWeightFile 在内存 FS 中翻转某权重文件首字节，模拟落盘后内容被改动
// （model.json 中的校验和仍为原始值）。
func tamperWeightFile(t *testing.T, ffsys fs.FS, path string) fs.FS {
	t.Helper()
	mfs, ok := ffsys.(fstest.MapFS)
	if !ok {
		t.Fatalf("夹具 FS 类型非 fstest.MapFS: %T", ffsys)
	}
	f, err := ffsys.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 12)
	n, _ := f.Read(raw)
	_ = f.Close()
	raw = raw[:n]
	raw[0] ^= 0xFF
	out := fstest.MapFS{}
	for name, file := range mfs {
		if name == path {
			out[name] = &fstest.MapFile{Data: raw}
		} else {
			out[name] = file
		}
	}
	return out
}

// TestUpdate_InsufficientSpace_OldVersionKept 空间不足以同时容纳新旧两套权重时，
// 更新整体失败，明确归类为 ErrInsufficientResources（区别于包不合规），
// 旧版本继续服务，失败后没有任何预占残留。
func TestUpdate_InsufficientSpace_OldVersionKept(t *testing.T) {
	pkg0 := testutil.LoadToy(t, testutil.SignedToy())
	w := pkg0.Bytes()
	// 预算只够旧权重 + 一个工作区，放不下第二套权重。
	rt := newTestRuntime(t, w+workspaceBytes(buildView(pkg0), 1))
	if err := rt.LoadModel("toy", pkg0); err != nil {
		t.Fatal(err)
	}
	sess, _ := rt.NewSession("toy", SessionOptions{})
	usedBefore := rt.UsedBytes()

	spec1 := testutil.SignedToy()
	spec1.Seed = 88
	pkg1 := testutil.LoadToyForUpdate(t, spec1)
	err := rt.UpdateModel("toy", pkg1)
	if !errors.Is(err, ErrInsufficientResources) {
		t.Fatalf("空间不足应报 ErrInsufficientResources，实际 %v", err)
	}
	t.Logf("判定依据: 预算=%d 已用=%d 新权重需=%d（需共存）-> %v；区别于包不合规错误",
		rt.cfg.MaxDeviceBytes, usedBefore, w, err)

	if rt.UsedBytes() != usedBefore {
		t.Fatalf("空间不足失败后 used 应保持 %d，实际 %d", usedBefore, rt.UsedBytes())
	}
	active, _ := rt.ActiveModel("toy")
	if active.Revision != pkg0.Revision() {
		t.Fatal("空间不足失败后当前版本不应改变")
	}
	if got := mustInfer(t, sess, toyInputs[2]); got == nil {
		t.Fatal("旧版本会话应继续正常推理")
	}
	// 释放工作区后空间足够，更新应成功——证明失败没有留下任何残留占用。
	_ = sess.Close()
	if err := rt.UpdateModel("toy", pkg1); err != nil {
		t.Fatalf("腾出空间后更新应成功（说明此前失败无残留）: %v", err)
	}
	a, _ := rt.ActiveModel("toy")
	if a.Revision != pkg1.Revision() {
		t.Fatal("更新成功后应生效新版本")
	}
	t.Logf("判定依据: 失败后释放工作区即可更新成功，used=%d，无残留预占", rt.UsedBytes())
}

// TestUpdate_UnknownModelAndClosedRuntime 名称/状态错误也要明确区分。
func TestUpdate_UnknownModelAndClosedRuntime(t *testing.T) {
	rt := newTestRuntime(t, 1<<30)
	pkg1 := testutil.LoadToyForUpdate(t, testutil.SignedToy())
	if err := rt.UpdateModel("ghost", pkg1); !errors.Is(err, ErrUnknownModel) {
		t.Fatalf("更新未加载模型应 ErrUnknownModel，实际 %v", err)
	}
	loadSignedToy(t, rt, "toy", testutil.DefaultToy())
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rt.UpdateModel("toy", pkg1); !errors.Is(err, ErrRuntimeClosed) {
		t.Fatalf("关闭后更新应 ErrRuntimeClosed，实际 %v", err)
	}
	if _, err := rt.RollbackModel("toy"); !errors.Is(err, ErrRuntimeClosed) {
		t.Fatalf("关闭后回退应 ErrRuntimeClosed，实际 %v", err)
	}
	t.Logf("判定依据: 未知模型->ErrUnknownModel；运行时关闭后更新/回退->ErrRuntimeClosed")
}
