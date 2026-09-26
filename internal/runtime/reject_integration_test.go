package runtime

import (
	"errors"
	"testing"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/testutil"
)

// TestUnsupportedPackages_NeverReachInference 端到端保证：
// 不受支持的模型包在 modelpkg.Load 阶段就被以可区分原因拒绝，
// 运行时永远拿不到 Package，自然不可能在推理中途才报错或给出错误精度数值。
func TestUnsupportedPackages_NeverReachInference(t *testing.T) {
	cases := []struct {
		name    string
		spec    testutil.ToySpec
		wantErr error
	}{
		{"v1+uint8组合不支持", func() testutil.ToySpec {
			s := testutil.DefaultToy()
			s.Version = modelpkg.V1
			s.Precision = modelpkg.UINT8
			s.LayerZP = 128
			return s
		}(), modelpkg.ErrVersionPrecisionMismatch},
		{"未知精度fp16", func() testutil.ToySpec {
			s := testutil.DefaultToy()
			s.Precision = modelpkg.Precision("fp16")
			return s
		}(), modelpkg.ErrUnsupportedPrecision},
		{"未知版本v9", func() testutil.ToySpec {
			s := testutil.DefaultToy()
			s.Version = modelpkg.Version("v9")
			return s
		}(), modelpkg.ErrUnsupportedVersion},
		{"量化尺度非法", func() testutil.ToySpec {
			s := testutil.DefaultToy()
			s.LayerScale[0] = 0
			return s
		}(), modelpkg.ErrMissingQuantParams},
	}

	rt := newTestRuntime(t, 1<<30)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ffsys, dir := testutil.ToyFS(tc.spec)
			pkg, err := modelpkg.Load(ffsys, dir)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("输入规格=%+v；期望加载阶段原因 %v，实际 err=%v pkg=%v",
					tc.spec, tc.wantErr, err, pkg != nil)
			}
			if pkg != nil {
				t.Fatalf("不得返回任何部分加载的包")
			}
			// 运行时无法登记一个不存在的包：同名建会话必为未知模型。
			if _, err := rt.NewSession("toy", SessionOptions{}); !errors.Is(err, ErrUnknownModel) {
				t.Fatalf("被拒模型不应能创建会话，实际 %v", err)
			}
			t.Logf("判定依据: %s -> Load 即返回 %v；运行时无此模型，推理路径根本不可达",
				tc.name, tc.wantErr)
		})
	}
}
