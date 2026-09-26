package modelpkg

import (
	"encoding/json"
	"errors"
	"io/fs"
	"testing"
	"testing/fstest"
)

// pkgBuilder 用于在 fstest.MapFS 上确定性地构造合法/非法模型包。
type pkgBuilder struct {
	dir       string
	loadDir   string // 若非空，Load 时使用该目录（用于目录缺失用例）
	version   Version
	precision Precision
	inputDim  int
	inputQP   QuantParams
	layers    []map[string]interface{} // 覆盖字段
	// corrupt 系列字段用于制造各类非法包。
	noJSON        bool
	badJSON       bool
	weightPatches map[string]patch
}

type patch struct {
	missing bool
	bytes   []byte // 若非 nil 则覆盖默认权重
}

func validBuilder() *pkgBuilder {
	return &pkgBuilder{
		dir:       "model",
		version:   V2,
		precision: INT8,
		inputDim:  3,
		inputQP:   QuantParams{Scale: 0.1, ZeroPoint: 0},
		layers: []map[string]interface{}{
			{"name": "fc1", "in_dim": 3, "out_dim": 4, "weight_file": "fc1.bin",
				"quant": map[string]interface{}{"scale": 0.05, "zero_point": int32(0)}},
			{"name": "fc2", "in_dim": 4, "out_dim": 2, "weight_file": "fc2.bin",
				"quant": map[string]interface{}{"scale": 0.02, "zero_point": int32(0)}},
		},
		weightPatches: map[string]patch{},
	}
}

func (b *pkgBuilder) effectiveDir() string {
	if b.loadDir != "" {
		return b.loadDir
	}
	return b.dir
}

func (b *pkgBuilder) fs(t *testing.T) fs.FS {
	t.Helper()
	mfs := fstest.MapFS{}
	if !b.noJSON {
		var content []byte
		var err error
		if b.badJSON {
			content = []byte("{not-json")
		} else {
			root := map[string]interface{}{
				"name":        "toy",
				"version":     string(b.version),
				"precision":   string(b.precision),
				"input_dim":   b.inputDim,
				"input_quant": b.inputQP,
				"layers":      b.layers,
			}
			content, err = json.Marshal(root)
			if err != nil {
				t.Fatalf("构造测试元数据失败: %v", err)
			}
		}
		mfs[b.dir+"/model.json"] = &fstest.MapFile{Data: content}
	}

	// 默认权重文件：每个层按 in*out 生成确定性字节。
	for _, l := range b.layers {
		wf, _ := l["weight_file"].(string)
		inD, _ := l["in_dim"].(int)
		outD, _ := l["out_dim"].(int)
		if wf == "" {
			continue
		}
		n := inD * outD
		w := make([]byte, n)
		for i := range w {
			w[i] = byte((i*7 + 11) % 256)
		}
		if p, ok := b.weightPatches[wf]; ok {
			if p.missing {
				continue
			}
			if p.bytes != nil {
				w = p.bytes
			}
		}
		mfs[b.dir+"/"+wf] = &fstest.MapFile{Data: w}
	}
	return mfs
}

func TestLoad_Success(t *testing.T) {
	b := validBuilder()
	pkg, err := Load(b.fs(t), b.effectiveDir())
	if err != nil {
		t.Fatalf("expected load success, got %v", err)
	}
	meta := pkg.Meta()
	t.Logf("判定依据: 合法包 version=%s precision=%s inputDim=%d layers=%d weightsBytes=%d -> 加载成功",
		meta.Version, meta.Precision, meta.InputDim, len(meta.Layers), pkg.Bytes())
	if len(meta.Layers) != 2 || meta.Layers[0].InDim != 3 {
		t.Fatalf("unexpected layers: %+v", meta.Layers)
	}
}

func TestLoad_Rejections(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(b *pkgBuilder)
		wantErr error
	}{
		{"目录缺失", func(b *pkgBuilder) { b.loadDir = "nope" }, ErrPackageNotFound},
		{"缺少model.json", func(b *pkgBuilder) { b.noJSON = true }, ErrPackageNotFound},
		{"JSON损坏", func(b *pkgBuilder) { b.badJSON = true }, ErrMalformedPackage},
		{"层规格缺失", func(b *pkgBuilder) {
			b.layers[0] = map[string]interface{}{"name": "fc1"}
		}, ErrMalformedPackage},
		{"版本不支持", func(b *pkgBuilder) { b.version = "v9" }, ErrUnsupportedVersion},
		{"精度不支持", func(b *pkgBuilder) { b.precision = "fp16" }, ErrUnsupportedPrecision},
		{"v1不支持uint8组合", func(b *pkgBuilder) {
			b.version = V1
			b.precision = UINT8
		}, ErrVersionPrecisionMismatch},
		{"模型输入量化缺失", func(b *pkgBuilder) { b.inputQP = QuantParams{} }, ErrMissingQuantParams},
		{"层scale缺失", func(b *pkgBuilder) {
			b.layers[0]["quant"] = QuantParams{Scale: 0}
		}, ErrMissingQuantParams},
		{"零点越界", func(b *pkgBuilder) {
			b.layers[0]["quant"] = QuantParams{Scale: 0.05, ZeroPoint: 999}
		}, ErrMissingQuantParams},
		{"层间维度不匹配", func(b *pkgBuilder) {
			b.layers[1]["in_dim"] = 7
		}, ErrDimMismatch},
		{"权重文件缺失(部分加载)", func(b *pkgBuilder) {
			b.weightPatches["fc2.bin"] = patch{missing: true}
		}, ErrPartialWeights},
		{"权重被截断(部分权重)", func(b *pkgBuilder) {
			b.weightPatches["fc1.bin"] = patch{bytes: []byte{1, 2, 3}}
		}, ErrPartialWeights},
		{"权重多于声明", func(b *pkgBuilder) {
			b.weightPatches["fc2.bin"] = patch{bytes: make([]byte, 4*2+5)}
		}, ErrDimMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := validBuilder()
			tc.mutate(b)
			pkg, err := Load(b.fs(t), b.effectiveDir())
			if pkg != nil {
				t.Fatalf("输入: %s；期望拒绝但返回了包（可能静默部分加载）", tc.name)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("输入: %s；期望原因 %v，实际 %v", tc.name, tc.wantErr, err)
			}
			t.Logf("输入: %s；判定依据: errors.Is(err,%v)=true；错误明细: %v",
				tc.name, tc.wantErr, err)
		})
	}
}

func TestSupportedMatrix(t *testing.T) {
	if got := SupportedPrecisions(V1); len(got) != 1 || got[0] != INT8 {
		t.Fatalf("v1 应仅支持 int8, got=%v", got)
	}
	got := SupportedPrecisions(V2)
	if len(got) != 2 {
		t.Fatalf("v2 应支持 int8/uint8, got=%v", got)
	}
	t.Logf("判定依据: 支持版本=%v，v1精度=%v，v2精度=%v", SupportedVersions(), SupportedPrecisions(V1), got)
}
