// Package testutil 为运行时测试提供确定性的内存模型包夹具。
// 所有模型都经由 modelpkg.Load 的完整校验路径构造，
// 保证测试用到的包与生产加载路径一致，绝不绕过校验。
package testutil

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"testing"
	"testing/fstest"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
)

// ToySpec 描述一个确定性玩具模型：两层全连接 2->3->2。
type ToySpec struct {
	Version   modelpkg.Version
	Precision modelpkg.Precision
	// LayerZP 每层零点（两层共用）。
	LayerZP int32
	// LayerScale 两层尺度。
	LayerScale [2]float64
	// Seed 为权重生成种子。为 0 时使用夹具默认种子 {fc1:21, fc2:7}；
	// 非 0 时 fc1 使用 Seed、fc2 使用 Seed+1，从而可用不同 Seed
	// 构造“版本/结构完全一致、仅权重内容不同”的新一代模型。
	Seed byte
	// WithChecksums 为 true 时在元数据中声明每个权重文件的 SHA-256。
	// 在线更新路径（LoadForUpdate）要求必须声明。
	WithChecksums bool
	// TamperFile 非空时，在写入文件系统前翻转该权重文件的首字节，
	// 但元数据仍声明原始校验和，用于模拟“权重内容被改动过”。
	TamperFile string
}

// DefaultToy 返回合法的 v2/int8 玩具模型规格。
func DefaultToy() ToySpec {
	return ToySpec{
		Version:    modelpkg.V2,
		Precision:  modelpkg.INT8,
		LayerZP:    0,
		LayerScale: [2]float64{0.1, 0.05},
	}
}

// ToyFS 按规格生成模型包文件系统（model.json + 两个权重文件）。
func ToyFS(spec ToySpec) (fs.FS, string) {
	dir := "toy"
	return buildFS(spec, dir), dir
}

func buildFS(spec ToySpec, dir string) fs.FS {
	type layerFile struct {
		name string
		in   int
		out  int
		seed byte
	}
	layers := []layerFile{
		{"fc1.bin", 2, 3, 21},
		{"fc2.bin", 3, 2, 7},
	}
	if spec.Seed != 0 {
		layers[0].seed = spec.Seed
		layers[1].seed = spec.Seed + 1
	}
	scales := spec.LayerScale

	meta := modelpkg.Metadata{
		Name:       "toy",
		Version:    spec.Version,
		Precision:  spec.Precision,
		InputDim:   2,
		InputQuant: modelpkg.QuantParams{Scale: 0.25, ZeroPoint: 0},
	}
	for i, lf := range layers {
		meta.Layers = append(meta.Layers, modelpkg.LayerSpec{
			Name:       lf.name[:len(lf.name)-4],
			InDim:      lf.in,
			OutDim:     lf.out,
			WeightFile: lf.name,
			Quant:      modelpkg.QuantParams{Scale: scales[i], ZeroPoint: spec.LayerZP},
		})
	}

	// 先生成原始权重字节；需要声明校验和时基于“未篡改”的字节计算摘要。
	raww := make(map[string][]byte, len(layers))
	for _, lf := range layers {
		n := lf.in * lf.out
		buf := make([]byte, n)
		for i := 0; i < n; i++ {
			buf[i] = byte((int(lf.seed) + i*13) % 256)
		}
		raww[lf.name] = buf
	}
	if spec.WithChecksums {
		meta.WeightSHA256 = make(map[string]string, len(layers))
		for _, lf := range layers {
			sum := sha256.Sum256(raww[lf.name])
			meta.WeightSHA256[lf.name] = hex.EncodeToString(sum[:])
		}
	}

	mfs := fstest.MapFS{}
	raw, _ := json.Marshal(meta)
	mfs[dir+"/model.json"] = &fstest.MapFile{Data: raw}
	for _, lf := range layers {
		buf := raww[lf.name]
		if spec.TamperFile == lf.name {
			// 模拟内容被改动：首字节翻转，但校验和仍是旧值。
			tampered := make([]byte, len(buf))
			copy(tampered, buf)
			tampered[0] ^= 0xFF
			buf = tampered
		}
		mfs[dir+"/"+lf.name] = &fstest.MapFile{Data: buf}
	}
	return mfs
}

// LoadToy 构造文件系统并走完整 Load 校验，返回模型包。
func LoadToy(t *testing.T, spec ToySpec) *modelpkg.Package {
	t.Helper()
	ffsys, dir := ToyFS(spec)
	pkg, err := modelpkg.Load(ffsys, dir)
	if err != nil {
		t.Fatalf("夹具模型加载失败（不应失败）: spec=%+v err=%v", spec, err)
	}
	return pkg
}

// LoadToyForUpdate 走在线更新专用的校验路径（强制校验和）。
func LoadToyForUpdate(t *testing.T, spec ToySpec) *modelpkg.Package {
	t.Helper()
	ffsys, dir := ToyFS(spec)
	pkg, err := modelpkg.LoadForUpdate(ffsys, dir)
	if err != nil {
		t.Fatalf("夹具模型更新加载失败（不应失败）: spec=%+v err=%v", spec, err)
	}
	return pkg
}

// SignedToy 返回一个带校验和、可用于在线更新的合法新规格：
// 调用方可在其基础上修改 Seed/版本/精度等字段。
func SignedToy() ToySpec {
	s := DefaultToy()
	s.WithChecksums = true
	return s
}
