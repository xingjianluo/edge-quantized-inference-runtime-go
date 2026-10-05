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
	// ModelVersion 业务侧模型版本号，写入元数据 model_version。
	ModelVersion string
	// SeedOffset 叠加在权重种子上的偏移：不同偏移产出不同权重内容，
	// 从而得到“结构相同但结果可区分”的不同代模型。
	SeedOffset byte
	// WithChecksum 为每层权重写入正确的 weight_sha256（用于防篡改校验路径）。
	WithChecksum bool
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
	scales := spec.LayerScale

	meta := modelpkg.Metadata{
		Name:         "toy",
		Version:      spec.Version,
		Precision:    spec.Precision,
		ModelVersion: spec.ModelVersion,
		InputDim:     2,
		InputQuant:   modelpkg.QuantParams{Scale: 0.25, ZeroPoint: 0},
	}
	weightBytes := map[string][]byte{}
	for _, lf := range layers {
		n := lf.in * lf.out
		buf := make([]byte, n)
		for i := 0; i < n; i++ {
			buf[i] = byte((int(lf.seed) + int(spec.SeedOffset) + i*13) % 256)
		}
		weightBytes[lf.name] = buf
	}
	for i, lf := range layers {
		layer := modelpkg.LayerSpec{
			Name:       lf.name[:len(lf.name)-4],
			InDim:      lf.in,
			OutDim:     lf.out,
			WeightFile: lf.name,
			Quant:      modelpkg.QuantParams{Scale: scales[i], ZeroPoint: spec.LayerZP},
		}
		if spec.WithChecksum {
			sum := sha256.Sum256(weightBytes[lf.name])
			layer.WeightSHA256 = hex.EncodeToString(sum[:])
		}
		meta.Layers = append(meta.Layers, layer)
	}

	mfs := fstest.MapFS{}
	raw, _ := json.Marshal(meta)
	mfs[dir+"/model.json"] = &fstest.MapFile{Data: raw}
	for name, buf := range weightBytes {
		mfs[dir+"/"+name] = &fstest.MapFile{Data: buf}
	}
	return mfs
}

// LoadToy 构造文件系统并走完整 Load 校验，返回模型包。
func LoadToy(t *testing.T, spec ToySpec) *modelpkg.Package {
	t.Helper()
	ffsys, dir := ToyFS(spec)
	pkg, err := modelpkg.Load(ffsys, dir)
	if err != nil {
		t.Fatalf("夹具模型加载失败（不应失败）: %v", err)
	}
	return pkg
}
