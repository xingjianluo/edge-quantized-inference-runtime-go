// 命令 genmodels 确定性地生成 examples/models 下的示例模型包：
// 合法两代（同 v2/int8 结构、不同权重内容，均带 SHA-256 校验和）、
// 以及故意非法的更新包（校验和不符、未声明校验和）。
// 重新运行只会逐字节复现同样的包，便于本地与 CI 校验。
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

type quant struct {
	Scale     float64 `json:"scale"`
	ZeroPoint int32   `json:"zero_point"`
}
type layer struct {
	Name       string `json:"name"`
	InDim      int    `json:"in_dim"`
	OutDim     int    `json:"out_dim"`
	WeightFile string `json:"weight_file"`
	Quant      quant  `json:"quant"`
}
type modelJSON struct {
	Name         string            `json:"name"`
	Version      string            `json:"version"`
	Precision    string            `json:"precision"`
	InputDim     int               `json:"input_dim"`
	InputQuant   quant             `json:"input_quant"`
	Layers       []layer           `json:"layers"`
	WeightSHA256 map[string]string `json:"weight_sha256,omitempty"`
}

func weights(in, out, seed int) []byte {
	buf := make([]byte, in*out)
	for i := range buf {
		buf[i] = byte((seed + i*13) % 256)
	}
	return buf
}

func checksumsFor(m modelJSON, seeds map[string]int) map[string]string {
	out := map[string]string{}
	for _, l := range m.Layers {
		b := weights(l.InDim, l.OutDim, seeds[l.WeightFile])
		sum := sha256.Sum256(b)
		out[l.WeightFile] = hex.EncodeToString(sum[:])
	}
	return out
}

// writeModel 写出一个模型包；signed 控制是否写入校验和声明；
// tamper 翻转 fc1 首字节（校验和若声明仍是原始值，用于模拟内容被改动）。
func writeModel(root string, m modelJSON, seeds map[string]int, signed, tamper bool) error {
	dir := filepath.Join(root, m.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if signed {
		m.WeightSHA256 = checksumsFor(m, seeds)
	}
	for _, l := range m.Layers {
		b := weights(l.InDim, l.OutDim, seeds[l.WeightFile])
		if tamper && l.WeightFile == "fc1.bin" {
			b[0] ^= 0xFF
		}
		if err := os.WriteFile(filepath.Join(dir, l.WeightFile), b, 0o644); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "model.json"), append(raw, '\n'), 0o644)
}

func baseModel(name string) (modelJSON, map[string]int) {
	m := modelJSON{
		Name:       name,
		Version:    "v2",
		Precision:  "int8",
		InputDim:   2,
		InputQuant: quant{Scale: 0.25, ZeroPoint: 0},
		Layers: []layer{
			{Name: "fc1", InDim: 2, OutDim: 3, WeightFile: "fc1.bin",
				Quant: quant{Scale: 0.1, ZeroPoint: 0}},
			{Name: "fc2", InDim: 3, OutDim: 2, WeightFile: "fc2.bin",
				Quant: quant{Scale: 0.05, ZeroPoint: 0}},
		},
	}
	return m, map[string]int{"fc1.bin": 21, "fc2.bin": 7}
}

func main() {
	root := flag.String("out", "examples/models", "输出目录")
	flag.Parse()

	// 第一代：与历史 toy-int8 权重一致，并补全 SHA-256 校验和（在线更新要求）。
	m1, seeds1 := baseModel("toy-int8")
	if err := writeModel(*root, m1, seeds1, true, false); err != nil {
		fail(err)
	}

	// 第二代：结构/版本完全一致，仅权重内容不同（不同种子）。
	m2, seeds2 := baseModel("toy-int8-v2")
	seeds2["fc1.bin"] = 99
	seeds2["fc2.bin"] = 100
	if err := writeModel(*root, m2, seeds2, true, false); err != nil {
		fail(err)
	}

	// 非法更新包 A：声明了校验和，但 fc1 内容被篡改 -> ErrChecksumMismatch。
	bad1, seedsBad1 := baseModel("bad-update-tampered")
	if err := writeModel(*root, bad1, seedsBad1, true, true); err != nil {
		fail(err)
	}

	// 非法更新包 B：不带任何校验和声明 -> ErrChecksumRequired。
	bad2, seedsBad2 := baseModel("bad-update-no-checksum")
	if err := writeModel(*root, bad2, seedsBad2, false, false); err != nil {
		fail(err)
	}

	fmt.Println("已生成: toy-int8（第一代）、toy-int8-v2（新一代）、" +
		"bad-update-tampered（校验和不符）、bad-update-no-checksum（未声明校验和）")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
