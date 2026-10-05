package modelpkg

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"strings"
)

// SupportedVersions 返回运行时支持的模型格式版本。
func SupportedVersions() []Version {
	return []Version{V1, V2}
}

// SupportedPrecisions 返回给定版本支持的量化精度；
// 版本本身不受支持时返回 nil。
func SupportedPrecisions(v Version) []Precision {
	switch v {
	case V1:
		return []Precision{INT8}
	case V2:
		return []Precision{INT8, UINT8}
	default:
		return nil
	}
}

// isSupportedVersion 判定版本是否受支持。
func isSupportedVersion(v Version) bool {
	for _, sv := range SupportedVersions() {
		if sv == v {
			return true
		}
	}
	return false
}

// isSupportedPrecision 判定某版本下的精度是否受支持。
func isSupportedPrecision(v Version, p Precision) bool {
	for _, sp := range SupportedPrecisions(v) {
		if sp == p {
			return true
		}
	}
	return false
}

// Load 从 fsys 中的 dir 目录完整加载并校验模型包。
//
// 包结构：
//
//	dir/model.json        元数据（版本、精度、输入维度、各层规格与量化参数）
//	dir/<weight_file>     每层一个原始小端字节权重文件
//
// 校验顺序保证失败原因可区分（errors.Is）：
//  1. 目录/文件缺失          -> ErrPackageNotFound
//  2. 元数据无法解析/结构非法 -> ErrMalformedPackage
//  3. 版本不受支持           -> ErrUnsupportedVersion
//  4. 精度不受支持           -> ErrUnsupportedPrecision
//  5. 版本+精度组合不支持    -> ErrVersionPrecisionMismatch
//  6. 量化参数缺失/非法      -> ErrMissingQuantParams
//  7. 权重缺失/被截断(部分)  -> ErrPartialWeights
//  8. 权重维度不匹配         -> ErrDimMismatch
//  9. 权重校验和不一致       -> ErrWeightCorrupted（声明了 weight_sha256 时）
//
// 任一失败都不会返回 Package，绝不静默降级或部分加载。
func Load(ffsys fs.FS, dir string) (*Package, error) {
	metaPath := fsPath(dir, "model.json")
	f, err := ffsys.Open(metaPath)
	if err != nil {
		return nil, fmt.Errorf("%w: open %s: %v", ErrPackageNotFound, metaPath, err)
	}
	data, err := io.ReadAll(f)
	_ = f.Close()
	if err != nil {
		return nil, fmt.Errorf("%w: read %s: %v", ErrMalformedPackage, metaPath, err)
	}

	var meta Metadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("%w: parse %s: %v", ErrMalformedPackage, metaPath, err)
	}
	if err := validateMetadata(meta); err != nil {
		return nil, err
	}

	// 指纹覆盖元数据原文与全部权重字节：任何一处内容变动都会改变指纹。
	fp := sha256.New()
	fp.Write(data)

	// 逐层读取权重；任何一层失败即整体失败，绝不返回部分权重的包。
	loaded := make([]LayerSpec, len(meta.Layers))
	var totalBytes int64
	for i, layer := range meta.Layers {
		wp := fsPath(dir, layer.WeightFile)
		raw, err := fs.ReadFile(ffsys, wp)
		if err != nil {
			return nil, fmt.Errorf("%w: layer %q: %v", ErrPartialWeights, layer.Name, err)
		}
		want := layer.WeightCount()
		got := 0
		switch meta.Precision {
		case INT8:
			got = len(raw)
		case UINT8:
			got = len(raw)
		}
		if got < want {
			// 权重被截断：只包含部分权重，明确拒绝。
			return nil, fmt.Errorf("%w: layer %q: have %d bytes, want %d",
				ErrPartialWeights, layer.Name, got, want)
		}
		if got > want {
			// 权重多于声明：维度对不上，同样拒绝，避免静默忽略尾部数据。
			return nil, fmt.Errorf("%w: layer %q: have %d bytes, want %d",
				ErrDimMismatch, layer.Name, got, want)
		}
		// 声明了校验和则必须逐字节一致：内容被改动过即拒绝。
		if layer.WeightSHA256 != "" {
			sum := sha256.Sum256(raw)
			if !strings.EqualFold(hex.EncodeToString(sum[:]), layer.WeightSHA256) {
				return nil, fmt.Errorf("%w: layer %q: weight file %s content altered",
					ErrWeightCorrupted, layer.Name, layer.WeightFile)
			}
		}
		writeFingerprint(fp, raw)
		if meta.Precision == INT8 {
			w := make([]int8, want)
			for j := 0; j < want; j++ {
				w[j] = int8(raw[j])
			}
			layer.Weights = w
		} else {
			u := make([]uint8, want)
			copy(u, raw)
			layer.UWeights = u
			layer.HasUWeights = true
		}
		totalBytes += int64(want)
		loaded[i] = layer
	}
	meta.Layers = loaded

	return &Package{meta: meta, bytes: totalBytes,
		fingerprint: hex.EncodeToString(fp.Sum(nil))}, nil
}

// writeFingerprint 把一层权重字节混入包指纹。
func writeFingerprint(h hash.Hash, raw []byte) { h.Write(raw) }

// validateMetadata 只做与权重文件无关的结构性与兼容性校验。
func validateMetadata(meta Metadata) error {
	if meta.Name == "" || meta.InputDim <= 0 || len(meta.Layers) == 0 {
		return fmt.Errorf("%w: name/input_dim/layers must be present and positive",
			ErrMalformedPackage)
	}

	// 版本。
	if !isSupportedVersion(meta.Version) {
		return fmt.Errorf("%w: %q (supported: v1,v2)",
			ErrUnsupportedVersion, meta.Version)
	}

	// 精度：先区分“精度名称本身未知”与“该版本下不允许此组合”。
	precisionKnown := meta.Precision == INT8 || meta.Precision == UINT8
	if !precisionKnown {
		return fmt.Errorf("%w: %q", ErrUnsupportedPrecision, meta.Precision)
	}
	if !isSupportedPrecision(meta.Version, meta.Precision) {
		// 例如 v1 + uint8：版本与精度都认识，但组合不被支持。
		return fmt.Errorf("%w: %s does not support %s",
			ErrVersionPrecisionMismatch, meta.Version, meta.Precision)
	}

	// 版本/精度确认后再校验量化参数（其值域依赖精度）。
	if meta.InputQuant.Scale <= 0 ||
		!zeroPointInRange(meta.Precision, meta.InputQuant.ZeroPoint) {
		return fmt.Errorf("%w: model input_quant missing or out of range",
			ErrMissingQuantParams)
	}

	// 逐层结构与量化参数、层间维度连续性。
	prevOut := meta.InputDim
	for i, layer := range meta.Layers {
		if layer.Name == "" || layer.InDim <= 0 || layer.OutDim <= 0 || layer.WeightFile == "" {
			return fmt.Errorf("%w: layer %d missing name/dims/weight_file",
				ErrMalformedPackage, i)
		}
		if layer.InDim != prevOut {
			return fmt.Errorf("%w: layer %q in_dim=%d but previous output/input=%d",
				ErrDimMismatch, layer.Name, layer.InDim, prevOut)
		}
		// scale 必须显式给出且为正；zero point 必须在量化值域内。
		if layer.Quant.Scale <= 0 {
			return fmt.Errorf("%w: layer %q: non-positive or missing scale %v",
				ErrMissingQuantParams, layer.Name, layer.Quant.Scale)
		}
		if !zeroPointInRange(meta.Precision, layer.Quant.ZeroPoint) {
			return fmt.Errorf("%w: layer %q: zero_point %d out of %s range",
				ErrMissingQuantParams, layer.Name, layer.Quant.ZeroPoint, meta.Precision)
		}
		prevOut = layer.OutDim
	}
	return nil
}

// zeroPointInRange 校验零点落在对应精度的可表示区间内。
func zeroPointInRange(p Precision, zp int32) bool {
	switch p {
	case INT8:
		return zp >= -128 && zp <= 127
	case UINT8:
		return zp >= 0 && zp <= 255
	default:
		return false
	}
}

// fsPath 拼接目录与文件名，避免直接做字符串假设。
func fsPath(dir, name string) string {
	if dir == "" || dir == "." {
		return name
	}
	return dir + "/" + name
}
