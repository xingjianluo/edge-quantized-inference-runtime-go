package modelpkg

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
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
//  9. 权重内容与校验和不符   -> ErrChecksumMismatch
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

	// 逐层读取权重；任何一层失败即整体失败，绝不返回部分权重的包。
	loaded := make([]LayerSpec, len(meta.Layers))
	var totalBytes int64
	revHasher := sha256.New()
	revHasher.Write([]byte(canonicalStructDigest(meta)))
	for i, layer := range meta.Layers {
		wp := fsPath(dir, layer.WeightFile)
		raw, err := fs.ReadFile(ffsys, wp)
		if err != nil {
			return nil, fmt.Errorf("%w: layer %q: %v", ErrPartialWeights, layer.Name, err)
		}
		want := layer.WeightCount()
		got := len(raw)
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
		// 维度确认无误后再核对内容校验和：任何字节级改动都在此被发现。
		if declared, ok := meta.WeightSHA256[layer.WeightFile]; ok {
			sum := sha256.Sum256(raw)
			if actual := hex.EncodeToString(sum[:]); !stringsEqualFold(actual, declared) {
				return nil, fmt.Errorf("%w: layer %q file %q: declared %s, actual %s",
					ErrChecksumMismatch, layer.Name, layer.WeightFile, declared, actual)
			}
		}
		// 指纹按层顺序纳入原始权重字节（层之间以层名分隔，消除拼接歧义）。
		revHasher.Write([]byte(layer.Name))
		revHasher.Write(raw)
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

	return &Package{
		meta:     meta,
		bytes:    totalBytes,
		revision: hex.EncodeToString(revHasher.Sum(nil)),
	}, nil
}

// LoadForUpdate 供“在线更新”使用的加载入口：除 Load 的全部校验外，
// 还要求模型包为每一层权重显式声明 SHA-256 校验和（ErrChecksumRequired），
// 并逐文件核对实际字节（ErrChecksumMismatch）。这样推送到设备的新一代
// 权重只要缺失完整性声明或内容被改动过，就到不了切换环节。
func LoadForUpdate(ffsys fs.FS, dir string) (*Package, error) {
	pkg, err := Load(ffsys, dir)
	if err != nil {
		return nil, err
	}
	meta := pkg.Meta()
	if len(meta.WeightSHA256) != len(meta.Layers) {
		return nil, fmt.Errorf("%w: package %q declares %d checksums for %d layers",
			ErrChecksumRequired, meta.Name, len(meta.WeightSHA256), len(meta.Layers))
	}
	return pkg, nil
}

// validateChecksumRefs 校验 WeightSHA256 只引用实际声明的权重文件，
// 且十六进制摘要长度/格式合法，避免用拼写错误的文件名“声明了个寂寞”。
func validateChecksumRefs(meta Metadata) error {
	if len(meta.WeightSHA256) == 0 {
		return nil
	}
	known := make(map[string]struct{}, len(meta.Layers))
	for _, l := range meta.Layers {
		known[l.WeightFile] = struct{}{}
	}
	for file, sum := range meta.WeightSHA256 {
		if _, ok := known[file]; !ok {
			return fmt.Errorf("%w: checksum declared for unknown weight file %q",
				ErrMalformedPackage, file)
		}
		if len(sum) != sha256.Size*2 {
			return fmt.Errorf("%w: weight file %q: checksum must be %d hex chars, got %d",
				ErrMalformedPackage, file, sha256.Size*2, len(sum))
		}
		if _, err := hex.DecodeString(sum); err != nil {
			return fmt.Errorf("%w: weight file %q: checksum not hex: %v",
				ErrMalformedPackage, file, err)
		}
	}
	return nil
}

// canonicalStructDigest 给出与权重字节无关的结构部分摘要输入，
// 使“结构不同但权重恰好相同”的包指纹也不同。
func canonicalStructDigest(meta Metadata) string {
	type l struct {
		Name       string      `json:"n"`
		InDim      int         `json:"i"`
		OutDim     int         `json:"o"`
		WeightFile string      `json:"f"`
		Quant      QuantParams `json:"q"`
	}
	c := struct {
		Name       string    `json:"name"`
		Version    Version   `json:"v"`
		Precision  Precision `json:"p"`
		InputDim   int       `json:"d"`
		InputQuant QuantParams
		Layers     []l
	}{
		Name:       meta.Name,
		Version:    meta.Version,
		Precision:  meta.Precision,
		InputDim:   meta.InputDim,
		InputQuant: meta.InputQuant,
	}
	for _, layer := range meta.Layers {
		c.Layers = append(c.Layers, l{
			Name: layer.Name, InDim: layer.InDim, OutDim: layer.OutDim,
			WeightFile: layer.WeightFile, Quant: layer.Quant,
		})
	}
	raw, _ := json.Marshal(c)
	return string(raw)
}

// stringsEqualFold 在不引入 strings 包的前提下做大小写不敏感比较
// （SHA-256 摘要规范为小写十六进制，兼容大写输入）。
func stringsEqualFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

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
	return validateChecksumRefs(meta)
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
