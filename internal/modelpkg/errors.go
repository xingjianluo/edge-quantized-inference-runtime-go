package modelpkg

import "errors"

// 加载阶段可区分的失败原因。调用方应使用 errors.Is 判定具体原因。
var (
	// ErrPackageNotFound 模型包目录或必需文件不存在。
	ErrPackageNotFound = errors.New("modelpkg: package not found")
	// ErrMalformedPackage 元数据缺失、无法解析或文件结构不完整。
	ErrMalformedPackage = errors.New("modelpkg: malformed package")
	// ErrUnsupportedVersion 模型版本不在支持列表内。
	ErrUnsupportedVersion = errors.New("modelpkg: unsupported model version")
	// ErrUnsupportedPrecision 量化精度不在支持列表内。
	ErrUnsupportedPrecision = errors.New("modelpkg: unsupported precision")
	// ErrVersionPrecisionMismatch 版本与精度的组合不受支持。
	ErrVersionPrecisionMismatch = errors.New("modelpkg: version/precision combination not supported")
	// ErrMissingQuantParams 量化参数（scale/zero point）缺失。
	ErrMissingQuantParams = errors.New("modelpkg: missing quantization parameters")
	// ErrDimMismatch 权重维度与声明的输入/层维度不匹配。
	ErrDimMismatch = errors.New("modelpkg: weight dimension mismatch")
	// ErrPartialWeights 权重文件只包含部分层，禁止部分加载。
	ErrPartialWeights = errors.New("modelpkg: partial weights")
	// ErrChecksumMismatch 权重文件内容与元数据声明的校验和不符（内容被改动过）。
	ErrChecksumMismatch = errors.New("modelpkg: weight checksum mismatch")
	// ErrChecksumRequired 在线更新要求模型包必须显式声明全部权重的校验和。
	ErrChecksumRequired = errors.New("modelpkg: weight checksum required for online update")
)
