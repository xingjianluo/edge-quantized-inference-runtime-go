package modelpkg

// Version 模型包格式版本。
type Version string

const (
	// V1 初代格式：仅支持 INT8。
	V1 Version = "v1"
	// V2 二代格式：支持 INT8 与 UINT8，要求每层显式量化参数。
	V2 Version = "v2"
)

// Precision 权重量化精度。
type Precision string

const (
	// INT8 有符号 8 比特量化。
	INT8 Precision = "int8"
	// UINT8 无符号 8 比特量化（仅 v2 支持）。
	UINT8 Precision = "uint8"
)

// QuantParams 单层量化参数：real_value = scale * (q - zeroPoint)。
type QuantParams struct {
	Scale     float64 `json:"scale"`
	ZeroPoint int32   `json:"zero_point"`
}

// LayerSpec 一层权重的规格与量化参数。
type LayerSpec struct {
	Name       string `json:"name"`
	InDim      int    `json:"in_dim"`
	OutDim     int    `json:"out_dim"`
	WeightFile string `json:"weight_file"`
	// WeightSHA256 可选：权重文件内容的 SHA-256（十六进制小写）。
	// 非空时加载阶段会校验实际内容，不一致以 ErrWeightCorrupted 拒绝。
	WeightSHA256 string      `json:"weight_sha256,omitempty"`
	Quant        QuantParams `json:"quant"`
	// Weights 为 INT8 权重；UWeights 为 UINT8 权重（HasUWeights 标记其存在）。
	// 二者按行主序存放，长度必须等于 InDim*OutDim。
	Weights     []int8  `json:"-"`
	UWeights    []uint8 `json:"-"`
	HasUWeights bool    `json:"-"`
}

// WeightCount 返回该层应有的权重元素数。
func (l LayerSpec) WeightCount() int { return l.InDim * l.OutDim }

// Metadata 模型包元数据（model.json）。
type Metadata struct {
	Name      string    `json:"name"`
	Version   Version   `json:"version"`
	Precision Precision `json:"precision"`
	// ModelVersion 业务侧模型版本号（如 "2026.10-r3"），仅作标识与展示，
	// 不参与兼容性校验；在线更新/回退时用于区分同名模型的不同代。
	ModelVersion string `json:"model_version,omitempty"`
	InputDim     int    `json:"input_dim"`
	// InputQuant 为输入特征的量化参数（float -> q8），必须显式给出。
	InputQuant QuantParams `json:"input_quant"`
	Layers     []LayerSpec `json:"layers"`
}

// Package 已通过完整校验、可直接用于推理的模型包。
// 加载成功即保证：版本/精度受支持、每层量化参数齐备、
// 权重维度与规格一致、不存在部分权重、声明了校验和的权重内容未被改动。
// 运行时不得再做静默降级。
type Package struct {
	meta Metadata
	// bytes 是该模型包占用的设备内存估算（字节）。
	bytes int64
	// fingerprint 是包内容（元数据 + 全部权重字节）的 SHA-256 十六进制摘要，
	// 加载时计算；结构相同但权重不同的两个包指纹必然不同。
	fingerprint string
}

// Meta 返回元数据只读副本。
func (p *Package) Meta() Metadata { return p.meta }

// Bytes 返回模型包内存占用估算（字节）。
func (p *Package) Bytes() int64 { return p.bytes }

// Fingerprint 返回包内容指纹（SHA-256 十六进制），用于区分不同代的模型包。
func (p *Package) Fingerprint() string { return p.fingerprint }
