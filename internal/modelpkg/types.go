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
	Name       string      `json:"name"`
	InDim      int         `json:"in_dim"`
	OutDim     int         `json:"out_dim"`
	WeightFile string      `json:"weight_file"`
	Quant      QuantParams `json:"quant"`
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
	InputDim  int       `json:"input_dim"`
	// InputQuant 为输入特征的量化参数（float -> q8），必须显式给出。
	InputQuant QuantParams `json:"input_quant"`
	Layers     []LayerSpec `json:"layers"`
	// WeightSHA256 可选：weight_file -> 该权重文件原始字节的 SHA-256（小写十六进制）。
	// 一旦在元数据中声明，加载时必须逐文件核对；不符即 ErrChecksumMismatch。
	// 在线更新路径要求必须声明，以防推送内容被改动过的权重。
	WeightSHA256 map[string]string `json:"weight_sha256,omitempty"`
}

// Package 已通过完整校验、可直接用于推理的模型包。
// 加载成功即保证：版本/精度受支持、每层量化参数齐备、
// 权重维度与规格一致、不存在部分权重。运行时不得再做静默降级。
type Package struct {
	meta Metadata
	// bytes 是该模型包占用的设备内存估算（字节）。
	bytes int64
	// revision 是对全部权重原始字节 + 模型结构的内容指纹（SHA-256 十六进制），
	// 用于区分“版本/精度字符串相同、但权重内容不同”的两代模型。
	revision string
}

// Meta 返回元数据只读副本。
func (p *Package) Meta() Metadata { return p.meta }

// Bytes 返回模型包内存占用估算（字节）。
func (p *Package) Bytes() int64 { return p.bytes }

// Revision 返回模型包内容指纹：只要权重或结构不同，指纹必然不同。
// 即使两个包的 version/precision 字符串完全一致也能据此区分。
func (p *Package) Revision() string { return p.revision }

// HasChecksums 报告该包是否声明了权重校验和。
// 在线更新只接受带校验和的包；本地直接 Load 旧格式包则保持兼容（可选声明）。
func (p *Package) HasChecksums() bool { return len(p.meta.WeightSHA256) > 0 }
