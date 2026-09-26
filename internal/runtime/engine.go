package runtime

import (
	"fmt"
	"math"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
)

// inferOne 是无副作用的纯函数确定性推理核。
//
// 量化约定（每层独立零点与尺度，real = scale*(q-zero)）：
//
//	acc    = Σ (wq - zw) * (aq - za)          // 纯 int64 整数累加
//	q_out  = round_fixed(acc * M) + z_out      // M = w_scale*a_scale/out_scale
//	M 以 mult/2^shift 定点表示，全程整数运算
//
// 因为不依赖浮点累加顺序、融合指令或 goroutine 调度：
//   - 同一输入在 Infer 与 InferBatch 下逐位一致；
//   - 重复执行、跨会话执行结果相同；
//   - INT8 与 UINT8 的权重解释不同，产出不同数值，精度可区分；
//   - 不支持的模型在加载阶段已被拒绝，推理核只面对合法模型。
func inferOne(model modelView, input []float32) ([]float32, error) {
	if len(input) != model.inputDim {
		return nil, fmt.Errorf("%w: model requires %d values, got %d",
			ErrInputDimMismatch, model.inputDim, len(input))
	}

	// 1) 输入量化：q = round(x / s_in) + z_in，饱和到精度值域。
	prevQ := make([]int32, model.inputDim)
	for i, x := range input {
		q := math.Round(float64(x)/model.inputScale) + float64(model.inputZero)
		prevQ[i] = clampQ(q, model.unsigned)
	}
	prevZero := model.inputZero

	// 2) 逐层量化矩阵乘（行主序权重）。
	for li := range model.layers {
		layer := &model.layers[li]
		outQ := make([]int32, layer.outDim)
		base := 0
		for o := 0; o < layer.outDim; o++ {
			var acc int64
			for i := 0; i < layer.inDim; i++ {
				wq := layer.weightQ(base + i)
				acc += int64(wq-layer.weightZero) * int64(prevQ[i]-prevZero)
			}
			base += layer.inDim

			scaled := mulScale(acc, layer.mult, layer.shift) + int64(layer.zero)
			v := clampQ(float64(scaled), model.unsigned)

			// 隐藏层 ReLU：real>=0 等价于 q>=z_out（scale 为正）。
			if li < len(model.layers)-1 && v < layer.zero {
				v = layer.zero
			}
			outQ[o] = v
		}
		prevQ = outQ
		prevZero = layer.zero
	}

	// 3) 末层反量化：float32(s_out*(q-z_out))。
	last := &model.layers[len(model.layers)-1]
	out := make([]float32, last.outDim)
	for i, q := range prevQ {
		out[i] = float32(last.outScale * float64(q-last.zero))
	}
	return out, nil
}

// modelView 推理核对模型的只读视图。
type modelView struct {
	inputDim   int
	unsigned   bool
	inputScale float64
	inputZero  int32
	layers     []layerView
}

type layerView struct {
	inDim      int
	outDim     int
	weights    []int8
	uweights   []uint8
	weightZero int32
	zero       int32
	outScale   float64
	// mult/shift 为定点乘子：round(acc * mult / 2^shift) ≈ acc * (w_scale*a_scale/out_scale)。
	mult  int64
	shift uint
}

// weightQ 按模型精度读取权重的量化整数值。
func (l *layerView) weightQ(idx int) int32 {
	if l.uweights != nil {
		return int32(l.uweights[idx])
	}
	return int32(l.weights[idx])
}

// buildView 从已校验模型包构造只读推理视图（不含任何可变状态）。
func buildView(pkg *modelpkg.Package) modelView {
	meta := pkg.Meta()
	v := modelView{
		inputDim:   meta.InputDim,
		unsigned:   meta.Precision == modelpkg.UINT8,
		inputScale: meta.InputQuant.Scale,
		inputZero:  meta.InputQuant.ZeroPoint,
	}
	actScale := meta.InputQuant.Scale
	for _, l := range meta.Layers {
		// M = 权重尺度 * 上一层激活尺度 / 本层输出尺度。
		mult, shift := quantizeMultiplier(l.Quant.Scale*actScale, l.Quant.Scale)
		lv := layerView{
			inDim:      l.InDim,
			outDim:     l.OutDim,
			weights:    l.Weights,
			uweights:   l.UWeights,
			weightZero: l.Quant.ZeroPoint,
			zero:       l.Quant.ZeroPoint,
			outScale:   l.Quant.Scale,
			mult:       mult,
			shift:      shift,
		}
		v.layers = append(v.layers, lv)
		actScale = l.Quant.Scale
	}
	return v
}

// quantizeMultiplier 计算定点乘子 mult/2^shift ≈ numerator/denominator。
// 用整数表达缩放，保证跨平台、跨调用逐位可复现。
func quantizeMultiplier(numerator, denominator float64) (int64, uint) {
	r := numerator / denominator
	if !(r > 0) || math.IsInf(r, 0) {
		return 0, 0
	}
	const shift = uint(31)
	mult := int64(math.Round(r * float64(int64(1)<<shift)))
	if mult <= 0 {
		return 1, shift // 退化为 1/2^31，避免除零并保持确定性
	}
	return mult, shift
}

// mulScale 计算 round(acc * mult / 2^shift)，纯整数四舍五入。
func mulScale(acc, mult int64, shift uint) int64 {
	if shift == 0 {
		return acc
	}
	prod := acc * mult
	half := int64(1) << (shift - 1)
	if prod >= 0 {
		return (prod + half) >> shift
	}
	return -((-prod + half) >> shift)
}

// clampQ 将量化值饱和到权重精度值域。
func clampQ(q float64, unsigned bool) int32 {
	v := int32(math.Round(q))
	if unsigned {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return v
	}
	if v < -128 {
		return -128
	}
	if v > 127 {
		return 127
	}
	return v
}

// workspaceBytes 估算 batchSize 条推理的会话工作区字节数（确定性记账口径）：
// 每条 (inputDim + 最大层 outDim) 个 int32。
func workspaceBytes(model modelView, batchSize int) int64 {
	maxOut := 0
	for _, l := range model.layers {
		if l.outDim > maxOut {
			maxOut = l.outDim
		}
	}
	return int64(model.inputDim+maxOut) * 4 * int64(batchSize)
}
