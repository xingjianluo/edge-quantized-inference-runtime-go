// 命令 edge-runtime 是本地边缘推理运行时的最小演示：
// 从磁盘加载一个量化模型包，校验通过后在两个独立会话上分别做
// 逐条与批量推理，校验结果一致并打印判定依据；加载失败则按可区分原因报告。
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/runtime"
)

func main() {
	dir := flag.String("model", "examples/models/toy-int8", "模型包目录")
	budget := flag.Int64("budget", 1<<20, "设备内存预算（字节）")
	flag.Parse()

	// 1) 加载与校验。任何不兼容都在这里被拒绝，绝不静默降级。
	pkg, err := modelpkg.Load(os.DirFS("."), *dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "模型包被拒绝: %v\n", err)
		fmt.Fprintf(os.Stderr, "原因归类: %s\n", classifyLoadErr(err))
		os.Exit(1)
	}
	m := pkg.Meta()
	fmt.Printf("已加载模型: name=%s version=%s precision=%s inputDim=%d weightBytes=%d\n",
		m.Name, m.Version, m.Precision, m.InputDim, pkg.Bytes())

	// 2) 构造运行时并登记模型（设备预算核算）。
	rt, err := runtime.New(runtime.Config{MaxDeviceBytes: *budget})
	if err != nil {
		fail("创建运行时", err)
	}
	if err := rt.LoadModel(m.Name, pkg); err != nil {
		fail("登记模型", err)
	}

	inputs := [][]float32{{0.1, -0.5}, {1.25, 0}, {-0.75, 0.5}, {2, -2}}

	// 3) 会话 A：逐条推理。
	sessA, err := rt.NewSession(m.Name, runtime.SessionOptions{})
	if err != nil {
		fail("创建会话A", err)
	}
	single := make([][]float32, len(inputs))
	for i, in := range inputs {
		out, err := sessA.Infer(context.Background(), in)
		if err != nil {
			fail("逐条推理", err)
		}
		single[i] = out
		fmt.Printf("会话A 逐条 输入=%v 输出=%v\n", in, out)
	}

	// 4) 会话 B：批量推理（与 A 不共享可变状态）。
	sessB, err := rt.NewSession(m.Name, runtime.SessionOptions{})
	if err != nil {
		fail("创建会话B", err)
	}
	batch, err := sessB.InferBatch(context.Background(), inputs)
	if err != nil {
		fail("批量推理", err)
	}

	// 5) 判定一致性。
	consistent := true
	for i := range inputs {
		if len(batch[i]) != len(single[i]) {
			consistent = false
			break
		}
		for j := range batch[i] {
			if batch[i][j] != single[i][j] {
				consistent = false
			}
		}
	}
	if !consistent {
		fmt.Println("判定: 批量与逐条结果不一致（不应发生）")
		os.Exit(2)
	}
	fmt.Println("判定依据: 会话A逐条 与 会话B批量 结果逐位一致；重复执行同样输入结论可复现")

	_ = sessA.Close()
	_ = sessB.Close()
	if err := rt.UnloadModel(m.Name); err != nil {
		fail("卸载模型", err)
	}
	if err := rt.Close(); err != nil {
		fail("关闭运行时", err)
	}
}

// classifyLoadErr 把加载错误映射为可读的可区分原因。
func classifyLoadErr(err error) string {
	switch {
	case errors.Is(err, modelpkg.ErrPackageNotFound):
		return "包/文件缺失"
	case errors.Is(err, modelpkg.ErrUnsupportedVersion):
		return "版本不受支持"
	case errors.Is(err, modelpkg.ErrUnsupportedPrecision):
		return "精度不受支持"
	case errors.Is(err, modelpkg.ErrVersionPrecisionMismatch):
		return "版本与精度组合不支持"
	case errors.Is(err, modelpkg.ErrMissingQuantParams):
		return "量化参数缺失或非法"
	case errors.Is(err, modelpkg.ErrDimMismatch):
		return "维度不匹配"
	case errors.Is(err, modelpkg.ErrPartialWeights):
		return "权重不完整（拒绝部分加载）"
	case errors.Is(err, modelpkg.ErrMalformedPackage):
		return "元数据损坏/结构非法"
	default:
		return "未知原因"
	}
}

func fail(stage string, err error) {
	fmt.Fprintf(os.Stderr, "%s失败: %v\n", stage, err)
	os.Exit(1)
}
