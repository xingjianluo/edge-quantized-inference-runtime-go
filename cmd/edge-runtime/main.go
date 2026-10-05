// 命令 edge-runtime 是本地边缘推理运行时的最小演示：
//   - demo=infer：从磁盘加载一个量化模型包，校验通过后在两个独立会话上分别做
//     逐条与批量推理，校验结果一致并打印判定依据；
//   - demo=update：在线把同名模型热切换到新一代包（不中断服务、旧会话不受影响），
//     再一键回退；并可演示不合规/空间不足时更新整体失败、旧版本继续服务。
//
// 加载/更新失败均按可区分原因报告（哨兵错误 + errors.Is）。
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
	demo := flag.String("demo", "infer", "演示场景: infer | update")
	dir := flag.String("model", "examples/models/toy-int8", "模型包目录（infer）")
	budget := flag.Int64("budget", 1<<20, "设备内存预算（字节）")
	first := flag.String("first", "examples/models/toy-int8", "第一代模型包目录（update）")
	next := flag.String("next", "examples/models/toy-int8-v2", "新一代模型包目录（update）")
	bad := flag.String("bad", "", "可选：非法更新包目录，演示安全失败（update）")
	flag.Parse()

	switch *demo {
	case "infer":
		demoInfer(*dir, *budget)
	case "update":
		demoUpdate(*first, *next, *bad, *budget)
	default:
		fmt.Fprintf(os.Stderr, "未知 demo=%q\n", *demo)
		os.Exit(2)
	}
}

func demoInfer(dir string, budget int64) {
	// 1) 加载与校验。任何不兼容都在这里被拒绝，绝不静默降级。
	pkg, err := modelpkg.Load(os.DirFS("."), dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "模型包被拒绝: %v\n", err)
		fmt.Fprintf(os.Stderr, "原因归类: %s\n", classifyLoadErr(err))
		os.Exit(1)
	}
	m := pkg.Meta()
	fmt.Printf("已加载模型: name=%s version=%s precision=%s inputDim=%d weightBytes=%d revision=%s\n",
		m.Name, m.Version, m.Precision, m.InputDim, pkg.Bytes(), pkg.Revision()[:12])

	// 2) 构造运行时并登记模型（设备预算核算）。
	rt, err := runtime.New(runtime.Config{MaxDeviceBytes: budget})
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
	if !slicesEqual(batch, single) {
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

func demoUpdate(firstDir, nextDir, badDir string, budget int64) {
	// 第一代经普通加载 + 登记。
	pkg0, err := modelpkg.Load(os.DirFS("."), firstDir)
	if err != nil {
		fail("加载第一代", fmt.Errorf("%v（原因: %s）", err, classifyLoadErr(err)))
	}
	name := pkg0.Meta().Name
	rt, err := runtime.New(runtime.Config{MaxDeviceBytes: budget})
	if err != nil {
		fail("创建运行时", err)
	}
	if err := rt.LoadModel(name, pkg0); err != nil {
		fail("登记第一代", err)
	}
	active, _ := rt.ActiveModel(name)
	fmt.Printf("[初始] 当前生效 revision=%s %s/%s 权重=%d字节 设备已用=%d\n",
		active.Revision[:12], active.FormatVersion, active.Precision,
		active.WeightBytes, rt.UsedBytes())

	inputs := [][]float32{{0.1, -0.5}, {1.25, 0}, {-0.75, 0.5}}

	// 旧会话：切换前创建，切换后继续用旧权重跑完。
	oldSess, err := rt.NewSession(name, runtime.SessionOptions{})
	if err != nil {
		fail("创建旧会话", err)
	}
	oldOut := make([][]float32, len(inputs))
	for i, in := range inputs {
		oldOut[i], _ = oldSess.Infer(context.Background(), in)
	}
	fmt.Printf("[旧会话] revision=%s 输入=%v 输出=%v（切换后这些结果不变）\n",
		oldSess.Revision()[:12], inputs, oldOut)

	// 可选：先试一个非法更新包，必须整体失败、旧版本继续服务。
	if badDir != "" {
		info, err := rt.UpdateModelFromFS(name, os.DirFS("."), badDir)
		switch {
		case err == nil:
			fail("非法更新竟然成功", fmt.Errorf("%+v", info))
		default:
			fmt.Printf("[非法更新] %s -> 被拒绝: %v（原因: %s）；旧版本继续服务\n",
				badDir, err, classifyUpdateErr(err))
		}
		if a, _ := rt.ActiveModel(name); a.Revision != pkg0.Revision() {
			fail("非法更新改变了当前版本", nil)
		}
	}

	// 在线热切换：原子生效，无不可用窗口。
	info, err := rt.UpdateModelFromFS(name, os.DirFS("."), nextDir)
	if err != nil {
		fail("在线更新", fmt.Errorf("%v（原因: %s）", err, classifyUpdateErr(err)))
	}
	fmt.Printf("[切换后] 当前生效 revision=%s %s/%s 权重=%d字节；新旧共存设备已用=%d\n",
		info.Revision[:12], info.FormatVersion, info.Precision, info.WeightBytes, rt.UsedBytes())

	newSess, err := rt.NewSession(name, runtime.SessionOptions{})
	if err != nil {
		fail("切换后新建会话（不应有不可用窗口）", err)
	}
	newOut := make([][]float32, len(inputs))
	for i, in := range inputs {
		newOut[i], _ = newSess.Infer(context.Background(), in)
	}
	fmt.Printf("[新会话] revision=%s 输入=%v 输出=%v\n", newSess.Revision()[:12], inputs, newOut)
	if sameResults(oldOut, newOut) {
		fmt.Println("判定: 两代结果完全相同（夹具下不应发生）")
		os.Exit(2)
	}
	// 旧会话重放：必须逐位等于切换前结果。
	for i, in := range inputs {
		got, _ := oldSess.Infer(context.Background(), in)
		if !eq(got, oldOut[i]) {
			fail("旧会话结果受切换影响", nil)
		}
	}
	fmt.Println("判定依据: 切换后新会话用新权重、结果与旧代可区分；旧会话重放逐位等于切换前")

	listVersions(rt, name)

	// 旧会话结束 -> 若旧代仍在回退窗口则保留，否则自动回收。
	_ = oldSess.Close()
	_ = newSess.Close()
	listVersions(rt, name)

	// 一键回退。
	rb, err := rt.RollbackModel(name)
	if err != nil {
		fail("回退", err)
	}
	fmt.Printf("[回退后] 当前生效 revision=%s；设备已用=%d（被顶出代已回收）\n",
		rb.Revision[:12], rt.UsedBytes())
	if _, err := rt.RollbackModel(name); !errors.Is(err, runtime.ErrNoRollbackTarget) {
		fail("第一代再回退应失败", err)
	}
	fmt.Println("判定依据: 一键回退到上一版本；已是第一代时返回 ErrNoRollbackTarget")

	if err := rt.UnloadModel(name); err != nil {
		fail("卸载", err)
	}
	if err := rt.Close(); err != nil {
		fail("关闭运行时", err)
	}
}

func listVersions(rt *runtime.Runtime, name string) {
	vs, err := rt.ModelVersions(name)
	if err != nil {
		fail("查询版本", err)
	}
	fmt.Printf("[版本列表] 设备已用=%d\n", rt.UsedBytes())
	for _, v := range vs {
		state := "可回退"
		if v.Active {
			state = "当前生效"
		}
		if v.Retiring {
			state = fmt.Sprintf("待回收(旧会话=%d)", v.ActiveSessions)
		}
		fmt.Printf("  - revision=%s %s/%s 权重=%d字节 %s\n",
			v.Revision[:12], v.FormatVersion, v.Precision, v.WeightBytes, state)
	}
}

func slicesEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !eq(a[i], b[i]) {
			return false
		}
	}
	return true
}

func sameResults(a, b [][]float32) bool { return slicesEqual(a, b) }

func eq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
	case errors.Is(err, modelpkg.ErrChecksumMismatch):
		return "权重内容与校验和不符（被改动过）"
	case errors.Is(err, modelpkg.ErrChecksumRequired):
		return "在线更新包未声明权重校验和"
	case errors.Is(err, modelpkg.ErrMalformedPackage):
		return "元数据损坏/结构非法"
	default:
		return "未知原因"
	}
}

// classifyUpdateErr 在加载原因之外，再补充运行时阶段的可区分原因。
func classifyUpdateErr(err error) string {
	switch {
	case errors.Is(err, runtime.ErrInsufficientResources):
		return "设备空间不足以同时容纳新旧两套权重"
	case errors.Is(err, runtime.ErrUnknownModel):
		return "同名模型尚未加载"
	case errors.Is(err, runtime.ErrRuntimeClosed):
		return "运行时已关闭"
	default:
		return classifyLoadErr(err)
	}
}

func fail(stage string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s失败: %v\n", stage, err)
	} else {
		fmt.Fprintf(os.Stderr, "%s失败\n", stage)
	}
	os.Exit(1)
}
