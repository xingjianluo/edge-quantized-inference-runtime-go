# 边缘量化推理运行时（Go）

一个纯本地、零外部依赖的边缘推理运行时模拟：把**本地模型包目录**当作设备上的量化模型部署物，
在不同精度与版本组合下提供一致、可解释的加载、并发推理与失败处理。

- 所有失败都带**机器可读的原因码**（`pkg/merrors`），不靠匹配错误文本。
- 不支持的包在**加载阶段**拒绝：不静默降级精度、不只加载部分权重。
- 会话之间**不共享可变状态**；单个会话的取消、超时、超预算只影响自身并干净释放。
- 同一批输入的**批量推理与逐条推理结论位级一致**，重复执行（含重新加载）结果相同。

## 目录结构

| 路径 | 职责 |
| --- | --- |
| `pkg/model` | 模型包类型：`Manifest`（`model.json`）、精度/版本常量、加载后的不可变 `LoadedModel` |
| `pkg/merrors` | 全部失败原因码（`Reason`）与带原因的错误类型，`ReasonOf(err)` 可判定 |
| `internal/packageio` | 模型包读写：`model.json`、`weights.bin`（int8/int4 打包）、`bias.bin`（float64 LE）、SHA-256 校验 |
| `pkg/loader` | 加载与全部校验（版本/精度/量化参数/维度/校验和/设备预算），通过后返回完整不可变模型 |
| `pkg/kernel` | 确定性量化推理：固定求和顺序的 float64 仿射量化算子、ReLU、softmax、argmax |
| `pkg/runtime` | 设备预算、模型生命周期、并发安全的会话创建/关闭、取消与超时隔离 |
| `cmd/eqperf` | 演示 CLI：生成模型包、逐条/批量推理对比、超时演示 |
| `internal/testutil` | 测试用模型包构造器 |

## 支持的模型版本与精度

| 项 | 支持值 |
| --- | --- |
| 包格式版本 | `edge-qmodel/v1`（`model.FormatV1`，唯一支持版本） |
| 权重精度 | `int8`（每元素 1 字节，范围 [-128,127]）、`int4`（每字节两个 nibble，低在前，范围 [-8,7]） |
| 输入精度 | `int8`、`int4`（激活在每层入口按上一层精度仿射量化并夹紧到表示范围） |
| 网络结构 | 1..N 个全连接层；隐藏层后接 ReLU，最后一层输出 logits；标签数须等于末层输出维 |
| 量化方式 | 仿射量化：`real = (q - zero_point) * scale`；权重与输入都必须给出合法 `scale` 与 `zero_point` |

模型包目录（v1）：

```
<package-dir>/
├── model.json   # Manifest：版本、精度、维度、每层量化参数、bias、两个 blob 的 sha256
├── weights.bin  # 各层权重按层顺序拼接；int8 每元素一字节，int4 两个元素一字节（低 nibble 在前）
└── bias.bin     # 各层 bias 按层顺序拼接，float64 小端、无填充
```

任意层可独立选择 int8/int4；不同层精度可混用（每层权重独立反量化）。

## 兼容与拒绝规则（全部发生在加载阶段）

校验顺序即下表，任一失败立即返回对应原因码，**设备状态不变、不发生部分加载**：

| 原因码 `merrors.Reason*` | 触发条件 |
| --- | --- |
| `package_not_found` | 目录不存在或缺少 `model.json` / blob 文件 |
| `unsupported_version` | `format_version` 不是 `edge-qmodel/v1`（未知版本绝不用当前语义解释） |
| `invalid_manifest` | 缺 `model_id`/`model_version`、无层、无标签、`input_dim<=0`、重名层、JSON 非法字段 |
| `unsupported_precision` | 权重精度或输入精度不在 {int8,int4}（如 fp16/int2/fp32）——**拒绝，不降级到 int8** |
| `missing_quantization` | 任一层或输入缺少量化参数：`scale<=0`，或 `zero_point` 超出该精度整数范围 |
| `dimension_mismatch` | 首层输入维≠`input_dim`；相邻层维度不衔接；bias 长度≠层输出维；标签数≠末层输出维；权重矩阵非正维 |
| `weight_corrupt` | 权重/bias blob 的 SHA-256 与清单不符、字节数与声明几何不符、int4 填充 nibble 非零、量化值越界、bias 非有限或与清单不一致、blob 截断/有多余尾字节 |
| `device_budget_exceeded` | 模型占用超过加载时给定的设备内存预算 |

推理阶段的拒绝（同样有独立原因码，且在任何数值计算前发生）：`input_shape`（特征维不符）、
`input_value`（NaN/Inf）、`session_budget_exceeded`（批量超过会话预算）、
`canceled`、`timeout`、`session_closed`、`model_not_found`、`model_busy`。

判定方式：

```go
if merrors.ReasonOf(err) == merrors.ReasonMissingQuantization { ... }
```

## 资源预算语义

`runtime.DeviceBudget`：

- `MemoryBytes`：**已加载模型占用 + 全部活跃会话预留工作内存**之和的上限（0 = 不限）。
  - 模型占用 = 打包权重字节 + float64 bias + 每行激活缓冲 + 固定簿记（4096B），由 `loader.ModelMemoryBytes` 给出。
  - 会话在 `NewSession` 时**一次性预留**（激活缓冲 + 2048B 簿记），在 `Close` 时**恰好释放一次**。
  - 模型加载或会话预留若会超预算，请求在改变任何状态前被拒绝；之前被拒绝的尝试不留任何部分预留。
- `MaxConcurrentSessions`：同时打开会话的上限（0 = 不限），超限以 `device_budget_exceeded` 拒绝。

`SessionOptions.MaxBatchSize` 是**单会话**预算：`InferBatch` 行数超限以
`session_budget_exceeded` 在计算开始前拒绝（与设备级 `device_budget_exceeded` 区分）。
批量行**顺序执行并复用同一块工作内存**，因此预留内存按单行计，不随批量大小膨胀。

## 一致性、可复现与隔离

- **确定性**：全部计算使用 float64、固定的行列求和顺序与确定的 softmax/argmax（平局取最小下标）；
  同一 `(模型, 输入)` 在任意会话、goroutine、运行时实例乃至重新加载后给出**位级相同**结果。
- **批量 = 逐条**：`InferBatch` 先整批校验，再逐行调用与 `Infer` 完全相同的计算路径，顺序输出。
- **精度如实上报**：`Prediction.Precision` 始终等于模型实际权重精度；不支持的模型无法通过加载，
  因此不存在"推理到一半才报错"或"返回错误精度数值"。
- **不可变模型**：`LoadedModel` 在 loader 返回前做了字节防御性拷贝，会话永不修改模型字节；
  每次推理的激活缓冲都是调用内的局部切片。
- **隔离**：取消/超时只终止对应调用（模拟计算延迟由 `ComputeLatency` 注入且可被 ctx 打断）；
  受影响会话可干净 `Close`，其他会话结果、已加载模型和后续加载均不受影响。
- **生命周期**：有活跃会话时 `UnloadModel`/`Close(runtime)` 返回 `model_busy`；`Session.Close` 幂等。

## 本地验证方法

需要 Go 1.25+（开发环境为 Go 1.26）。

```bash
# 全部单元测试（含竞态检测）
go test -race ./...

# 查看输入与判定依据（测试日志逐条打印输入、logits、原因码）
go test ./... -v

# 静态检查
go vet ./...
```

测试覆盖：

- `pkg/loader`：12 类不兼容包按不同原因拒绝（版本/两种精度/缺量化参数×3/各类维度错×4/权重与 bias 损坏）、
  设备预算边界（need-1 拒绝、need 成功）、目录缺失。
- `pkg/kernel`：int8/int4 确定性分类、批量与逐条一致、重复执行与重新加载一致、错误输入/坏批量提前拒绝、精度字段如实。
- `pkg/runtime`：24 个并发会话无串扰无泄漏、会话取消隔离与可重建、超时与取消原因可区分、
  设备内存/并发数预算拒绝不留残留、会话批量预算、关闭与卸载生命周期。
- `cmd/eqperf`：CLI 端到端（生成→逐条/批量一致、超时退出码为 `[timeout]`、fp32 生成被 `[unsupported_precision]` 拒绝）。

### 演示 CLI

```bash
go run ./cmd/eqperf generate -dir /tmp/demo -precision int8
go run ./cmd/eqperf infer    -dir /tmp/demo -inputs "1,0;0,1;0.7,0.3"

# int4 权重组合
go run ./cmd/eqperf generate -dir /tmp/demo-int4 -precision int4
go run ./cmd/eqperf infer    -dir /tmp/demo-int4 -inputs "1,0;0,1"

# 超时演示（200ms 模拟计算 vs 20ms 截止 -> error [timeout]，非零退出）
go run ./cmd/eqperf infer -dir /tmp/demo -latency 200ms -timeout 20ms
```

预期输出（逐条与批量两段完全对应）：

```
one-by-one inference:
  [1 0] -> label=a confidence=0.7311 logits=[1 0] precision=int8
  [0 1] -> label=b confidence=0.7311 logits=[0 1] precision=int8
  [0.7 0.3] -> label=a confidence=0.5987 logits=[0.7 0.3] precision=int8
batch inference:
  ...（与逐条逐行相同）
```
