# Edge Quantized Inference Runtime（Go）

一个纯 Go 的**本地边缘量化推理运行时**模拟实现。用本地“模型包”模拟设备上的量化模型部署，
目标是让**不同精度与版本组合**下的加载、并发推理与失败处理都产生**一致、可复现、可解释**的结果。

- 模型包在**加载阶段**完成全部校验：量化参数缺失、维度不匹配、版本/精度（含二者组合）不受支持、
  权重缺失或被截断、权重内容与声明的校验和不符，都会以**可区分的错误原因**被拒绝；
  **绝不静默降级精度，绝不部分加载权重**。
- 多个推理会话可高并发运行；会话之间**不共享可变状态**。单个会话的取消、超时、超出自身资源预算
  **只影响其自身**，并能干净释放，不会污染其他会话的结果，也不会导致后续加载失败。
- **在线更新与回退**：同名模型可不中断服务地切换到新一代模型包；切换前已存在的会话继续用旧权重
  跑完，新建会话立即使用新版本；更新失败（包不合规或空间不足）不影响在役版本；更新后可一键回退。
- 推理核是**纯函数式定点整数运算**：同一批输入在“批量推理”和“逐条推理”下结论逐位一致，
  重复执行、跨会话执行结果相同。不受支持的模型在加载时即被拒，永远到不了推理阶段；
  NaN/±Inf 等异常输入被明确拒绝，不会产出看似正常实则无意义的数值。

## 目录结构

```
internal/modelpkg/   模型包类型、加载与校验（错误原因可 errors.Is 区分）
internal/runtime/    运行时：设备资源预算、会话生命周期、取消/超时、确定性推理核、在线更新与回退
internal/testutil/   测试夹具（经真实 Load 校验路径构造内存模型包）
cmd/edge-runtime/    可运行的最小演示：加载 + 逐条/批量一致性判定
examples/models/     示例模型包（合法 int8/uint8 与多个故意非法的包）
```

## 支持的模型版本与精度

| 版本 | 支持精度 | 说明 |
|------|----------|------|
| `v1` | `int8`   | 初代格式，仅支持有符号 8 比特量化 |
| `v2` | `int8`、`uint8` | 二代格式，支持有符号与无符号 8 比特量化，每层需显式量化参数 |

- 精度名称本身未知（如 `fp16`）→ `ErrUnsupportedPrecision`。
- 版本与精度都认识但组合不被允许（如 `v1` + `uint8`）→ `ErrVersionPrecisionMismatch`。
  两者被刻意区分为不同错误，便于定位“是不认识还是不允许”。

`int8` 与 `uint8` 的权重解释不同（含各自零点），对同一输入会产出**不同的数值结果**，
从而可验证运行时没有偷偷把 `uint8` 当 `int8` 算。

## 模型包格式

每个模型包是一个目录：

```
<model-dir>/model.json     元数据
<model-dir>/<weight_file>  每层一个原始字节权重文件（每权重 1 字节，行主序，长度必须 = in_dim*out_dim）
```

`model.json` 关键字段：

```json
{
  "name": "toy-int8",
  "version": "v2",
  "precision": "int8",
  "model_version": "2026.10-r2",
  "input_dim": 2,
  "input_quant": {"scale": 0.25, "zero_point": 0},
  "layers": [
    {"name": "fc1", "in_dim": 2, "out_dim": 3, "weight_file": "fc1.bin",
     "weight_sha256": "<可选：fc1.bin 内容的 SHA-256 十六进制>",
     "quant": {"scale": 0.1, "zero_point": 0}},
    {"name": "fc2", "in_dim": 3, "out_dim": 2, "weight_file": "fc2.bin",
     "quant": {"scale": 0.05, "zero_point": 0}}
  ]
}
```

量化约定：`real_value = scale * (q - zero_point)`。模型级 `input_quant` 负责把 float 输入量化为 8 比特；
每层 `quant` 给出权重/输出尺度与零点。

- `model_version`（可选）：业务侧模型版本号，仅作标识，不参与兼容性校验；
  在线更新/回退时用于区分同名模型的不同代（见下文）。
- `weight_sha256`（可选，逐层）：声明后加载阶段会校验权重文件实际内容的 SHA-256，
  不一致即判定**内容被改动过**，以 `ErrWeightCorrupted` 拒绝。
- 每个成功加载的包都会计算**内容指纹**（`Package.Fingerprint()`，元数据+全部权重字节的 SHA-256）：
  结构相同但权重不同的两个包指纹必然不同，可据此确认“当前生效的到底是哪一版权重”。

## 加载校验与拒绝规则（全部发生在加载阶段）

按固定顺序校验，错误均可用 `errors.Is` 区分（见 `internal/modelpkg/errors.go`）：

| 检查 | 错误哨兵 | 触发条件 |
|------|----------|----------|
| 目录/文件存在性 | `ErrPackageNotFound` | 目录或 `model.json`/权重文件不存在 |
| 元数据可解析与结构完整 | `ErrMalformedPackage` | JSON 损坏；name/input_dim/layers 缺失；层缺 name/dims/weight_file |
| 版本支持 | `ErrUnsupportedVersion` | version 不在 `v1,v2` |
| 精度支持 | `ErrUnsupportedPrecision` | precision 不是 `int8/uint8` |
| 版本×精度组合 | `ErrVersionPrecisionMismatch` | 如 `v1` + `uint8` |
| 量化参数 | `ErrMissingQuantParams` | 模型或层 `scale<=0`/缺失；零点超出该精度值域 |
| 层间维度连续 | `ErrDimMismatch` | 某层 `in_dim` 不等于上一层 `out_dim`/模型 `input_dim` |
| 权重完整（禁止部分加载） | `ErrPartialWeights` | 权重文件缺失，或字节数**少于** `in_dim*out_dim` |
| 权重维度精确匹配 | `ErrDimMismatch` | 权重字节数**多于**声明（不静默忽略尾部） |
| 权重内容未被改动 | `ErrWeightCorrupted` | 声明了 `weight_sha256` 但权重实际内容的 SHA-256 不符 |

要点：**任一检查失败都返回 `nil` 包 + 带原因的错误**。运行时因此不可能拿到“半加载”的模型，
也就不会在推理中途才暴露不兼容，更不会返回错误精度的数值。

## 资源预算语义

预算模拟设备上的一块共享“设备内存”，单位**字节**，由 `runtime.Config.MaxDeviceBytes` 限定：

```
已用设备字节 = 全部已加载模型权重字节 + 全部活跃会话预占的工作区字节
```

- **模型加载**：登记时立即预占权重字节数；剩余空间不足返回 `ErrInsufficientResources`，
  且运行时状态保持不变（模型完全不可见、不可建会话）。
- **会话创建**：创建时按会话的 `MaxBatchBytes` 预算，预占其允许的**最大整批工作区**
  （记账口径 `(inputDim + 最大层 outDim) * 4 字节/条`）。设备剩余空间不足则建会话失败，
  不影响已有会话。
- **会话级批次预算**：`SessionOptions.MaxBatchBytes` 限制单次批量大小。一批输入的工作区估算
  超过该值时，本次整批以 `ErrSessionBudgetExceeded` 被拒，**不返回部分结果**，但仅影响该会话；
  同一会话在预算内的后续调用照常成功。
- **释放**：`Session.Close()` 等待进行中的推理退出后归还预占字节；`UnloadModel` 仅在无活跃会话时
  允许；全部归还后可继续加载/建会话，不残留占用。
- **多代共存**：在线更新后，同一模型名下可能同时驻留多代权重（见下节），
  **所有驻留代的权重字节都如实计入设备预算**，`Runtime.UsedBytes()` 可查。

### 取消、超时与隔离

- `Session.Cancel()`：进行中的推理在协作点以 `ErrSessionCanceled` 中止，此后该会话拒绝新推理；
  不影响其他会话。
- `SessionOptions.Timeout`：仅约束**单次调用**（含整批）。超时返回 `ErrSessionTimeout`，
  会话本身不被终结，下一次调用仍可正常推理。
- 传入已取消的 `context.Context`：映射为 `ErrSessionCanceled`。
- 会话只持有对**只读**模型视图的引用，自身的输入/激活/生命周期状态互不共享；
  `Close` 通过 `WaitGroup` 保证后台推理退出后再释放预算。

## 在线更新与回退

同名对外模型可以不中断服务地切换到新一代模型包（`Runtime.UpdateModel`），
并在需要时一键退回（`Runtime.RollbackModel`）。

### 哪些操作不中断服务

- `UpdateModel` / `RollbackModel` 都是**单次原子指针交换**：不存在“模型短暂不可用”的窗口；
  切换瞬间新建会话只会拿到旧版或新版之一，绝不会因“正在更新”而失败。
- 切换前已存在的会话**绑定创建时的那一代**，继续用原权重跑完，结果完全不受影响。
- 切换成功后，新建会话立即使用新一代。

### 更新失败的两种可区分原因

1. **包本身不合规**：新包必须通过与首次加载完全相同的 `modelpkg.Load` 校验
   （版本/精度组合、量化参数、层间维度、权重完整性、`weight_sha256` 防篡改）。
   不合规的包在 `Load` 阶段就以对应哨兵错误被拒绝，`UpdateModel` 根本不会被调用到，
   在役版本不受任何影响。
2. **设备空间不足**：更新期间新旧两套权重需要同时驻留，剩余空间放不下新权重时，
   `UpdateModel` 整体失败并返回 `ErrInsufficientResources`，运行时状态不变，
   原版本继续对外服务；释放空间（如关闭部分会话）后同一更新可以重试成功。

### 回退与版本查询

- 更新成功后，上一个版本作为 **standby（回退目标）** 继续驻留；
  `RollbackModel(name)` 把它重新置为对外生效版本（原生效版本转为 standby，可再次来回切换）。
  回退同样是原子交换，不增减设备记账、不中断任何会话。
- 没有可回退目标时返回 `ErrNoRollbackTarget`（与空间不足、包不合规均可区分）。
- 版本查询：
  - `ActiveVersion(name)`：当前对外生效的一代（代编号、`model_version`、内容指纹、活跃会话数、权重字节）。
  - `ModelVersions(name)`：全部驻留代，角色为 `active` / `standby`（可回退目标）/ `draining`（等待回收）。

### 新旧共存期间的记账与回收

```
已用设备字节 = 所有驻留代权重（active + standby + draining）+ 所有活跃会话预占工作区
```

- **standby（上一版本）始终保留驻留并记账**，保证随时可以一键回退。
- 比回退目标更旧的代在再次更新时**退役（draining）**：不再接受新会话，
  等绑定它的旧会话全部 `Close` 后**自动回收**其权重字节；回收后设备可继续加载、继续建会话，不残留占用。
- `UnloadModel` 要求该名下所有代的会话都结束，卸载时归还全部驻留代权重。

### 更新后的结果可信性

- 推理核是纯函数定点运算：同一批输入在更新后的新会话里重复跑结果逐位一致，
  且与“直接 `LoadModel` 加载同一版本”得到的结论逐位一致。
- 不同代即使结构完全相同，权重不同则**内容指纹不同、输出数值不同**，
  可通过 `ActiveVersion` 的指纹与输出双重确认没有用错版本。
- 异常输入（NaN、±Inf）在任何版本下都被明确拒绝（`ErrInputNotFinite`，
  与维度错误 `ErrInputDimMismatch` 可区分），不会产出看似正常实则无意义的数值。

## 一致性与可复现

推理核（`internal/runtime/engine.go`）全程使用 **int64 整数累加 + 定点乘子**（`mult/2^shift`），
不依赖浮点累加顺序或 SIMD 融合指令：

- `Infer(input)` 与 `InferBatch([]{input})` 共用同一条执行路径，结果**逐位相同**；
- 同一会话重复执行、不同会话执行，结果一致；
- `int8` 与 `uint8` 模型输出可区分，防止精度被静默替换。

## 本地验证方法

```bash
# 1) 编译、静态检查、全部单测（含竞态检测）
go build ./...
go vet ./...
go test ./... -race -count=1

# 2) 运行演示：合法 int8 / uint8 模型（打印逐条 vs 批量的一致性判定）
go run ./cmd/edge-runtime -model examples/models/toy-int8
go run ./cmd/edge-runtime -model examples/models/toy-uint8

# 3) 验证非法包按可区分原因在加载阶段被拒绝
go run ./cmd/edge-runtime -model examples/models/bad-v1-uint8        # 版本×精度组合不支持
go run ./cmd/edge-runtime -model examples/models/bad-missing-quant   # 量化参数缺失
go run ./cmd/edge-runtime -model examples/models/bad-partial-weights # 权重不完整

# 4) 只看在线更新/回退相关单测（日志含场景、输入与判定依据）
go test ./internal/runtime/ -run 'TestUpdate|TestRollback|TestRetired|TestAbnormal' -v
```

## 测试覆盖

- `internal/modelpkg/loader_test.go`：15 种加载拒绝场景（含权重校验和不符），逐一对 `errors.Is` 断言；
  另验证正确校验和通过、指纹确定且对内容敏感。
- `internal/runtime/determinism_test.go`：逐条两遍、批量、跨会话一致性；维度错误立即拒绝；int8/uint8 结果可区分。
- `internal/runtime/budget_test.go`：设备预算不足且状态不变、会话创建挤占、批次超预算只影响自身、活跃会话禁止卸载。
- `internal/runtime/concurrent_test.go`：16 会话 × 25 轮高并发结果不被污染（`-race`）；一个会话被取消时其他会话保持正确。
- `internal/runtime/cancel_test.go`：超时只作用于单次调用、取消终结会话、外部 ctx 取消映射、运行时关闭语义。
- `internal/runtime/reject_integration_test.go`：不兼容包端到端在加载阶段被拒，推理路径不可达。
- `internal/runtime/update_test.go`：在线更新与回退——
  切换后新会话用新版本且与直接加载该版本逐位一致、不同代结果可区分；
  切换期间旧会话结果不受影响；更新/回退与并发建会话交错无不可用窗口；
  不合规新包（版本×精度、量化参数、权重被篡改）与空间不足均保持旧版本且原因可区分；
  一键回退与版本查询；退役代在旧会话结束后自动回收、设备可继续加载；
  NaN/±Inf 异常输入被明确拒绝。

所有失败均以**哨兵错误 + `errors.Is`** 表达原因，单测日志打印输入向量与判定依据。
