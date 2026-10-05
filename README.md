# Edge Quantized Inference Runtime（Go）

一个纯 Go 的**本地边缘量化推理运行时**模拟实现。用本地“模型包”模拟设备上的量化模型部署，
目标是让**不同精度与版本组合**下的加载、并发推理与失败处理都产生**一致、可复现、可解释**的结果。

- 模型包在**加载阶段**完成全部校验：量化参数缺失、维度不匹配、版本/精度（含二者组合）不受支持、
  权重缺失或被截断，都会以**可区分的错误原因**被拒绝；**绝不静默降级精度，绝不部分加载权重**。
- 多个推理会话可高并发运行；会话之间**不共享可变状态**。单个会话的取消、超时、超出自身资源预算
  **只影响其自身**，并能干净释放，不会污染其他会话的结果，也不会导致后续加载失败。
- 支持**在线更新与一键回退**：同名模型可不停机热切换到新一代权重——切换是原子的指针交换，
  **没有模型短暂不可用的窗口**；切换前在跑的旧会话继续用旧权重跑完、结果不受影响，
  切换后新建的会话使用新权重。旧版本保留一代可回退，超出回退窗口的旧代在最后一个旧会话
  结束后**自动回收**。更新可安全失败（新包不合规或空间不足时整体失败、旧版本继续服务、
  失败原因可区分），且不残留任何资源占用。
- 推理核是**纯函数式定点整数运算**：同一批输入在“批量推理”和“逐条推理”下结论逐位一致，
  重复执行、跨会话执行结果相同。不受支持的模型在加载时即被拒，永远到不了推理阶段。
- `NaN`/`±Inf` 这类本身得不到有意义结果的输入会被**明确拒绝**（`ErrInvalidInput`），
  不会静默饱和成量程端点、产出“看起来正常”的数值。

## 目录结构

```
internal/modelpkg/   模型包类型、加载与校验（错误原因可 errors.Is 区分）
internal/runtime/    运行时：设备资源预算、会话生命周期、取消/超时、在线切换/回退、确定性推理核
internal/testutil/   测试夹具（经真实 Load 校验路径构造内存模型包）
cmd/edge-runtime/    可运行的最小演示：加载 + 逐条/批量一致性、在线更新/回退
cmd/genmodels/       确定性生成 examples/models 下的示例包（含两代权重与非法更新包）
examples/models/     示例模型包（合法两代 int8/uint8 与多个故意非法的包）
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
  "input_dim": 2,
  "input_quant": {"scale": 0.25, "zero_point": 0},
  "layers": [
    {"name": "fc1", "in_dim": 2, "out_dim": 3, "weight_file": "fc1.bin",
     "quant": {"scale": 0.1, "zero_point": 0}},
    {"name": "fc2", "in_dim": 3, "out_dim": 2, "weight_file": "fc2.bin",
     "quant": {"scale": 0.05, "zero_point": 0}}
  ]
}
```

量化约定：`real_value = scale * (q - zero_point)`。模型级 `input_quant` 负责把 float 输入量化为 8 比特；
每层 `quant` 给出权重/输出尺度与零点。

### 权重校验和（在线更新强制）

`model.json` 可选字段 `weight_sha256` 给出每个权重文件原始字节的 SHA-256（小写十六进制）：

```json
"weight_sha256": {
  "fc1.bin": "2c4d3aff41d6...",
  "fc2.bin": "2fa46bcc9c9d..."
}
```

- 普通 `Load` 时该字段**可选**（兼容旧格式）；一旦声明，就逐文件核对实际字节，
  任何字节级改动 -> `ErrChecksumMismatch`；引用了不存在的权重文件或摘要不是 64 位十六进制
  -> `ErrMalformedPackage`。
- 在线更新入口 `LoadForUpdate` / `Runtime.UpdateModelFromFS` **强制每一层都必须声明校验和**，
  否则 `ErrChecksumRequired`——推送到设备的权重若无法证明“内容没被改动过”，到不了切换环节。
- 每个成功加载的包都有 `Package.Revision()` 内容指纹（对全部权重字节 + 模型结构求 SHA-256）：
  即使两代的 `version/precision` 字符串完全一致，只要权重或结构不同，指纹必然不同，
  从结果与指纹上都不会“悄悄用错版本”。

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
| 权重内容完整性 | `ErrChecksumMismatch` | 声明了 `weight_sha256` 但实际字节摘要不符（内容被改动） |
| 更新包完整性声明 | `ErrChecksumRequired` | 在线更新包未给每一层权重声明 SHA-256 校验和 |

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

### 取消、超时与隔离

- `Session.Cancel()`：进行中的推理在协作点以 `ErrSessionCanceled` 中止，此后该会话拒绝新推理；
  不影响其他会话。
- `SessionOptions.Timeout`：仅约束**单次调用**（含整批）。超时返回 `ErrSessionTimeout`，
  会话本身不被终结，下一次调用仍可正常推理。
- 传入已取消的 `context.Context`：映射为 `ErrSessionCanceled`。
- 会话只持有对**只读**模型视图的引用，自身的输入/激活/生命周期状态互不共享；
  `Close` 通过 `WaitGroup` 保证后台推理退出后再释放预算。

## 在线更新与回退（不停机）

同一个对外模型名内部维护一个**代际栈**（generation stack）。每次 `UpdateModel` 压入新一代，
`RollbackModel` 弹出当前代并让上一代重新生效。会话在创建时**绑定具体一代**，
之后的切换/回退不改变该绑定。

### 操作接口

| API | 语义 |
|-----|------|
| `UpdateModelFromFS(name, fsys, dir)` | 一站式在线更新：先按更新专用规则（强制校验和）加载+完整校验，通过后原子切换；失败返回 `modelpkg` 的可区分原因 |
| `UpdateModel(name, pkg)` | 用**已校验**的包切换；包未声明校验和 -> `ErrChecksumRequired`，空间不足 -> `ErrInsufficientResources` |
| `RollbackModel(name)` | 一键回退到紧邻上一代，返回回退后生效版本的 `VersionInfo`；已在最早一代 -> `ErrNoRollbackTarget` |
| `ActiveModel(name)` | 查询当前对新建会话生效的版本（指纹、格式版本、精度、权重字节） |
| `ModelVersions(name)` | 列出全部驻留代：栈内自旧到新，`Active=true` 为当前版本，其余非 `Retiring` 代即**可回退版本**；被顶出仅等旧会话结束的代为 `Retiring=true` |

### 哪些操作会中断服务，哪些不会

- **不会中断**：`UpdateModel` / `RollbackModel`。切换只是一次持锁的**当前代指针交换**——
  切换期间任何 `NewSession` 要么拿到旧代、要么拿到新代，**绝不会**遇到“模型不存在/短暂不可用/
  新建会话失败”的窗口。
- 切换前已存在的会话继续引用旧代不可变权重，跑完的结果与切换前**逐位相同**；
  切换后新建的会话才使用新代权重。
- 会中断/报错的只有真正的异常路径：模型名不存在（`ErrUnknownModel`）、运行时已 `Close`
  （`ErrRuntimeClosed`）、更新包校验失败或空间不足时的**那次更新调用本身**——且旧版本照常服务。
- `UnloadModel` 仍然是**显式停机操作**：任一驻留代（含等待回收的代）还有会话时拒绝；
  在线更新场景通常不需要调用它。

### 更新失败的两类原因（整体失败、旧版本继续服务）

1. **包本身不合规**——由 `modelpkg` 哨兵错误区分：版本/精度组合不支持
   （`ErrUnsupportedVersion` / `ErrUnsupportedPrecision` / `ErrVersionPrecisionMismatch`）、
   量化参数缺失（`ErrMissingQuantParams`）、层间维度对不上（`ErrDimMismatch`）、
   权重不完整（`ErrPartialWeights`）、内容与校验和不符（`ErrChecksumMismatch`）、
   更新包未声明校验和（`ErrChecksumRequired`）。
2. **设备空间不够**——`ErrInsufficientResources`：更新前要求**同时容纳新旧两套权重**
   （旧代可能还要服务旧会话），剩余空间不足时整次更新失败。

两类失败都**不改变当前对外版本、不预占任何字节、无残留**；旧会话继续推理、
新会话仍可创建且仍使用旧权重。推荐直接用 `UpdateModelFromFS`，加载原因会原样包在错误链里，
`errors.Is` 即可区分上述全部原因。

### 回退策略与版本可见性

- 代际栈**始终只保留紧邻的上一代**作为回退目标（即“一键退回到上一个版本”）。
- 连续第二次更新会把最早一代顶出回退窗口：它若已无会话，权重**当场归还**、不再出现在
  `ModelVersions` 中；若仍有旧会话，则标记 `Retiring` 继续驻留记账，等最后一个旧会话结束自动回收。
- 回退把当前代顶出、上一代重新生效；被顶出的代同样按“有无旧会话”立即或延迟回收。
  回退后再次 `UpdateModel` 即等于推送新的一代（可用于“回退后修好再推”）。

### 新旧共存期间的资源记账与回收

```
已用设备字节 = 所有驻留代权重（栈内代 + Retiring 代）+ 所有活跃会话预占工作区
```

- 切换成功的那一刻即预占新一代权重；旧代权重在旧会话存活期间继续计入 `UsedBytes`，
  记账如实反映“两套权重同时驻留”。
- 旧代的最后一个会话 `Close` 时，运行时自动归还该代权重字节并将其移出版本列表；
  回收后设备空间可继续用于更新/加载/建会话，**不残留占用**。
- 因驻留变多而占满预算时，被拒绝的只是“新的建会话/更新请求”（`ErrInsufficientResources`），
  已有旧会话推理不受影响。

## 结果可信：版本可区分与异常输入拒绝

- 新会话结果与“直接加载该版本”的推理结论逐位一致（同一条确定性整数执行路径）；
  同代重复推理、跨会话推理也一致。
- 两代即使 `version/precision` 字符串与网络结构完全相同，只要权重内容不同，
  `Revision()` 指纹与推理输出都不同，不会悄悄用错版本。
- 输入向量含 `NaN` 或 `±Inf` 时以 `ErrInvalidInput` 拒绝（不返回任何结果，批量也不返回部分结果）；
  维度不符仍是 `ErrInputDimMismatch`，两类原因可区分。拒绝异常输入不影响后续正常推理。

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
go run ./cmd/edge-runtime -demo infer -model examples/models/toy-int8
go run ./cmd/edge-runtime -demo infer -model examples/models/toy-uint8

# 3) 在线更新与一键回退（热切换、旧会话不受影响、新旧共存记账、回退回收）
go run ./cmd/edge-runtime -demo update \
  -first examples/models/toy-int8 -next examples/models/toy-int8-v2

# 4) 更新安全失败：新包被改动过 / 未声明校验和，旧版本继续服务
go run ./cmd/edge-runtime -demo update -bad examples/models/bad-update-tampered
go run ./cmd/edge-runtime -demo update -bad examples/models/bad-update-no-checksum

# 5) 空间不足以同时容纳新旧两套权重（预算只够旧权重+一份工作区）
go run ./cmd/edge-runtime -demo update -budget 43

# 6) 旧的加载阶段拒绝场景仍然成立
go run ./cmd/edge-runtime -demo infer -model examples/models/bad-v1-uint8        # 版本×精度组合不支持
go run ./cmd/edge-runtime -demo infer -model examples/models/bad-missing-quant   # 量化参数缺失
go run ./cmd/edge-runtime -demo infer -model examples/models/bad-partial-weights # 权重不完整

# 示例包都是确定性生成的，需要时可逐字节复现：
go run ./cmd/genmodels
```

## 测试覆盖

- `internal/modelpkg/loader_test.go`：14 种加载拒绝场景，逐一对 `errors.Is` 断言并打印输入与判定依据。
- `internal/runtime/determinism_test.go`：逐条两遍、批量、跨会话一致性；维度错误立即拒绝；int8/uint8 结果可区分。
- `internal/runtime/budget_test.go`：设备预算不足且状态不变、会话创建挤占、批次超预算只影响自身、活跃会话禁止卸载。
- `internal/runtime/concurrent_test.go`：16 会话 × 25 轮高并发结果不被污染（`-race`）；一个会话被取消时其他会话保持正确。
- `internal/runtime/cancel_test.go`：超时只作用于单次调用、取消终结会话、外部 ctx 取消映射、运行时关闭语义。
- `internal/runtime/reject_integration_test.go`：不兼容包端到端在加载阶段被拒，推理路径不可达。

在线更新与回退新增覆盖（均带场景/输入/判定依据日志，`-race` 下验证）：

- `internal/runtime/update_test.go`：热切换后新会话用新权重、旧会话重放逐位不变；
  12 客户端在切换全程持续建会话/推理，无任何“模型不可用”窗口、结果按各自代际基准核对。
- `internal/runtime/rollback_test.go`：一键回退恢复上一代（结果等于该版本直接加载）、
  第一代再回退报 `ErrNoRollbackTarget`；连续更新只保留一代可回退、更早代立即或延迟回收；
  新旧共存记账精确到字节、最后旧会话结束自动回收、回收后可继续加载建会话无残留。
- `internal/runtime/update_reject_test.go`：版本/精度组合不支持、量化参数缺失、
  校验和不符/未声明校验和、空间不足四类失败时旧版本继续服务、记账不变、新会话仍用旧权重；
  空间不足归类为 `ErrInsufficientResources` 以区别于包不合规；失败后腾出空间即可更新成功。
- `internal/runtime/invalid_input_test.go`：NaN/±Inf 逐条与批量都报 `ErrInvalidInput`、
  不返回部分结果，与 `ErrInputDimMismatch` 可区分，拒绝后正常输入仍可复现。
- `internal/modelpkg/checksum_test.go`：校验和不符、更新强制校验和、非法引用/非法摘要格式、
  内容指纹对“仅权重不同”可区分、相同内容指纹可复现。

所有失败均以**哨兵错误 + `errors.Is`**表达原因，单测日志打印场景、输入向量与判定依据。
