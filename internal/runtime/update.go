package runtime

import (
	"fmt"
	"sort"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
)

// VersionInfo 描述某对外模型名下驻留的一代模型包。
type VersionInfo struct {
	// Generation 该模型名下的代数编号，从 1 开始单调递增。
	Generation int
	// ModelVersion 模型包元数据里声明的业务版本号（可为空）。
	ModelVersion string
	// Fingerprint 模型包内容指纹，用于区分结构相同但权重不同的版本。
	Fingerprint string
	// Role 该代当前角色：active（对外生效）/ standby（可回退目标）/ draining（等待回收）。
	Role string
	// ActiveSessions 仍绑定该代的活跃会话数。
	ActiveSessions int
	// WeightBytes 该代权重占用的设备字节数（已记账）。
	WeightBytes int64
}

// 代角色取值。
const (
	RoleActive   = "active"
	RoleStandby  = "standby"
	RoleDraining = "draining"
)

// UpdateModel 不中断服务地把同名对外模型切换到新一代模型包。
//
// 语义：
//   - 新包必须已经通过 modelpkg.Load 的全部校验（版本/精度组合、量化参数、
//     层间维度、权重完整性与校验和）；不合规的包在 Load 阶段就已被拒绝，
//     本函数不会看到，更不会影响在役版本。
//   - 设备剩余空间必须能同时容纳新旧两套权重，否则整体失败并返回
//     ErrInsufficientResources，原版本继续对外服务，运行时状态不变。
//   - 切换是单次原子指针交换：不存在模型短暂不可用的窗口，切换瞬间
//     新建会话只会拿到旧版或新版之一，绝不失败于“正在更新”。
//   - 切换成功后：新建会话使用新一代；切换前已存在的会话仍绑定原代，
//     用原权重跑完，结果不受影响。
//   - 原 active 代成为 standby（一键回退目标，继续驻留并记账）；
//     原 standby 代退役，待其旧会话全部结束后自动回收权重字节。
//
// 失败原因可区分：模型未加载 -> ErrUnknownModel；空间不足 ->
// ErrInsufficientResources；运行时已关闭 -> ErrRuntimeClosed。
func (rt *Runtime) UpdateModel(name string, pkg *modelpkg.Package) error {
	if name == "" || pkg == nil {
		return fmt.Errorf("runtime: update model requires non-empty name and non-nil package")
	}
	need := pkg.Bytes()

	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	// 新旧两套权重（以及可能仍在排空的退役代）需同时放得下。
	if rt.used+need > rt.cfg.MaxDeviceBytes {
		return fmt.Errorf("%w: update %q needs %d more bytes for new weights, %d of %d used",
			ErrInsufficientResources, name, need, rt.used, rt.cfg.MaxDeviceBytes)
	}

	g := e.newGeneration(pkg)
	// 原 standby 退役：有旧会话则转入 draining 等待回收，无会话立即回收。
	if e.standby != nil {
		e.retired[e.standby] = struct{}{}
	}
	e.standby = e.active
	e.active = g
	rt.used += need
	rt.reclaimLocked(e)
	return nil
}

// RollbackModel 一键退回：把 standby（上一个版本）重新置为对外生效版本，
// 原 active 转为 standby（可再次回退/前滚）。纯指针交换，不增减设备记账，
// 不中断任何会话；新建会话立即使用回退后的版本。
//
// 没有可回退目标时返回 ErrNoRollbackTarget，状态不变。
func (rt *Runtime) RollbackModel(name string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	if e.standby == nil {
		return fmt.Errorf("%w: model %q", ErrNoRollbackTarget, name)
	}
	e.active, e.standby = e.standby, e.active
	return nil
}

// ActiveVersion 返回当前对外生效的版本信息；模型不存在返回 ErrUnknownModel。
func (rt *Runtime) ActiveVersion(name string) (VersionInfo, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	e, ok := rt.models[name]
	if !ok {
		return VersionInfo{}, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	return versionInfo(e.active, RoleActive), nil
}

// ModelVersions 返回该模型名下全部驻留版本（active / standby / draining），
// 按代数升序；模型不存在返回 ErrUnknownModel。
// 其中 Role==RoleStandby 的即当前可一键回退的目标。
func (rt *Runtime) ModelVersions(name string) ([]VersionInfo, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	e, ok := rt.models[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	infos := []VersionInfo{versionInfo(e.active, RoleActive)}
	if e.standby != nil {
		infos = append(infos, versionInfo(e.standby, RoleStandby))
	}
	for g := range e.retired {
		infos = append(infos, versionInfo(g, RoleDraining))
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].Generation < infos[j].Generation })
	return infos, nil
}

// versionInfo 汇总一代的对外信息（调用方须持锁）。
func versionInfo(g *generation, role string) VersionInfo {
	return VersionInfo{
		Generation:     g.id,
		ModelVersion:   g.pkg.Meta().ModelVersion,
		Fingerprint:    g.pkg.Fingerprint(),
		Role:           role,
		ActiveSessions: len(g.sessions),
		WeightBytes:    g.pkg.Bytes(),
	}
}
