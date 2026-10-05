package runtime

import (
	"errors"
	"fmt"
	"io/fs"
	"sync"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
)

// Config 设备级资源预算。预算针对运行时进程内共享的“设备内存”，单位字节；
// 已加载模型权重占用与活跃会话预占的工作区合并计入同一池子。
type Config struct {
	// MaxDeviceBytes 设备可用于“模型权重 + 会话工作区”的总内存上限，必须为正。
	MaxDeviceBytes int64
}

// Runtime 模拟一台边缘设备上的推理运行时，负责模型登记与设备级预算核算。
// 所有跨会话共享的只有不可变模型权重；会话自身的可变状态彼此隔离。
type Runtime struct {
	mu     sync.Mutex
	models map[string]*modelEntry
	used   int64
	cfg    Config
	closed bool
}

// generation 同一对外模型名的一代权重。一经创建即不可变；
// 切换/回退只改变 modelEntry.active 指针，老会话继续引用自己那一代跑完。
type generation struct {
	id       int64
	entry    *modelEntry
	pkg      *modelpkg.Package
	view     modelView
	sessions map[*Session]struct{}
	// detached 表示该代已不在代际栈上（被更新顶掉或被回退），
	// 仅因仍有旧会话而驻留；最后一个旧会话结束时立即回收其权重字节。
	detached bool
}

// modelEntry 一个对外模型名的代际记录：
// gens 为代际栈（末尾为当前对外生效的一代），retired 为等待旧会话
// 结束后回收的已摘除代。
type modelEntry struct {
	name    string
	gens    []*generation
	retired []*generation
	active  *generation
	nextID  int64
}

// VersionInfo 某一代权重的对外可查信息。
type VersionInfo struct {
	ModelName string
	// Revision 内容指纹：权重或结构不同则不同，即使版本/精度字符串相同也能区分。
	Revision      string
	FormatVersion modelpkg.Version
	Precision     modelpkg.Precision
	WeightBytes   int64
	// Active 为 true 表示当前新建会话使用这一代。
	Active bool
	// Retiring 为 true 表示该代已被切换/回退顶出，仅因旧会话未结束而暂驻，
	// 不可回退、也不对外服务；旧会话全部结束后自动回收。
	Retiring bool
	// ActiveSessions 仍绑定在这一代上的会话数（含进行中）。
	ActiveSessions int
}

// UsedBytes 返回当前已记账的设备字节数（权重 + 活跃会话工作区），主要用于测试。
func (rt *Runtime) UsedBytes() int64 {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.used
}

// New 以设备预算构造运行时。非正预算非法。
func New(cfg Config) (*Runtime, error) {
	if cfg.MaxDeviceBytes <= 0 {
		return nil, fmt.Errorf("runtime: invalid budget MaxDeviceBytes=%d (must be > 0)",
			cfg.MaxDeviceBytes)
	}
	return &Runtime{
		models: make(map[string]*modelEntry),
		cfg:    cfg,
	}, nil
}

// LoadModel 将已校验模型包登记到设备并预占其权重字节数。
//   - 运行时已关闭            -> ErrRuntimeClosed
//   - 同名模型已存在          -> 错误（不覆盖；在线切换请用 UpdateModel）
//   - 剩余设备内存不足        -> ErrInsufficientResources，状态不变
//
// 加载失败绝不部分登记：要么成功可推理，要么完全不可见。
func (rt *Runtime) LoadModel(name string, pkg *modelpkg.Package) error {
	if name == "" || pkg == nil {
		return fmt.Errorf("runtime: load model requires non-empty name and non-nil package")
	}
	need := pkg.Bytes()

	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	if _, ok := rt.models[name]; ok {
		return fmt.Errorf("runtime: model %q already loaded", name)
	}
	if rt.used+need > rt.cfg.MaxDeviceBytes {
		return fmt.Errorf("%w: need %d bytes for model %q, %d of %d used",
			ErrInsufficientResources, need, name, rt.used, rt.cfg.MaxDeviceBytes)
	}
	e := &modelEntry{name: name}
	g := rt.newGeneration(e, pkg)
	e.gens = []*generation{g}
	e.active = g
	rt.models[name] = e
	rt.used += need
	return nil
}

// newGeneration 必须在持有 rt.mu 时调用。
func (rt *Runtime) newGeneration(e *modelEntry, pkg *modelpkg.Package) *generation {
	e.nextID++
	return &generation{
		id:       e.nextID,
		entry:    e,
		pkg:      pkg,
		view:     buildView(pkg),
		sessions: make(map[*Session]struct{}),
	}
}

// UpdateModelFromFS 是在线更新的一站式入口：先按更新专用规则
// （含强制权重校验和）从 fsys 的 dir 目录加载并完整校验新包，通过后再
// 执行不停机切换。加载失败时返回 modelpkg 的可区分原因，运行时状态不变；
// 成功时返回新代的版本信息。
func (rt *Runtime) UpdateModelFromFS(name string, fsys fs.FS, dir string) (VersionInfo, error) {
	pkg, err := modelpkg.LoadForUpdate(fsys, dir)
	if err != nil {
		return VersionInfo{}, fmt.Errorf("runtime: update %q rejected: %w", name, err)
	}
	if err := rt.UpdateModel(name, pkg); err != nil {
		return VersionInfo{}, err
	}
	rt.mu.Lock()
	info := versionInfo(rt.models[name].active)
	rt.mu.Unlock()
	return info, nil
}

// UpdateModel 在不中断服务的前提下，把对外模型名 name 切换到新一代权重 pkg：
//   - 切换是原子的指针交换：期间任何新建会话要么拿到旧代、要么拿到新代，
//     绝不会遇到“模型短暂不存在/不可用”的窗口；
//   - 切换前已存在的会话继续绑定旧代权重跑完，结果不受影响；
//   - 新旧两套权重在旧会话全部结束前同时驻留并各自记账，旧代自动回收。
//
// 失败语义（全部为整体失败，当前对外版本继续服务，状态不变、无残留预占）：
//   - pkg 未声明权重校验和 -> modelpkg.ErrChecksumRequired（无法证明内容未被改动）；
//   - 设备剩余空间不足以同时容纳新旧两套权重 -> ErrInsufficientResources；
//   - 模型名不存在 -> ErrUnknownModel；运行时已关闭 -> ErrRuntimeClosed。
//
// 包本身不合规（版本/精度组合不支持、量化参数缺失、维度对不上、权重不完整、
// 内容与校验和不符）应在调用前经 modelpkg.LoadForUpdate 发现，原因由
// modelpkg 的哨兵错误区分，不会进入切换环节。
func (rt *Runtime) UpdateModel(name string, pkg *modelpkg.Package) error {
	if name == "" || pkg == nil {
		return fmt.Errorf("runtime: update model requires non-empty name and non-null package")
	}
	if !pkg.HasChecksums() {
		return fmt.Errorf("%w: model %q update package declares no weight checksums",
			modelpkg.ErrChecksumRequired, name)
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	need := pkg.Bytes()
	if rt.used+need > rt.cfg.MaxDeviceBytes {
		// 必须同时容纳“旧代（可能仍有旧会话）+ 新代”才允许切换。
		return fmt.Errorf("%w: update %q needs %d more bytes for new weights (coexist), %d of %d used",
			ErrInsufficientResources, name, need, rt.used, rt.cfg.MaxDeviceBytes)
	}

	g := rt.newGeneration(e, pkg)
	// 只保留紧邻的上一代作为回退目标；更早的代不再可回退并立即摘除：
	// 无旧会话当场归还字节，仍有旧会话则保留权重、待其全部结束后自动回收。
	old := e.gens
	e.gens = nil
	if len(old) > 0 {
		// 除栈顶（紧邻上一代）外全部摘除。
		for _, beyond := range old[:len(old)-1] {
			rt.detachGeneration(e, beyond)
		}
		e.gens = append(e.gens, old[len(old)-1])
	}
	e.gens = append(e.gens, g)
	e.active = g
	rt.used += need
	return nil
}

// RollbackModel 一键把对外模型名回退到代际栈中的上一代：
// 当前代被顶出（无旧会话时立即归还字节；有旧会话则等其结束后自动回收），
// 上一代重新成为新建会话使用的版本。没有更早的版本时返回 ErrNoRollbackTarget。
// 返回回退后生效版本的信息。
func (rt *Runtime) RollbackModel(name string) (VersionInfo, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return VersionInfo{}, ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return VersionInfo{}, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	if len(e.gens) < 2 {
		return VersionInfo{}, fmt.Errorf("%w: model %q is at its first generation",
			ErrNoRollbackTarget, name)
	}
	top := e.gens[len(e.gens)-1]
	e.gens = e.gens[:len(e.gens)-1]
	prev := e.gens[len(e.gens)-1]
	e.active = prev
	rt.detachGeneration(e, top)
	return versionInfo(prev), nil
}

// detachGeneration 把一代从对外栈摘除：无旧会话立即回收权重字节，
// 否则登记到 retired 等最后一个会话结束时回收。必须持有 rt.mu。
func (rt *Runtime) detachGeneration(e *modelEntry, g *generation) {
	g.detached = true
	if len(g.sessions) == 0 {
		rt.used -= g.pkg.Bytes()
		return
	}
	e.retired = append(e.retired, g)
}

// ActiveModel 查询某模型名当前对外生效版本的信息。
func (rt *Runtime) ActiveModel(name string) (VersionInfo, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return VersionInfo{}, ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return VersionInfo{}, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	return versionInfo(e.active), nil
}

// ModelVersions 列出某模型名所有驻留代的信息（代际栈自旧到新，末尾为当前
// 生效版本）；Active 标记当前对外版本，其余栈内版本即为“可回退版本”。
// 已被顶出、仅等旧会话结束后回收的代以 Retiring=true 附在末尾。
func (rt *Runtime) ModelVersions(name string) ([]VersionInfo, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	out := make([]VersionInfo, 0, len(e.gens)+len(e.retired))
	for _, g := range e.gens {
		out = append(out, versionInfo(g))
	}
	for _, g := range e.retired {
		info := versionInfo(g)
		info.Retiring = true
		out = append(out, info)
	}
	return out, nil
}

// versionInfo 必须在持有 rt.mu 时调用。
func versionInfo(g *generation) VersionInfo {
	meta := g.pkg.Meta()
	return VersionInfo{
		ModelName:      g.entry.name,
		Revision:       g.pkg.Revision(),
		FormatVersion:  meta.Version,
		Precision:      meta.Precision,
		WeightBytes:    g.pkg.Bytes(),
		Active:         g.entry.active == g && !g.detached,
		Retiring:       g.detached,
		ActiveSessions: len(g.sessions),
	}
}

// UnloadModel 卸载模型并归还全部驻留权重字节数。任一驻留代（含等待回收的
// 已摘除代）仍有会话时拒绝卸载；不存在的模型返回 ErrUnknownModel。
func (rt *Runtime) UnloadModel(name string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	e, ok := rt.models[name]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	n := 0
	for _, g := range e.gens {
		n += len(g.sessions)
	}
	for _, g := range e.retired {
		n += len(g.sessions)
	}
	if n > 0 {
		return fmt.Errorf("runtime: model %q has %d active session(s), close them first", name, n)
	}
	for _, g := range e.gens {
		rt.used -= g.pkg.Bytes()
	}
	for _, g := range e.retired {
		rt.used -= g.pkg.Bytes()
	}
	delete(rt.models, name)
	return nil
}

// NewSession 基于某模型当前生效的一代创建独立推理会话，并立即预占工作区预算。
//   - 模型不存在                         -> ErrUnknownModel
//   - 设备剩余内存不足会话预留工作区     -> ErrInsufficientResources
//
// 会话在创建时绑定具体一代；之后的切换/回退不改变该绑定。
func (rt *Runtime) NewSession(modelName string, opts SessionOptions) (*Session, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, ErrRuntimeClosed
	}
	e, ok := rt.models[modelName]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, modelName)
	}
	g := e.active
	reserve := opts.reserveBytes(g.view)
	if reserve < 0 {
		return nil, fmt.Errorf("runtime: invalid session options")
	}
	if rt.used+reserve > rt.cfg.MaxDeviceBytes {
		return nil, fmt.Errorf("%w: session for %q needs %d workspace bytes, %d of %d used",
			ErrInsufficientResources, modelName, reserve, rt.used, rt.cfg.MaxDeviceBytes)
	}

	s := &Session{
		rt:       rt,
		gen:      g,
		opts:     opts,
		reserve:  reserve,
		maxBatch: opts.maxBatch(g.view),
		done:     make(chan struct{}),
	}
	g.sessions[s] = struct{}{}
	rt.used += reserve
	return s, nil
}

// release 在会话结束时归还其预占工作区并从绑定代的会话集合中摘除。
// 若绑定代已被顶出且这是最后一个旧会话，同时回收该代权重字节。仅执行一次。
func (rt *Runtime) release(s *Session) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if s.released {
		return
	}
	s.released = true
	rt.used -= s.reserve
	g := s.gen
	delete(g.sessions, s)
	if g.detached && len(g.sessions) == 0 {
		rt.used -= g.pkg.Bytes()
		e := g.entry
		for i, r := range e.retired {
			if r == g {
				e.retired = append(e.retired[:i], e.retired[i+1:]...)
				break
			}
		}
	}
}

// Close 关闭运行时。若仍有活跃会话则拒绝，避免在后台使用中拆除设备；
// 关闭后一切加载/更新/回退/建会话操作返回 ErrRuntimeClosed。
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	for _, e := range rt.models {
		for _, g := range e.gens {
			if len(g.sessions) > 0 {
				return fmt.Errorf("runtime: cannot close, %d active session(s)", len(g.sessions))
			}
		}
		for _, g := range e.retired {
			if len(g.sessions) > 0 {
				return fmt.Errorf("runtime: cannot close, %d active session(s)", len(g.sessions))
			}
		}
	}
	rt.closed = true
	rt.models = nil
	rt.used = 0
	return nil
}

// Is 使用 errors.Is 的便捷封装，供调用方/测试区分原因。
func Is(err, target error) bool { return errors.Is(err, target) }
