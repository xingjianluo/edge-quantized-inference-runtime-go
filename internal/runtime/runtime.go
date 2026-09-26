package runtime

import (
	"errors"
	"fmt"
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
	models map[string]*loadedModel
	used   int64
	cfg    Config
	closed bool
}

// loadedModel 运行时内部登记的一个模型。
type loadedModel struct {
	name     string
	pkg      *modelpkg.Package
	view     modelView
	sessions map[*Session]struct{}
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
		models: make(map[string]*loadedModel),
		cfg:    cfg,
	}, nil
}

// LoadModel 将已校验模型包登记到设备并预占其权重字节数。
//   - 运行时已关闭            -> ErrRuntimeClosed
//   - 同名模型已存在          -> 错误（不覆盖）
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
	rt.models[name] = &loadedModel{
		name:     name,
		pkg:      pkg,
		view:     buildView(pkg),
		sessions: make(map[*Session]struct{}),
	}
	rt.used += need
	return nil
}

// UnloadModel 卸载模型并归还权重字节数。仍有活跃会话时拒绝卸载，
// 以避免会话引用到已释放模型；不存在的模型返回 ErrUnknownModel。
func (rt *Runtime) UnloadModel(name string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	m, ok := rt.models[name]
	if !ok {
		return fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	if n := len(m.sessions); n > 0 {
		return fmt.Errorf("runtime: model %q has %d active session(s), close them first", name, n)
	}
	rt.used -= m.pkg.Bytes()
	delete(rt.models, name)
	return nil
}

// NewSession 基于已加载模型创建独立推理会话，并立即预占工作区预算。
//   - 模型不存在                         -> ErrUnknownModel
//   - 设备剩余内存不足会话预留工作区     -> ErrInsufficientResources
//
// 返回后会话持有自己的 modelView 引用（只读）与独立状态，互不共享可变数据。
func (rt *Runtime) NewSession(modelName string, opts SessionOptions) (*Session, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, ErrRuntimeClosed
	}
	m, ok := rt.models[modelName]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, modelName)
	}
	reserve := opts.reserveBytes(m.view)
	if reserve < 0 {
		return nil, fmt.Errorf("runtime: invalid session options")
	}
	if rt.used+reserve > rt.cfg.MaxDeviceBytes {
		return nil, fmt.Errorf("%w: session for %q needs %d workspace bytes, %d of %d used",
			ErrInsufficientResources, modelName, reserve, rt.used, rt.cfg.MaxDeviceBytes)
	}

	s := &Session{
		rt:       rt,
		model:    m,
		opts:     opts,
		reserve:  reserve,
		maxBatch: opts.maxBatch(m.view),
		done:     make(chan struct{}),
	}
	m.sessions[s] = struct{}{}
	rt.used += reserve
	return s, nil
}

// release 在会话结束时归还其预占工作区并从模型的会话集合中摘除。
// 仅执行一次；由 Session.Close 调用。
func (rt *Runtime) release(s *Session) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if !s.released {
		s.released = true
		rt.used -= s.reserve
		delete(s.model.sessions, s)
	}
}

// Close 关闭运行时。若仍有活跃会话则拒绝，避免在后台使用中拆除设备；
// 关闭后一切加载/卸载/建会话操作返回 ErrRuntimeClosed。
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	for _, m := range rt.models {
		if len(m.sessions) > 0 {
			return fmt.Errorf("runtime: cannot close, %d active session(s)", len(m.sessions))
		}
	}
	rt.closed = true
	rt.models = nil
	rt.used = 0
	return nil
}

// Is 使用 errors.Is 的便捷封装，供调用方/测试区分原因。
func Is(err, target error) bool { return errors.Is(err, target) }
