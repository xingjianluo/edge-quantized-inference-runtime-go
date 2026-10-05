package runtime

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/internal/modelpkg"
)

// SessionOptions 单个会话的预算与超时设置。
type SessionOptions struct {
	// MaxBatchBytes 本会话允许的单批工作区内存上限（字节）。
	// 一批输入按 workspaceBytes 估算，超过则整批拒绝（ErrSessionBudgetExceeded），
	// 仅影响该会话自身；<=0 表示不限制会话级预算（仍受设备总预算约束）。
	MaxBatchBytes int64
	// Timeout 单次推理（含整批）最长墙钟时间；<=0 表示不限制。
	Timeout time.Duration
}

// workspaceForOne 单条输入的工作区记账字节数。
func workspaceForOne(view modelView) int64 { return workspaceBytes(view, 1) }

// reserveBytes 会话创建时向设备预占的工作区字节数：
// 取会话预算允许的最大整批工作区，使会话在预算内推理时设备记账稳定。
func (o SessionOptions) reserveBytes(view modelView) int64 {
	one := workspaceForOne(view)
	if one <= 0 {
		return 0
	}
	if o.MaxBatchBytes <= 0 {
		return one
	}
	maxN := o.MaxBatchBytes / one
	if maxN < 1 {
		return 0
	}
	return maxN * one
}

// maxBatch 返回会话允许的单批最大条数；0 表示任何推理都会超会话预算。
func (o SessionOptions) maxBatch(view modelView) int {
	one := workspaceForOne(view)
	if one <= 0 {
		return 0
	}
	if o.MaxBatchBytes <= 0 {
		return int(^uint(0)>>1) / int(one)
	}
	return int(o.MaxBatchBytes / one)
}

// Session 相互隔离的推理会话：持有独立生命周期与错误状态，
// 不与其他会话共享可变数据（模型权重为只读共享，不影响隔离性）。
// 会话在创建时绑定具体一代权重，之后的在线切换/回退不改变该绑定。
type Session struct {
	rt   *Runtime
	gen  *generation
	opts SessionOptions

	reserve  int64
	maxBatch int

	mu       sync.Mutex
	closed   bool
	canceled bool
	released bool

	// inflight 让 Close 等待进行中的推理退出，保证预算干净归还。
	inflight sync.WaitGroup
	// done 在会话被 Cancel/Close 时关闭。
	done chan struct{}
}

// Infer 对单条输入执行确定性量化推理。
func (s *Session) Infer(ctx context.Context, input []float32) ([]float32, error) {
	results, err := s.run(ctx, [][]float32{input})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// InferBatch 对一批输入逐条执行与 Infer 完全相同的确定性计算，
// results[i] 与单独 Infer(inputs[i]) 逐位一致。
// 取消/超时/超预算使整批失败、不返回部分结果，且只影响本会话。
func (s *Session) InferBatch(ctx context.Context, inputs [][]float32) ([][]float32, error) {
	if len(inputs) == 0 {
		return [][]float32{}, nil
	}
	return s.run(ctx, inputs)
}

// run 是单条与批量共享的唯一执行路径，从结构上保证两种调用结论一致。
func (s *Session) run(ctx context.Context, inputs [][]float32) ([][]float32, error) {
	n := len(inputs)

	s.mu.Lock()
	if s.closed || s.released {
		s.mu.Unlock()
		return nil, ErrSessionClosed
	}
	if s.canceled {
		s.mu.Unlock()
		return nil, ErrSessionCanceled
	}
	if n > s.maxBatch {
		need := workspaceBytes(s.gen.view, n)
		s.mu.Unlock()
		return nil, fmt.Errorf("%w: batch of %d needs %d bytes, session budget %d bytes",
			ErrSessionBudgetExceeded, n, need, s.opts.MaxBatchBytes)
	}
	s.inflight.Add(1)
	done := s.done
	view := s.gen.view
	timeout := s.opts.Timeout
	s.mu.Unlock()
	defer s.inflight.Done()

	// 超时仅作用于本次调用。使用 context.WithTimeout 让超时携带确定的
	// DeadlineExceeded 原因；会话自身取消（done）拥有更高优先级。
	var (
		callCtx    context.Context
		cancelCall context.CancelFunc
	)
	if timeout > 0 {
		callCtx, cancelCall = context.WithTimeout(ctx, timeout)
	} else {
		callCtx, cancelCall = context.WithCancel(ctx)
	}
	defer cancelCall()

	// classify 在每个协作点判定当前是否应中止以及可区分原因。
	// 先看会话是否被显式取消（done），再看本次调用 context：
	// DeadlineExceeded -> 超时；其余（外部 ctx 取消）-> 取消。
	classify := func() error {
		select {
		case <-done:
			return ErrSessionCanceled
		default:
		}
		if err := callCtx.Err(); err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				return ErrSessionTimeout
			}
			return ErrSessionCanceled
		}
		return nil
	}

	results := make([][]float32, n)
	for i, in := range inputs {
		if err := classify(); err != nil {
			return nil, err
		}
		out, err := inferOne(view, in)
		if err != nil {
			return nil, err
		}
		results[i] = out
	}
	if err := classify(); err != nil {
		return nil, err
	}
	return results, nil
}

// Cancel 取消会话：进行中的推理在协作点以 ErrSessionCanceled 失败，
// 之后本会话拒绝新推理。不影响其他会话；设备预算由 Close 归还。
func (s *Session) Cancel() {
	s.mu.Lock()
	if s.closed || s.canceled {
		s.mu.Unlock()
		return
	}
	s.canceled = true
	done := s.done
	s.mu.Unlock()
	close(done)
}

// Close 关闭会话：等待进行中的推理退出，干净归还预占设备预算。可重复调用。
func (s *Session) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	if !s.canceled {
		s.canceled = true
		done := s.done
		s.mu.Unlock()
		close(done)
	} else {
		s.mu.Unlock()
	}

	s.inflight.Wait()
	s.rt.release(s)
	return nil
}

// ModelName 返回会话绑定的模型名（便于日志）。
func (s *Session) ModelName() string { return s.gen.entry.name }

// Revision 返回会话当前使用的权重代的内容指纹。
// 由于会话绑定创建时的一代，切换前建的旧会话与切换后建的新会话
// 返回的指纹不同，可直接用于判定“结果来自哪一代权重”。
func (s *Session) Revision() string { return s.gen.pkg.Revision() }

// FormatVersion 返回会话绑定代的模型格式版本字符串。
func (s *Session) FormatVersion() modelpkg.Version { return s.gen.pkg.Meta().Version }

// Precision 返回会话绑定代的量化精度。
func (s *Session) Precision() modelpkg.Precision { return s.gen.pkg.Meta().Precision }
