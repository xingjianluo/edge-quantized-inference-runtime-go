// Package runtime simulates an edge device that loads quantized model
// packages and serves isolated inference sessions under a device resource
// budget.
//
// Isolation rules:
//
//   - models loaded into a Runtime are immutable; sessions never mutate model
//     bytes, and all per-inference buffers are allocated inside the call,
//   - canceling, timing out or closing one session is local to that session:
//     other sessions and the device's loaded models are untouched,
//   - a session's working memory is reserved once at creation against the
//     device budget and released exactly once on Close,
//   - a request whose batch exceeds the session budget is rejected up front,
//     before any compute, with reason_session_budget_exceeded.
package runtime

import (
	"context"
	"sync"
	"time"

	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/kernel"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/loader"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/merrors"
	"github.com/xingjianluo/edge-quantized-inference-runtime-go/pkg/model"
)

// perSessionFixedOverhead accounts for the session object and its reserved
// scratch buffers used for a single inference row.
const perSessionFixedOverhead int64 = 2048

// DeviceBudget describes the simulated on-device resource limits.
type DeviceBudget struct {
	// MemoryBytes limits total device memory occupied by loaded models plus
	// active session working memory. Zero means unlimited.
	MemoryBytes int64
	// MaxConcurrentSessions limits simultaneously open sessions. Zero means
	// unlimited.
	MaxConcurrentSessions int
}

// Config configures a Runtime.
type Config struct {
	Budget DeviceBudget
}

// SessionOptions configures one inference session.
type SessionOptions struct {
	// MaxBatchSize bounds InferBatch length (0 defaults to 1).
	MaxBatchSize int
	// ComputeLatency inserts a deterministic per-row delay so cancellation
	// and timeout behavior is observable without depending on machine speed.
	ComputeLatency time.Duration
}

// Stats is a point-in-time snapshot of runtime resource usage.
type Stats struct {
	LoadedModelBytes int64
	SessionBytes     int64
	ActiveSessions   int
	LoadedModelCount int
}

type loadedEntry struct {
	model *model.LoadedModel
	bytes int64
}

// Runtime is the edge inference runtime. A Runtime is safe for concurrent use.
type Runtime struct {
	mu           sync.Mutex
	cfg          Config
	closed       bool
	models       map[string]*loadedEntry
	sessionCount int
	sessionBytes int64
}

// Session is one isolated inference session.
type Session struct {
	rt         *Runtime
	mu         sync.Mutex
	model      *model.LoadedModel
	reserved   int64
	maxBatch   int
	latency    time.Duration
	closed     bool
	activeRows int
}

// New creates a Runtime with the given device configuration.
func New(cfg Config) *Runtime {
	return &Runtime{
		cfg:    cfg,
		models: make(map[string]*loadedEntry),
	}
}

// LoadModel loads and accounts a model package. The load fails atomically
// with a distinguishable reason before any device state changes if the package
// is invalid or the device budget cannot accommodate it.
func (rt *Runtime) LoadModel(dir string) (*model.LoadedModel, error) {
	m, err := loader.Load(dir, loader.Options{})
	if err != nil {
		return nil, err
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, merrors.New(merrors.ReasonSessionClosed,
			"runtime is closed")
	}
	if _, exists := rt.models[m.Manifest.ModelID]; exists {
		return nil, merrors.New(merrors.ReasonInvalidManifest,
			"model %q is already loaded", m.Manifest.ModelID)
	}
	need := loader.ModelMemoryBytes(m)
	if rt.cfg.Budget.MemoryBytes > 0 && rt.usedMemoryLocked()+need > rt.cfg.Budget.MemoryBytes {
		return nil, merrors.New(merrors.ReasonDeviceBudgetExceeded,
			"loading model %q (%d bytes) would exceed device budget %d (%d in use)",
			m.Manifest.ModelID, need, rt.cfg.Budget.MemoryBytes, rt.usedMemoryLocked())
	}
	rt.models[m.Manifest.ModelID] = &loadedEntry{model: m, bytes: need}
	return m, nil
}

// UnloadModel removes a loaded model. It fails with reason_model_busy while
// sessions referencing it are open and with reason_model_not_found when the
// ID is unknown.
func (rt *Runtime) UnloadModel(modelID string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if _, ok := rt.models[modelID]; !ok {
		return merrors.New(merrors.ReasonModelNotFound,
			"model %q is not loaded", modelID)
	}
	if rt.sessionCount > 0 {
		return merrors.New(merrors.ReasonModelBusy,
			"cannot unload %q: %d session(s) open", modelID, rt.sessionCount)
	}
	delete(rt.models, modelID)
	return nil
}

// Model returns the loaded model or nil if absent.
func (rt *Runtime) Model(modelID string) *model.LoadedModel {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if e, ok := rt.models[modelID]; ok {
		return e.model
	}
	return nil
}

// Stats returns a snapshot of current resource usage.
func (rt *Runtime) Stats() Stats {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return Stats{
		LoadedModelBytes: rt.modelBytesLocked(),
		SessionBytes:     rt.sessionBytes,
		ActiveSessions:   rt.sessionCount,
		LoadedModelCount: len(rt.models),
	}
}

// Close releases the runtime. It fails with reason_model_busy if sessions are
// still open; after Close all new model/session requests are rejected.
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.sessionCount > 0 {
		return merrors.New(merrors.ReasonModelBusy,
			"cannot close runtime: %d session(s) still open", rt.sessionCount)
	}
	rt.closed = true
	rt.models = map[string]*loadedEntry{}
	return nil
}

func (rt *Runtime) modelBytesLocked() int64 {
	var total int64
	for _, e := range rt.models {
		total += e.bytes
	}
	return total
}

func (rt *Runtime) usedMemoryLocked() int64 {
	return rt.modelBytesLocked() + rt.sessionBytes
}

// NewSession opens an isolated session for the named model and reserves its
// working memory against the device budget. A rejected reservation changes no
// state.
func (rt *Runtime) NewSession(modelID string, opts SessionOptions) (*Session, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, merrors.New(merrors.ReasonSessionClosed, "runtime is closed")
	}
	entry, ok := rt.models[modelID]
	if !ok {
		return nil, merrors.New(merrors.ReasonModelNotFound,
			"model %q is not loaded", modelID)
	}
	if rt.cfg.Budget.MaxConcurrentSessions > 0 &&
		rt.sessionCount >= rt.cfg.Budget.MaxConcurrentSessions {
		return nil, merrors.New(merrors.ReasonDeviceBudgetExceeded,
			"max concurrent sessions %d reached",
			rt.cfg.Budget.MaxConcurrentSessions)
	}

	maxBatch := opts.MaxBatchSize
	if maxBatch <= 0 {
		maxBatch = 1
	}
	// Reserved working memory covers one inference row's activation buffers
	// plus bookkeeping; batch rows run sequentially and reuse this space.
	reserved := int64(entry.model.ActivationsBytes) + perSessionFixedOverhead
	if rt.cfg.Budget.MemoryBytes > 0 && rt.usedMemoryLocked()+reserved > rt.cfg.Budget.MemoryBytes {
		return nil, merrors.New(merrors.ReasonDeviceBudgetExceeded,
			"session for %q needs %d reserved bytes, device budget %d has %d free",
			modelID, reserved, rt.cfg.Budget.MemoryBytes,
			rt.cfg.Budget.MemoryBytes-rt.usedMemoryLocked())
	}

	rt.sessionCount++
	rt.sessionBytes += reserved
	s := &Session{
		rt:       rt,
		model:    entry.model,
		reserved: reserved,
		maxBatch: maxBatch,
		latency:  opts.ComputeLatency,
	}
	return s, nil
}

// Model returns the immutable model this session is bound to.
func (s *Session) Model() *model.LoadedModel { return s.model }

// ReservedBytes reports the device memory reserved for this session.
func (s *Session) ReservedBytes() int64 { return s.reserved }

// Infer runs one inference under ctx. Cancellation or timeout affects only
// this call; resources are goroutine-local and released when the call returns.
func (s *Session) Infer(ctx context.Context, input []float64) (kernel.Prediction, error) {
	if err := s.enter(ctx, 1); err != nil {
		return kernel.Prediction{}, err
	}
	defer s.leave()
	return s.runOne(ctx, input)
}

// InferBatch runs a batch sequentially in input order. Cancellation is
// observed before each row, so a canceled or timed-out batch returns
// immediately and never disturbs other sessions.
func (s *Session) InferBatch(ctx context.Context, inputs [][]float64) ([]kernel.Prediction, error) {
	if err := s.enter(ctx, len(inputs)); err != nil {
		return nil, err
	}
	defer s.leave()

	out := make([]kernel.Prediction, 0, len(inputs))
	for i, in := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, mapContextErr(ctx)
		}
		p, err := s.runOne(ctx, in)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
		_ = i
	}
	return out, nil
}

// enter validates session liveness, context and the per-session batch budget.
func (s *Session) enter(ctx context.Context, rows int) error {
	if err := ctx.Err(); err != nil {
		return mapContextErr(ctx)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return merrors.New(merrors.ReasonSessionClosed,
			"session for model %q is closed", s.model.Manifest.ModelID)
	}
	if rows <= 0 {
		return merrors.New(merrors.ReasonInputShape,
			"empty inference batch")
	}
	if rows > s.maxBatch {
		return merrors.New(merrors.ReasonSessionBudgetExceeded,
			"batch of %d exceeds session max batch size %d",
			rows, s.maxBatch)
	}
	s.activeRows = rows
	return nil
}

func (s *Session) leave() {
	s.mu.Lock()
	s.activeRows = 0
	s.mu.Unlock()
}

func (s *Session) runOne(ctx context.Context, input []float64) (kernel.Prediction, error) {
	// Observability point for cancellation: simulated device latency. The
	// wait is interruptible by ctx, distinguishing cancel from timeout.
	if s.latency > 0 {
		select {
		case <-ctx.Done():
			return kernel.Prediction{}, mapContextErr(ctx)
		case <-time.After(s.latency):
		}
	}
	return kernel.Infer(s.model, input)
}

// Close releases the session and returns its reserved resources exactly once.
// It is idempotent.
func (s *Session) Close() error {
	s.mu.Lock()
	already := s.closed
	s.closed = true
	s.mu.Unlock()
	if already {
		return nil
	}

	rt := s.rt
	rt.mu.Lock()
	rt.sessionCount--
	rt.sessionBytes -= s.reserved
	rt.mu.Unlock()
	return nil
}

func mapContextErr(ctx context.Context) error {
	switch ctx.Err() {
	case context.DeadlineExceeded:
		return merrors.New(merrors.ReasonTimeout, "inference deadline exceeded")
	case context.Canceled:
		return merrors.New(merrors.ReasonCanceled, "inference canceled")
	default:
		return merrors.New(merrors.ReasonCanceled, "context done without a reason")
	}
}
