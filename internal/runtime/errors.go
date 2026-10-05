package runtime

import "errors"

// 运行时阶段可区分的失败原因。调用方应使用 errors.Is 判定。
var (
	// ErrUnknownModel 模型尚未加载或已被卸载。
	ErrUnknownModel = errors.New("runtime: unknown model")
	// ErrInsufficientResources 设备全局资源预算不足以加载模型或创建会话。
	ErrInsufficientResources = errors.New("runtime: insufficient device resources")
	// ErrRuntimeClosed 运行时已关闭，拒绝新的加载与推理。
	ErrRuntimeClosed = errors.New("runtime: runtime closed")
	// ErrInputDimMismatch 输入向量维度与模型声明不一致。
	ErrInputDimMismatch = errors.New("runtime: input dimension mismatch")
	// ErrSessionClosed 会话已关闭，不能再发起推理。
	ErrSessionClosed = errors.New("runtime: session closed")
	// ErrSessionCanceled 会话被显式取消，正在进行的推理中止。
	ErrSessionCanceled = errors.New("runtime: session canceled")
	// ErrSessionTimeout 会话推理超过其超时预算。
	ErrSessionTimeout = errors.New("runtime: session timeout")
	// ErrSessionBudgetExceeded 会话推理超出其单批资源预算（仅影响自身）。
	ErrSessionBudgetExceeded = errors.New("runtime: session resource budget exceeded")
	// ErrNoRollbackTarget 模型没有可回退到的上一个版本。
	ErrNoRollbackTarget = errors.New("runtime: no rollback target")
	// ErrInputNotFinite 输入含 NaN 或 ±Inf，无法得到有意义的推理结果，明确拒绝。
	ErrInputNotFinite = errors.New("runtime: input contains NaN or Inf")
)
