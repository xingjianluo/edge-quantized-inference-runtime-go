// Package merrors defines the distinguishable failure reasons used across the
// edge inference runtime. Every failure returned by the loader, runtime or a
// session carries one of the sentinel reasons below so callers and tests can
// branch on the exact cause instead of matching on message text.
package merrors

import (
	"errors"
	"fmt"
)

// Reason identifies a failure category.
type Reason string

const (
	// ReasonUnsupportedVersion: the package format version is not supported.
	ReasonUnsupportedVersion Reason = "unsupported_version"
	// ReasonUnsupportedPrecision: the requested weight or input precision is
	// not supported by the package format version.
	ReasonUnsupportedPrecision Reason = "unsupported_precision"
	// ReasonMissingQuantization: a tensor is missing scale/zero-point
	// quantization parameters required to interpret its values.
	ReasonMissingQuantization Reason = "missing_quantization"
	// ReasonDimensionMismatch: two tensors that must agree in shape do not
	// (e.g. layer output dim vs next layer input dim).
	ReasonDimensionMismatch Reason = "dimension_mismatch"
	// ReasonInvalidManifest: a manifest is missing required fields or
	// contains a structurally invalid value.
	ReasonInvalidManifest Reason = "invalid_manifest"
	// ReasonWeightCorrupt: a weight/bias blob is missing bytes, contains an
	// out-of-range quantized value, or fails checksum verification.
	ReasonWeightCorrupt Reason = "weight_corrupt"
	// ReasonPackageNotFound: the model package directory does not exist.
	ReasonPackageNotFound Reason = "package_not_found"
	// ReasonDeviceBudgetExceeded: loading a model would exceed the device
	// memory budget.
	ReasonDeviceBudgetExceeded Reason = "device_budget_exceeded"
	// ReasonSessionBudgetExceeded: a single session's input/batch would
	// exceed the per-session resource budget.
	ReasonSessionBudgetExceeded Reason = "session_budget_exceeded"
	// ReasonModelNotFound: the referenced model ID is not loaded.
	ReasonModelNotFound Reason = "model_not_found"
	// ReasonModelBusy: an operation on a model is refused because sessions
	// are still active.
	ReasonModelBusy Reason = "model_busy"
	// ReasonInputShape: an inference input has the wrong dimension.
	ReasonInputShape Reason = "input_shape"
	// ReasonInputValue: an inference input contains a non-finite value.
	ReasonInputValue Reason = "input_value"
	// ReasonCanceled: the session context was canceled.
	ReasonCanceled Reason = "canceled"
	// ReasonTimeout: the session deadline elapsed before inference finished.
	ReasonTimeout Reason = "timeout"
	// ReasonSessionClosed: inference was attempted on a closed session.
	ReasonSessionClosed Reason = "session_closed"
)

// Error is a failure carrying an explicit, machine-readable reason.
type Error struct {
	Reason  Reason
	Message string
}

func (e *Error) Error() string { return string(e.Reason) + ": " + e.Message }

// New builds an error with the given reason and message.
func New(reason Reason, format string, args ...any) error {
	return &Error{Reason: reason, Message: fmt.Sprintf(format, args...)}
}

// ReasonOf extracts the reason from an error produced by this package. It
// returns "" if the error does not carry a reason.
// ReasonOf extracts the reason from an error produced by this package. It
// returns "" if the error does not carry a reason.
func ReasonOf(err error) Reason {
	var me *Error
	if errors.As(err, &me) {
		return me.Reason
	}
	return ""
}
