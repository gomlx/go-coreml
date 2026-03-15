//go:build darwin && cgo

// Package coreml provides a CoreML backend for GoMLX.
// The backend is only registered on Darwin (macOS) with CGO enabled.
// On other platforms, the package can still be imported but the backend
// will not be available.
package coreml

import (
	"github.com/gomlx/gomlx/backends"
)

func init() {
	backends.Register(BackendName, New)
}
