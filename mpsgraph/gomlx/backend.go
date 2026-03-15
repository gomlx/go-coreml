// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

// Package mpsgraph implements a GoMLX backend using Apple's MPSGraph framework
// for GPU-accelerated tensor computation on Apple Silicon.
package mpsgraph

import (
	"sync"

	"github.com/gomlx/go-darwinml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mpsgraph"

// Backend implements the backends.Backend interface using Apple's MPSGraph.
type Backend struct {
	ctx         *bridge.Context
	deviceName  string
	mu          sync.RWMutex
	isFinalized bool
}

// Verify interface compliance.
var _ backends.Backend = &Backend{}

// New creates a new MPSGraph backend. config is currently unused.
func New(config string) (backends.Backend, error) {
	ctx, err := bridge.NewContext()
	if err != nil {
		return nil, errors.Wrap(err, "creating MPSGraph backend")
	}
	b := &Backend{
		ctx:        ctx,
		deviceName: ctx.DeviceName(),
	}
	return b, nil
}

// Name returns the backend name.
func (b *Backend) Name() string { return BackendName }

// String returns the backend name.
func (b *Backend) String() string { return b.Name() }

// Description returns a human-readable description.
func (b *Backend) Description() string {
	return "MPSGraph GPU backend (" + b.deviceName + ")"
}

// NumDevices returns 1 (single GPU on Apple Silicon).
func (b *Backend) NumDevices() int { return 1 }

// DeviceDescription returns the Metal device name.
func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	return b.deviceName
}

// Capabilities returns the set of supported ops and dtypes.
func (b *Backend) Capabilities() backends.Capabilities {
	return backendCapabilities
}

// Builder creates a new computation builder, reusing the backend's Metal device.
func (b *Backend) Builder(name string) backends.Builder {
	ctx, err := bridge.NewContextWithDevice(b.ctx.DeviceHandle())
	if err != nil {
		// Graph building functions panic on error per GoMLX convention.
		panic(errors.Wrapf(err, "creating MPSGraph builder %q", name))
	}
	return newBuilder(b, name, ctx)
}

// Finalize releases all backend resources.
func (b *Backend) Finalize() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.isFinalized {
		return
	}
	b.isFinalized = true
	if b.ctx != nil {
		b.ctx.Destroy()
		b.ctx = nil
	}
}

// IsFinalized returns whether the backend has been finalized.
func (b *Backend) IsFinalized() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.isFinalized
}

// --- DataInterface implementation ---

// BufferFinalize releases a buffer's resources.
func (b *Backend) BufferFinalize(buffer backends.Buffer) error {
	if b.isFinalized {
		return nil
	}
	buf, ok := buffer.(*gpuBuffer)
	if !ok {
		return errors.Errorf("BufferFinalize: expected *gpuBuffer, got %T", buffer)
	}
	buf.valid = false
	buf.flat = nil
	return nil
}

// BufferShape returns the shape of a buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	buf, ok := buffer.(*gpuBuffer)
	if !ok {
		return shapes.Invalid(), errors.Errorf("BufferShape: expected *gpuBuffer, got %T", buffer)
	}
	return buf.shape, nil
}

// BufferDeviceNum returns the device number (always 0 for single device).
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	return 0, nil
}

// BufferToFlatData copies buffer data into a Go flat slice.
func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) error {
	buf, ok := buffer.(*gpuBuffer)
	if !ok {
		return errors.Errorf("BufferToFlatData: expected *gpuBuffer, got %T", buffer)
	}
	return bufferCopyToFlat(buf, flat)
}

// BufferFromFlatData creates a buffer from a Go flat slice.
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	return bufferFromFlat(flat, shape)
}

// HasSharedBuffers returns true since we use Go-managed memory for buffers.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer creates a new buffer with direct access to its memory.
func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any, err error) {
	buf := newBuffer(shape)
	return buf, buf.flat, nil
}

// BufferData returns direct access to the buffer's flat data.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	buf, ok := buffer.(*gpuBuffer)
	if !ok {
		return nil, errors.Errorf("BufferData: expected *gpuBuffer, got %T", buffer)
	}
	return buf.flat, nil
}

// BufferCopyToDevice copies buffer to another device (not supported with single device).
func (b *Backend) BufferCopyToDevice(source backends.Buffer, deviceNum backends.DeviceNum) (backends.Buffer, error) {
	return nil, errors.New("BufferCopyToDevice: only one device supported")
}
