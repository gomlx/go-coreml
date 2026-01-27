//go:build darwin && cgo

package coreml

import (
	"math"
	"reflect"
	"strings"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// internalDType returns the dtype used internally for storage.
// Int64 is stored as Int32 because CoreML doesn't support Int64 well.
func internalDType(dtype dtypes.DType) dtypes.DType {
	if dtype == dtypes.Int64 {
		return dtypes.Int32
	}
	return dtype
}

// Compile-time check:
var _ backends.DataInterface = (*Backend)(nil)

// Buffer for CoreML backend holds a shape and a reference to the flat data.
//
// The flat data is CPU-backed and can be shared with CoreML models.
type Buffer struct {
	shape shapes.Shape
	valid bool

	// flat is always a slice of the underlying data type (shape.DType).
	flat any
}

type bufferPoolKey struct {
	dtype  dtypes.DType
	length int
}

// getBufferPool for given dtype/length.
func (b *Backend) getBufferPool(dtype dtypes.DType, length int) *sync.Pool {
	key := bufferPoolKey{dtype: dtype, length: length}
	poolInterface, ok := b.bufferPools.Load(key)
	if !ok {
		poolInterface, _ = b.bufferPools.LoadOrStore(key, &sync.Pool{
			New: func() interface{} {
				return &Buffer{
					flat:  reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), length, length).Interface(),
					shape: shapes.Make(dtype, length),
				}
			},
		})
	}
	return poolInterface.(*sync.Pool)
}

// getBuffer from the backend pool of buffers.
// Important: it's not necessarily initialized with zero, since it can reuse old buffers.
// Note: Uses internal dtype (Int64→Int32) for actual storage.
//
// See also Buffer.Zeros to initialize it with zeros, if needed.
func (b *Backend) getBuffer(dtype dtypes.DType, length int) *Buffer {
	if b.isFinalized {
		return nil
	}
	// Use internal dtype for storage (Int64→Int32)
	storageDType := internalDType(dtype)
	pool := b.getBufferPool(storageDType, length)
	buf := pool.Get().(*Buffer)
	buf.valid = true
	return buf
}

// getBufferForShape is a wrapper for getBuffer that also sets the buffer shape accordingly.
func (b *Backend) getBufferForShape(shape shapes.Shape) *Buffer {
	if b.isFinalized {
		return nil
	}
	buf := b.getBuffer(shape.DType, shape.Size())
	buf.shape = shape
	return buf
}

// putBuffer back into the backend pool of buffers.
// After this any references to buffer should be dropped.
func (b *Backend) putBuffer(buffer *Buffer) {
	if b.isFinalized {
		return
	}
	if buffer == nil || !buffer.shape.Ok() {
		return
	}
	buffer.valid = false
	pool := b.getBufferPool(buffer.shape.DType, buffer.shape.Size())
	pool.Put(buffer)
}

// copyFlat assumes both flat slices are of the same underlying type.
func copyFlat(flatDst, flatSrc any) {
	reflect.Copy(reflect.ValueOf(flatDst), reflect.ValueOf(flatSrc))
}

// cloneBuffer using the pool to allocate a new one.
// Returns an error if the buffer is nil or has already been finalized.
func (b *Backend) cloneBuffer(buffer *Buffer) (*Buffer, error) {
	if buffer == nil || buffer.flat == nil || !buffer.shape.Ok() || !buffer.valid {
		// the buffer is already empty or finalized.
		var issues []string
		if buffer != nil {
			if buffer.flat == nil {
				issues = append(issues, "buffer.flat was nil")
			}
			if !buffer.shape.Ok() {
				issues = append(issues, "buffer.shape was invalid")
			}
			if !buffer.valid {
				issues = append(issues, "buffer was marked as invalid")
			}
		} else {
			issues = append(issues, "buffer was nil")
		}
		return nil, errors.Errorf("cloneBuffer(%p): %s -- buffer was already finalized", buffer, strings.Join(issues, ", "))
	}
	newBuffer := b.getBuffer(buffer.shape.DType, buffer.shape.Size())
	newBuffer.shape = buffer.shape.Clone()
	copyFlat(newBuffer.flat, buffer.flat)
	return newBuffer, nil
}

// NewBuffer creates the buffer with a newly allocated flat space.
func (b *Backend) NewBuffer(shape shapes.Shape) *Buffer {
	if b.isFinalized {
		return nil
	}
	buffer := b.getBuffer(shape.DType, shape.Size())
	buffer.shape = shape.Clone()
	return buffer
}

// BufferFinalize allows the client to inform backend that buffer is no longer needed and associated resources can be
// freed immediately.
//
// A finalized buffer should never be used again. Preferably, the caller should set its references to it to nil.
func (b *Backend) BufferFinalize(backendBuffer backends.Buffer) error {
	buffer := backendBuffer.(*Buffer)
	if b.isFinalized {
		buffer.flat = nil // Accelerates GC.
		return errors.Errorf("BufferFinalize(%p): backend is already finalized", backendBuffer)
	}
	if buffer == nil || buffer.flat == nil || !buffer.shape.Ok() || !buffer.valid {
		// The buffer is already empty.
		var issues []string
		if buffer != nil {
			if buffer.flat == nil {
				issues = append(issues, "buffer.flat was nil")
			}
			if !buffer.shape.Ok() {
				issues = append(issues, "buffer.shape was invalid")
			}
			if !buffer.valid {
				issues = append(issues, "buffer was marked as invalid")
			}
		} else {
			issues = append(issues, "buffer was nil")
		}
		return errors.Errorf("BufferFinalize(%p): %s -- buffer was already finalized!?\n", buffer, strings.Join(issues, ", "))
	}
	b.putBuffer(buffer)
	return nil
}

// BufferShape returns the shape for the buffer.
func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return shapes.Invalid(), errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	return buf.shape, nil
}

// BufferDeviceNum returns the deviceNum for the buffer.
// CoreML backend only supports device 0 (single device, CPU-backed).
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	_, ok := buffer.(*Buffer)
	if !ok {
		return 0, errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}
	return 0, nil
}

// BufferToFlatData transfers the flat values of the buffer to the Go flat array.
// The slice flat must have the exact number of elements required to store the backends.Buffer shape.
//
// Note: If the shape is Int64, the buffer internally stores Int32 (CoreML limitation).
// This function handles the conversion back to Int64 when needed.
//
// See also BufferFromFlatData, BufferShape, and shapes.Shape.Size.
func (b *Backend) BufferToFlatData(backendBuffer backends.Buffer, flat any) error {
	buf, ok := backendBuffer.(*Buffer)
	if !ok {
		return errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}

	// Handle Int64 shape with Int32 storage
	if buf.shape.DType == dtypes.Int64 {
		// Buffer stores Int32 internally, but caller expects Int64
		int32Data, ok := buf.flat.([]int32)
		if !ok {
			return errors.Errorf("buffer has Int64 shape but internal storage is not []int32")
		}
		int64Data, ok := flat.([]int64)
		if !ok {
			return errors.Errorf("flat slice must be []int64 for Int64 shape, got %T", flat)
		}
		for i, v := range int32Data {
			int64Data[i] = int64(v)
		}
		return nil
	}

	copyFlat(flat, buf.flat)
	return nil
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding backends.Buffer.
//
// Note: If the shape is Int64, the buffer will internally store as Int32 (CoreML limitation).
// This function handles the conversion from Int64 to Int32.
func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	if b.isFinalized {
		return nil, errors.Errorf("backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, errors.Errorf("backend (%s) only supports deviceNum 0, cannot create buffer on deviceNum %d (shape=%s)",
			b.Name(), deviceNum, shape)
	}
	if dtypes.FromGoType(reflect.TypeOf(flat).Elem()) != shape.DType {
		return nil, errors.Errorf("flat data type (%s) does not match shape DType (%s)",
			reflect.TypeOf(flat).Elem(), shape.DType)
	}
	buffer := b.NewBuffer(shape)

	// Handle Int64 shape: convert to Int32 for storage
	if shape.DType == dtypes.Int64 {
		int64Data, ok := flat.([]int64)
		if !ok {
			return nil, errors.Errorf("flat slice must be []int64 for Int64 shape, got %T", flat)
		}
		int32Data, ok := buffer.flat.([]int32)
		if !ok {
			return nil, errors.Errorf("buffer storage should be []int32 for Int64 shape")
		}
		for i, v := range int64Data {
			if v < math.MinInt32 || v > math.MaxInt32 {
				return nil, errors.Errorf("Int64 value %d at index %d exceeds Int32 range", v, i)
			}
			int32Data[i] = int32(v)
		}
		return buffer, nil
	}

	copyFlat(buffer.flat, flat)
	return buffer, nil
}

// HasSharedBuffers returns whether the backend supports "shared buffers": these are buffers
// that can be used directly by the engine and has a local address that can be read or mutated
// directly by the client.
//
// CoreML backend uses CPU-backed buffers, so it supports shared buffers.
func (b *Backend) HasSharedBuffers() bool {
	return true
}

// NewSharedBuffer returns a "shared buffer" that can be both used as input for execution of
// computations and directly read or mutated by the clients.
//
// It panics if the backend doesn't support shared buffers -- see HasSharedBuffers.
//
// The shared buffer should not be mutated while it is used by an execution.
// Also, the shared buffer cannot be "donated" during execution.
//
// When done, to release the memory, call BufferFinalize on the returned buffer.
//
// It returns a handle to the buffer and a slice of the corresponding data type pointing
// to the shared data.
//
// Note: For Int64 shapes, the buffer internally stores Int32 (CoreML limitation).
// The returned flat slice will be []int64 (allocated and converted), but mutations to it
// will NOT be reflected in the buffer. Use BufferFromFlatData to update the buffer with
// []int64 data.
func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (buffer backends.Buffer, flat any, err error) {
	if b.isFinalized {
		return nil, nil, errors.Errorf("backend is already finalized")
	}
	if deviceNum != 0 {
		return nil, nil, errors.Errorf("backend (%s) only supports deviceNum 0, cannot create buffer on deviceNum %d (shape=%s)",
			b.Name(), deviceNum, shape)
	}
	goBuffer := b.NewBuffer(shape)

	// Handle Int64 shape with Int32 storage
	// We return a []int64 slice to match what the caller expects, but it's not truly shared
	// since we need to convert between Int32 and Int64
	if shape.DType == dtypes.Int64 {
		int32Data, ok := goBuffer.flat.([]int32)
		if !ok {
			return nil, nil, errors.Errorf("buffer has Int64 shape but internal storage is not []int32")
		}
		// Allocate and convert to []int64
		int64Data := make([]int64, len(int32Data))
		for i, v := range int32Data {
			int64Data[i] = int64(v)
		}
		return goBuffer, int64Data, nil
	}

	return goBuffer, goBuffer.flat, nil
}

// BufferData returns a slice pointing to the buffer storage memory directly.
//
// This only works if HasSharedBuffers is true, that is, if the backend engine runs on CPU, or
// shares CPU memory.
//
// The returned slice becomes invalid after the buffer is destroyed.
//
// Note: For Int64 shapes, the buffer internally stores Int32 (CoreML limitation).
// This function will allocate and return a converted []int64 slice to match the
// expected dtype. For truly zero-copy access to Int64 shapes, you would need to
// work with the underlying []int32 storage directly.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	if b.isFinalized {
		return nil, errors.Errorf("backend is already finalized")
	}
	buf, ok := buffer.(*Buffer)
	if !ok {
		return nil, errors.Errorf("buffer is not a %q backend buffer", BackendName)
	}

	// Handle Int64 shape with Int32 storage
	// We need to return []int64 to match what the caller expects based on shape.DType
	if buf.shape.DType == dtypes.Int64 {
		int32Data, ok := buf.flat.([]int32)
		if !ok {
			return nil, errors.Errorf("buffer has Int64 shape but internal storage is not []int32")
		}
		// Allocate and convert to []int64
		int64Data := make([]int64, len(int32Data))
		for i, v := range int32Data {
			int64Data[i] = int64(v)
		}
		return int64Data, nil
	}

	return buf.flat, nil
}

// BufferCopyToDevice implements the backends.Backend interface.
// CoreML backend only supports a single device, so this operation is not supported.
func (b *Backend) BufferCopyToDevice(source backends.Buffer, deviceNum backends.DeviceNum) (
	bufferOnDevice backends.Buffer, err error) {
	return nil, errors.Errorf("backend %q: multi-device not supported on this backend",
		BackendName)
}
