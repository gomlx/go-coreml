// Package bridge provides low-level cgo bindings to CoreML.
//
// This package wraps the Objective-C++ CoreML bridge and exposes
// Go-friendly APIs for model loading, tensor operations, and inference.
package bridge

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework CoreML
#include "bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// DType represents a CoreML data type.
type DType int

const (
	DTypeFloat32 DType = C.COREML_DTYPE_FLOAT32
	DTypeFloat16 DType = C.COREML_DTYPE_FLOAT16
	DTypeInt32   DType = C.COREML_DTYPE_INT32
	DTypeInt64   DType = C.COREML_DTYPE_INT64
	DTypeBool    DType = C.COREML_DTYPE_BOOL
)

// ComputeUnits specifies which compute units to use.
type ComputeUnits int

const (
	ComputeAll       ComputeUnits = C.COREML_COMPUTE_ALL
	ComputeCPUOnly   ComputeUnits = C.COREML_COMPUTE_CPU_ONLY
	ComputeCPUAndGPU ComputeUnits = C.COREML_COMPUTE_CPU_AND_GPU
	ComputeCPUAndANE ComputeUnits = C.COREML_COMPUTE_CPU_AND_ANE
)

// SetComputeUnits sets the global compute units for model loading.
func SetComputeUnits(units ComputeUnits) {
	C.coreml_set_compute_units(C.CoreMLComputeUnits(units))
}

// Model represents a loaded CoreML model.
type Model struct {
	handle C.CoreMLModel
}

// LoadModel loads a CoreML model from a .mlmodelc directory.
func LoadModel(path string) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var err C.CoreMLError
	handle := C.coreml_load_model(cPath, &err)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to load model: %s", msg)
	}

	return &Model{handle: handle}, nil
}

// Close releases the model resources.
func (m *Model) Close() {
	if m.handle != nil {
		C.coreml_free_model(m.handle)
		m.handle = nil
	}
}

// InputCount returns the number of inputs.
func (m *Model) InputCount() int {
	return int(C.coreml_model_input_count(m.handle))
}

// OutputCount returns the number of outputs.
func (m *Model) OutputCount() int {
	return int(C.coreml_model_output_count(m.handle))
}

// InputName returns the name of the input at the given index.
func (m *Model) InputName(index int) string {
	cName := C.coreml_model_input_name(m.handle, C.int(index))
	if cName == nil {
		return ""
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name
}

// OutputName returns the name of the output at the given index.
func (m *Model) OutputName(index int) string {
	cName := C.coreml_model_output_name(m.handle, C.int(index))
	if cName == nil {
		return ""
	}
	name := C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name
}

// Tensor represents a multi-dimensional array for CoreML.
type Tensor struct {
	handle C.CoreMLTensor
}

// NewTensor creates a new tensor with the given shape and data type.
func NewTensor(shape []int64, dtype DType) (*Tensor, error) {
	var err C.CoreMLError
	handle := C.coreml_tensor_create(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		&err,
	)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to create tensor: %s", msg)
	}
	return &Tensor{handle: handle}, nil
}

// NewTensorWithData creates a tensor and copies data into it.
func NewTensorWithData(shape []int64, dtype DType, data unsafe.Pointer) (*Tensor, error) {
	var err C.CoreMLError
	handle := C.coreml_tensor_create_with_data(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		data,
		&err,
	)
	if handle == nil {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return nil, fmt.Errorf("failed to create tensor with data: %s", msg)
	}
	return &Tensor{handle: handle}, nil
}

// Close releases the tensor resources.
func (t *Tensor) Close() {
	if t.handle != nil {
		C.coreml_tensor_free(t.handle)
		t.handle = nil
	}
}

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int {
	return int(C.coreml_tensor_rank(t.handle))
}

// Dim returns the size of the given dimension.
func (t *Tensor) Dim(axis int) int64 {
	return int64(C.coreml_tensor_dim(t.handle, C.int(axis)))
}

// Shape returns the shape as a slice.
func (t *Tensor) Shape() []int64 {
	rank := t.Rank()
	shape := make([]int64, rank)
	for i := 0; i < rank; i++ {
		shape[i] = t.Dim(i)
	}
	return shape
}

// DType returns the data type.
func (t *Tensor) DType() DType {
	return DType(C.coreml_tensor_dtype(t.handle))
}

// DataPtr returns an unsafe pointer to the underlying data.
func (t *Tensor) DataPtr() unsafe.Pointer {
	return C.coreml_tensor_data(t.handle)
}

// SizeBytes returns the total size in bytes.
func (t *Tensor) SizeBytes() int64 {
	return int64(C.coreml_tensor_size_bytes(t.handle))
}

// Predict runs inference with the given inputs and outputs.
func (m *Model) Predict(inputNames []string, inputs []*Tensor, outputNames []string, outputs []*Tensor) error {
	if len(inputNames) != len(inputs) {
		return fmt.Errorf("input names count (%d) != inputs count (%d)", len(inputNames), len(inputs))
	}
	if len(outputNames) != len(outputs) {
		return fmt.Errorf("output names count (%d) != outputs count (%d)", len(outputNames), len(outputs))
	}

	// Convert input names
	cInputNames := make([]*C.char, len(inputNames))
	for i, name := range inputNames {
		cInputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cInputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	// Convert output names
	cOutputNames := make([]*C.char, len(outputNames))
	for i, name := range outputNames {
		cOutputNames[i] = C.CString(name)
	}
	defer func() {
		for _, name := range cOutputNames {
			C.free(unsafe.Pointer(name))
		}
	}()

	// Convert tensors
	cInputs := make([]C.CoreMLTensor, len(inputs))
	for i, t := range inputs {
		cInputs[i] = t.handle
	}

	cOutputs := make([]C.CoreMLTensor, len(outputs))
	for i, t := range outputs {
		cOutputs[i] = t.handle
	}

	var err C.CoreMLError
	ok := C.coreml_model_predict(
		m.handle,
		(**C.char)(unsafe.Pointer(&cInputNames[0])),
		(*C.CoreMLTensor)(unsafe.Pointer(&cInputs[0])),
		C.int(len(inputs)),
		(**C.char)(unsafe.Pointer(&cOutputNames[0])),
		(*C.CoreMLTensor)(unsafe.Pointer(&cOutputs[0])),
		C.int(len(outputs)),
		&err,
	)

	if !ok {
		msg := "unknown error"
		if err.message != nil {
			msg = C.GoString(err.message)
			C.free(unsafe.Pointer(err.message))
		}
		return fmt.Errorf("prediction failed: %s", msg)
	}

	return nil
}
