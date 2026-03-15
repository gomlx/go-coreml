// Package bridge provides low-level cgo bindings to CoreML.
package bridge

import (
	"testing"
	"unsafe"
)

func TestTensorCreate(t *testing.T) {
	shape := []int64{2, 3}
	tensor, err := NewTensor(shape, DTypeFloat32)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	if tensor.Rank() != 2 {
		t.Errorf("expected rank 2, got %d", tensor.Rank())
	}
	if tensor.Dim(0) != 2 {
		t.Errorf("expected dim 0 = 2, got %d", tensor.Dim(0))
	}
	if tensor.Dim(1) != 3 {
		t.Errorf("expected dim 1 = 3, got %d", tensor.Dim(1))
	}
	if tensor.DType() != DTypeFloat32 {
		t.Errorf("expected dtype Float32, got %d", tensor.DType())
	}
	if tensor.SizeBytes() != 2*3*4 {
		t.Errorf("expected size 24 bytes, got %d", tensor.SizeBytes())
	}
}

func TestTensorCreateWithData(t *testing.T) {
	shape := []int64{2, 2}
	data := []float32{1.0, 2.0, 3.0, 4.0}

	tensor, err := NewTensorWithData(shape, DTypeFloat32, unsafe.Pointer(&data[0]))
	if err != nil {
		t.Fatalf("NewTensorWithData failed: %v", err)
	}
	defer tensor.Close()

	// Read data back
	ptr := tensor.DataPtr()
	result := (*[4]float32)(ptr)[:]

	for i, expected := range data {
		if result[i] != expected {
			t.Errorf("data[%d] = %f, expected %f", i, result[i], expected)
		}
	}
}

func TestTensorShape(t *testing.T) {
	shape := []int64{3, 4, 5}
	tensor, err := NewTensor(shape, DTypeFloat32)
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	got := tensor.Shape()
	if len(got) != len(shape) {
		t.Fatalf("expected shape length %d, got %d", len(shape), len(got))
	}
	for i, v := range shape {
		if got[i] != v {
			t.Errorf("shape[%d] = %d, expected %d", i, got[i], v)
		}
	}
}

func TestComputeUnits(t *testing.T) {
	// Just verify these don't panic
	SetComputeUnits(ComputeAll)
	SetComputeUnits(ComputeCPUOnly)
	SetComputeUnits(ComputeCPUAndGPU)
	SetComputeUnits(ComputeCPUAndANE)
	SetComputeUnits(ComputeAll) // Reset
}

func TestDTypeConstants(t *testing.T) {
	// Verify dtype constants are correctly defined
	tests := []struct {
		dtype DType
		name  string
	}{
		{DTypeFloat32, "float32"},
		{DTypeFloat16, "float16"},
		{DTypeInt32, "int32"},
	}

	for _, tt := range tests {
		shape := []int64{1}
		tensor, err := NewTensor(shape, tt.dtype)
		if err != nil {
			t.Fatalf("NewTensor(%s) failed: %v", tt.name, err)
		}
		if tensor.DType() != tt.dtype {
			t.Errorf("expected dtype %s (%d), got %d", tt.name, tt.dtype, tensor.DType())
		}
		tensor.Close()
	}
}
