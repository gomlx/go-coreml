package bridge

import (
	"testing"
	"unsafe"
)

func TestScalarTensor(t *testing.T) {
	// Test creating a scalar tensor (rank 0)
	shape := []int64{}
	tensor, err := NewTensor(shape, DTypeFloat32)
	if err != nil {
		t.Fatalf("Error creating scalar tensor: %v", err)
	}
	defer tensor.Close()

	t.Logf("Scalar tensor created successfully!")
	t.Logf("Rank: %d", tensor.Rank())
	t.Logf("Shape: %v", tensor.Shape())
	t.Logf("Size bytes: %d", tensor.SizeBytes())

	if tensor.Rank() != 0 {
		t.Errorf("Expected rank 0, got %d", tensor.Rank())
	}
	if tensor.SizeBytes() != 4 {
		t.Errorf("Expected 4 bytes for float32 scalar, got %d", tensor.SizeBytes())
	}
}

func TestScalarTensorWithData(t *testing.T) {
	// Test creating a scalar tensor with data
	shape := []int64{}
	data := []float32{42.5}

	tensor, err := NewTensorWithData(shape, DTypeFloat32, unsafe.Pointer(&data[0]))
	if err != nil {
		t.Fatalf("Error creating scalar tensor with data: %v", err)
	}
	defer tensor.Close()

	// Read back the data
	ptr := tensor.DataPtr()
	result := *(*float32)(ptr)

	if result != 42.5 {
		t.Errorf("Expected 42.5, got %f", result)
	}
	t.Logf("Scalar value: %f", result)
}
