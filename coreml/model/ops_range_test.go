package model

import (
	"testing"

	"github.com/gomlx/go-darwinml/proto/coreml/milspec"
)

func TestRange1D(t *testing.T) {
	b := NewBuilder("range_1d_test")

	// Create scalar constants for start, end, step
	start := b.Const("start", Int32, []int64{}, []int32{0})
	end := b.Const("end", Int32, []int64{}, []int32{10})
	step := b.Const("step", Int32, []int64{}, []int32{1})

	// Generate range [0, 1, 2, ..., 9]
	rangeVal := b.Range1D(start, end, step)

	// Output the range
	b.Output("range_out", rangeVal)

	// Verify output is 1D with dynamic shape
	if len(rangeVal.shape) != 1 {
		t.Fatalf("expected 1D shape, got %d dimensions", len(rangeVal.shape))
	}

	// Build and verify
	program := b.Build()
	mainFunc := program.Functions["range_1d_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Find range_1d operation
	var rangeOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "range_1d" {
			rangeOp = op
			break
		}
	}

	if rangeOp == nil {
		t.Fatal("expected range_1d operation in program")
	}

	// Verify inputs
	if rangeOp.Inputs["start"] == nil {
		t.Error("expected 'start' input in range_1d operation")
	}
	if rangeOp.Inputs["end"] == nil {
		t.Error("expected 'end' input in range_1d operation")
	}
	if rangeOp.Inputs["step"] == nil {
		t.Error("expected 'step' input in range_1d operation")
	}
}

func TestRange1DFloat(t *testing.T) {
	b := NewBuilder("range_1d_float")

	// Create scalar constants with float values
	start := b.Const("start", Float32, []int64{}, []float32{0.0})
	end := b.Const("end", Float32, []int64{}, []float32{5.0})
	step := b.Const("step", Float32, []int64{}, []float32{0.5})

	// Generate range [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
	rangeVal := b.Range1D(start, end, step)

	// Verify dtype matches start
	if rangeVal.dtype != Float32 {
		t.Errorf("expected Float32 dtype, got %v", rangeVal.dtype)
	}

	// Output the range
	b.Output("range_out", rangeVal)

	// Build
	_ = b.Build()
}

func TestRange1DNegativeStep(t *testing.T) {
	b := NewBuilder("range_1d_negative")

	// Create scalar constants for counting down
	start := b.Const("start", Int32, []int64{}, []int32{10})
	end := b.Const("end", Int32, []int64{}, []int32{0})
	step := b.Const("step", Int32, []int64{}, []int32{-1})

	// Generate range [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
	rangeVal := b.Range1D(start, end, step)

	// Output the range
	b.Output("range_out", rangeVal)

	// Verify it builds
	_ = b.Build()
}
