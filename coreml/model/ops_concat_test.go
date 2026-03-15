package model

import (
	"testing"

	"github.com/gomlx/go-darwinml/proto/coreml/milspec"
)

func TestConcat(t *testing.T) {
	b := NewBuilder("concat_test")

	// Create three tensors with different sizes along axis 1
	x := b.Input("x", Float32, 2, 3, 4)  // [2, 3, 4]
	y := b.Input("y", Float32, 2, 5, 4)  // [2, 5, 4]
	z := b.Input("z", Float32, 2, 2, 4)  // [2, 2, 4]

	// Concatenate along axis 1
	concat := b.Concat([]*Value{x, y, z}, 1)

	// Output should be [2, 10, 4] (3+5+2=10)
	b.Output("concat_out", concat)

	// Verify output shape
	expectedShape := []int64{2, 10, 4}
	if len(concat.shape) != len(expectedShape) {
		t.Fatalf("expected shape length %d, got %d", len(expectedShape), len(concat.shape))
	}
	for i, dim := range expectedShape {
		if concat.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, concat.shape[i])
		}
	}

	// Build and verify
	program := b.Build()
	mainFunc := program.Functions["concat_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Find concat operation
	var concatOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "concat" {
			concatOp = op
			break
		}
	}

	if concatOp == nil {
		t.Fatal("expected concat operation in program")
	}

	// Verify that values argument is a list with 3 bindings
	valuesArg := concatOp.Inputs["values"]
	if valuesArg == nil {
		t.Fatal("expected 'values' input in concat operation")
	}

	if len(valuesArg.Arguments) != 3 {
		t.Errorf("expected 3 values in concat, got %d", len(valuesArg.Arguments))
	}

	// Verify axis argument exists
	axisArg := concatOp.Inputs["axis"]
	if axisArg == nil {
		t.Fatal("expected 'axis' input in concat operation")
	}
}

func TestConcatNegativeAxis(t *testing.T) {
	b := NewBuilder("concat_neg_axis")

	// Create two tensors
	x := b.Input("x", Float32, 2, 3, 4)
	y := b.Input("y", Float32, 2, 3, 5)

	// Concatenate along last axis using negative index
	concat := b.Concat([]*Value{x, y}, -1)

	// Output should be [2, 3, 9] (4+5=9)
	expectedShape := []int64{2, 3, 9}
	for i, dim := range expectedShape {
		if concat.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, concat.shape[i])
		}
	}
}

func TestConcatAxis0(t *testing.T) {
	b := NewBuilder("concat_axis0")

	// Create two tensors with different batch sizes
	x := b.Input("x", Float32, 2, 3, 4)
	y := b.Input("y", Float32, 5, 3, 4)

	// Concatenate along axis 0 (batch dimension)
	concat := b.Concat([]*Value{x, y}, 0)

	// Output should be [7, 3, 4] (2+5=7)
	expectedShape := []int64{7, 3, 4}
	for i, dim := range expectedShape {
		if concat.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, concat.shape[i])
		}
	}
}

func TestConcatSingleTensor(t *testing.T) {
	b := NewBuilder("concat_single")

	// Concat with single tensor should work (identity-like operation)
	x := b.Input("x", Float32, 2, 3, 4)
	concat := b.Concat([]*Value{x}, 1)

	// Output shape should match input
	for i, dim := range x.shape {
		if concat.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, concat.shape[i])
		}
	}
}

func TestConcatWithConstants(t *testing.T) {
	b := NewBuilder("concat_const")

	// Mix variables and constants
	x := b.Input("x", Float32, 2, 3)
	constVal := b.Const("const", Float32, []int64{2, 2}, []float32{1, 2, 3, 4})

	// Concatenate along axis 1
	concat := b.Concat([]*Value{x, constVal}, 1)

	// Output should be [2, 5] (3+2=5)
	expectedShape := []int64{2, 5}
	for i, dim := range expectedShape {
		if concat.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, concat.shape[i])
		}
	}

	// Build and verify constant is embedded
	program := b.Build()
	mainFunc := program.Functions["concat_const"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	var concatOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "concat" {
			concatOp = op
			break
		}
	}

	if concatOp == nil {
		t.Fatal("expected concat operation")
	}

	// Check that one of the values is embedded
	valuesArg := concatOp.Inputs["values"]
	hasEmbeddedValue := false
	for _, binding := range valuesArg.Arguments {
		if binding.GetValue() != nil {
			hasEmbeddedValue = true
			break
		}
	}
	if !hasEmbeddedValue {
		t.Error("expected at least one embedded constant value")
	}
}
