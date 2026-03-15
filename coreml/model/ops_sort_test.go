package model

import (
	"testing"
)

// TestArgsort tests the Argsort operation.
func TestArgsort(t *testing.T) {
	b := NewBuilder("argsort_test")

	// Create input tensor
	x := b.Input("x", Float32, 5)

	// Argsort ascending
	indicesAsc := b.Argsort(x, 0, false)

	// Argsort descending
	indicesDesc := b.Argsort(x, 0, true)

	// Output both
	b.Output("indices_asc", indicesAsc)
	b.Output("indices_desc", indicesDesc)

	program := b.Build()

	// Verify program structure
	mainFunc, ok := program.Functions["argsort_test"]
	if !ok {
		t.Fatal("expected 'argsort_test' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Verify argsort operations are present
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["argsort"] < 2 {
		t.Errorf("expected at least 2 argsort operations, got %d", opTypeCount["argsort"])
	}

	// Verify output shapes
	if indicesAsc.DType() != Int32 {
		t.Errorf("expected argsort output dtype Int32, got %v", indicesAsc.DType())
	}
	if len(indicesAsc.Shape()) != 1 || indicesAsc.Shape()[0] != 5 {
		t.Errorf("expected argsort output shape [5], got %v", indicesAsc.Shape())
	}
}

// TestArgsort2D tests Argsort on a 2D tensor along different axes.
func TestArgsort2D(t *testing.T) {
	b := NewBuilder("argsort_2d_test")

	// Create 2D input tensor [3, 4]
	x := b.Input("x", Float32, 3, 4)

	// Argsort along axis 0 (sort each column)
	indicesAxis0 := b.Argsort(x, 0, false)

	// Argsort along axis 1 (sort each row)
	indicesAxis1 := b.Argsort(x, 1, false)

	// Argsort along axis -1 (last axis, same as axis 1)
	indicesAxisNeg1 := b.Argsort(x, -1, true)

	b.Output("indices_axis0", indicesAxis0)
	b.Output("indices_axis1", indicesAxis1)
	b.Output("indices_axis_neg1", indicesAxisNeg1)

	program := b.Build()

	// Verify shapes
	if len(indicesAxis0.Shape()) != 2 || indicesAxis0.Shape()[0] != 3 || indicesAxis0.Shape()[1] != 4 {
		t.Errorf("expected argsort along axis 0 output shape [3, 4], got %v", indicesAxis0.Shape())
	}
	if len(indicesAxis1.Shape()) != 2 || indicesAxis1.Shape()[0] != 3 || indicesAxis1.Shape()[1] != 4 {
		t.Errorf("expected argsort along axis 1 output shape [3, 4], got %v", indicesAxis1.Shape())
	}

	// Verify program compiles
	mainFunc, ok := program.Functions["argsort_2d_test"]
	if !ok {
		t.Fatal("expected 'argsort_2d_test' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Verify argsort operations are present
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["argsort"] < 3 {
		t.Errorf("expected at least 3 argsort operations, got %d", opTypeCount["argsort"])
	}
}

// TestTopK tests the TopK operation.
func TestTopK(t *testing.T) {
	b := NewBuilder("topk_test")

	// Create input tensor
	x := b.Input("x", Float32, 10)

	// TopK: get top 3 largest values
	values, indices := b.TopK(x, 3, 0, false)

	b.Output("values", values)
	_ = indices // indices is a placeholder for now

	program := b.Build()

	// Verify shapes
	if len(values.Shape()) != 1 || values.Shape()[0] != 3 {
		t.Errorf("expected topk values shape [3], got %v", values.Shape())
	}
	if values.DType() != Float32 {
		t.Errorf("expected topk values dtype Float32, got %v", values.DType())
	}

	// Verify program compiles
	mainFunc, ok := program.Functions["topk_test"]
	if !ok {
		t.Fatal("expected 'topk_test' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Verify topk operation is present
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["topk"] < 1 {
		t.Errorf("expected at least 1 topk operation, got %d", opTypeCount["topk"])
	}
}

// TestTopK2D tests TopK on a 2D tensor.
func TestTopK2D(t *testing.T) {
	b := NewBuilder("topk_2d_test")

	// Create 2D input tensor [5, 10]
	x := b.Input("x", Float32, 5, 10)

	// TopK along axis 1: get top 3 largest values from each row
	values, _ := b.TopK(x, 3, 1, false)

	// TopK along axis 0: get top 2 smallest (ascending) values from each column
	valuesAsc, _ := b.TopK(x, 2, 0, true)

	b.Output("values_axis1", values)
	b.Output("values_axis0_asc", valuesAsc)

	program := b.Build()

	// Verify shapes
	if len(values.Shape()) != 2 || values.Shape()[0] != 5 || values.Shape()[1] != 3 {
		t.Errorf("expected topk along axis 1 output shape [5, 3], got %v", values.Shape())
	}
	if len(valuesAsc.Shape()) != 2 || valuesAsc.Shape()[0] != 2 || valuesAsc.Shape()[1] != 10 {
		t.Errorf("expected topk along axis 0 output shape [2, 10], got %v", valuesAsc.Shape())
	}

	// Verify program compiles
	mainFunc, ok := program.Functions["topk_2d_test"]
	if !ok {
		t.Fatal("expected 'topk_2d_test' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Verify topk operations are present
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["topk"] < 2 {
		t.Errorf("expected at least 2 topk operations, got %d", opTypeCount["topk"])
	}
}

// TestArgsortGatherSort tests using Argsort + GatherAlongAxis to sort a tensor.
// This is the recommended workaround for sort on CoreML.
func TestArgsortGatherSort(t *testing.T) {
	b := NewBuilder("argsort_gather_sort_test")

	// Create input tensor
	x := b.Input("x", Float32, 5)

	// Get sorted indices
	indices := b.Argsort(x, 0, false)

	// Use GatherAlongAxis to reorder the tensor according to sorted indices
	sorted := b.GatherAlongAxis(x, indices, 0)

	b.Output("sorted", sorted)

	program := b.Build()

	// Verify output shape matches input shape
	if len(sorted.Shape()) != 1 || sorted.Shape()[0] != 5 {
		t.Errorf("expected sorted output shape [5], got %v", sorted.Shape())
	}
	if sorted.DType() != Float32 {
		t.Errorf("expected sorted output dtype Float32, got %v", sorted.DType())
	}

	// Verify program compiles
	mainFunc, ok := program.Functions["argsort_gather_sort_test"]
	if !ok {
		t.Fatal("expected 'argsort_gather_sort_test' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Verify both argsort and gather_along_axis operations are present
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["argsort"] < 1 {
		t.Errorf("expected at least 1 argsort operation, got %d", opTypeCount["argsort"])
	}
	if opTypeCount["gather_along_axis"] < 1 {
		t.Errorf("expected at least 1 gather_along_axis operation, got %d", opTypeCount["gather_along_axis"])
	}
}
