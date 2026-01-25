//go:build darwin && cgo

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestGatherSimpleSingleAxis tests the simple single-axis gather pattern
// (embedding lookup style) where we gather along one axis with collapsed output.
// This is the most common pattern: params[3, 5], indices[2] -> output[2, 5]
func TestGatherSimpleSingleAxis(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_simple")
	mainFn := builder.Main()

	// Params shape: [3, 5] - like an embedding table with 3 embeddings of size 5
	// Indices shape: [2, 1] - 2 indices, each pointing to one embedding
	paramsShape := shapes.Make(dtypes.Float32, 3, 5)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Gather: this is the standard pattern used by GoMLX's Gather function
	// - indexVectorAxis: 1 (the last axis of indices contains the index)
	// - offsetOutputAxes: [1] (output axis 1 corresponds to the slice dimension)
	// - collapsedSliceAxes: [0] (collapse axis 0 of params, which we index into)
	// - startIndexMap: [0] (index into axis 0 of params)
	// - sliceSizes: [1, 5] (take 1 element from axis 0, all 5 from axis 1)
	result, err := mainFn.Gather(
		params, indices,
		1,          // indexVectorAxis
		[]int{1},   // offsetOutputAxes
		[]int{0},   // collapsedSliceAxes
		[]int{0},   // startIndexMap
		[]int{1, 5}, // sliceSizes
		false,      // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	// Params: 3 embeddings, each of size 5
	// Row 0: [0, 1, 2, 3, 4]
	// Row 1: [10, 11, 12, 13, 14]
	// Row 2: [20, 21, 22, 23, 24]
	paramsData := []float32{
		0, 1, 2, 3, 4,
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
	}
	// Indices: [1, 2] - get embeddings 1 and 2
	indicesData := []int32{1, 2}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected output shape: [2, 5]
	// Row 0: embedding 1 -> [10, 11, 12, 13, 14]
	// Row 1: embedding 2 -> [20, 21, 22, 23, 24]
	outputData := make([]float32, 10)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{10, 11, 12, 13, 14, 20, 21, 22, 23, 24}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("Simple gather output: %v", outputData)
}

// TestGatherScalarIndices tests gather with scalar indices (after expansion)
func TestGatherScalarIndices(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_scalar_indices")
	mainFn := builder.Main()

	// Params shape: [4, 3] - 4 rows of 3 elements each
	// Indices shape: [1] - a single index (scalar-like)
	paramsShape := shapes.Make(dtypes.Float32, 4, 3)
	indicesShape := shapes.Make(dtypes.Int32, 1, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Gather single element
	result, err := mainFn.Gather(
		params, indices,
		1,          // indexVectorAxis
		[]int{1},   // offsetOutputAxes
		[]int{0},   // collapsedSliceAxes
		[]int{0},   // startIndexMap
		[]int{1, 3}, // sliceSizes
		false,      // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	paramsData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	indicesData := []int32{2} // Get row 2

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected: [7, 8, 9]
	outputData := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{7, 8, 9}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("Scalar indices gather output: %v", outputData)
}

// TestGatherMultiDimIndices tests gather with multi-dimensional indices
// e.g., params[10, 8], indices[3, 4, 1] -> output[3, 4, 8]
func TestGatherMultiDimIndices(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_multi_dim_indices")
	mainFn := builder.Main()

	// Params shape: [5, 4] - 5 rows of 4 elements
	// Indices shape: [2, 3, 1] - 2x3 grid of indices
	paramsShape := shapes.Make(dtypes.Float32, 5, 4)
	indicesShape := shapes.Make(dtypes.Int32, 2, 3, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Gather with multi-dim indices
	result, err := mainFn.Gather(
		params, indices,
		2,           // indexVectorAxis (last axis)
		[]int{2},    // offsetOutputAxes
		[]int{0},    // collapsedSliceAxes
		[]int{0},    // startIndexMap
		[]int{1, 4}, // sliceSizes
		false,       // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	paramsData := []float32{
		0, 1, 2, 3,
		10, 11, 12, 13,
		20, 21, 22, 23,
		30, 31, 32, 33,
		40, 41, 42, 43,
	}
	// Indices: 2x3 grid with values selecting different rows
	indicesData := []int32{
		0, 1, 2, // First row of 2x3 grid
		3, 4, 0, // Second row of 2x3 grid
	}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected output shape: [2, 3, 4]
	// Each index maps to a row from params
	outputData := make([]float32, 2*3*4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected:
	// [0,0,:] -> row 0: [0, 1, 2, 3]
	// [0,1,:] -> row 1: [10, 11, 12, 13]
	// [0,2,:] -> row 2: [20, 21, 22, 23]
	// [1,0,:] -> row 3: [30, 31, 32, 33]
	// [1,1,:] -> row 4: [40, 41, 42, 43]
	// [1,2,:] -> row 0: [0, 1, 2, 3]
	expected := []float32{
		0, 1, 2, 3,
		10, 11, 12, 13,
		20, 21, 22, 23,
		30, 31, 32, 33,
		40, 41, 42, 43,
		0, 1, 2, 3,
	}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("Multi-dim indices gather succeeded, output shape: [2, 3, 4]")
}

// TestGatherGatherAlongNonZeroAxis tests gathering along a non-zero axis
// e.g., params[2, 5, 3], gather along axis 1 -> output[2, num_indices, 3]
func TestGatherAlongNonZeroAxis(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_nonzero_axis")
	mainFn := builder.Main()

	// Params shape: [2, 4, 3] - 2 batches, 4 items per batch, 3 features per item
	// Indices shape: [2, 1] - 2 indices to gather from axis 1
	paramsShape := shapes.Make(dtypes.Float32, 2, 4, 3)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Gather along axis 1 (middle axis)
	result, err := mainFn.Gather(
		params, indices,
		1,             // indexVectorAxis
		[]int{0, 2},   // offsetOutputAxes (axis 0 and axis 2 are kept)
		[]int{1},      // collapsedSliceAxes (collapse axis 1)
		[]int{1},      // startIndexMap (index into axis 1)
		[]int{2, 1, 3}, // sliceSizes (full on 0, 1 on 1, full on 2)
		false,         // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	// Batch 0: 4 items with 3 features each
	// Batch 1: 4 items with 3 features each
	paramsData := []float32{
		// Batch 0
		0, 1, 2, // item 0
		3, 4, 5, // item 1
		6, 7, 8, // item 2
		9, 10, 11, // item 3
		// Batch 1
		100, 101, 102, // item 0
		103, 104, 105, // item 1
		106, 107, 108, // item 2
		109, 110, 111, // item 3
	}
	// Indices: get items 1 and 3
	indicesData := []int32{1, 3}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected output shape: [2, 2, 3]
	// Gathered items 1 and 3 from each batch
	outputData := make([]float32, 2*2*3)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected:
	// From batch 0: item 1 [3,4,5], item 3 [9,10,11]
	// From batch 1: item 1 [103,104,105], item 3 [109,110,111]
	expected := []float32{
		3, 4, 5,
		9, 10, 11,
		103, 104, 105,
		109, 110, 111,
	}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("Non-zero axis gather succeeded")
}

// TestGatherNDPattern tests the multi-axis gather pattern that maps to GatherND
// This pattern has multiple collapsed axes indexing contiguously from axis 0
func TestGatherNDPattern(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_nd_pattern")
	mainFn := builder.Main()

	// Params shape: [3, 4, 5] - a 3D tensor
	// We want to index into the first 2 dimensions (axes 0 and 1)
	// Indices shape: [2, 2] - 2 indices, each with 2 coordinates
	paramsShape := shapes.Make(dtypes.Float32, 3, 4, 5)
	indicesShape := shapes.Make(dtypes.Int32, 2, 2)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// GatherND pattern: index into axes 0 and 1, collapse both
	// - startIndexMap: [0, 1] - indices point to axes 0 and 1
	// - collapsedSliceAxes: [0, 1] - collapse both indexed axes
	// - sliceSizes: [1, 1, 5] - take single elements from axes 0,1, full from axis 2
	result, err := mainFn.Gather(
		params, indices,
		1,            // indexVectorAxis (last axis of indices)
		[]int{1},     // offsetOutputAxes (output axis 1 for the slice dimension)
		[]int{0, 1},  // collapsedSliceAxes
		[]int{0, 1},  // startIndexMap
		[]int{1, 1, 5}, // sliceSizes
		false,        // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data: 3x4x5 tensor
	// We'll use a pattern where paramsData[i][j][k] = i*100 + j*10 + k
	paramsData := make([]float32, 3*4*5)
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				paramsData[i*4*5+j*5+k] = float32(i*100 + j*10 + k)
			}
		}
	}

	// Indices: select [1,2] and [2,3]
	// [1,2] selects paramsData[1][2][:] = [120, 121, 122, 123, 124]
	// [2,3] selects paramsData[2][3][:] = [230, 231, 232, 233, 234]
	indicesData := []int32{
		1, 2, // first index: [1, 2]
		2, 3, // second index: [2, 3]
	}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected output shape: [2, 5]
	// Row 0: paramsData[1][2][:] = [120, 121, 122, 123, 124]
	// Row 1: paramsData[2][3][:] = [230, 231, 232, 233, 234]
	outputData := make([]float32, 2*5)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{
		120, 121, 122, 123, 124,
		230, 231, 232, 233, 234,
	}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("GatherND pattern test succeeded, output: %v", outputData)
}

// TestGatherSlicesWithSliceSize1 tests the GatherSlices pattern with slice_size=1
// This extracts slices without collapsing the gathered dimension.
// params[3, 5], indices[2, 1] -> output[2, 1, 5]
func TestGatherSlicesWithSliceSize1(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_slices_size1")
	mainFn := builder.Main()

	// Params shape: [3, 5] - like an embedding table with 3 embeddings of size 5
	// Indices shape: [2, 1] - 2 indices, each pointing to one embedding (with indexVectorAxis=1)
	paramsShape := shapes.Make(dtypes.Float32, 3, 5)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// GatherSlices pattern: no collapsed axes, slice_size=1 on the gathered axis
	// This should produce output shape [2, 1, 5] (not [2, 5] like regular Gather)
	// - indexVectorAxis: 1 (last axis of indices)
	// - offsetOutputAxes: [1, 2] (all input axes map to output axes 1 and 2)
	// - collapsedSliceAxes: [] (no collapsing - GatherSlices pattern)
	// - startIndexMap: [0] (index into axis 0 of params)
	// - sliceSizes: [1, 5] (take 1 element from axis 0, all 5 from axis 1)
	result, err := mainFn.Gather(
		params, indices,
		1,          // indexVectorAxis
		[]int{1, 2}, // offsetOutputAxes
		[]int{},    // no collapsed axes - GatherSlices pattern
		[]int{0},   // startIndexMap
		[]int{1, 5}, // sliceSizes (slice_size=1 on gathered axis)
		false,      // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	// Params: 3 embeddings, each of size 5
	// Row 0: [0, 1, 2, 3, 4]
	// Row 1: [10, 11, 12, 13, 14]
	// Row 2: [20, 21, 22, 23, 24]
	paramsData := []float32{
		0, 1, 2, 3, 4,
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
	}
	// Indices: [1, 2] - get embeddings 1 and 2
	indicesData := []int32{1, 2}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected output shape: [2, 1, 5]
	// Row 0: [[10, 11, 12, 13, 14]] (embedding 1, with extra dimension)
	// Row 1: [[20, 21, 22, 23, 24]] (embedding 2, with extra dimension)
	outputData := make([]float32, 2*1*5)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := []float32{10, 11, 12, 13, 14, 20, 21, 22, 23, 24}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("GatherSlices (slice_size=1) output: %v", outputData)
}

// TestGatherSlicesNonZeroAxis tests GatherSlices on a non-zero axis with slice_size=1
// params[2, 4, 3], indices[2, 1] -> output[2, 2, 1, 3]
func TestGatherSlicesNonZeroAxis(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_slices_nonzero_axis")
	mainFn := builder.Main()

	// Params shape: [2, 4, 3] - 2 batches, 4 items per batch, 3 features per item
	// Indices shape: [2, 1] - 2 indices to gather from axis 1
	paramsShape := shapes.Make(dtypes.Float32, 2, 4, 3)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// GatherSlices along axis 1 (middle axis) with no collapse
	// Output shape: [2, 2, 1, 3]
	// - indexVectorAxis: 1 (last axis of indices)
	// - offsetOutputAxes: [1, 2, 3] (all input axes map to output axes 1, 2, 3)
	// - collapsedSliceAxes: [] (no collapsing)
	// - startIndexMap: [1] (index into axis 1)
	// - sliceSizes: [2, 1, 3] (full on axis 0, slice_size=1 on axis 1, full on axis 2)
	result, err := mainFn.Gather(
		params, indices,
		1,              // indexVectorAxis
		[]int{1, 2, 3}, // offsetOutputAxes
		[]int{},        // no collapsed axes
		[]int{1},       // startIndexMap (index into axis 1)
		[]int{2, 1, 3}, // sliceSizes
		false,          // indicesAreSorted
	)
	if err != nil {
		t.Fatalf("Gather() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data
	// Batch 0: 4 items with 3 features each
	// Batch 1: 4 items with 3 features each
	paramsData := []float32{
		// Batch 0
		0, 1, 2, // item 0
		3, 4, 5, // item 1
		6, 7, 8, // item 2
		9, 10, 11, // item 3
		// Batch 1
		100, 101, 102, // item 0
		103, 104, 105, // item 1
		106, 107, 108, // item 2
		109, 110, 111, // item 3
	}
	// Indices: get items 1 and 3
	indicesData := []int32{1, 3}

	paramsBuf, err := backend.BufferFromFlatData(0, paramsData, paramsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for params failed: %v", err)
	}
	defer backend.BufferFinalize(paramsBuf)

	indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for indices failed: %v", err)
	}
	defer backend.BufferFinalize(indicesBuf)

	outputs, err := exec.Execute([]backends.Buffer{paramsBuf, indicesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	// Expected output shape: [2, 2, 1, 3]
	// Gathered items 1 and 3 from each batch, keeping the size-1 dimension
	outputData := make([]float32, 2*2*1*3)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected (flattened):
	// Output shape is [2, 2, 1, 3] (batch_from_indices=2, batch_from_params=2, gathered_axis=1, features=3)
	// The order is: for each batch_from_params, for each index, the slice with size 1
	// batch 0: index 0 (item 1) [3,4,5], index 1 (item 3) [9,10,11]
	// batch 1: index 0 (item 1) [103,104,105], index 1 (item 3) [109,110,111]
	expected := []float32{
		3, 4, 5,
		9, 10, 11,
		103, 104, 105,
		109, 110, 111,
	}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-5 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
	t.Logf("GatherSlices on non-zero axis succeeded, output: %v", outputData)
}

// TestGatherSlicesWithSliceSizeGreaterThan1 tests that GatherSlices with slice_size > 1
// returns an appropriate error (not yet supported)
func TestGatherSlicesWithSliceSizeGreaterThan1(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_slices_size_gt_1")
	mainFn := builder.Main()

	paramsShape := shapes.Make(dtypes.Float32, 4, 3, 2)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Try a GatherSlices pattern with slice_size=2 (> 1) - should fail
	_, err = mainFn.Gather(
		params, indices,
		1,              // indexVectorAxis
		[]int{1, 2, 3}, // offsetOutputAxes
		[]int{},        // no collapsed axes - GatherSlices pattern
		[]int{0},       // startIndexMap
		[]int{2, 3, 2}, // sliceSizes (slice_size=2 on axis 0, which is > 1)
		false,          // indicesAreSorted
	)
	if err == nil {
		t.Fatalf("Expected error for GatherSlices with slice_size > 1, but got nil")
	}

	// Verify error message is helpful
	errMsg := err.Error()
	if !contains(errMsg, "slice_size > 1") && !contains(errMsg, "GatherSlices") {
		t.Errorf("Error message should mention slice_size > 1 or GatherSlices, got: %s", errMsg)
	}
	t.Logf("Got expected error for slice_size > 1: %v", err)
}

// TestGatherUnsupportedPatternError tests that unsupported gather patterns
// return appropriate error messages
func TestGatherUnsupportedPatternError(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("gather_unsupported")
	mainFn := builder.Main()

	paramsShape := shapes.Make(dtypes.Float32, 4, 3, 2)
	indicesShape := shapes.Make(dtypes.Int32, 2, 1)

	params, err := mainFn.Parameter("params", paramsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for params failed: %v", err)
	}

	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for indices failed: %v", err)
	}

	// Try a GatherSlices pattern (no collapsed axes) - should fail with helpful error
	_, err = mainFn.Gather(
		params, indices,
		1,             // indexVectorAxis
		[]int{1, 2, 3}, // offsetOutputAxes
		[]int{},       // no collapsed axes - GatherSlices pattern
		[]int{0},      // startIndexMap
		[]int{2, 3, 2}, // sliceSizes
		false,         // indicesAreSorted
	)
	if err == nil {
		t.Fatalf("Expected error for unsupported Gather pattern, but got nil")
	}

	// Verify error message is helpful
	errMsg := err.Error()
	if !contains(errMsg, "GatherSlices") && !contains(errMsg, "not yet supported") {
		t.Errorf("Error message should mention GatherSlices or unsupported, got: %s", errMsg)
	}
	t.Logf("Got expected error for unsupported pattern: %v", err)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
