//go:build darwin && cgo

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package coreml

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestIdentity tests the Identity operation.
func TestIdentity(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_identity")
	mainFn := builder.Main()

	// Create input parameter
	shape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Apply Identity operation
	result, err := mainFn.Identity(x)
	if err != nil {
		t.Fatalf("Identity() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	inputBuffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("Identity: outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}
}

// TestClamp tests the Clamp operation.
func TestClamp(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_clamp")
	mainFn := builder.Main()

	// Create input parameter
	shape := shapes.Make(dtypes.Float32, 6)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Create min and max constants
	minVal, err := mainFn.Constant([]float32{2.0})
	if err != nil {
		t.Fatalf("Constant(min) failed: %v", err)
	}
	maxVal, err := mainFn.Constant([]float32{4.0})
	if err != nil {
		t.Fatalf("Constant(max) failed: %v", err)
	}

	// Apply Clamp operation
	result, err := mainFn.Clamp(minVal, x, maxVal)
	if err != nil {
		t.Fatalf("Clamp() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: values from 0 to 5
	inputData := []float32{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
	inputBuffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output: values clamped to [2.0, 4.0]
	expected := []float32{2.0, 2.0, 2.0, 3.0, 4.0, 4.0}
	outputData := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("Clamp: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestLogicalAnd tests the LogicalAnd operation.
func TestLogicalAnd(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_logical_and")
	mainFn := builder.Main()

	// CoreML doesn't support Bool as input/output type, so we use Float32 and convert
	shape := shapes.Make(dtypes.Float32, 4)
	lhsFloat, err := mainFn.Parameter("lhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(lhs) failed: %v", err)
	}
	rhsFloat, err := mainFn.Parameter("rhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(rhs) failed: %v", err)
	}

	// Convert Float32 inputs to Bool using comparison with 0.5
	zeroHalf, _ := mainFn.Constant([]float32{0.5}, 1)
	lhs, _ := mainFn.GreaterThan(lhsFloat, zeroHalf)
	rhs, _ := mainFn.GreaterThan(rhsFloat, zeroHalf)

	// Apply LogicalAnd operation
	andResult, err := mainFn.LogicalAnd(lhs, rhs)
	if err != nil {
		t.Fatalf("LogicalAnd() failed: %v", err)
	}

	// Convert Bool result back to Float32
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)
	result, _ := mainFn.Where(andResult, trueConst, falseConst)

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: truth table for AND (using 0.0 for false, 1.0 for true)
	lhsData := []float32{0.0, 0.0, 1.0, 1.0}
	rhsData := []float32{0.0, 1.0, 0.0, 1.0}
	expected := []float32{0.0, 0.0, 0.0, 1.0}

	lhsBuffer, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(lhs) failed: %v", err)
	}
	defer backend.BufferFinalize(lhsBuffer)

	rhsBuffer, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(rhs) failed: %v", err)
	}
	defer backend.BufferFinalize(rhsBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{lhsBuffer, rhsBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("LogicalAnd: outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}

// TestLogicalOr tests the LogicalOr operation.
func TestLogicalOr(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_logical_or")
	mainFn := builder.Main()

	// CoreML doesn't support Bool as input/output type, so we use Float32 and convert
	shape := shapes.Make(dtypes.Float32, 4)
	lhsFloat, err := mainFn.Parameter("lhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(lhs) failed: %v", err)
	}
	rhsFloat, err := mainFn.Parameter("rhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(rhs) failed: %v", err)
	}

	// Convert Float32 inputs to Bool using comparison with 0.5
	zeroHalf, _ := mainFn.Constant([]float32{0.5}, 1)
	lhs, _ := mainFn.GreaterThan(lhsFloat, zeroHalf)
	rhs, _ := mainFn.GreaterThan(rhsFloat, zeroHalf)

	// Apply LogicalOr operation
	orResult, err := mainFn.LogicalOr(lhs, rhs)
	if err != nil {
		t.Fatalf("LogicalOr() failed: %v", err)
	}

	// Convert Bool result back to Float32
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)
	result, _ := mainFn.Where(orResult, trueConst, falseConst)

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: truth table for OR (using 0.0 for false, 1.0 for true)
	lhsData := []float32{0.0, 0.0, 1.0, 1.0}
	rhsData := []float32{0.0, 1.0, 0.0, 1.0}
	expected := []float32{0.0, 1.0, 1.0, 1.0}

	lhsBuffer, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(lhs) failed: %v", err)
	}
	defer backend.BufferFinalize(lhsBuffer)

	rhsBuffer, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(rhs) failed: %v", err)
	}
	defer backend.BufferFinalize(rhsBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{lhsBuffer, rhsBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("LogicalOr: outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}

// TestLogicalNot tests the LogicalNot operation.
func TestLogicalNot(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_logical_not")
	mainFn := builder.Main()

	// CoreML doesn't support Bool as input/output type, so we use Float32 and convert
	shape := shapes.Make(dtypes.Float32, 2)
	xFloat, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Convert Float32 input to Bool using comparison with 0.5
	zeroHalf, _ := mainFn.Constant([]float32{0.5}, 1)
	x, _ := mainFn.GreaterThan(xFloat, zeroHalf)

	// Apply LogicalNot operation
	notResult, err := mainFn.LogicalNot(x)
	if err != nil {
		t.Fatalf("LogicalNot() failed: %v", err)
	}

	// Convert Bool result back to Float32
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0}, 2)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0}, 2)
	result, _ := mainFn.Where(notResult, trueConst, falseConst)

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input (using 0.0 for false, 1.0 for true)
	inputData := []float32{0.0, 1.0}
	expected := []float32{1.0, 0.0}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 2)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("LogicalNot: outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}

// TestIsNaN tests the IsNaN operation.
func TestIsNaN(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_isnan")
	mainFn := builder.Main()

	// Create input parameter
	shape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Apply IsNaN operation
	isNaNResult, err := mainFn.IsNaN(x)
	if err != nil {
		t.Fatalf("IsNaN() failed: %v", err)
	}

	// Convert Bool to Float32 using Where (CoreML doesn't support Bool output)
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)
	result, err := mainFn.Where(isNaNResult, trueConst, falseConst)
	if err != nil {
		t.Fatalf("Where() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: some NaN and non-NaN values
	nan := float32(math.NaN())
	inputData := []float32{1.0, nan, 3.0, nan}
	expected := []float32{0.0, 1.0, 0.0, 1.0} // 1.0 = true, 0.0 = false

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("IsNaN: outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}

// TestIsFinite tests the IsFinite operation.
func TestIsFinite(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_isfinite")
	mainFn := builder.Main()

	// Create input parameter
	shape := shapes.Make(dtypes.Float32, 5)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Apply IsFinite operation
	isFiniteResult, err := mainFn.IsFinite(x)
	if err != nil {
		t.Fatalf("IsFinite() failed: %v", err)
	}

	// Convert Bool to Float32 using Where (CoreML doesn't support Bool output)
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0, 1.0}, 5)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0, 0.0}, 5)
	result, err := mainFn.Where(isFiniteResult, trueConst, falseConst)
	if err != nil {
		t.Fatalf("Where() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: mix of finite, NaN, and Inf values
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	inputData := []float32{1.0, nan, inf, negInf, 3.14}
	expected := []float32{1.0, 0.0, 0.0, 0.0, 1.0} // 1.0 = true, 0.0 = false

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 5)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("IsFinite: outputData[%d] = %v, want %v", i, outputData[i], expected[i])
		}
	}
}

// TestRem tests the Rem (modulo) operation.
func TestRem(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_rem")
	mainFn := builder.Main()

	// Create input parameters
	shape := shapes.Make(dtypes.Float32, 5)
	lhs, err := mainFn.Parameter("lhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(lhs) failed: %v", err)
	}
	rhs, err := mainFn.Parameter("rhs", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(rhs) failed: %v", err)
	}

	// Apply Rem operation
	result, err := mainFn.Rem(lhs, rhs)
	if err != nil {
		t.Fatalf("Rem() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input
	lhsData := []float32{7.0, 10.0, 5.0, 8.5, 15.0}
	rhsData := []float32{3.0, 3.0, 2.0, 3.0, 4.0}
	expected := []float32{1.0, 1.0, 1.0, 2.5, 3.0}

	lhsBuffer, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(lhs) failed: %v", err)
	}
	defer backend.BufferFinalize(lhsBuffer)

	rhsBuffer, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(rhs) failed: %v", err)
	}
	defer backend.BufferFinalize(rhsBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{lhsBuffer, rhsBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	outputData := make([]float32, 5)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		diff := outputData[i] - expected[i]
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("Rem: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestBroadcastInDim tests the BroadcastInDim operation.
func TestBroadcastInDim(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_broadcast")
	mainFn := builder.Main()

	// Create input parameter: [2] vector
	inputShape := shapes.Make(dtypes.Float32, 2)
	x, err := mainFn.Parameter("x", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %v", err)
	}

	// Broadcast to [2, 3] shape: [[a, a, a], [b, b, b]]
	outputShape := shapes.Make(dtypes.Float32, 2, 3)
	result, err := mainFn.BroadcastInDim(x, outputShape, []int{0})
	if err != nil {
		t.Fatalf("BroadcastInDim() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input: [1.0, 2.0]
	inputData := []float32{1.0, 2.0}
	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}
	defer backend.BufferFinalize(inputBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output: [[1, 1, 1], [2, 2, 2]]
	expected := []float32{1.0, 1.0, 1.0, 2.0, 2.0, 2.0}
	outputData := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("BroadcastInDim: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestDynamicSlice tests the DynamicSlice operation.
func TestDynamicSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_slice")
	mainFn := builder.Main()

	// Create input parameter: [6] array
	dataShape := shapes.Make(dtypes.Float32, 6)
	data, err := mainFn.Parameter("data", dataShape, nil)
	if err != nil {
		t.Fatalf("Parameter(data) failed: %v", err)
	}

	// Create start index as a constant (CoreML doesn't support scalar inputs well)
	// Start at index 2: should extract [30, 40, 50]
	startIdx, err := mainFn.Constant([]int32{2})
	if err != nil {
		t.Fatalf("Constant(start) failed: %v", err)
	}

	// Apply DynamicSlice: extract 3 elements starting at startIdx
	result, err := mainFn.DynamicSlice(data, []backends.Value{startIdx}, []int{3})
	if err != nil {
		t.Fatalf("DynamicSlice() failed: %v", err)
	}

	// Return the result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Prepare input data
	inputData := []float32{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}
	dataBuffer, err := backend.BufferFromFlatData(0, inputData, dataShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(data) failed: %v", err)
	}
	defer backend.BufferFinalize(dataBuffer)

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{dataBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Verify output
	expected := []float32{30.0, 40.0, 50.0}
	outputData := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("DynamicSlice: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestLogicalXor tests the LogicalXor operation.
func TestLogicalXor(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_logical_xor")
	mainFn := builder.Main()

	// Create input parameters as Float32 (CoreML doesn't support Bool I/O)
	shape := shapes.Make(dtypes.Float32, 4)
	a, err := mainFn.Parameter("a", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(a) failed: %v", err)
	}
	b, err := mainFn.Parameter("b", shape, nil)
	if err != nil {
		t.Fatalf("Parameter(b) failed: %v", err)
	}

	// Convert to bool using > 0.5
	threshold, _ := mainFn.Constant([]float32{0.5, 0.5, 0.5, 0.5}, 4)
	aBool, _ := mainFn.GreaterThan(a, threshold)
	bBool, _ := mainFn.GreaterThan(b, threshold)

	// Apply LogicalXor
	xorResult, err := mainFn.LogicalXor(aBool, bBool)
	if err != nil {
		t.Fatalf("LogicalXor() failed: %v", err)
	}

	// Convert bool result to float32 using Where
	trueConst, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	falseConst, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0, 0.0}, 4)
	result, _ := mainFn.Where(xorResult, trueConst, falseConst)

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test: [true, true, false, false] XOR [true, false, true, false] = [false, true, true, false]
	aData := []float32{1.0, 1.0, 0.0, 0.0}
	bData := []float32{1.0, 0.0, 1.0, 0.0}
	aBuffer, _ := backend.BufferFromFlatData(0, aData, shape)
	defer backend.BufferFinalize(aBuffer)
	bBuffer, _ := backend.BufferFromFlatData(0, bData, shape)
	defer backend.BufferFinalize(bBuffer)

	outputs, err := exec.Execute([]backends.Buffer{aBuffer, bBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	expected := []float32{0.0, 1.0, 1.0, 0.0}
	outputData := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if outputData[i] != expected[i] {
			t.Errorf("LogicalXor: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

// TestBatchNormForInference tests the BatchNormForInference operation.
func TestBatchNormForInference(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_batchnorm")
	mainFn := builder.Main()

	// BatchNorm on a [2, 3, 2] tensor with feature axis 1 (3 features)
	// CoreML batch_norm requires rank 3-5 tensors
	operandShape := shapes.Make(dtypes.Float32, 2, 3, 2)
	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter(operand) failed: %v", err)
	}

	// Scale, offset, mean, variance each have shape [3] (per feature)
	paramShape := shapes.Make(dtypes.Float32, 3)
	scale, _ := mainFn.Constant([]float32{1.0, 2.0, 0.5}, 3)   // scaling per feature
	offset, _ := mainFn.Constant([]float32{0.0, 1.0, -1.0}, 3) // bias per feature
	mean, _ := mainFn.Constant([]float32{0.0, 0.0, 0.0}, 3)    // mean per feature
	variance, _ := mainFn.Constant([]float32{1.0, 1.0, 1.0}, 3) // variance per feature

	epsilon := float32(1e-5)
	featureAxis := 1

	result, err := mainFn.BatchNormForInference(operand, scale, offset, mean, variance, epsilon, featureAxis)
	if err != nil {
		t.Fatalf("BatchNormForInference() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Input shape: [2, 3, 2] - batch=2, channels=3, spatial=2
	// With mean=0, var=1, epsilon≈0:
	// normalized = (x - mean) / sqrt(var + eps) ≈ x
	// output = scale * normalized + offset
	// For channel 0: scale=1, offset=0 → output = x
	// For channel 1: scale=2, offset=1 → output = 2*x + 1
	// For channel 2: scale=0.5, offset=-1 → output = 0.5*x - 1
	inputData := []float32{
		// batch 0
		1, 2, // channel 0
		3, 4, // channel 1
		5, 6, // channel 2
		// batch 1
		7, 8,   // channel 0
		9, 10,  // channel 1
		11, 12, // channel 2
	}
	inputBuffer, _ := backend.BufferFromFlatData(0, inputData, operandShape)
	defer backend.BufferFinalize(inputBuffer)

	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Expected:
	// channel 0: x * 1 + 0 = x → [1, 2, 7, 8]
	// channel 1: x * 2 + 1 → [7, 9, 19, 21]
	// channel 2: x * 0.5 - 1 → [1.5, 2, 4.5, 5]
	expected := []float32{
		// batch 0
		1, 2,     // channel 0: 1*1+0, 2*1+0
		7, 9,     // channel 1: 3*2+1, 4*2+1
		1.5, 2.0, // channel 2: 5*0.5-1, 6*0.5-1
		// batch 1
		7, 8,     // channel 0
		19, 21,   // channel 1: 9*2+1, 10*2+1
		4.5, 5.0, // channel 2: 11*0.5-1, 12*0.5-1
	}
	outputData := make([]float32, 12)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	_ = paramShape // silence unused warning
	for i := range expected {
		if diff := outputData[i] - expected[i]; diff > 1e-4 || diff < -1e-4 {
			t.Errorf("BatchNormForInference: outputData[%d] = %f, want %f", i, outputData[i], expected[i])
		}
	}
}

//======================================================================================================================
// Edge Case and Validation Tests
//======================================================================================================================

// TestRemNegativeOperands tests Rem with negative operands to verify floor modulo semantics.
func TestRemNegativeOperands(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_rem_negative")
	mainFn := builder.Main()

	// Test various combinations of positive/negative operands
	shape := shapes.Make(dtypes.Float32, 4)
	lhs, _ := mainFn.Parameter("lhs", shape, nil)
	rhs, _ := mainFn.Parameter("rhs", shape, nil)

	result, err := mainFn.Rem(lhs, rhs)
	if err != nil {
		t.Fatalf("Rem() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test cases: Rem(lhs, rhs) with floor modulo semantics
	// Rem(7, 3) = 1, Rem(-7, 3) = 2, Rem(7, -3) = -2, Rem(-7, -3) = -1
	lhsData := []float32{7, -7, 7, -7}
	rhsData := []float32{3, 3, -3, -3}
	lhsBuffer, _ := backend.BufferFromFlatData(0, lhsData, shape)
	defer backend.BufferFinalize(lhsBuffer)
	rhsBuffer, _ := backend.BufferFromFlatData(0, rhsData, shape)
	defer backend.BufferFinalize(rhsBuffer)

	outputs, err := exec.Execute([]backends.Buffer{lhsBuffer, rhsBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	defer backend.BufferFinalize(outputs[0])

	// Floor modulo: result has same sign as divisor (rhs)
	expected := []float32{1, 2, -2, -1}
	outputData := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	for i := range expected {
		if math.Abs(float64(outputData[i]-expected[i])) > 1e-5 {
			t.Errorf("Rem negative: outputData[%d] = %f, want %f (lhs=%f, rhs=%f)",
				i, outputData[i], expected[i], lhsData[i], rhsData[i])
		}
	}
}

// TestBroadcastInDimValidation tests error handling for invalid BroadcastInDim inputs.
func TestBroadcastInDimValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("broadcast_axes_out_of_bounds", func(t *testing.T) {
		builder := backend.Builder("test_broadcast_oob")
		mainFn := builder.Main()

		inputShape := shapes.Make(dtypes.Float32, 3)
		x, _ := mainFn.Parameter("x", inputShape, nil)

		outputShape := shapes.Make(dtypes.Float32, 2, 3)
		// broadcastAxes[0] = 5 is out of bounds for output rank 2
		_, err := mainFn.BroadcastInDim(x, outputShape, []int{5})
		if err == nil {
			t.Error("BroadcastInDim should fail with out-of-bounds broadcast axis")
		}
	})

	t.Run("dimension_mismatch", func(t *testing.T) {
		builder := backend.Builder("test_broadcast_dim_mismatch")
		mainFn := builder.Main()

		inputShape := shapes.Make(dtypes.Float32, 3) // input dim = 3
		x, _ := mainFn.Parameter("x", inputShape, nil)

		outputShape := shapes.Make(dtypes.Float32, 2, 5) // output dim at axis 1 = 5
		// Input dim 3 != output dim 5, and input dim is not 1, so this should fail
		_, err := mainFn.BroadcastInDim(x, outputShape, []int{1})
		if err == nil {
			t.Error("BroadcastInDim should fail with incompatible dimensions")
		}
	})

	t.Run("axes_length_mismatch", func(t *testing.T) {
		builder := backend.Builder("test_broadcast_axes_len")
		mainFn := builder.Main()

		inputShape := shapes.Make(dtypes.Float32, 2, 3) // rank 2
		x, _ := mainFn.Parameter("x", inputShape, nil)

		outputShape := shapes.Make(dtypes.Float32, 2, 3, 4)
		// broadcastAxes has 3 elements but input rank is 2
		_, err := mainFn.BroadcastInDim(x, outputShape, []int{0, 1, 2})
		if err == nil {
			t.Error("BroadcastInDim should fail when axes length doesn't match input rank")
		}
	})
}

// TestDynamicSliceValidation tests error handling for invalid DynamicSlice inputs.
func TestDynamicSliceValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("start_indices_length_mismatch", func(t *testing.T) {
		builder := backend.Builder("test_dynslice_idx_len")
		mainFn := builder.Main()

		dataShape := shapes.Make(dtypes.Float32, 6)
		data, _ := mainFn.Parameter("data", dataShape, nil)

		// No start indices provided for rank-1 operand
		_, err := mainFn.DynamicSlice(data, []backends.Value{}, []int{3})
		if err == nil {
			t.Error("DynamicSlice should fail when startIndices length doesn't match operand rank")
		}
	})

	t.Run("slice_dims_length_mismatch", func(t *testing.T) {
		builder := backend.Builder("test_dynslice_dims_len")
		mainFn := builder.Main()

		dataShape := shapes.Make(dtypes.Float32, 6)
		data, _ := mainFn.Parameter("data", dataShape, nil)
		startIdx, _ := mainFn.Constant([]int32{2})

		// sliceDims has 2 elements but operand rank is 1
		_, err := mainFn.DynamicSlice(data, []backends.Value{startIdx}, []int{3, 2})
		if err == nil {
			t.Error("DynamicSlice should fail when sliceDims length doesn't match operand rank")
		}
	})

	t.Run("non_scalar_start_index", func(t *testing.T) {
		builder := backend.Builder("test_dynslice_nonscalar")
		mainFn := builder.Main()

		dataShape := shapes.Make(dtypes.Float32, 6)
		data, _ := mainFn.Parameter("data", dataShape, nil)

		// Start index with shape [2] is not scalar
		startIdx, _ := mainFn.Constant([]int32{2, 3}, 2)

		_, err := mainFn.DynamicSlice(data, []backends.Value{startIdx}, []int{3})
		if err == nil {
			t.Error("DynamicSlice should fail with non-scalar start index")
		}
	})
}

// TestBatchNormValidation tests error handling for invalid BatchNormForInference inputs.
func TestBatchNormValidation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	t.Run("feature_axis_out_of_bounds", func(t *testing.T) {
		builder := backend.Builder("test_bn_axis_oob")
		mainFn := builder.Main()

		operandShape := shapes.Make(dtypes.Float32, 2, 3, 4)
		operand, _ := mainFn.Parameter("operand", operandShape, nil)
		scale, _ := mainFn.Constant([]float32{1, 1, 1}, 3)
		offset, _ := mainFn.Constant([]float32{0, 0, 0}, 3)
		mean, _ := mainFn.Constant([]float32{0, 0, 0}, 3)
		variance, _ := mainFn.Constant([]float32{1, 1, 1}, 3)

		// featureAxis = 5 is out of bounds for rank-3 operand
		_, err := mainFn.BatchNormForInference(operand, scale, offset, mean, variance, 1e-5, 5)
		if err == nil {
			t.Error("BatchNormForInference should fail with out-of-bounds feature axis")
		}
	})

	t.Run("param_shape_mismatch", func(t *testing.T) {
		builder := backend.Builder("test_bn_param_shape")
		mainFn := builder.Main()

		operandShape := shapes.Make(dtypes.Float32, 2, 3, 4) // 3 features at axis 1
		operand, _ := mainFn.Parameter("operand", operandShape, nil)
		scale, _ := mainFn.Constant([]float32{1, 1, 1, 1, 1}, 5) // Wrong: 5 elements instead of 3
		offset, _ := mainFn.Constant([]float32{0, 0, 0}, 3)
		mean, _ := mainFn.Constant([]float32{0, 0, 0}, 3)
		variance, _ := mainFn.Constant([]float32{1, 1, 1}, 3)

		_, err := mainFn.BatchNormForInference(operand, scale, offset, mean, variance, 1e-5, 1)
		if err == nil {
			t.Error("BatchNormForInference should fail when parameter shape doesn't match feature count")
		}
	})
}
