//go:build darwin && cgo

package coreml

import (
	"math"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// TestSimpleCNN tests a simple CNN-like pattern:
// Input -> Conv2D -> MaxPool -> output
func TestSimpleCNN(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("simple_cnn")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	// Kernel shape: [1, 1, 2, 2] (out_channels=1, in_channels=1, kH=2, kW=2)
	kernelShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	kernel, err := mainFn.Parameter("kernel", kernelShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for kernel failed: %v", err)
	}

	// ConvGeneral with NCHW layout (which is CoreML's native layout)
	axesConfig := backends.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         []int{2, 3},
		KernelInputChannels:  1,
		KernelOutputChannels: 0,
		KernelSpatial:        []int{2, 3},
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        []int{2, 3},
	}

	// No padding, stride 1
	convResult, err := mainFn.ConvGeneral(
		input, kernel, axesConfig,
		[]int{1, 1}, // strides
		nil,         // paddings (valid)
		nil,         // inputDilations
		nil,         // kernelDilations
		1,           // channelGroupCount
		1,           // batchGroupCount
	)
	if err != nil {
		t.Fatalf("ConvGeneral() failed: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{convResult}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	kernelData := []float32{
		1, 0,
		0, 1,
	}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for input failed: %v", err)
	}
	kernelBuffer, err := backend.BufferFromFlatData(0, kernelData, kernelShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for kernel failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer, kernelBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected: convolution with identity-like kernel (1 on diagonal)
	// For a 2x2 kernel [1,0; 0,1] on 4x4 input with valid padding:
	// Output should be 3x3 where each output is sum of diagonal elements
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 3, 3)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// First output element: 1*1 + 0*2 + 0*5 + 1*6 = 7
	if len(outputData) > 0 && math.Abs(float64(outputData[0]-7)) > 0.01 {
		t.Errorf("Expected first output ~7, got %f", outputData[0])
	}
}

// TestDotGeneralSimpleMatMul tests DotGeneral with standard 2D matrix multiplication.
// This is equivalent to: [M, K] x [K, N] -> [M, N]
func TestDotGeneralSimpleMatMul(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("dotgeneral_simple")
	mainFn := builder.Main()

	// Create input shapes: [2, 3] x [3, 4] -> [2, 4]
	lhsShape := shapes.Make(dtypes.Float32, 2, 3)
	rhsShape := shapes.Make(dtypes.Float32, 3, 4)

	lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for lhs failed: %v", err)
	}

	rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for rhs failed: %v", err)
	}

	// DotGeneral: contract axis 1 of lhs with axis 0 of rhs
	result, err := mainFn.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	if err != nil {
		t.Fatalf("DotGeneral() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data
	// lhs = [[1, 2, 3], [4, 5, 6]]
	lhsData := []float32{1, 2, 3, 4, 5, 6}
	// rhs = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
	rhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for lhs failed: %v", err)
	}

	rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for rhs failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 8)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected: [[38, 44, 50, 56], [83, 98, 113, 128]]
	// Row 0: 1*1+2*5+3*9=38, 1*2+2*6+3*10=44, 1*3+2*7+3*11=50, 1*4+2*8+3*12=56
	// Row 1: 4*1+5*5+6*9=83, 4*2+5*6+6*10=98, 4*3+5*7+6*11=113, 4*4+5*8+6*12=128
	expected := []float32{38, 44, 50, 56, 83, 98, 113, 128}
	for i, exp := range expected {
		if math.Abs(float64(outputData[i]-exp)) > 1e-3 {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], exp)
		}
	}
}

// TestBatchedMatMul tests batched matrix multiplication for attention-like patterns.
func TestBatchedMatMul(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("batched_matmul")
	mainFn := builder.Main()

	// Batched matmul: [B, M, K] x [B, K, N] -> [B, M, N]
	// Shape: [2, 3, 4] x [2, 4, 5] -> [2, 3, 5]
	lhsShape := shapes.Make(dtypes.Float32, 2, 3, 4)
	rhsShape := shapes.Make(dtypes.Float32, 2, 4, 5)

	lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for lhs failed: %v", err)
	}

	rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for rhs failed: %v", err)
	}

	// DotGeneral: batch axis 0, contract axis 2 of lhs with axis 1 of rhs
	result, err := mainFn.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0})
	if err != nil {
		t.Fatalf("DotGeneral() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data (initialize with simple pattern)
	lhsData := make([]float32, 2*3*4)
	rhsData := make([]float32, 2*4*5)
	for i := range lhsData {
		lhsData[i] = float32(i+1) * 0.1
	}
	for i := range rhsData {
		rhsData[i] = float32(i+1) * 0.1
	}

	lhsBuf, err := backend.BufferFromFlatData(0, lhsData, lhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for lhs failed: %v", err)
	}

	rhsBuf, err := backend.BufferFromFlatData(0, rhsData, rhsShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for rhs failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 2*3*5)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Verify output shape is correct (2x3x5 = 30 elements)
	if len(outputData) != 30 {
		t.Errorf("Expected 30 output elements, got %d", len(outputData))
	}

	// Verify values are reasonable (should be positive, increasing pattern)
	for i, v := range outputData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("outputData[%d] = %f is invalid", i, v)
		}
	}
}

// TestDotGeneralVectorDot tests DotGeneral with vector dot product (inner product).
func TestDotGeneralVectorDot(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("vector_dot")
	mainFn := builder.Main()

	// Vector dot product: [4] dot [4] -> scalar
	vecShape := shapes.Make(dtypes.Float32, 4)

	a, err := mainFn.Parameter("a", vecShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for a failed: %v", err)
	}

	b, err := mainFn.Parameter("b", vecShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for b failed: %v", err)
	}

	// Dot product: a · b = sum(a[i] * b[i])
	// DotGeneral with contracting axis 0 on both sides
	result, err := mainFn.DotGeneral(a, []int{0}, nil, b, []int{0}, nil)
	if err != nil {
		t.Fatalf("DotGeneral() failed: %v", err)
	}

	// Return scalar result
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Test data: a = [1, 2, 3, 4], b = [2, 3, 4, 5]
	// Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
	aData := []float32{1, 2, 3, 4}
	bData := []float32{2, 3, 4, 5}

	aBuf, err := backend.BufferFromFlatData(0, aData, vecShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for a failed: %v", err)
	}
	defer backend.BufferFinalize(aBuf)

	bBuf, err := backend.BufferFromFlatData(0, bData, vecShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for b failed: %v", err)
	}
	defer backend.BufferFinalize(bBuf)

	outputs, err := exec.Execute([]backends.Buffer{aBuf, bBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Get scalar output
	outputData := make([]float32, 1)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	expected := float32(40.0)
	if outputData[0] != expected {
		t.Errorf("Expected %f, got %f", expected, outputData[0])
	}
	t.Logf("Vector dot product result: %f (expected %f)", outputData[0], expected)
}

// TestAttentionBlock tests a simplified self-attention mechanism.
func TestAttentionBlock(t *testing.T) {
	// Simplified attention: Q @ K^T / sqrt(d) @ V
	// For simplicity, just test Q @ K^T pattern
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("attention")
	mainFn := builder.Main()

	// Q, K shapes: [batch, seq_len, d_model] = [1, 4, 8]
	qShape := shapes.Make(dtypes.Float32, 1, 4, 8)
	kShape := shapes.Make(dtypes.Float32, 1, 4, 8)

	q, err := mainFn.Parameter("q", qShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for q failed: %v", err)
	}

	k, err := mainFn.Parameter("k", kShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for k failed: %v", err)
	}

	// Q @ K^T: [1, 4, 8] @ [1, 8, 4] -> [1, 4, 4]
	// batch axis: 0, contract Q axis 2 with K axis 2 (transposing K)
	// This gives us Q @ K^T semantics
	result, err := mainFn.DotGeneral(q, []int{2}, []int{0}, k, []int{2}, []int{0})
	if err != nil {
		t.Fatalf("DotGeneral() failed: %v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input data
	qData := make([]float32, 1*4*8)
	kData := make([]float32, 1*4*8)
	for i := range qData {
		qData[i] = float32(i) * 0.1
	}
	for i := range kData {
		kData[i] = float32(i) * 0.1
	}

	qBuf, err := backend.BufferFromFlatData(0, qData, qShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for q failed: %v", err)
	}

	kBuf, err := backend.BufferFromFlatData(0, kData, kShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for k failed: %v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{qBuf, kBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	outputData := make([]float32, 1*4*4)
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Verify output shape is correct (1x4x4 = 16 elements)
	if len(outputData) != 16 {
		t.Errorf("Expected 16 output elements, got %d", len(outputData))
	}

	// Verify values are reasonable
	for i, v := range outputData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("outputData[%d] = %f is invalid", i, v)
		}
	}
}

// TestReduceWindowMaxPool tests ReduceWindow with MaxPool semantics.
func TestReduceWindowMaxPool(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("reduce_window_max_pool")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	// MaxPool with 2x2 window, stride 2
	// Window dimensions: [1, 1, 2, 2] (batch=1, channels=1, spatial=2x2)
	// Strides: [1, 1, 2, 2]
	result, err := mainFn.ReduceWindow(
		input,
		backends.ReduceOpMax,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,               // baseDilations
		nil,               // windowDilations
		nil,               // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected shape: [1, 1, 2, 2] (4x4 input with 2x2 window and stride 2)
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected values: max of each 2x2 window
	// Window 1: max(1,2,5,6) = 6
	// Window 2: max(3,4,7,8) = 8
	// Window 3: max(9,10,13,14) = 14
	// Window 4: max(11,12,15,16) = 16
	expected := []float32{6, 8, 14, 16}
	for i, v := range expected {
		if len(outputData) > i && math.Abs(float64(outputData[i]-v)) > 0.01 {
			t.Errorf("Expected output[%d] = %f, got %f", i, v, outputData[i])
		}
	}
}

// TestReduceWindowSumPool tests ReduceWindow with Sum (via AvgPool scaling).
func TestReduceWindowSumPool(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("reduce_window_sum_pool")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	// SumPool with 2x2 window, stride 2
	result, err := mainFn.ReduceWindow(
		input,
		backends.ReduceOpSum,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,               // baseDilations
		nil,               // windowDilations
		nil,               // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected shape: [1, 1, 2, 2]
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected values: sum of each 2x2 window
	// Window 1: 1+2+5+6 = 14
	// Window 2: 3+4+7+8 = 22
	// Window 3: 9+10+13+14 = 46
	// Window 4: 11+12+15+16 = 54
	expected := []float32{14, 22, 46, 54}
	for i, v := range expected {
		if len(outputData) > i && math.Abs(float64(outputData[i]-v)) > 0.01 {
			t.Errorf("Expected output[%d] = %f, got %f", i, v, outputData[i])
		}
	}
}

// TestReduceWindowMinPool tests ReduceWindow with MinPool semantics.
// MinPool is implemented via the negation trick: MinPool(x) = -MaxPool(-x)
func TestReduceWindowMinPool(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("reduce_window_min_pool")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	// MinPool with 2x2 window, stride 2
	// Window dimensions: [1, 1, 2, 2] (batch=1, channels=1, spatial=2x2)
	// Strides: [1, 1, 2, 2]
	result, err := mainFn.ReduceWindow(
		input,
		backends.ReduceOpMin,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,               // baseDilations
		nil,               // windowDilations
		nil,               // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected shape: [1, 1, 2, 2] (4x4 input with 2x2 window and stride 2)
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected values: min of each 2x2 window
	// Window 1: min(1,2,5,6) = 1
	// Window 2: min(3,4,7,8) = 3
	// Window 3: min(9,10,13,14) = 9
	// Window 4: min(11,12,15,16) = 11
	expected := []float32{1, 3, 9, 11}
	for i, v := range expected {
		if len(outputData) > i && math.Abs(float64(outputData[i]-v)) > 0.01 {
			t.Errorf("Expected output[%d] = %f, got %f", i, v, outputData[i])
		}
	}
}

// TestReduceWindowMinPoolNegativeValues tests MinPool with negative values
// to verify the negation trick works correctly.
func TestReduceWindowMinPoolNegativeValues(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("reduce_window_min_pool_neg")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	// MinPool with 2x2 window, stride 2
	result, err := mainFn.ReduceWindow(
		input,
		backends.ReduceOpMin,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,               // baseDilations
		nil,               // windowDilations
		nil,               // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffer with negative values
	inputData := []float32{
		-5, -2, -3, -4,
		-1, -6, -7, -8,
		-9, -10, -11, -12,
		-13, -14, -15, -16,
	}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Expected shape: [1, 1, 2, 2]
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Expected values: min of each 2x2 window
	// Window 1: min(-5,-2,-1,-6) = -6
	// Window 2: min(-3,-4,-7,-8) = -8
	// Window 3: min(-9,-10,-13,-14) = -14
	// Window 4: min(-11,-12,-15,-16) = -16
	expected := []float32{-6, -8, -14, -16}
	for i, v := range expected {
		if len(outputData) > i && math.Abs(float64(outputData[i]-v)) > 0.01 {
			t.Errorf("Expected output[%d] = %f, got %f", i, v, outputData[i])
		}
	}
}

// TestConvGeneralInputDilationError tests that ConvGeneral returns a helpful error
// when input dilation > 1 is requested (which CoreML doesn't support).
func TestConvGeneralInputDilationError(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("conv_input_dilation_error")
	mainFn := builder.Main()

	// Input shape: [1, 1, 4, 4] (batch=1, channels=1, height=4, width=4)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	// Kernel shape: [1, 1, 2, 2] (out_channels=1, in_channels=1, kH=2, kW=2)
	kernelShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	kernel, err := mainFn.Parameter("kernel", kernelShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for kernel failed: %v", err)
	}

	// NCHW layout
	axesConfig := backends.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         []int{2, 3},
		KernelInputChannels:  1,
		KernelOutputChannels: 0,
		KernelSpatial:        []int{2, 3},
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        []int{2, 3},
	}

	// Try to use input dilation > 1 (should fail with helpful error)
	_, err = mainFn.ConvGeneral(
		input, kernel, axesConfig,
		[]int{1, 1}, // strides
		nil,         // paddings (valid)
		[]int{2, 2}, // inputDilations > 1 - NOT SUPPORTED
		nil,         // kernelDilations
		1,           // channelGroupCount
		1,           // batchGroupCount
	)

	// Verify we get an error
	if err == nil {
		t.Fatal("ConvGeneral with inputDilations > 1 should have failed, but it didn't")
	}

	// Verify the error message is helpful
	errMsg := err.Error()

	// Check that the error mentions key concepts
	if !strings.Contains(errMsg, "input dilation") {
		t.Errorf("Error message should mention 'input dilation', got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "base dilation") {
		t.Errorf("Error message should mention 'base dilation' (alternative name), got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "CoreML") {
		t.Errorf("Error message should mention 'CoreML', got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "kernel dilation") && !strings.Contains(errMsg, "KERNEL dilation") {
		t.Errorf("Error message should explain that KERNEL dilation IS supported, got: %s", errMsg)
	}
	if !strings.Contains(errMsg, "orkaround") { // Workaround or workarounds
		t.Errorf("Error message should suggest workarounds, got: %s", errMsg)
	}

	t.Logf("Got expected helpful error message: %s", errMsg)
}

// TestConvGeneralKernelDilationWorks tests that kernel dilation (which IS supported) works correctly.
func TestConvGeneralKernelDilationWorks(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("conv_kernel_dilation")
	mainFn := builder.Main()

	// Input shape: [1, 1, 5, 5] (batch=1, channels=1, height=5, width=5)
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 5, 5)
	// Kernel shape: [1, 1, 2, 2] (out_channels=1, in_channels=1, kH=2, kW=2)
	kernelShape := shapes.Make(dtypes.Float32, 1, 1, 2, 2)

	input, err := mainFn.Parameter("input", inputShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for input failed: %v", err)
	}

	kernel, err := mainFn.Parameter("kernel", kernelShape, nil)
	if err != nil {
		t.Fatalf("Parameter() for kernel failed: %v", err)
	}

	// NCHW layout
	axesConfig := backends.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         []int{2, 3},
		KernelInputChannels:  1,
		KernelOutputChannels: 0,
		KernelSpatial:        []int{2, 3},
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        []int{2, 3},
	}

	// Use kernel dilation = 2 (which IS supported by CoreML)
	// A 2x2 kernel with dilation 2 acts like a 3x3 kernel with center cross pattern
	convResult, err := mainFn.ConvGeneral(
		input, kernel, axesConfig,
		[]int{1, 1}, // strides
		nil,         // paddings (valid)
		nil,         // inputDilations (not used)
		[]int{2, 2}, // kernelDilations = 2 - SUPPORTED
		1,           // channelGroupCount
		1,           // batchGroupCount
	)
	if err != nil {
		t.Fatalf("ConvGeneral with kernelDilations should work, but got error: %v", err)
	}

	// Set output
	if err := mainFn.Return([]backends.Value{convResult}, nil); err != nil {
		t.Fatalf("Return() failed: %v", err)
	}

	// Compile
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}
	defer exec.Finalize()

	// Create input buffers - all ones for simplicity
	inputData := make([]float32, 25)
	for i := range inputData {
		inputData[i] = 1.0
	}
	// Kernel: identity-like diagonal kernel
	kernelData := []float32{1, 0, 0, 1}

	inputBuffer, err := backend.BufferFromFlatData(0, inputData, inputShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for input failed: %v", err)
	}
	kernelBuffer, err := backend.BufferFromFlatData(0, kernelData, kernelShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() for kernel failed: %v", err)
	}

	// Execute
	outputs, err := exec.Execute([]backends.Buffer{inputBuffer, kernelBuffer}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// With 5x5 input, 2x2 kernel with dilation 2 (effective 3x3), valid padding:
	// Output should be 3x3
	// Each output element is sum of 4 input elements at dilated positions
	// With all 1s input and kernel [1,0;0,1], output should be 2.0 everywhere
	expectedShape := shapes.Make(dtypes.Float32, 1, 1, 3, 3)
	outputData := make([]float32, expectedShape.Size())
	if err := backend.BufferToFlatData(outputs[0], outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %v", err)
	}

	// Verify output values are correct (should be 2.0 = 1*1 + 0 + 0 + 1*1)
	for i, v := range outputData {
		if math.Abs(float64(v-2.0)) > 0.01 {
			t.Errorf("Expected output[%d] = 2.0, got %f", i, v)
		}
	}
	t.Logf("Kernel dilation works correctly, output: %v", outputData)
}
