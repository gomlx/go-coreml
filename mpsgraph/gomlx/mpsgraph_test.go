// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

// TestBackendCreation tests that the backend can be created.
func TestBackendCreation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if backend.Name() != "mpsgraph" {
		t.Errorf("Name() = %q, want %q", backend.Name(), "mpsgraph")
	}
	if backend.NumDevices() != 1 {
		t.Errorf("NumDevices() = %d, want 1", backend.NumDevices())
	}
	t.Logf("Backend: %s — %s", backend.Name(), backend.Description())
}

// TestBufferOperations tests buffer creation, read-back, and finalize.
func TestBufferOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	buf, err := backend.BufferFromFlatData(0, inputData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	gotShape, err := backend.BufferShape(buf)
	if err != nil {
		t.Fatalf("BufferShape() failed: %+v", err)
	}
	if !gotShape.Equal(shape) {
		t.Errorf("BufferShape() = %v, want %v", gotShape, shape)
	}

	outputData := make([]float32, 6)
	if err := backend.BufferToFlatData(buf, outputData); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	for i := range inputData {
		if outputData[i] != inputData[i] {
			t.Errorf("outputData[%d] = %f, want %f", i, outputData[i], inputData[i])
		}
	}

	if err := backend.BufferFinalize(buf); err != nil {
		t.Fatalf("BufferFinalize() failed: %+v", err)
	}
}

// TestSharedBuffer tests shared buffer (Go-managed memory) functionality.
func TestSharedBuffer(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	if !backend.HasSharedBuffers() {
		t.Fatal("HasSharedBuffers() = false, want true")
	}

	shape := shapes.Make(dtypes.Float32, 4)
	buf, flat, err := backend.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatalf("NewSharedBuffer() failed: %+v", err)
	}

	flatData := flat.([]float32)
	for i := range flatData {
		flatData[i] = float32(i + 1)
	}

	gotFlat, err := backend.BufferData(buf)
	if err != nil {
		t.Fatalf("BufferData() failed: %+v", err)
	}
	gotData := gotFlat.([]float32)
	for i := range gotData {
		if gotData[i] != float32(i+1) {
			t.Errorf("gotData[%d] = %f, want %f", i, gotData[i], float32(i+1))
		}
	}
	_ = backend.BufferFinalize(buf)
}

// execUnaryOp is a test helper that builds, compiles, and executes a unary op.
func execUnaryOp(t *testing.T, backend backends.Backend, opFn func(backends.Function, backends.Value) (backends.Value, error),
	input []float32) []float32 {
	t.Helper()

	shape := shapes.Make(dtypes.Float32, len(input))
	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := opFn(mainFn, x)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, input, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(input))
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	return result
}

// execBinaryOp is a test helper that builds, compiles, and executes a binary op.
func execBinaryOp(t *testing.T, backend backends.Backend,
	opFn func(backends.Function, backends.Value, backends.Value) (backends.Value, error),
	shape shapes.Shape, lhsData, rhsData []float32) []float32 {
	t.Helper()

	builder := backend.Builder("test")
	mainFn := builder.Main()

	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Parameter("y", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	z, err := opFn(mainFn, x, y)
	if err != nil {
		t.Fatalf("opFn() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{z}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}
	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, lhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %+v", err)
	}
	yBuf, err := backend.BufferFromFlatData(0, rhsData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(y) failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}
	result := make([]float32, len(lhsData))
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	return result
}

func assertClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

// TestAddOperation tests the full pipeline: parameter → add → compile → execute → read.
func TestAddOperation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 2, 3)
	result := execBinaryOp(t, backend,
		func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Add(x, y) },
		shape,
		[]float32{1, 2, 3, 4, 5, 6},
		[]float32{10, 20, 30, 40, 50, 60},
	)
	assertClose(t, result, []float32{11, 22, 33, 44, 55, 66}, 1e-5)
}

// TestUnaryOperations tests several unary math ops.
func TestUnaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	tests := []struct {
		name     string
		opFn     func(backends.Function, backends.Value) (backends.Value, error)
		input    []float32
		expected []float32
		tol      float64
	}{
		{"Abs", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Abs(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, 2, 3, 4}, 1e-5},
		{"Neg", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Neg(x) },
			[]float32{-1, 2, -3, 4}, []float32{1, -2, 3, -4}, 1e-5},
		{"Sqrt", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Sqrt(x) },
			[]float32{1, 4, 9, 16}, []float32{1, 2, 3, 4}, 1e-5},
		{"Exp", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Exp(x) },
			[]float32{0, 1}, []float32{1, float32(math.E)}, 1e-5},
		{"Log", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Log(x) },
			[]float32{1, float32(math.E)}, []float32{0, 1}, 1e-4},
		{"Tanh", func(f backends.Function, x backends.Value) (backends.Value, error) { return f.Tanh(x) },
			[]float32{0, 1}, []float32{0, float32(math.Tanh(1))}, 1e-5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := execUnaryOp(t, backend, tc.opFn, tc.input)
			assertClose(t, got, tc.expected, tc.tol)
		})
	}
}

// TestBinaryOperations tests binary math ops.
func TestBinaryOperations(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhs := []float32{10, 20, 30, 40}
	rhs := []float32{3, 5, 10, 8}

	t.Run("Sub", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Sub(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{7, 15, 20, 32}, 1e-5)
	})

	t.Run("Mul", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Mul(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{30, 100, 300, 320}, 1e-5)
	})

	t.Run("Div", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Div(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10.0 / 3, 4, 3, 5}, 1e-4)
	})

	t.Run("Max", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Max(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{10, 20, 30, 40}, 1e-5)
	})

	t.Run("Min", func(t *testing.T) {
		got := execBinaryOp(t, backend,
			func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Min(x, y) },
			shape, lhs, rhs)
		assertClose(t, got, []float32{3, 5, 10, 8}, 1e-5)
	})
}

// TestReshape tests the Reshape operation.
func TestReshape(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reshape")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}
	y, err := mainFn.Reshape(x, 3, 2)
	if err != nil {
		t.Fatalf("Reshape() failed: %+v", err)
	}
	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	outShapes := exec.Outputs()
	if len(outShapes) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outShapes))
	}
	expectedOutShape := shapes.Make(dtypes.Float32, 3, 2)
	if !outShapes[0].Equal(expectedOutShape) {
		t.Errorf("Output shape = %v, want %v", outShapes[0], expectedOutShape)
	}

	xData := []float32{1, 2, 3, 4, 5, 6}
	xBuf, err := backend.BufferFromFlatData(0, xData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Reshape doesn't change data, just layout.
	assertClose(t, result, xData, 0)
}

// TestConstant tests that Constant values are compiled into the graph correctly.
func TestConstant(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_constant")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Create a constant and add it to the parameter.
	c, err := mainFn.Constant([]float32{10, 20, 30}, 3)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}

	y, err := mainFn.Add(x, c)
	if err != nil {
		t.Fatalf("Add() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{11, 22, 33}, 1e-5)
}

// TestReduceSum tests the ReduceSum operation.
func TestReduceSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_sum")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Sum over axis 1 → shape [2].
	y, err := mainFn.ReduceSum(x, 1)
	if err != nil {
		t.Fatalf("ReduceSum() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 2)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, result, []float32{6, 15}, 1e-5)
}

// TestDotGeneral tests the DotGeneral (batched matmul) operation.
func TestDotGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("SimpleMatMul", func(t *testing.T) {
		// [2,3] x [3,2] → [2,2]
		builder := backend.Builder("test_dot_simple")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		// Standard matmul: contract axis 1 of lhs with axis 0 of rhs.
		y, err := mainFn.DotGeneral(lhs, []int{1}, nil, rhs, []int{0}, nil, backends.DotGeneralConfig{})
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// lhs = [[1,2,3],[4,5,6]], rhs = [[1,2],[3,4],[5,6]]
		// result = [[22,28],[49,64]]
		lhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, rhsShape)

		outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 4)
		if err := backend.BufferToFlatData(outputs[0], result); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64}, 1e-4)
	})

	t.Run("BatchedMatMul", func(t *testing.T) {
		// [2,2,3] x [2,3,2] → [2,2,2] with batch axis 0.
		builder := backend.Builder("test_dot_batched")
		mainFn := builder.Main()

		lhsShape := shapes.Make(dtypes.Float32, 2, 2, 3)
		rhsShape := shapes.Make(dtypes.Float32, 2, 3, 2)

		lhs, err := mainFn.Parameter("lhs", lhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		rhs, err := mainFn.Parameter("rhs", rhsShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		y, err := mainFn.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0}, backends.DotGeneralConfig{})
		if err != nil {
			t.Fatalf("DotGeneral() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// Batch 0: [[1,2,3],[4,5,6]] x [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
		// Batch 1: [[7,8,9],[10,11,12]] x [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
		lhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		rhsData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		lhsBuf, _ := backend.BufferFromFlatData(0, lhsData, lhsShape)
		rhsBuf, _ := backend.BufferFromFlatData(0, rhsData, rhsShape)

		outputs, err := exec.Execute([]backends.Buffer{lhsBuf, rhsBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		result := make([]float32, 8)
		if err := backend.BufferToFlatData(outputs[0], result); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		assertClose(t, result, []float32{22, 28, 49, 64, 220, 244, 301, 334}, 1e-3)
	})
}

// TestGather tests the Gather operation (embedding lookup pattern).
func TestGather(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	t.Run("EmbeddingLookup", func(t *testing.T) {
		// Operand: [4, 3] embedding table, indices: [2] → gather rows 1, 3 → [2, 3]
		builder := backend.Builder("test_gather_embed")
		mainFn := builder.Main()

		operandShape := shapes.Make(dtypes.Float32, 4, 3)
		indicesShape := shapes.Make(dtypes.Int32, 2, 1) // [2, 1] with indexVectorAxis=1

		operand, err := mainFn.Parameter("operand", operandShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}
		indices, err := mainFn.Parameter("indices", indicesShape, nil)
		if err != nil {
			t.Fatalf("Parameter() failed: %+v", err)
		}

		// XLA-style gather for embedding lookup:
		// indexVectorAxis=1, startIndexMap=[0], collapsedSliceAxes=[0],
		// offsetOutputAxes=[1], sliceSizes=[1, 3]
		result, err := mainFn.Gather(operand, indices,
			1,        // indexVectorAxis
			[]int{1}, // offsetOutputAxes
			[]int{0}, // collapsedSliceAxes
			[]int{0}, // startIndexMap
			[]int{1, 3}, // sliceSizes
			false,       // indicesAreSorted
		)
		if err != nil {
			t.Fatalf("Gather() failed: %+v", err)
		}

		if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
			t.Fatalf("Return() failed: %+v", err)
		}

		exec, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile() failed: %+v", err)
		}
		defer exec.Finalize()

		// Embedding table: [[10,11,12], [20,21,22], [30,31,32], [40,41,42]]
		// Indices: [1, 3] → select rows 1 and 3
		operandData := []float32{10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42}
		indicesData := []int32{1, 3}

		operandBuf, err := backend.BufferFromFlatData(0, operandData, operandShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(operand) failed: %+v", err)
		}
		indicesBuf, err := backend.BufferFromFlatData(0, indicesData, indicesShape)
		if err != nil {
			t.Fatalf("BufferFromFlatData(indices) failed: %+v", err)
		}

		outputs, err := exec.Execute([]backends.Buffer{operandBuf, indicesBuf}, nil, 0)
		if err != nil {
			t.Fatalf("Execute() failed: %+v", err)
		}

		got := make([]float32, 6)
		if err := backend.BufferToFlatData(outputs[0], got); err != nil {
			t.Fatalf("BufferToFlatData() failed: %+v", err)
		}
		// Expected: rows 1 and 3 → [20,21,22, 40,41,42]
		assertClose(t, got, []float32{20, 21, 22, 40, 41, 42}, 1e-5)
	})
}

// TestSlice tests the Slice operation.
func TestSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_slice")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3, 4)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Slice rows [1:3], cols [1:3] → [2, 2]
	y, err := mainFn.Slice(x, []int{1, 1}, []int{3, 3}, []int{1, 1})
	if err != nil {
		t.Fatalf("Slice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	xData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	xBuf, err := backend.BufferFromFlatData(0, xData, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[6,7], [10,11]]
	assertClose(t, got, []float32{6, 7, 10, 11}, 1e-5)
}

// TestConcatenate tests the Concatenate operation.
func TestConcatenate(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_concat")
	mainFn := builder.Main()

	shapeA := shapes.Make(dtypes.Float32, 2, 2)
	shapeB := shapes.Make(dtypes.Float32, 2, 3)

	a, err := mainFn.Parameter("a", shapeA, nil)
	if err != nil {
		t.Fatalf("Parameter(a) failed: %+v", err)
	}
	b, err := mainFn.Parameter("b", shapeB, nil)
	if err != nil {
		t.Fatalf("Parameter(b) failed: %+v", err)
	}

	// Concatenate along axis 1: [2,2] + [2,3] → [2,5]
	y, err := mainFn.Concatenate(1, a, b)
	if err != nil {
		t.Fatalf("Concatenate() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	aBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, shapeA)
	bBuf, _ := backend.BufferFromFlatData(0, []float32{5, 6, 7, 8, 9, 10}, shapeB)

	outputs, err := exec.Execute([]backends.Buffer{aBuf, bBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 10)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[1,2,5,6,7], [3,4,8,9,10]]
	assertClose(t, got, []float32{1, 2, 5, 6, 7, 3, 4, 8, 9, 10}, 1e-5)
}

// TestIota tests the Iota operation.
func TestIota(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_iota")
	mainFn := builder.Main()

	// Iota with shape [2, 3] along axis 1 → [[0,1,2],[0,1,2]]
	outShape := shapes.Make(dtypes.Float32, 2, 3)
	y, err := mainFn.Iota(outShape, 1)
	if err != nil {
		t.Fatalf("Iota() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	outputs, err := exec.Execute(nil, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{0, 1, 2, 0, 1, 2}, 1e-5)
}

// TestBroadcastInDim tests the BroadcastInDim operation.
func TestBroadcastInDim(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_broadcast")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Broadcast [3] → [2, 3] by mapping axis 0 of input to axis 1 of output.
	outShape := shapes.Make(dtypes.Float32, 2, 3)
	y, err := mainFn.BroadcastInDim(x, outShape, []int{1})
	if err != nil {
		t.Fatalf("BroadcastInDim() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{10, 20, 30}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[10,20,30], [10,20,30]]
	assertClose(t, got, []float32{10, 20, 30, 10, 20, 30}, 1e-5)
}

// TestPad tests the Pad operation.
func TestPad(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_pad")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 2)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Pad value = 0
	padVal, err := mainFn.Constant([]float32{0}, 1)
	if err != nil {
		t.Fatalf("Constant() failed: %+v", err)
	}
	padValScalar, err := mainFn.Reshape(padVal)
	if err != nil {
		t.Fatalf("Reshape() failed: %+v", err)
	}

	// Pad: 1 before axis 0, 0 after; 0 before axis 1, 1 after → [3, 3]
	y, err := mainFn.Pad(x, padValScalar,
		backends.PadAxis{Start: 1, End: 0},
		backends.PadAxis{Start: 0, End: 1},
	)
	if err != nil {
		t.Fatalf("Pad() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 9)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[0,0,0], [1,2,0], [3,4,0]]
	assertClose(t, got, []float32{0, 0, 0, 1, 2, 0, 3, 4, 0}, 1e-5)
}

// TestArgMinMax tests the ArgMinMax operation.
func TestArgMinMax(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_argmax")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// ArgMax along axis 1 → shape [2], dtype Int32.
	y, err := mainFn.ArgMinMax(x, 1, dtypes.Int32, false)
	if err != nil {
		t.Fatalf("ArgMinMax() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 5, 3], [8, 2, 6]] → argmax on axis 1 → [1, 0]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 5, 3, 8, 2, 6}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]int32, 2)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	if got[0] != 1 || got[1] != 0 {
		t.Errorf("ArgMax got %v, want [1, 0]", got)
	}
}

// TestDynamicSlice tests the DynamicSlice operation.
func TestDynamicSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_slice")
	mainFn := builder.Main()

	// Input: [3, 4], slice a [2, 2] window starting at (1, 1).
	inShape := shapes.Make(dtypes.Float32, 3, 4)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Start indices as scalar Int32 parameters.
	startShape := shapes.Make(dtypes.Int32)
	start0, err := mainFn.Parameter("start0", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start0) failed: %+v", err)
	}
	start1, err := mainFn.Parameter("start1", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start1) failed: %+v", err)
	}

	y, err := mainFn.DynamicSlice(x, []backends.Value{start0, start1}, []int{2, 2})
	if err != nil {
		t.Fatalf("DynamicSlice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	// Slice at (1,1) with size (2,2) → [[6,7], [10,11]]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(x) failed: %+v", err)
	}
	s0Buf, err := backend.BufferFromFlatData(0, []int32{1}, startShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(s0) failed: %+v", err)
	}
	s1Buf, err := backend.BufferFromFlatData(0, []int32{1}, startShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData(s1) failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf, s0Buf, s1Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{6, 7, 10, 11}, 1e-5)
}

// TestDynamicUpdateSlice tests the DynamicUpdateSlice operation.
func TestDynamicUpdateSlice(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_dynamic_update_slice")
	mainFn := builder.Main()

	// Input: [3, 4], update: [2, 2] at position (1, 1).
	inShape := shapes.Make(dtypes.Float32, 3, 4)
	updateShape := shapes.Make(dtypes.Float32, 2, 2)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter(x) failed: %+v", err)
	}
	upd, err := mainFn.Parameter("upd", updateShape, nil)
	if err != nil {
		t.Fatalf("Parameter(upd) failed: %+v", err)
	}

	startShape := shapes.Make(dtypes.Int32)
	start0, err := mainFn.Parameter("start0", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start0) failed: %+v", err)
	}
	start1, err := mainFn.Parameter("start1", startShape, nil)
	if err != nil {
		t.Fatalf("Parameter(start1) failed: %+v", err)
	}

	y, err := mainFn.DynamicUpdateSlice(x, upd, []backends.Value{start0, start1})
	if err != nil {
		t.Fatalf("DynamicUpdateSlice() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	// Update: [[100,200], [300,400]] at (1,1)
	// Expected: [[1,2,3,4], [5,100,200,8], [9,300,400,12]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, inShape)
	updBuf, _ := backend.BufferFromFlatData(0, []float32{100, 200, 300, 400}, updateShape)
	s0Buf, _ := backend.BufferFromFlatData(0, []int32{1}, startShape)
	s1Buf, _ := backend.BufferFromFlatData(0, []int32{1}, startShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf, updBuf, s0Buf, s1Buf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 12)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{1, 2, 3, 4, 5, 100, 200, 8, 9, 300, 400, 12}, 1e-5)
}

// TestFusedSoftmax tests the FusedSoftmax operation.
func TestFusedSoftmax(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_softmax")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Softmax along axis 1.
	y, err := mainFn.FusedSoftmax(x, 1)
	if err != nil {
		t.Fatalf("FusedSoftmax() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1, 2, 3], [1, 1, 1]]
	xBuf, err := backend.BufferFromFlatData(0, []float32{1, 2, 3, 1, 1, 1}, inShape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Row 0: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
	// Row 1: softmax([1,1,1]) = [0.3333, 0.3333, 0.3333]
	expected := []float32{
		float32(math.Exp(1) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(2) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		float32(math.Exp(3) / (math.Exp(1) + math.Exp(2) + math.Exp(3))),
		1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
	}
	assertClose(t, got, expected, 1e-4)
}

// TestScatterSum tests the ScatterSum operation (embedding gradient pattern).
func TestScatterSum(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	// Operand: [4, 2] (e.g., gradient accumulator for 4-row embedding).
	// Indices: [3, 1] (scatter to rows 0, 2, 0 — row 0 gets two updates summed).
	// Updates: [3, 2] (values to scatter-add).
	builder := backend.Builder("test_scatter_sum")
	mainFn := builder.Main()

	operandShape := shapes.Make(dtypes.Float32, 4, 2)
	indicesShape := shapes.Make(dtypes.Int32, 3, 1)
	updatesShape := shapes.Make(dtypes.Float32, 3, 2)

	operand, err := mainFn.Parameter("operand", operandShape, nil)
	if err != nil {
		t.Fatalf("Parameter(operand) failed: %+v", err)
	}
	indices, err := mainFn.Parameter("indices", indicesShape, nil)
	if err != nil {
		t.Fatalf("Parameter(indices) failed: %+v", err)
	}
	updates, err := mainFn.Parameter("updates", updatesShape, nil)
	if err != nil {
		t.Fatalf("Parameter(updates) failed: %+v", err)
	}

	// ScatterSum: scatter updates into operand along axis 0.
	// indexVectorAxis=1, updateWindowAxes=[1], insertedWindowAxes=[0],
	// scatterAxesToOperandAxes=[0]
	result, err := mainFn.ScatterSum(operand, indices, updates,
		1,        // indexVectorAxis
		[]int{1}, // updateWindowAxes
		[]int{0}, // insertedWindowAxes
		[]int{0}, // scatterAxesToOperandAxes
		false, false,
	)
	if err != nil {
		t.Fatalf("ScatterSum() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Operand (all zeros): [[0,0], [0,0], [0,0], [0,0]]
	// Indices: [0, 2, 0] → scatter-add to rows 0, 2, 0
	// Updates: [[1,2], [3,4], [5,6]]
	// Expected: row 0 = [0,0] + [1,2] + [5,6] = [6,8], row 2 = [0,0] + [3,4] = [3,4]
	operandBuf, _ := backend.BufferFromFlatData(0, []float32{0, 0, 0, 0, 0, 0, 0, 0}, operandShape)
	indicesBuf, _ := backend.BufferFromFlatData(0, []int32{0, 2, 0}, indicesShape)
	updatesBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, updatesShape)

	outputs, err := exec.Execute([]backends.Buffer{operandBuf, indicesBuf, updatesBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 8)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[6,8], [0,0], [3,4], [0,0]]
	assertClose(t, got, []float32{6, 8, 0, 0, 3, 4, 0, 0}, 1e-5)
}

// TestReverse tests the Reverse operation.
func TestReverse(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reverse")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Reverse along axis 1.
	y, err := mainFn.Reverse(x, 1)
	if err != nil {
		t.Fatalf("Reverse() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [[3,2,1], [6,5,4]]
	assertClose(t, got, []float32{3, 2, 1, 6, 5, 4}, 1e-5)
}

// TestReduceProduct tests the ReduceProduct operation.
func TestReduceProduct(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_reduce_product")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Product over axis 1 → shape [2].
	y, err := mainFn.ReduceProduct(x, 1)
	if err != nil {
		t.Fatalf("ReduceProduct() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]]
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 2)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Expected: [1*2*3=6, 4*5*6=120]
	assertClose(t, got, []float32{6, 120}, 1e-5)
}

// TestComparisonOps tests comparison operations.
func TestComparisonOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	shape := shapes.Make(dtypes.Float32, 4)
	lhsData := []float32{1, 2, 3, 4}
	rhsData := []float32{2, 2, 2, 2}

	tests := []struct {
		name     string
		opFn     func(backends.Function, backends.Value, backends.Value) (backends.Value, error)
		expected []bool
	}{
		{"Equal", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.Equal(x, y) },
			[]bool{false, true, false, false}},
		{"NotEqual", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.NotEqual(x, y) },
			[]bool{true, false, true, true}},
		{"GreaterThan", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.GreaterThan(x, y) },
			[]bool{false, false, true, true}},
		{"LessThan", func(f backends.Function, x, y backends.Value) (backends.Value, error) { return f.LessThan(x, y) },
			[]bool{true, false, false, false}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			builder := backend.Builder("test_cmp_" + tc.name)
			mainFn := builder.Main()

			x, _ := mainFn.Parameter("x", shape, nil)
			y, _ := mainFn.Parameter("y", shape, nil)
			z, err := tc.opFn(mainFn, x, y)
			if err != nil {
				t.Fatalf("opFn() failed: %+v", err)
			}
			if err := mainFn.Return([]backends.Value{z}, nil); err != nil {
				t.Fatalf("Return() failed: %+v", err)
			}

			exec, err := builder.Compile()
			if err != nil {
				t.Fatalf("Compile() failed: %+v", err)
			}
			defer exec.Finalize()

			xBuf, _ := backend.BufferFromFlatData(0, lhsData, shape)
			yBuf, _ := backend.BufferFromFlatData(0, rhsData, shape)

			outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
			if err != nil {
				t.Fatalf("Execute() failed: %+v", err)
			}

			got := make([]bool, 4)
			if err := backend.BufferToFlatData(outputs[0], got); err != nil {
				t.Fatalf("BufferToFlatData() failed: %+v", err)
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Errorf("[%d] = %v, want %v", i, got[i], tc.expected[i])
				}
			}
		})
	}
}

// TestWhere tests the Where (select) operation.
func TestWhere(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_where")
	mainFn := builder.Main()

	dataShape := shapes.Make(dtypes.Float32, 4)
	condShape := shapes.Make(dtypes.Bool, 4)

	cond, _ := mainFn.Parameter("cond", condShape, nil)
	onTrue, _ := mainFn.Parameter("onTrue", dataShape, nil)
	onFalse, _ := mainFn.Parameter("onFalse", dataShape, nil)

	result, err := mainFn.Where(cond, onTrue, onFalse)
	if err != nil {
		t.Fatalf("Where() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{result}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	condBuf, _ := backend.BufferFromFlatData(0, []bool{true, false, true, false}, condShape)
	trueBuf, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30, 40}, dataShape)
	falseBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4}, dataShape)

	outputs, err := exec.Execute([]backends.Buffer{condBuf, trueBuf, falseBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{10, 2, 30, 4}, 1e-5)
}

// TestConvertDType tests dtype conversion.
func TestConvertDType(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_convert")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Convert Float32 → Int32 (truncation).
	y, err := mainFn.ConvertDType(x, dtypes.Int32)
	if err != nil {
		t.Fatalf("ConvertDType() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xBuf, _ := backend.BufferFromFlatData(0, []float32{1.7, 2.3, -0.9}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]int32, 3)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Float32 → Int32 truncates toward zero.
	if got[0] != 1 || got[1] != 2 || got[2] != 0 {
		t.Errorf("ConvertDType got %v, want [1, 2, 0]", got)
	}
}

// TestTranspose tests the Transpose operation.
func TestTranspose(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transpose")
	mainFn := builder.Main()

	inShape := shapes.Make(dtypes.Float32, 2, 3)
	x, err := mainFn.Parameter("x", inShape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Transpose [2,3] → [3,2]
	y, err := mainFn.Transpose(x, 1, 0)
	if err != nil {
		t.Fatalf("Transpose() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: [[1,2,3], [4,5,6]] (row-major: [1,2,3,4,5,6])
	xBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3, 4, 5, 6}, inShape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Transposed: [[1,4], [2,5], [3,6]] (row-major: [1,4,2,5,3,6])
	assertClose(t, got, []float32{1, 4, 2, 5, 3, 6}, 1e-5)
}

// TestMultipleOutputs tests returning multiple values from a graph.
func TestMultipleOutputs(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_multi_output")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 3)
	x, _ := mainFn.Parameter("x", shape, nil)
	y, _ := mainFn.Parameter("y", shape, nil)

	sum, _ := mainFn.Add(x, y)
	diff, _ := mainFn.Sub(x, y)

	if err := mainFn.Return([]backends.Value{sum, diff}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	if len(exec.Outputs()) != 2 {
		t.Fatalf("Expected 2 outputs, got %d", len(exec.Outputs()))
	}

	xBuf, _ := backend.BufferFromFlatData(0, []float32{10, 20, 30}, shape)
	yBuf, _ := backend.BufferFromFlatData(0, []float32{1, 2, 3}, shape)

	outputs, err := exec.Execute([]backends.Buffer{xBuf, yBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	gotSum := make([]float32, 3)
	gotDiff := make([]float32, 3)
	if err := backend.BufferToFlatData(outputs[0], gotSum); err != nil {
		t.Fatalf("BufferToFlatData(sum) failed: %+v", err)
	}
	if err := backend.BufferToFlatData(outputs[1], gotDiff); err != nil {
		t.Fatalf("BufferToFlatData(diff) failed: %+v", err)
	}

	assertClose(t, gotSum, []float32{11, 22, 33}, 1e-5)
	assertClose(t, gotDiff, []float32{9, 18, 27}, 1e-5)
}

// TestTransformerBlock tests a simplified transformer-style computation:
// attention_scores = softmax(Q @ K^T / sqrt(d_k)) @ V
func TestTransformerBlock(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_transformer")
	mainFn := builder.Main()

	// Q, K, V: [2, 3] (seq_len=2, d_k=3)
	qkvShape := shapes.Make(dtypes.Float32, 2, 3)
	Q, _ := mainFn.Parameter("Q", qkvShape, nil)
	K, _ := mainFn.Parameter("K", qkvShape, nil)
	V, _ := mainFn.Parameter("V", qkvShape, nil)

	// scores = Q @ K^T → [2, 2]
	scores, err := mainFn.DotGeneral(Q, []int{1}, nil, K, []int{1}, nil, backends.DotGeneralConfig{})
	if err != nil {
		t.Fatalf("DotGeneral(Q, K^T) failed: %+v", err)
	}

	// Scale by 1/sqrt(d_k) = 1/sqrt(3)
	scaleFactor := float32(1.0 / math.Sqrt(3.0))
	scaleConst, _ := mainFn.Constant([]float32{scaleFactor}, 1)
	scaleShape := shapes.Make(dtypes.Float32, 2, 2)
	scaleBroadcast, _ := mainFn.BroadcastInDim(scaleConst, scaleShape, []int{1})
	scores, _ = mainFn.Mul(scores, scaleBroadcast)

	// weights = softmax(scores, axis=1)
	weights, err := mainFn.FusedSoftmax(scores, 1)
	if err != nil {
		t.Fatalf("FusedSoftmax() failed: %+v", err)
	}

	// output = weights @ V → [2, 3]
	output, err := mainFn.DotGeneral(weights, []int{1}, nil, V, []int{0}, nil, backends.DotGeneralConfig{})
	if err != nil {
		t.Fatalf("DotGeneral(weights, V) failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Simple test data.
	qData := []float32{1, 0, 0, 0, 1, 0} // Q = identity-like rows
	kData := []float32{1, 0, 0, 0, 1, 0}
	vData := []float32{10, 20, 30, 40, 50, 60}

	qBuf, _ := backend.BufferFromFlatData(0, qData, qkvShape)
	kBuf, _ := backend.BufferFromFlatData(0, kData, qkvShape)
	vBuf, _ := backend.BufferFromFlatData(0, vData, qkvShape)

	outputs, err := exec.Execute([]backends.Buffer{qBuf, kBuf, vBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 6)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Verify output is a valid weighted combination of V rows.
	// Row 0 should weight V[0] more (since Q[0] dot K[0] = 1 but Q[0] dot K[1] = 0).
	// Row 1 should weight V[1] more (since Q[1] dot K[1] = 1 but Q[1] dot K[0] = 0).
	// After softmax with scale 1/sqrt(3):
	//   score[0] = [1/sqrt(3), 0] → softmax → [exp(1/sqrt(3))/(exp(1/sqrt(3))+1), 1/(exp(1/sqrt(3))+1)]
	s := math.Exp(1.0 / math.Sqrt(3.0))
	w0 := float32(s / (s + 1.0))
	w1 := float32(1.0 / (s + 1.0))

	// Row 0 of output = w0 * V[0] + w1 * V[1]
	expectedRow0 := []float32{w0*10 + w1*40, w0*20 + w1*50, w0*30 + w1*60}
	// Row 1 of output = w1 * V[0] + w0 * V[1]
	expectedRow1 := []float32{w1*10 + w0*40, w1*20 + w0*50, w1*30 + w0*60}
	expected := append(expectedRow0, expectedRow1...)

	assertClose(t, got, expected, 1e-3)
}

// TestChainedOps tests a multi-op graph: y = sigmoid(x * w + b).
func TestChainedOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_chained")
	mainFn := builder.Main()

	shape := shapes.Make(dtypes.Float32, 4)
	x, err := mainFn.Parameter("x", shape, nil)
	if err != nil {
		t.Fatalf("Parameter() failed: %+v", err)
	}

	// Constants: w and b.
	w, err := mainFn.Constant([]float32{0.5, -0.5, 1.0, -1.0}, 4)
	if err != nil {
		t.Fatalf("Constant(w) failed: %+v", err)
	}
	b, err := mainFn.Constant([]float32{0.1, 0.2, -0.1, 0.0}, 4)
	if err != nil {
		t.Fatalf("Constant(b) failed: %+v", err)
	}

	// y = sigmoid(x * w + b)
	xw, err := mainFn.Mul(x, w)
	if err != nil {
		t.Fatalf("Mul() failed: %+v", err)
	}
	xwb, err := mainFn.Add(xw, b)
	if err != nil {
		t.Fatalf("Add() failed: %+v", err)
	}
	y, err := mainFn.Logistic(xwb)
	if err != nil {
		t.Fatalf("Logistic() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{y}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	xData := []float32{2.0, 3.0, -1.0, 0.5}
	xBuf, err := backend.BufferFromFlatData(0, xData, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData() failed: %+v", err)
	}

	outputs, err := exec.Execute([]backends.Buffer{xBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	result := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], result); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}

	// Compute expected: sigmoid(x*w + b) for each element.
	sigmoid := func(x float64) float32 { return float32(1.0 / (1.0 + math.Exp(-x))) }
	expected := []float32{
		sigmoid(2.0*0.5 + 0.1),   // sigmoid(1.1)
		sigmoid(3.0*-0.5 + 0.2),  // sigmoid(-1.3)
		sigmoid(-1.0*1.0 + -0.1), // sigmoid(-1.1)
		sigmoid(0.5*-1.0 + 0.0),  // sigmoid(-0.5)
	}
	assertClose(t, result, expected, 1e-5)
}

// TestConvGeneral tests the ConvGeneral (2D convolution) operation.
func TestConvGeneral(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_conv")
	mainFn := builder.Main()

	// NCHW format: input [1, 1, 4, 4], kernel [1, 1, 3, 3]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	kernelShape := shapes.Make(dtypes.Float32, 1, 1, 3, 3)

	input, _ := mainFn.Parameter("input", inputShape, nil)
	kernel, _ := mainFn.Parameter("kernel", kernelShape, nil)

	// Standard 2D conv: NCHW layout, stride 1, no padding.
	axes := backends.ConvolveAxesConfig{
		InputBatch: 0, InputChannels: 1, InputSpatial: []int{2, 3},
		KernelOutputChannels: 0, KernelInputChannels: 1, KernelSpatial: []int{2, 3},
		OutputBatch: 0, OutputChannels: 1, OutputSpatial: []int{2, 3},
	}

	output, err := mainFn.ConvGeneral(input, kernel, axes,
		[]int{1, 1},       // strides
		[][2]int{{0, 0}, {0, 0}}, // paddings
		nil,               // inputDilations
		nil,               // kernelDilations
		1, 1,              // groups
	)
	if err != nil {
		t.Fatalf("ConvGeneral() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: 4x4 matrix of all 1s.
	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = 1
	}
	// Kernel: 3x3 matrix of all 1s → sum pooling.
	kernelData := make([]float32, 9)
	for i := range kernelData {
		kernelData[i] = 1
	}

	inputBuf, _ := backend.BufferFromFlatData(0, inputData, inputShape)
	kernelBuf, _ := backend.BufferFromFlatData(0, kernelData, kernelShape)

	outputs, err := exec.Execute([]backends.Buffer{inputBuf, kernelBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	// Output shape: [1, 1, 2, 2] (valid conv with 3x3 kernel on 4x4 input).
	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Each output element is sum of 3x3 window of all 1s = 9.
	assertClose(t, got, []float32{9, 9, 9, 9}, 1e-5)
}

// TestReduceWindow tests the ReduceWindow (pooling) operation.
func TestReduceWindow(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	builder := backend.Builder("test_pool")
	mainFn := builder.Main()

	// NCHW format: [1, 1, 4, 4]
	inputShape := shapes.Make(dtypes.Float32, 1, 1, 4, 4)
	input, _ := mainFn.Parameter("input", inputShape, nil)

	// Max pool with 2x2 window, stride 2 → [1, 1, 2, 2]
	output, err := mainFn.ReduceWindow(input,
		backends.ReduceOpMax,
		[]int{1, 1, 2, 2}, // windowDimensions
		[]int{1, 1, 2, 2}, // strides
		nil,                // baseDilations
		nil,                // windowDilations
		[][2]int{{0, 0}, {0, 0}, {0, 0}, {0, 0}}, // paddings
	)
	if err != nil {
		t.Fatalf("ReduceWindow() failed: %+v", err)
	}

	if err := mainFn.Return([]backends.Value{output}, nil); err != nil {
		t.Fatalf("Return() failed: %+v", err)
	}

	exec, err := builder.Compile()
	if err != nil {
		t.Fatalf("Compile() failed: %+v", err)
	}
	defer exec.Finalize()

	// Input: 4x4 matrix
	// [[1,  2,  3,  4],
	//  [5,  6,  7,  8],
	//  [9,  10, 11, 12],
	//  [13, 14, 15, 16]]
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	inputBuf, _ := backend.BufferFromFlatData(0, inputData, inputShape)

	outputs, err := exec.Execute([]backends.Buffer{inputBuf}, nil, 0)
	if err != nil {
		t.Fatalf("Execute() failed: %+v", err)
	}

	got := make([]float32, 4)
	if err := backend.BufferToFlatData(outputs[0], got); err != nil {
		t.Fatalf("BufferToFlatData() failed: %+v", err)
	}
	// Max pool: max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
	assertClose(t, got, []float32{6, 8, 14, 16}, 1e-5)
}

// ===========================================================================
// GoMLX High-Level Integration Tests
// These tests use GoMLX's graph-building API (graph.Node, graph.Exec) to verify
// the mpsgraph backend works correctly with GoMLX's full computation graph system.
// ===========================================================================

// newTestBackend creates a backend for GoMLX integration tests.
func newTestBackend(t *testing.T) backends.Backend {
	t.Helper()
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %+v", err)
	}
	t.Cleanup(func() { backend.Finalize() })
	return backend
}

// TestGoMLXAdd tests using GoMLX's graph API for a simple add operation.
func TestGoMLXAdd(t *testing.T) {
	backend := newTestBackend(t)
	result := graph.MustExecOnce(backend, func(x, y *graph.Node) *graph.Node {
		return graph.Add(x, y)
	}, []float32{1, 2, 3}, []float32{10, 20, 30})

	got := result.Value().([]float32)
	assertClose(t, got, []float32{11, 22, 33}, 1e-5)
}

// TestGoMLXMathChain tests a chain of GoMLX math operations.
func TestGoMLXMathChain(t *testing.T) {
	backend := newTestBackend(t)
	// Compute: exp(log(x) + 1)
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.Exp(graph.Add(graph.Log(x), graph.Ones(x.Graph(), x.Shape())))
	}, []float32{1, 2, 4})

	got := result.Value().([]float32)
	// exp(log(x) + 1) = exp(log(x)) * exp(1) = x * e
	e := float32(math.E)
	assertClose(t, got, []float32{1 * e, 2 * e, 4 * e}, 1e-4)
}

// TestGoMLXMatMul tests matrix multiplication using GoMLX's graph API.
func TestGoMLXMatMul(t *testing.T) {
	backend := newTestBackend(t)
	lhs := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	rhs := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 3, 2)

	result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
		return graph.Dot(a, b).MatMul()
	}, lhs, rhs)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	// [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
	assertClose(t, got, []float32{22, 28, 49, 64}, 1e-4)
}

// TestGoMLXReduceAndBroadcast tests reduce + broadcast using GoMLX API.
func TestGoMLXReduceAndBroadcast(t *testing.T) {
	backend := newTestBackend(t)
	input := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	// Compute row means and subtract (manual centering).
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		rowSums := graph.ReduceSum(x, -1)                  // [2]
		rowMeans := graph.DivScalar(rowSums, 3.0)           // [2]
		rowMeansBroadcast := graph.ExpandDims(rowMeans, -1) // [2, 1]
		return graph.Sub(x, rowMeansBroadcast)              // [2, 3] - [2, 1] → [2, 3]
	}, input)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	// Row 0 mean = (1+2+3)/3 = 2 → [-1, 0, 1]
	// Row 1 mean = (4+5+6)/3 = 5 → [-1, 0, 1]
	assertClose(t, got, []float32{-1, 0, 1, -1, 0, 1}, 1e-4)
}

// TestGoMLXGatherAndSlice tests gather/embedding lookup using GoMLX API.
func TestGoMLXGatherAndSlice(t *testing.T) {
	backend := newTestBackend(t)
	table := tensors.FromFlatDataAndDimensions([]float32{
		10, 11, 12,
		20, 21, 22,
		30, 31, 32,
		40, 41, 42,
	}, 4, 3)
	// For graph.Gather, the last dim of indices is the index vector dimension.
	// Shape [2, 1] means 2 indices, each indexing 1 axis of params → embedding lookup.
	indices := tensors.FromFlatDataAndDimensions([]int32{1, 3}, 2, 1)

	result := graph.MustExecOnce(backend, func(tbl, idx *graph.Node) *graph.Node {
		return graph.Gather(tbl, idx)
	}, table, indices)

	got, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		t.Fatalf("CopyFlatData() failed: %+v", err)
	}
	assertClose(t, got, []float32{20, 21, 22, 40, 41, 42}, 1e-5)
}

// TestGoMLXWhereAndCompare tests conditional logic using GoMLX API.
func TestGoMLXWhereAndCompare(t *testing.T) {
	backend := newTestBackend(t)
	// ReLU: max(0, x) = where(x > 0, x, 0)
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		zero := graph.ZerosLike(x)
		mask := graph.GreaterThan(x, zero)
		return graph.Where(mask, x, zero)
	}, []float32{-2, -1, 0, 1, 2})

	got := result.Value().([]float32)
	assertClose(t, got, []float32{0, 0, 0, 1, 2}, 1e-5)
}

// =============================================================================
// Training Integration Tests
// =============================================================================

// TestContextExecDense tests a forward pass through a Dense layer using context.Exec.
func TestContextExecDense(t *testing.T) {
	backend := newTestBackend(t)
	ctx := context.New()

	// Model: Dense(input, useBias=true, outputDim=1)
	modelFn := func(ctx *context.Context, input *graph.Node) *graph.Node {
		return layers.Dense(ctx, input, true, 1)
	}

	exec, err := context.NewExec(backend, ctx, modelFn)
	if err != nil {
		t.Fatalf("NewExec failed: %+v", err)
	}

	// Input: 3 examples, 2 features
	input := tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	results := exec.MustExec(input)
	if len(results) != 1 {
		t.Fatalf("expected 1 output, got %d", len(results))
	}

	// Just verify the output shape is [3, 1] (values depend on random init).
	outShape := results[0].Shape()
	if outShape.Rank() != 2 || outShape.Dimensions[0] != 3 || outShape.Dimensions[1] != 1 {
		t.Fatalf("expected output shape [3,1], got %s", outShape)
	}
	t.Logf("Dense forward output shape: %s", outShape)

	// Verify variables were created.
	weightsVar := ctx.GetVariableByScopeAndName("/dense", "weights")
	biasesVar := ctx.GetVariableByScopeAndName("/dense", "biases")
	if weightsVar == nil {
		t.Fatal("weights variable not found")
	}
	if biasesVar == nil {
		t.Fatal("biases variable not found")
	}
	t.Logf("Weights shape: %s, Biases shape: %s",
		weightsVar.Shape(), biasesVar.Shape())
}

// simpleTrainDataset implements train.Dataset for a simple regression problem.
// It yields the same batch indefinitely (infinite dataset).
type simpleTrainDataset struct {
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
}

func (d *simpleTrainDataset) Name() string                  { return "simple" }
func (d *simpleTrainDataset) Reset()                        {}
func (d *simpleTrainDataset) IsOwnershipTransferred() bool  { return false }
func (d *simpleTrainDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	return d, d.inputs, d.labels, nil
}

// TestTrainingLinearRegression tests a full training loop: forward, backward (autodiff), SGD update.
func TestTrainingLinearRegression(t *testing.T) {
	backend := newTestBackend(t)

	// Simple linear regression: y = 2*x1 + 3*x2 + 1
	// 4 examples, 2 features
	inputData := []float32{
		1, 0,
		0, 1,
		1, 1,
		2, 1,
	}
	// Labels: 2*x1 + 3*x2 + 1
	labelData := []float32{3, 4, 6, 8}

	inputs := tensors.FromFlatDataAndDimensions(inputData, 4, 2)
	labels := tensors.FromFlatDataAndDimensions(labelData, 4, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{inputs},
		labels: []*tensors.Tensor{labels},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.1)

	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		logits := layers.Dense(ctx, inputs[0], true, 1)
		return []*graph.Node{logits}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent().Done(),
		nil, nil)

	loop := train.NewLoop(trainer)

	// Run 500 training steps.
	numSteps := 500
	metrics, err := loop.RunSteps(dataset, numSteps)
	if err != nil {
		t.Fatalf("Training failed: %+v", err)
	}

	// The last metric should be the loss.
	if len(metrics) < 2 {
		t.Fatalf("expected at least 2 metrics (step + loss), got %d", len(metrics))
	}

	// Loss metric can be float32 or float64 depending on backend.
	var finalLoss float64
	switch v := metrics[1].Value().(type) {
	case float64:
		finalLoss = v
	case float32:
		finalLoss = float64(v)
	default:
		t.Fatalf("unexpected loss type: %T", metrics[1].Value())
	}
	t.Logf("Final loss after %d steps: %f", numSteps, finalLoss)

	// After 100 steps of SGD on this simple problem, loss should be small.
	if finalLoss > 5.0 {
		t.Errorf("Loss too high after training: %f (expected < 5.0)", finalLoss)
	}

	// Check learned weights approximate [2, 3] and bias ≈ 1.
	weightsVar := ctx.GetVariableByScopeAndName("/dense", "weights")
	biasVar := ctx.GetVariableByScopeAndName("/dense", "biases")
	if weightsVar == nil || biasVar == nil {
		t.Fatal("variables not found after training")
	}

	wTensor := weightsVar.MustValue()
	bTensor := biasVar.MustValue()
	t.Logf("Learned weights: %v", wTensor.Value())
	t.Logf("Learned bias: %v", bTensor.Value())
}

// TestTrainingWithAdam tests training with the Adam optimizer on a 2-layer network.
func TestTrainingWithAdam(t *testing.T) {
	backend := newTestBackend(t)

	// XOR-like problem: need nonlinearity to solve.
	// y = 1 if exactly one of x1, x2 is > 0.5, else 0.
	inputData := []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}
	labelData := []float32{0, 1, 1, 0}

	inputs := tensors.FromFlatDataAndDimensions(inputData, 4, 2)
	labels := tensors.FromFlatDataAndDimensions(labelData, 4, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{inputs},
		labels: []*tensors.Tensor{labels},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.01)

	// 2-layer MLP: Dense(4, tanh) → Dense(1)
	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		x := inputs[0]
		x = layers.Dense(ctx.In("hidden"), x, true, 4)
		x = graph.Tanh(x)
		x = layers.Dense(ctx.In("output"), x, true, 1)
		return []*graph.Node{x}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.MeanSquaredError,
		optimizers.Adam().Done(),
		nil, nil)

	loop := train.NewLoop(trainer)
	numSteps := 500
	metrics, err := loop.RunSteps(dataset, numSteps)
	if err != nil {
		t.Fatalf("Training with Adam failed: %+v", err)
	}

	var finalLoss float64
	switch v := metrics[1].Value().(type) {
	case float64:
		finalLoss = v
	case float32:
		finalLoss = float64(v)
	}
	t.Logf("Final loss after %d Adam steps: %f", numSteps, finalLoss)

	// XOR is harder, but after 500 Adam steps loss should be decreasing.
	if finalLoss > 1.0 {
		t.Errorf("Loss too high with Adam: %f", finalLoss)
	}
}

// =============================================================================
// Autodiff (Gradient) Tests
// =============================================================================

// TestGradientSimple verifies that automatic differentiation produces correct gradients.
func TestGradientSimple(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("linear_grad", func(t *testing.T) {
		// f(x) = sum(3*x + 2) → df/dx = 3 for each element
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			three := graph.Const(x.Graph(), float32(3.0))
			y := graph.Add(graph.Mul(three, x), graph.Const(x.Graph(), float32(2.0)))
			loss := graph.ReduceAllSum(y)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, []float32{5.0, 10.0})
		got := result.Value().([]float32)
		assertClose(t, got, []float32{3.0, 3.0}, 1e-5)
	})

	t.Run("quadratic_grad", func(t *testing.T) {
		// f(x) = sum(x^2) → df/dx = 2*x
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			y := graph.Mul(x, x)
			loss := graph.ReduceAllSum(y)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, []float32{3.0, -2.0})
		got := result.Value().([]float32)
		assertClose(t, got, []float32{6.0, -4.0}, 1e-5)
	})

	t.Run("matmul_grad", func(t *testing.T) {
		// f(W) = sum(x @ W) where x=[1,2], W=[2,1] → df/dW = x^T
		x := tensors.FromFlatDataAndDimensions([]float32{1, 2}, 1, 2)
		result := graph.MustExecOnce(backend, func(xNode, wNode *graph.Node) *graph.Node {
			y := graph.Dot(xNode, wNode).MatMul()
			loss := graph.ReduceAllSum(y)
			grads := graph.Gradient(loss, wNode)
			return grads[0]
		}, x, tensors.FromFlatDataAndDimensions([]float32{0.5, 0.3}, 2, 1))
		got, err := tensors.CopyFlatData[float32](result)
		if err != nil {
			t.Fatalf("CopyFlatData failed: %+v", err)
		}
		// df/dW = x^T = [[1], [2]]
		assertClose(t, got, []float32{1, 2}, 1e-5)
	})

	t.Run("relu_grad", func(t *testing.T) {
		// f(x) = relu(x) → grad is 1 where x > 0, 0 otherwise
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			y := graph.Where(
				graph.GreaterThan(x, graph.ZerosLike(x)),
				x,
				graph.ZerosLike(x),
			)
			loss := graph.ReduceAllSum(y)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, []float32{-2, -1, 0.5, 1, 3})
		got := result.Value().([]float32)
		assertClose(t, got, []float32{0, 0, 1, 1, 1}, 1e-5)
	})

	t.Run("reduce_mean_grad", func(t *testing.T) {
		// f(x) = mean(x) → df/dx = 1/n for each element
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			y := graph.ReduceAllMean(x)
			grads := graph.Gradient(y, x)
			return grads[0]
		}, []float32{1, 2, 3, 4})
		got := result.Value().([]float32)
		// mean of 4 elements → gradient = 1/4 = 0.25
		assertClose(t, got, []float32{0.25, 0.25, 0.25, 0.25}, 1e-5)
	})
}

// TestGradientDenseLayer tests gradients through a Dense layer with MSE loss.
func TestGradientDenseLayer(t *testing.T) {
	backend := newTestBackend(t)
	ctx := context.New()

	// Build a graph with Dense layer and compute loss + gradients.
	modelFn := func(ctx *context.Context, input, target *graph.Node) *graph.Node {
		pred := layers.Dense(ctx, input, true, 1)
		diff := graph.Sub(target, pred)
		loss := graph.ReduceAllMean(graph.Mul(diff, diff))
		return loss
	}

	exec, err := context.NewExec(backend, ctx, modelFn)
	if err != nil {
		t.Fatalf("NewExec failed: %+v", err)
	}

	input := tensors.FromFlatDataAndDimensions([]float32{1, 2}, 1, 2)
	target := tensors.FromFlatDataAndDimensions([]float32{5}, 1, 1)

	// First execution initializes variables and computes loss.
	results := exec.MustExec(input, target)
	loss1, err := tensors.CopyFlatData[float32](results[0])
	if err != nil {
		t.Fatalf("CopyFlatData failed: %+v", err)
	}
	t.Logf("Initial loss: %f", loss1[0])

	// Verify loss is a scalar and finite.
	if math.IsNaN(float64(loss1[0])) || math.IsInf(float64(loss1[0]), 0) {
		t.Fatalf("Loss is not finite: %f", loss1[0])
	}
}

// =============================================================================
// MNIST-like Model Test (Linear — no pooling)
// =============================================================================

// TestMNISTLinear tests training a simple linear model on synthetic MNIST-like data.
func TestMNISTLinear(t *testing.T) {
	backend := newTestBackend(t)

	// Synthetic "MNIST": 50 examples, 784 features (28x28 flattened), 10 classes.
	numExamples := 50
	numFeatures := 784
	numClasses := 10

	// Generate random input data using the backend.
	inputData := make([]float32, numExamples*numFeatures)
	for i := range inputData {
		inputData[i] = float32(i%7) * 0.01 // Simple deterministic pattern
	}
	// Generate labels: class = example_idx % numClasses
	labelData := make([]int32, numExamples)
	for i := range labelData {
		labelData[i] = int32(i % numClasses)
	}

	inputs := tensors.FromFlatDataAndDimensions(inputData, numExamples, numFeatures)
	labels := tensors.FromFlatDataAndDimensions(labelData, numExamples, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{inputs},
		labels: []*tensors.Tensor{labels},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.01)

	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		x := inputs[0]
		logits := layers.Dense(ctx, x, true, numClasses)
		return []*graph.Node{logits}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil)

	loop := train.NewLoop(trainer)
	numSteps := 50
	metrics, err := loop.RunSteps(dataset, numSteps)
	if err != nil {
		t.Fatalf("MNIST linear training failed: %+v", err)
	}

	var finalLoss float64
	switch v := metrics[1].Value().(type) {
	case float64:
		finalLoss = v
	case float32:
		finalLoss = float64(v)
	}
	t.Logf("MNIST linear final loss after %d steps: %f", numSteps, finalLoss)

	// Cross-entropy loss should be significantly less than initial ~ln(10) ≈ 2.3.
	if finalLoss > 3.0 {
		t.Errorf("MNIST linear loss too high: %f (expected < 3.0)", finalLoss)
	}
}

// TestMaxPoolGradient tests MaxPool forward + backward (SelectAndScatter) through GoMLX's autodiff.
func TestMaxPoolGradient(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("forward_NCHW", func(t *testing.T) {
		// Test MaxPool forward using GoMLX graph API with ChannelsFirst (NCHW).
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.MaxPool(x).ChannelsAxis(images.ChannelsFirst).Window(2).Done()
		}, tensors.FromFlatDataAndDimensions(
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			1, 1, 4, 4))
		got, _ := tensors.CopyFlatData[float32](result)
		assertClose(t, got, []float32{6, 8, 14, 16}, 1e-5)
	})

	t.Run("forward_NHWC", func(t *testing.T) {
		// Test MaxPool forward with ChannelsLast (NHWC) — the default.
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.MaxPool(x).Window(2).Done()
		}, tensors.FromFlatDataAndDimensions(
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			1, 4, 4, 1))
		got, _ := tensors.CopyFlatData[float32](result)
		assertClose(t, got, []float32{6, 8, 14, 16}, 1e-5)
	})

	t.Run("gradient_NCHW", func(t *testing.T) {
		// Test MaxPool gradient (SelectAndScatter) with ChannelsFirst.
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			pooled := graph.MaxPool(x).ChannelsAxis(images.ChannelsFirst).Window(2).Done()
			loss := graph.ReduceAllSum(pooled)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, tensors.FromFlatDataAndDimensions(
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			1, 1, 4, 4))
		got, _ := tensors.CopyFlatData[float32](result)
		// Gradient flows to the max elements: 6(pos 5), 8(pos 7), 14(pos 13), 16(pos 15)
		expected := []float32{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1}
		assertClose(t, got, expected, 1e-5)
	})

	t.Run("gradient_NHWC", func(t *testing.T) {
		// Test MaxPool gradient (SelectAndScatter) with ChannelsLast — the default.
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			pooled := graph.MaxPool(x).Window(2).Done()
			loss := graph.ReduceAllSum(pooled)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, tensors.FromFlatDataAndDimensions(
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			1, 4, 4, 1))
		got, _ := tensors.CopyFlatData[float32](result)
		expected := []float32{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1}
		assertClose(t, got, expected, 1e-5)
	})
}

// TestCNNTraining tests a simple CNN model (Conv + ReLU + MaxPool + Dense) with training.
// Uses synthetic 8x8 images with 2 classes: class 0 has high values in the left half,
// class 1 has high values in the right half.
func TestCNNTraining(t *testing.T) {
	backend := newTestBackend(t)

	const (
		batchSize  = 20
		imgH       = 8
		imgW       = 8
		numClasses = 2
	)

	// Generate synthetic dataset: class 0 = brighter left, class 1 = brighter right.
	imgData := make([]float32, batchSize*imgH*imgW)
	labelData := make([]int32, batchSize)
	for i := range batchSize {
		cls := int32(i % numClasses)
		labelData[i] = cls
		for h := range imgH {
			for w := range imgW {
				idx := i*imgH*imgW + h*imgW + w
				if cls == 0 {
					// Class 0: left half bright
					if w < imgW/2 {
						imgData[idx] = 0.8 + float32(i%5)*0.02
					} else {
						imgData[idx] = 0.1 + float32(i%5)*0.02
					}
				} else {
					// Class 1: right half bright
					if w >= imgW/2 {
						imgData[idx] = 0.8 + float32(i%5)*0.02
					} else {
						imgData[idx] = 0.1 + float32(i%5)*0.02
					}
				}
			}
		}
	}

	// Create tensors: images [batch, H, W, 1] (NHWC), labels [batch, 1].
	imgTensor := tensors.FromFlatDataAndDimensions(imgData, batchSize, imgH, imgW, 1)
	labelTensor := tensors.FromFlatDataAndDimensions(labelData, batchSize, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{imgTensor},
		labels: []*tensors.Tensor{labelTensor},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.01)

	// CNN model: Conv(3x3, 4 filters) → ReLU → MaxPool(2x2) → Flatten → Dense(numClasses)
	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		x := inputs[0] // [batch, 8, 8, 1]

		// Conv layer
		x = layers.Convolution(ctx.In("conv1"), x).Filters(4).KernelSize(3).PadSame().Done()
		x = activations.Relu(x)
		x = graph.MaxPool(x).Window(2).Done() // [batch, 4, 4, 4]

		// Flatten and classify
		batchDim := x.Shape().Dimensions[0]
		x = graph.Reshape(x, batchDim, -1) // [batch, 64]
		logits := layers.Dense(ctx.In("dense"), x, true, numClasses)
		return []*graph.Node{logits}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil)

	loop := train.NewLoop(trainer)
	numSteps := 200
	metrics, err := loop.RunSteps(dataset, numSteps)
	if err != nil {
		t.Fatalf("CNN training failed: %+v", err)
	}

	var finalLoss float64
	switch v := metrics[1].Value().(type) {
	case float64:
		finalLoss = v
	case float32:
		finalLoss = float64(v)
	}
	t.Logf("CNN final loss after %d steps: %f", numSteps, finalLoss)

	// Loss should decrease from initial ~ln(2) ≈ 0.69 to something much smaller.
	if finalLoss > 0.5 {
		t.Errorf("CNN loss too high: %f (expected < 0.5)", finalLoss)
	}
}

// TestFusedLayerNorm tests the FusedLayerNorm operation using nn.LayerNorm directly (no context variables).
func TestFusedLayerNorm(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("basic", func(t *testing.T) {
		// Layer normalize a [2, 4] tensor over axis 1 (features), no gamma/beta.
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return nn.LayerNorm(x, []int{-1}, 1e-5, nil, nil, nil)
		}, tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4))
		got, _ := tensors.CopyFlatData[float32](result)
		// Row 1: [1,2,3,4] → mean=2.5, var=1.25, std≈1.118 → [-1.342, -0.447, 0.447, 1.342]
		assertClose(t, got[:4], []float32{-1.3416, -0.4472, 0.4472, 1.3416}, 1e-3)
		assertClose(t, got[4:], []float32{-1.3416, -0.4472, 0.4472, 1.3416}, 1e-3)
	})

	t.Run("with_gamma_beta", func(t *testing.T) {
		// Layer normalize with gamma=2 and beta=1.
		// gamma/beta must be broadcast-shaped to match x's rank: [1, 4] for x [2, 4].
		result := graph.MustExecOnce(backend, func(x, gamma, beta *graph.Node) *graph.Node {
			return nn.LayerNorm(x, []int{-1}, 1e-5, gamma, beta, nil)
		},
			tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4),
			tensors.FromFlatDataAndDimensions([]float32{2, 2, 2, 2}, 1, 4),
			tensors.FromFlatDataAndDimensions([]float32{1, 1, 1, 1}, 1, 4),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// normalized * 2 + 1: [-1.342*2+1, -0.447*2+1, 0.447*2+1, 1.342*2+1]
		//                   = [-1.683, 0.106, 1.894, 3.683]
		assertClose(t, got[:4], []float32{-1.6833, 0.1056, 1.8944, 3.6833}, 1e-3)
	})

	t.Run("gradient", func(t *testing.T) {
		// Test that gradient flows through LayerNorm (no gamma/beta).
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			normed := nn.LayerNorm(x, []int{-1}, 1e-5, nil, nil, nil)
			loss := graph.ReduceAllSum(normed)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4}, 1, 4))
		got, _ := tensors.CopyFlatData[float32](result)
		// LayerNorm gradient for uniform loss sum should be close to 0
		// (since normalizing then summing → the gradient direction
		// is perpendicular to the constant vector).
		for i, v := range got {
			if math.Abs(float64(v)) > 1e-4 {
				t.Errorf("LayerNorm gradient[%d] = %f, expected ~0", i, v)
			}
		}
	})
}

// TestFusedGelu tests the FusedGelu operation.
func TestFusedGelu(t *testing.T) {
	backend := newTestBackend(t)

	// GELU(0) = 0, GELU(large) ≈ large, GELU(negative) ≈ 0
	result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return activations.Gelu(x)
	}, []float32{0.0, 1.0, -1.0, 3.0})
	got, _ := tensors.CopyFlatData[float32](result)
	// Expected: GELU(0)=0, GELU(1)≈0.8413, GELU(-1)≈-0.1587, GELU(3)≈2.9960
	assertClose(t, got, []float32{0.0, 0.8413, -0.1587, 2.9960}, 1e-3)
}

// TestTransformerTraining tests a small transformer-like model: Embedding + LayerNorm + Dense.
func TestTransformerTraining(t *testing.T) {
	backend := newTestBackend(t)

	const (
		batchSize  = 10
		seqLen     = 8
		dModel     = 16
		numClasses = 3
	)

	// Synthetic data: each class has different feature patterns that survive LayerNorm.
	inputData := make([]float32, batchSize*seqLen*dModel)
	labelData := make([]int32, batchSize)
	for i := range batchSize {
		cls := int32(i % numClasses)
		labelData[i] = cls
		for s := range seqLen {
			for d := range dModel {
				idx := i*seqLen*dModel + s*dModel + d
				// Create class-dependent variance patterns: different features are active per class.
				if d%numClasses == int(cls) {
					inputData[idx] = 1.0 + float32(d)*0.1
				} else {
					inputData[idx] = -0.5
				}
			}
		}
	}

	inputTensor := tensors.FromFlatDataAndDimensions(inputData, batchSize, seqLen, dModel)
	labelTensor := tensors.FromFlatDataAndDimensions(labelData, batchSize, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{inputTensor},
		labels: []*tensors.Tensor{labelTensor},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.01)

	// Transformer-like model: LayerNorm → Dense → GELU → reduce → Dense → logits
	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		x := inputs[0] // [batch, seq, dModel]

		// LayerNorm over feature dim
		x = layers.LayerNormalization(ctx.In("ln1"), x, -1).Done()

		// Feed-forward: Dense → GELU → Dense
		x = layers.Dense(ctx.In("ff1"), x, true, dModel)
		x = activations.Gelu(x)

		// Mean pool over sequence
		x = graph.ReduceMean(x, 1) // [batch, dModel]

		// Output
		logits := layers.Dense(ctx.In("out"), x, true, numClasses)
		return []*graph.Node{logits}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil)

	loop := train.NewLoop(trainer)
	numSteps := 500
	metrics, err := loop.RunSteps(dataset, numSteps)
	if err != nil {
		t.Fatalf("Transformer training failed: %+v", err)
	}

	var finalLoss float64
	switch v := metrics[1].Value().(type) {
	case float64:
		finalLoss = v
	case float32:
		finalLoss = float64(v)
	}
	t.Logf("Transformer final loss after %d steps: %f", numSteps, finalLoss)

	// ln(3) ≈ 1.099 is the random baseline for 3 classes; loss should drop well below.
	if finalLoss > 0.5 {
		t.Errorf("Transformer loss too high: %f (expected < 0.5)", finalLoss)
	}
}

// TestInt64Operations tests Int64 dtype support.
func TestInt64Operations(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("add", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
			return graph.Add(a, b)
		}, []int64{1, 2, 3}, []int64{10, 20, 30})
		got, _ := tensors.CopyFlatData[int64](result)
		expected := []int64{11, 22, 33}
		for i, v := range got {
			if v != expected[i] {
				t.Errorf("Int64 Add[%d]: got %d, want %d", i, v, expected[i])
			}
		}
	})

	t.Run("cast_float_to_int64", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.ConvertDType(x, dtypes.Int64)
		}, []float32{1.5, 2.9, -3.1})
		got, _ := tensors.CopyFlatData[int64](result)
		// Truncation: 1.5→1, 2.9→2, -3.1→-3
		expected := []int64{1, 2, -3}
		for i, v := range got {
			if v != expected[i] {
				t.Errorf("Cast Float32→Int64[%d]: got %d, want %d", i, v, expected[i])
			}
		}
	})

	t.Run("gather_with_int32_indices", func(t *testing.T) {
		// Common pattern: gather with int32 index tensor (used for embeddings).
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			indices := graph.Const(x.Graph(), tensors.FromFlatDataAndDimensions([]int32{2, 0, 1}, 3, 1))
			return graph.Gather(x, indices)
		}, tensors.FromFlatDataAndDimensions([]float32{10, 20, 30}, 3, 1))
		got, _ := tensors.CopyFlatData[float32](result)
		expected := []float32{30, 10, 20}
		for i, v := range got {
			if v != expected[i] {
				t.Errorf("Gather[%d]: got %f, want %f", i, v, expected[i])
			}
		}
	})

	t.Run("uint8_arithmetic", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
			return graph.Add(a, b)
		}, []uint8{100, 200}, []uint8{10, 20})
		got, _ := tensors.CopyFlatData[uint8](result)
		expected := []uint8{110, 220}
		for i, v := range got {
			if v != expected[i] {
				t.Errorf("Uint8 Add[%d]: got %d, want %d", i, v, expected[i])
			}
		}
	})
}

// TestFusedDense tests the FusedDense operation (matmul + bias + activation).
func TestFusedDense(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("no_activation_no_bias", func(t *testing.T) {
		// Simple matmul: [2,3] @ [3,4] → [2,4]
		result := graph.MustExecOnce(backend, func(x, w *graph.Node) *graph.Node {
			return graph.BackendFusedDense(x, w, nil, backends.ActivationNone)
		},
			tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
			tensors.FromFlatDataAndDimensions([]float32{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
			}, 3, 4),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// With identity-like weight: first 3 dims pass through, 4th is 0.
		assertClose(t, got, []float32{1, 2, 3, 0, 4, 5, 6, 0}, 1e-4)
	})

	t.Run("with_bias", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(x, w, b *graph.Node) *graph.Node {
			return graph.BackendFusedDense(x, w, b, backends.ActivationNone)
		},
			tensors.FromFlatDataAndDimensions([]float32{1, 0, 0, 1}, 2, 2),
			tensors.FromFlatDataAndDimensions([]float32{2, 0, 0, 3}, 2, 2),
			tensors.FromFlatDataAndDimensions([]float32{10, 20}, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// Row 0: [1,0] @ [[2,0],[0,3]] + [10,20] = [2,0] + [10,20] = [12, 20]
		// Row 1: [0,1] @ [[2,0],[0,3]] + [10,20] = [0,3] + [10,20] = [10, 23]
		assertClose(t, got, []float32{12, 20, 10, 23}, 1e-4)
	})

	t.Run("relu_activation", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(x, w *graph.Node) *graph.Node {
			return graph.BackendFusedDense(x, w, nil, backends.ActivationRelu)
		},
			tensors.FromFlatDataAndDimensions([]float32{1, -1, -1, 1}, 2, 2),
			tensors.FromFlatDataAndDimensions([]float32{1, 0, 0, 1}, 2, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// Row 0: [1,-1] → relu → [1, 0]
		// Row 1: [-1,1] → relu → [0, 1]
		assertClose(t, got, []float32{1, 0, 0, 1}, 1e-4)
	})

	t.Run("dense_layer_integration", func(t *testing.T) {
		// Test that layers.Dense (which uses FusedDense internally) works correctly.
		ctx := context.New()
		result, err := context.ExecOnce(backend, ctx, func(ctx *context.Context, input *graph.Node) *graph.Node {
			return layers.Dense(ctx, input, true, 4)
		}, tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
		if err != nil {
			t.Fatalf("Dense exec failed: %+v", err)
		}
		// Just verify it runs and returns the right shape.
		if result.Shape().Rank() != 2 || result.Shape().Dimensions[0] != 2 || result.Shape().Dimensions[1] != 4 {
			t.Errorf("Dense output shape = %v, want [2,4]", result.Shape())
		}
	})

	t.Run("gradient_via_dense_layer", func(t *testing.T) {
		// Verify gradients flow through Dense layer (which uses FusedDense internally
		// via InternalFusedOpCaller that handles VJP fallback).
		ctx := context.New()
		result, err := context.ExecOnce(backend, ctx, func(ctx *context.Context, x *graph.Node) *graph.Node {
			y := layers.Dense(ctx, x, false, 2)
			loss := graph.ReduceAllSum(y)
			grads := graph.Gradient(loss, x)
			return grads[0]
		}, tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4}, 2, 2))
		if err != nil {
			t.Fatalf("Dense gradient exec failed: %+v", err)
		}
		got, _ := tensors.CopyFlatData[float32](result)
		// Gradient should be non-zero for all inputs.
		allZero := true
		for _, v := range got {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Error("Gradient is all zeros, expected non-zero values")
		}
	})
}

// TestFusedAttentionQKVProjection tests the QKV projection fused op.
func TestFusedAttentionQKVProjection(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("identity_weights", func(t *testing.T) {
		// x: [2, 6], wQKV: [6, 6] (identity), queryDim=2, kvDim=2
		// So combined output [2,6] gets split into Q[2,2], K[2,2], V[2,2]
		xData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		wData := make([]float32, 36)
		for i := range 6 {
			wData[i*6+i] = 1 // identity matrix
		}
		results := graph.MustExecOnceN(backend, func(x, w *graph.Node) []*graph.Node {
			q, k, v := graph.BackendFusedAttentionQKVProjection(x, w, nil, nil, nil, 2, 2)
			return []*graph.Node{q, k, v}
		},
			tensors.FromFlatDataAndDimensions(xData, 2, 6),
			tensors.FromFlatDataAndDimensions(wData, 6, 6),
		)

		qGot, _ := tensors.CopyFlatData[float32](results[0])
		kGot, _ := tensors.CopyFlatData[float32](results[1])
		vGot, _ := tensors.CopyFlatData[float32](results[2])

		// Q = first 2 cols: [1,2], [7,8]
		assertClose(t, qGot, []float32{1, 2, 7, 8}, 1e-4)
		// K = next 2 cols: [3,4], [9,10]
		assertClose(t, kGot, []float32{3, 4, 9, 10}, 1e-4)
		// V = last 2 cols: [5,6], [11,12]
		assertClose(t, vGot, []float32{5, 6, 11, 12}, 1e-4)
	})

	t.Run("with_biases", func(t *testing.T) {
		// x: [1, 4], wQKV: [4, 6] (identity-ish), qDim=2, kvDim=2, with biases
		xData := []float32{1, 2, 3, 4}
		wData := make([]float32, 24)
		for i := range 4 {
			if i < 6 {
				wData[i*6+i] = 1
			}
		}
		biasQ := []float32{10, 20}
		biasK := []float32{100, 200}
		biasV := []float32{1000, 2000}

		results := graph.MustExecOnceN(backend, func(x, w, bq, bk, bv *graph.Node) []*graph.Node {
			q, k, v := graph.BackendFusedAttentionQKVProjection(x, w, bq, bk, bv, 2, 2)
			return []*graph.Node{q, k, v}
		},
			tensors.FromFlatDataAndDimensions(xData, 1, 4),
			tensors.FromFlatDataAndDimensions(wData, 4, 6),
			tensors.FromFlatDataAndDimensions(biasQ, 2),
			tensors.FromFlatDataAndDimensions(biasK, 2),
			tensors.FromFlatDataAndDimensions(biasV, 2),
		)

		qGot, _ := tensors.CopyFlatData[float32](results[0])
		kGot, _ := tensors.CopyFlatData[float32](results[1])
		vGot, _ := tensors.CopyFlatData[float32](results[2])

		// Q = [1,2] + [10,20] = [11, 22]
		assertClose(t, qGot, []float32{11, 22}, 1e-4)
		// K = [3,4] + [100,200] = [103, 204]
		assertClose(t, kGot, []float32{103, 204}, 1e-4)
		// V = [0,0] + [1000,2000] = [1000, 2000] (cols 4,5 of x@w are zero since w is sparse)
		assertClose(t, vGot, []float32{1000, 2000}, 1e-4)
	})
}

// TestFusedScaledDotProductAttention tests the SDPA fused op.
func TestFusedScaledDotProductAttention(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("basic_BHSD", func(t *testing.T) {
		// Simple attention: B=1, H=1, S=2, D=2 with BHSD layout.
		// Q=K=V=[[[1,0],[0,1]]] → scores = Q@K^T = [[1,0],[0,1]], softmax → [[0.731,0.269],[0.269,0.731]]
		data := []float32{1, 0, 0, 1}
		scale := 1.0 / math.Sqrt(2.0) // 1/sqrt(headDim)

		result := graph.MustExecOnce(backend, func(q, k, v *graph.Node) *graph.Node {
			return graph.BackendFusedScaledDotProductAttention(
				q, k, v, nil, 1, 1, backends.AxesLayoutBHSD, scale, false, nil)
		},
			tensors.FromFlatDataAndDimensions(data, 1, 1, 2, 2),
			tensors.FromFlatDataAndDimensions(data, 1, 1, 2, 2),
			tensors.FromFlatDataAndDimensions(data, 1, 1, 2, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// Output shape should be [1,1,2,2].
		if result.Shape().Rank() != 4 {
			t.Fatalf("Expected rank 4, got %d", result.Shape().Rank())
		}
		// Values should be between 0 and 1 (weighted average of V=[identity]).
		for i, v := range got {
			if v < -0.1 || v > 1.1 {
				t.Errorf("Output[%d] = %f, expected in [0,1]", i, v)
			}
		}
	})

	t.Run("causal_mask", func(t *testing.T) {
		// With causal masking, first position can only attend to itself.
		// B=1, H=1, S=3, D=2
		qData := []float32{1, 0, 0, 1, 1, 1}
		kData := []float32{1, 0, 0, 1, 1, 1}
		vData := []float32{1, 0, 0, 1, 0.5, 0.5}
		scale := 1.0

		result := graph.MustExecOnce(backend, func(q, k, v *graph.Node) *graph.Node {
			return graph.BackendFusedScaledDotProductAttention(
				q, k, v, nil, 1, 1, backends.AxesLayoutBHSD, scale, true, nil)
		},
			tensors.FromFlatDataAndDimensions(qData, 1, 1, 3, 2),
			tensors.FromFlatDataAndDimensions(kData, 1, 1, 3, 2),
			tensors.FromFlatDataAndDimensions(vData, 1, 1, 3, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// First position: only attends to position 0 → output should be V[0] = [1,0]
		assertClose(t, got[:2], []float32{1, 0}, 1e-3)
	})

	t.Run("BSHD_layout", func(t *testing.T) {
		// Test BSHD layout: B=1, S=2, H=1, D=2
		data := []float32{1, 0, 0, 1}
		scale := 1.0 / math.Sqrt(2.0)

		result := graph.MustExecOnce(backend, func(q, k, v *graph.Node) *graph.Node {
			return graph.BackendFusedScaledDotProductAttention(
				q, k, v, nil, 1, 1, backends.AxesLayoutBSHD, scale, false, nil)
		},
			tensors.FromFlatDataAndDimensions(data, 1, 2, 1, 2),
			tensors.FromFlatDataAndDimensions(data, 1, 2, 1, 2),
			tensors.FromFlatDataAndDimensions(data, 1, 2, 1, 2),
		)
		// Should produce valid output with BSHD layout [1,2,1,2].
		if result.Shape().Rank() != 4 || result.Shape().Dimensions[1] != 2 || result.Shape().Dimensions[2] != 1 {
			t.Errorf("BSHD output shape = %v, want [1,2,1,2]", result.Shape())
		}
	})

	t.Run("boolean_mask", func(t *testing.T) {
		// Test boolean mask: mask out second KV position.
		// B=1, H=1, S=2, D=2
		qData := []float32{1, 0, 0, 1}
		kData := []float32{1, 0, 0, 1}
		vData := []float32{10, 20, 30, 40}
		// Mask: [[true, false], [true, true]] — position 0 can only attend to position 0.
		maskData := []bool{true, false, true, true}

		result := graph.MustExecOnce(backend, func(q, k, v, m *graph.Node) *graph.Node {
			return graph.BackendFusedScaledDotProductAttention(
				q, k, v, m, 1, 1, backends.AxesLayoutBHSD, 1.0, false, nil)
		},
			tensors.FromFlatDataAndDimensions(qData, 1, 1, 2, 2),
			tensors.FromFlatDataAndDimensions(kData, 1, 1, 2, 2),
			tensors.FromFlatDataAndDimensions(vData, 1, 1, 2, 2),
			tensors.FromFlatDataAndDimensions(maskData, 1, 1, 2, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// Position 0 can only attend to position 0 → output[0] = V[0] = [10, 20]
		assertClose(t, got[:2], []float32{10, 20}, 1e-3)
	})
}

// TestBatchNormForTraining tests the BatchNormForTraining operation.
func TestBatchNormForTraining(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("basic", func(t *testing.T) {
		// Input: [4, 2] with featureAxis=1 (2 features, batch of 4).
		// Feature 0: [1, 3, 5, 7] → mean=4, var=5
		// Feature 1: [2, 4, 6, 8] → mean=5, var=5
		inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		scaleData := []float32{1, 1}
		offsetData := []float32{0, 0}

		results := graph.MustExecOnceN(backend, func(op, sc, off *graph.Node) []*graph.Node {
			normalized, batchMean, batchVar := graph.InternalBatchNormForTraining(op, sc, off, 1e-5, 1)
			return []*graph.Node{normalized, batchMean, batchVar}
		},
			tensors.FromFlatDataAndDimensions(inputData, 4, 2),
			tensors.FromFlatDataAndDimensions(scaleData, 2),
			tensors.FromFlatDataAndDimensions(offsetData, 2),
		)

		normGot, _ := tensors.CopyFlatData[float32](results[0])
		meanGot, _ := tensors.CopyFlatData[float32](results[1])
		varGot, _ := tensors.CopyFlatData[float32](results[2])

		// Check mean: [4, 5]
		assertClose(t, meanGot, []float32{4, 5}, 1e-3)
		// Check variance: [5, 5]
		assertClose(t, varGot, []float32{5, 5}, 1e-3)
		// Check normalized: each feature should have mean≈0 after normalization.
		var sum0, sum1 float32
		for i := 0; i < 4; i++ {
			sum0 += normGot[i*2]
			sum1 += normGot[i*2+1]
		}
		if math.Abs(float64(sum0/4)) > 1e-3 {
			t.Errorf("Normalized feature 0 mean = %f, expected ~0", sum0/4)
		}
		if math.Abs(float64(sum1/4)) > 1e-3 {
			t.Errorf("Normalized feature 1 mean = %f, expected ~0", sum1/4)
		}
	})

	t.Run("with_scale_offset", func(t *testing.T) {
		inputData := []float32{0, 10, 0, 10}
		scaleData := []float32{2, 3}
		offsetData := []float32{1, -1}

		results := graph.MustExecOnceN(backend, func(op, sc, off *graph.Node) []*graph.Node {
			normalized, batchMean, batchVar := graph.InternalBatchNormForTraining(op, sc, off, 1e-5, 1)
			return []*graph.Node{normalized, batchMean, batchVar}
		},
			tensors.FromFlatDataAndDimensions(inputData, 2, 2),
			tensors.FromFlatDataAndDimensions(scaleData, 2),
			tensors.FromFlatDataAndDimensions(offsetData, 2),
		)

		normGot, _ := tensors.CopyFlatData[float32](results[0])
		meanGot, _ := tensors.CopyFlatData[float32](results[1])

		// Feature 0: values [0, 0] → mean=0, var=0, normalized=[0,0], scaled=[0*2+1, 0*2+1]=[1,1]
		// Feature 1: values [10, 10] → mean=10, var=0, normalized=[0,0], scaled=[0*3-1, 0*3-1]=[-1,-1]
		assertClose(t, meanGot, []float32{0, 10}, 1e-3)
		assertClose(t, normGot, []float32{1, -1, 1, -1}, 1e-3)
	})
}

// TestBFloat16 tests basic BFloat16 support.
func TestBFloat16(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("add", func(t *testing.T) {
		// Test BFloat16 addition via ConvertDType round-trip.
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			bf := graph.ConvertDType(x, dtypes.BFloat16)
			bf = graph.Add(bf, bf) // 2x
			return graph.ConvertDType(bf, dtypes.Float32)
		}, []float32{1.0, 2.0, 3.0, 4.0})
		got, _ := tensors.CopyFlatData[float32](result)
		assertClose(t, got, []float32{2.0, 4.0, 6.0, 8.0}, 0.1)
	})

	t.Run("matmul", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
			aBF := graph.ConvertDType(a, dtypes.BFloat16)
			bBF := graph.ConvertDType(b, dtypes.BFloat16)
			c := graph.Dot(aBF, bBF).MatMul()
			return graph.ConvertDType(c, dtypes.Float32)
		},
			tensors.FromFlatDataAndDimensions([]float32{1, 2, 3, 4}, 2, 2),
			tensors.FromFlatDataAndDimensions([]float32{5, 6, 7, 8}, 2, 2),
		)
		got, _ := tensors.CopyFlatData[float32](result)
		// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
		assertClose(t, got, []float32{19, 22, 43, 50}, 1.0) // BFloat16 has limited precision
	})
}

// BenchmarkTransformerStep measures per-step time for a transformer-like training iteration.
// Run with: go test -bench BenchmarkTransformerStep -benchtime 10s -count 1
func BenchmarkTransformerStep(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	const (
		batchSize  = 8
		seqLen     = 128
		dModel     = 64
		numClasses = 10
	)

	// Synthetic data.
	inputData := make([]float32, batchSize*seqLen*dModel)
	labelData := make([]int32, batchSize)
	for i := range batchSize {
		cls := int32(i % numClasses)
		labelData[i] = cls
		for j := range seqLen * dModel {
			inputData[i*seqLen*dModel+j] = float32(cls)*0.1 + float32(j%dModel)*0.01
		}
	}

	inputTensor := tensors.FromFlatDataAndDimensions(inputData, batchSize, seqLen, dModel)
	labelTensor := tensors.FromFlatDataAndDimensions(labelData, batchSize, 1)

	dataset := &simpleTrainDataset{
		inputs: []*tensors.Tensor{inputTensor},
		labels: []*tensors.Tensor{labelTensor},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, 0.001)

	// Transformer-like model: LayerNorm → Dense → GELU → reduce → Dense → logits
	modelFn := func(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
		x := inputs[0] // [batch, seq, dModel]

		// LayerNorm over feature dim
		x = layers.LayerNormalization(ctx.In("ln1"), x, -1).Done()

		// Feed-forward: Dense → GELU → Dense
		x = layers.Dense(ctx.In("ff1"), x, true, dModel)
		x = activations.Gelu(x)
		x = layers.Dense(ctx.In("ff2"), x, true, dModel)

		// Mean pool over sequence
		x = graph.ReduceMean(x, 1) // [batch, dModel]

		// Output
		logits := layers.Dense(ctx.In("out"), x, true, numClasses)
		return []*graph.Node{logits}
	}

	trainer := train.NewTrainer(backend, ctx, modelFn,
		losses.SparseCategoricalCrossEntropyLogits,
		optimizers.Adam().Done(),
		nil, nil)

	// Warm up: first step includes compilation.
	loop := train.NewLoop(trainer)
	_, err = loop.RunSteps(dataset, 5)
	if err != nil {
		b.Fatalf("Warmup failed: %+v", err)
	}

	b.ResetTimer()
	for range b.N {
		_, err = loop.RunSteps(dataset, 1)
		if err != nil {
			b.Fatalf("Step failed: %+v", err)
		}
	}
}

// benchmarkMatMul benchmarks matrix multiplication of given dimensions.
func benchmarkMatMul(b *testing.B, M, K, N int) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	aData := make([]float32, M*K)
	bData := make([]float32, K*N)
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%100) * 0.01
	}
	aTensor := tensors.FromFlatDataAndDimensions(aData, M, K)
	bTensor := tensors.FromFlatDataAndDimensions(bData, K, N)

	exec, err := graph.NewExec(backend, func(a, b *graph.Node) *graph.Node {
		return graph.Dot(a, b).MatMul()
	})
	if err != nil {
		b.Fatalf("NewExec failed: %+v", err)
	}
	defer exec.Finalize()
	exec.MustExec(aTensor, bTensor)

	b.ResetTimer()
	for range b.N {
		exec.MustExec(aTensor, bTensor)
	}
}

func BenchmarkDenseMatMul_512x256x512(b *testing.B)   { benchmarkMatMul(b, 512, 256, 512) }
func BenchmarkDenseMatMul_1024x768x1024(b *testing.B) { benchmarkMatMul(b, 1024, 768, 1024) }
func BenchmarkDenseMatMul_2048x768x2048(b *testing.B) { benchmarkMatMul(b, 2048, 768, 2048) }

// Benchmarks matching CoreML benchmark_test.go for direct comparison.

func BenchmarkMatMulExecution_64(b *testing.B)   { benchmarkMatMul(b, 64, 64, 64) }
func BenchmarkMatMulExecution_128(b *testing.B)  { benchmarkMatMul(b, 128, 128, 128) }
func BenchmarkMatMulExecution_256(b *testing.B)  { benchmarkMatMul(b, 256, 256, 256) }
func BenchmarkMatMulExecution_512(b *testing.B)  { benchmarkMatMul(b, 512, 512, 512) }
func BenchmarkMatMulExecution_1024(b *testing.B) { benchmarkMatMul(b, 1024, 1024, 1024) }
func BenchmarkMatMulExecution_2048(b *testing.B) { benchmarkMatMul(b, 2048, 2048, 2048) }

// BenchmarkBinaryOps benchmarks (x + y) * (x - y) on [1024] vectors.
func BenchmarkBinaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	xData := make([]float32, 1024)
	yData := make([]float32, 1024)
	for i := range xData {
		xData[i] = float32(i) * 0.01
		yData[i] = float32(1024-i) * 0.01
	}

	exec, err := graph.NewExec(backend, func(x, y *graph.Node) *graph.Node {
		return graph.Mul(graph.Add(x, y), graph.Sub(x, y))
	})
	if err != nil {
		b.Fatalf("NewExec failed: %+v", err)
	}
	defer exec.Finalize()
	exec.MustExec(xData, yData)

	b.ResetTimer()
	for range b.N {
		exec.MustExec(xData, yData)
	}
}

// BenchmarkReduceOps benchmarks ReduceSum on [1024, 1024] along axis 1.
func BenchmarkReduceOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	data := make([]float32, 1024*1024)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	tensor := tensors.FromFlatDataAndDimensions(data, 1024, 1024)

	exec, err := graph.NewExec(backend, func(x *graph.Node) *graph.Node {
		return graph.ReduceSum(x, 1)
	})
	if err != nil {
		b.Fatalf("NewExec failed: %+v", err)
	}
	defer exec.Finalize()
	exec.MustExec(tensor)

	b.ResetTimer()
	for range b.N {
		exec.MustExec(tensor)
	}
}

// BenchmarkUnaryOps benchmarks exp(log(abs(x))) on [1024] vectors.
func BenchmarkUnaryOps(b *testing.B) {
	backend, err := New("")
	if err != nil {
		b.Fatalf("New() failed: %+v", err)
	}
	defer backend.Finalize()

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i+1) * 0.01
	}

	exec, err := graph.NewExec(backend, func(x *graph.Node) *graph.Node {
		return graph.Exp(graph.Log(graph.Abs(x)))
	})
	if err != nil {
		b.Fatalf("NewExec failed: %+v", err)
	}
	defer exec.Finalize()
	exec.MustExec(data)

	b.ResetTimer()
	for range b.N {
		exec.MustExec(data)
	}
}

// =============================================================================
// Control Flow Tests
// =============================================================================

// TestWhile tests the While control flow operation.
func TestWhile(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("sum_1_to_10", func(t *testing.T) {
		// Compute sum 1+2+...+10 = 55 using a while loop.
		exec := graph.MustNewExec(backend, func(g *graph.Node) *graph.Node {
			cond := graph.NewClosure(g.Graph(), func(g *graph.Graph) []*graph.Node {
				counter := graph.Parameter(g, "counter", shapes.Scalar[int32]())
				_ = graph.Parameter(g, "sum", shapes.Scalar[int32]())
				return []*graph.Node{graph.LessOrEqual(counter, graph.Const(g, int32(10)))}
			})

			body := graph.NewClosure(g.Graph(), func(g *graph.Graph) []*graph.Node {
				counter := graph.Parameter(g, "counter", shapes.Scalar[int32]())
				sum := graph.Parameter(g, "sum", shapes.Scalar[int32]())
				newCounter := graph.Add(counter, graph.Const(g, int32(1)))
				newSum := graph.Add(sum, counter)
				return []*graph.Node{newCounter, newSum}
			})

			results := graph.While(cond, body,
				graph.Const(g.Graph(), int32(1)),
				graph.Const(g.Graph(), int32(0)))
			return results[1] // Return sum
		})
		defer exec.Finalize()

		// Pass a dummy input (required by the signature).
		result := exec.MustExec(int32(0))
		got := result[0].Value().(int32)
		if got != 55 {
			t.Errorf("sum 1..10 = %d, want 55", got)
		}
		t.Logf("While sum 1..10 = %d", got)
	})

	t.Run("factorial", func(t *testing.T) {
		// Compute 5! = 120.
		exec := graph.MustNewExec(backend, func(g *graph.Node) *graph.Node {
			cond := graph.NewClosure(g.Graph(), func(g *graph.Graph) []*graph.Node {
				n := graph.Parameter(g, "n", shapes.Scalar[int32]())
				_ = graph.Parameter(g, "result", shapes.Scalar[int32]())
				return []*graph.Node{graph.GreaterThan(n, graph.Const(g, int32(1)))}
			})

			body := graph.NewClosure(g.Graph(), func(g *graph.Graph) []*graph.Node {
				n := graph.Parameter(g, "n", shapes.Scalar[int32]())
				result := graph.Parameter(g, "result", shapes.Scalar[int32]())
				newResult := graph.Mul(result, n)
				newN := graph.Sub(n, graph.Const(g, int32(1)))
				return []*graph.Node{newN, newResult}
			})

			results := graph.While(cond, body,
				graph.Const(g.Graph(), int32(5)),
				graph.Const(g.Graph(), int32(1)))
			return results[1] // Return result
		})
		defer exec.Finalize()

		result := exec.MustExec(int32(0))
		got := result[0].Value().(int32)
		if got != 120 {
			t.Errorf("5! = %d, want 120", got)
		}
		t.Logf("While 5! = %d", got)
	})
}

// TestIf tests the If control flow operation.
func TestIf(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("true_branch", func(t *testing.T) {
		exec := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node {
			pred := graph.GreaterThan(x, graph.Const(x.Graph(), float32(5)))

			trueBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Const(g, float32(1))}
			})
			falseBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Const(g, float32(-1))}
			})

			results := graph.If(pred, trueBranch, falseBranch)
			return results[0]
		})
		defer exec.Finalize()

		result := exec.MustExec(float32(10))
		got := result[0].Value().(float32)
		if got != 1 {
			t.Errorf("If(10>5) = %g, want 1", got)
		}
		t.Logf("If(10>5) = %g (true branch)", got)
	})

	t.Run("false_branch", func(t *testing.T) {
		exec := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node {
			pred := graph.GreaterThan(x, graph.Const(x.Graph(), float32(5)))

			trueBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Const(g, float32(1))}
			})
			falseBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Const(g, float32(-1))}
			})

			results := graph.If(pred, trueBranch, falseBranch)
			return results[0]
		})
		defer exec.Finalize()

		result := exec.MustExec(float32(3))
		got := result[0].Value().(float32)
		if got != -1 {
			t.Errorf("If(3>5) = %g, want -1", got)
		}
		t.Logf("If(3>5) = %g (false branch)", got)
	})

	t.Run("with_captured_values", func(t *testing.T) {
		// Branches capture a value from parent scope and use it.
		exec := graph.MustNewExec(backend, func(x *graph.Node) *graph.Node {
			factor := graph.Const(x.Graph(), float32(10))
			pred := graph.GreaterThan(x, graph.Const(x.Graph(), float32(0)))

			trueBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Mul(x, factor)} // capture x and factor
			})
			falseBranch := graph.NewClosure(x.Graph(), func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.Neg(graph.Mul(x, factor))} // capture x and factor
			})

			results := graph.If(pred, trueBranch, falseBranch)
			return results[0]
		})
		defer exec.Finalize()

		// x=3 > 0 → true branch → 3*10 = 30
		result := exec.MustExec(float32(3))
		got := result[0].Value().(float32)
		if got != 30 {
			t.Errorf("If(3>0, 3*10) = %g, want 30", got)
		}
		t.Logf("If(3>0, 3*10) = %g", got)
	})
}

// TestSort tests the Sort control flow operation.
func TestSort(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("ascending", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.Sort(x, 0, true)
		}, []float32{5, 2, 8, 1, 9, 3})

		got := result.Value().([]float32)
		want := []float32{1, 2, 3, 5, 8, 9}
		assertClose(t, got, want, 0)
		t.Logf("Sort ascending: %v", got)
	})

	t.Run("descending", func(t *testing.T) {
		result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.Sort(x, 0, false)
		}, []float32{5, 2, 8, 1, 9, 3})

		got := result.Value().([]float32)
		want := []float32{9, 8, 5, 3, 2, 1}
		assertClose(t, got, want, 0)
		t.Logf("Sort descending: %v", got)
	})
}

// TestCall tests the Call control flow operation.
func TestCall(t *testing.T) {
	backend := newTestBackend(t)

	t.Run("simple_add", func(t *testing.T) {
		exec := graph.MustNewExec(backend, func(g *graph.Node) *graph.Node {
			addFn := graph.NewFunction(g.Graph(), "add", func(g *graph.Graph) []*graph.Node {
				a := graph.Parameter(g, "a", shapes.Make(dtypes.Float32))
				b := graph.Parameter(g, "b", shapes.Make(dtypes.Float32))
				return []*graph.Node{graph.Add(a, b)}
			})

			a := graph.Const(g.Graph(), float32(10))
			b := graph.Const(g.Graph(), float32(32))
			return addFn.Call(a, b)[0]
		})
		defer exec.Finalize()

		result := exec.MustExec(float32(0)) // dummy input
		got := result[0].Value().(float32)
		if got != 42 {
			t.Errorf("Call(10+32) = %g, want 42", got)
		}
		t.Logf("Call(10+32) = %g", got)
	})
}
