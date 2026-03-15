//go:build darwin && cgo

package coreml

import (
	"testing"

	"github.com/gomlx/go-darwinml/coreml/model"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestIntegerOps tests individual integer operations through the CoreML backend
// to find which ones crash.
func TestIntegerOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	// Test each operation individually
	tests := []struct {
		name string
		fn   func(x *graph.Node) *graph.Node
	}{
		{"Identity", func(x *graph.Node) *graph.Node { return graph.Identity(x) }},
		{"Add_scalar", func(x *graph.Node) *graph.Node { return graph.AddScalar(x, 1) }},
		{"Neg", func(x *graph.Node) *graph.Node { return graph.Neg(x) }},
		{"Abs", func(x *graph.Node) *graph.Node { return graph.Abs(x) }},
		{"Reshape", func(x *graph.Node) *graph.Node { return graph.Reshape(x, 1, 2) }},
		{"ExpandDims", func(x *graph.Node) *graph.Node { return graph.ExpandDims(x, 0) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := tensors.FromFlatDataAndDimensions([]int32{3, 5}, 2)
			defer input.FinalizeAll()

			result := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
				return tt.fn(x)
			}, input)
			defer result.FinalizeAll()
			t.Logf("OK: shape=%v", result.Shape())
		})
	}
}

// TestIntegerBinaryOps tests binary integer operations.
func TestIntegerBinaryOps(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	tests := []struct {
		name string
		fn   func(a, b *graph.Node) *graph.Node
	}{
		{"Add", func(a, b *graph.Node) *graph.Node { return graph.Add(a, b) }},
		{"Sub", func(a, b *graph.Node) *graph.Node { return graph.Sub(a, b) }},
		{"Mul", func(a, b *graph.Node) *graph.Node { return graph.Mul(a, b) }},
		{"Div", func(a, b *graph.Node) *graph.Node { return graph.Div(a, b) }},
		{"Mod", func(a, b *graph.Node) *graph.Node { return graph.Mod(a, b) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := tensors.FromFlatDataAndDimensions([]int32{10, 15}, 2)
			b := tensors.FromFlatDataAndDimensions([]int32{3, 4}, 2)
			defer a.FinalizeAll()
			defer b.FinalizeAll()

			result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
				return tt.fn(a, b)
			}, a, b)
			defer result.FinalizeAll()
			result.MustConstFlatData(func(flat any) {
				t.Logf("OK: %v", flat)
			})
		})
	}
}

// TestRangeCountComputation reproduces the exact computation chain used by
// onnx-gomlx's rangeCount function: Ceil((limit - start) / delta) with
// Int64 inputs. This was producing 2147483647 (MaxInt32) instead of 24.
// Note: CoreML requires rank >= 1, so we use [1]-shaped tensors instead of scalars.
func TestRangeCountComputation(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	// Test simple identity with Int64
	t.Run("Identity_Int64", func(t *testing.T) {
		a := tensors.FromFlatDataAndDimensions([]int64{42}, 1)
		defer a.FinalizeAll()

		result := graph.MustExecOnce(backend, func(a *graph.Node) *graph.Node {
			return graph.Identity(a)
		}, a)
		defer result.FinalizeAll()

		var got int64
		result.MustConstFlatData(func(flat any) { got = flat.([]int64)[0] })
		t.Logf("Identity_Int64: got=%d, shape=%v", got, result.Shape())
		if got != 42 {
			t.Errorf("Identity_Int64: got %d, want 42", got)
		}
	})

	// Test Sub with Int64
	t.Run("Sub_Int64", func(t *testing.T) {
		a := tensors.FromFlatDataAndDimensions([]int64{24}, 1)
		b := tensors.FromFlatDataAndDimensions([]int64{0}, 1)
		defer a.FinalizeAll()
		defer b.FinalizeAll()

		result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
			return graph.Sub(a, b)
		}, a, b)
		defer result.FinalizeAll()

		var got int64
		result.MustConstFlatData(func(flat any) { got = flat.([]int64)[0] })
		t.Logf("Sub_Int64: got=%d, shape=%v", got, result.Shape())
		if got != 24 {
			t.Errorf("Sub_Int64: got %d, want 24", got)
		}
	})

	// Test ConvertDType Int64 → Float64 (both downcast internally)
	t.Run("Cast_Int64_to_Float", func(t *testing.T) {
		a := tensors.FromFlatDataAndDimensions([]int64{24}, 1)
		defer a.FinalizeAll()

		result := graph.MustExecOnce(backend, func(a *graph.Node) *graph.Node {
			return graph.ConvertDType(a, dtypes.Float64)
		}, a)
		defer result.FinalizeAll()

		var got float64
		result.MustConstFlatData(func(flat any) {
			got = flat.([]float64)[0]
		})
		t.Logf("Cast_Int64_to_Float: got=%v, shape=%v", got, result.Shape())
		if got != 24.0 {
			t.Errorf("Cast_Int64_to_Float: got %v, want 24.0", got)
		}
	})

	// Test with Int32 (bypasses Int64 conversion)
	t.Run("FullChain_Int32", func(t *testing.T) {
		start := tensors.FromFlatDataAndDimensions([]int32{0}, 1)
		limit := tensors.FromFlatDataAndDimensions([]int32{24}, 1)
		delta := tensors.FromFlatDataAndDimensions([]int32{1}, 1)
		defer start.FinalizeAll()
		defer limit.FinalizeAll()
		defer delta.FinalizeAll()

		result := graph.MustExecOnce(backend, func(start, limit, delta *graph.Node) *graph.Node {
			amount := graph.Sub(limit, start)
			amountFloat := graph.ConvertDType(amount, dtypes.Float32)
			deltaFloat := graph.ConvertDType(delta, dtypes.Float32)
			count := graph.Ceil(graph.Div(amountFloat, deltaFloat))
			return graph.ConvertDType(count, dtypes.Int32)
		}, start, limit, delta)
		defer result.FinalizeAll()

		var got int32
		result.MustConstFlatData(func(flat any) { got = flat.([]int32)[0] })
		t.Logf("FullChain_Int32: got=%d, shape=%v", got, result.Shape())
		if got != 24 {
			t.Errorf("FullChain_Int32: got %d, want 24", got)
		}
	})

	// Test the full rangeCount chain with Int64
	t.Run("FullChain_Int64", func(t *testing.T) {
		start := tensors.FromFlatDataAndDimensions([]int64{0}, 1)
		limit := tensors.FromFlatDataAndDimensions([]int64{24}, 1)
		delta := tensors.FromFlatDataAndDimensions([]int64{1}, 1)
		defer start.FinalizeAll()
		defer limit.FinalizeAll()
		defer delta.FinalizeAll()

		result := graph.MustExecOnce(backend, func(start, limit, delta *graph.Node) *graph.Node {
			amount := graph.Sub(limit, start)
			amountFloat := graph.ConvertDType(amount, dtypes.Float64)
			deltaFloat := graph.ConvertDType(delta, dtypes.Float64)
			count := graph.Ceil(graph.Div(amountFloat, deltaFloat))
			return graph.ConvertDType(count, dtypes.Int64)
		}, start, limit, delta)
		defer result.FinalizeAll()

		var got int64
		result.MustConstFlatData(func(flat any) { got = flat.([]int64)[0] })
		t.Logf("FullChain_Int64: got=%d, shape=%v", got, result.Shape())
		if got != 24 {
			t.Errorf("FullChain_Int64: got %d, want 24", got)
		}
	})
}

// TestIntegerConcat tests int32 concatenation.
func TestIntegerConcat(t *testing.T) {
	backend, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer backend.Finalize()

	a := tensors.FromFlatDataAndDimensions([]int32{0, 0}, 2)
	b := tensors.FromFlatDataAndDimensions([]int32{2, 1}, 2)
	defer a.FinalizeAll()
	defer b.FinalizeAll()

	result := graph.MustExecOnce(backend, func(a, b *graph.Node) *graph.Node {
		return graph.Concatenate([]*graph.Node{a, b}, 0)
	}, a, b)
	defer result.FinalizeAll()

	result.MustConstFlatData(func(flat any) {
		got := flat.([]int32)
		t.Logf("Int32Concat result: %v", got)
	})
}

// TestInt64BufferRoundtrip tests that Int64 data survives the buffer→runtime→buffer path.
func TestInt64BufferRoundtrip(t *testing.T) {
	be, err := New("")
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	defer be.Finalize()
	backend := be.(*Backend)

	// Step 1: Create an Int64 buffer and check internal storage.
	shape := shapes.Make(dtypes.Int64, 1)
	buf, err := backend.BufferFromFlatData(0, []int64{42}, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %v", err)
	}
	cmlBuf := buf.(*Buffer)
	t.Logf("Buffer internal flat type=%T, value=%v, shape=%v", cmlBuf.flat, cmlBuf.flat, cmlBuf.shape)

	// Step 2: Convert buffer to runtime data (same as Execute does).
	runtimeData, err := bufferToRuntimeData(cmlBuf)
	if err != nil {
		t.Fatalf("bufferToRuntimeData failed: %v", err)
	}
	t.Logf("Runtime data type=%T, value=%v", runtimeData, runtimeData)

	// Step 3: Run a pure Int32 identity model through the runtime directly.
	mb := model.NewBuilder("main")
	x := mb.Input("x", model.Int32, 1)
	mb.Output("y", x)

	rt := backend.runtime
	exec, err := rt.Compile(mb)
	if err != nil {
		t.Fatalf("Runtime compile failed: %v", err)
	}
	defer exec.Close()

	outputs, err := exec.Run(map[string]any{"x": runtimeData})
	if err != nil {
		t.Fatalf("Runtime run failed: %v", err)
	}
	t.Logf("Runtime output type=%T, value=%v", outputs["y"], outputs["y"])

	// Step 4: Convert output back to buffer.
	outBuf, err := backend.runtimeDataToBuffer(outputs["y"], shape)
	if err != nil {
		t.Fatalf("runtimeDataToBuffer failed: %v", err)
	}
	t.Logf("Output buffer flat type=%T, value=%v, shape=%v", outBuf.flat, outBuf.flat, outBuf.shape)

	// Step 5: Read back as Int64.
	outFlat := make([]int64, 1)
	if err := backend.BufferToFlatData(outBuf, outFlat); err != nil {
		t.Fatalf("BufferToFlatData failed: %v", err)
	}
	t.Logf("Final output: %v", outFlat)
	if outFlat[0] != 42 {
		t.Errorf("Expected 42, got %d", outFlat[0])
	}
}
