package runtime

import (
	"testing"

	"github.com/gomlx/go-darwinml/coreml/internal/bridge"
	"github.com/gomlx/go-darwinml/coreml/model"
)

// skipIfNotMacOS skips the test if not running on macOS.
func skipIfNotMacOS(t *testing.T) {
	t.Helper()
	// The runtime package only works on macOS due to CoreML dependency
	// This is enforced by the cgo build constraints
}

func TestRuntimeCompileAndRun(t *testing.T) {
	skipIfNotMacOS(t)

	// Build a simple ReLU model
	b := model.NewBuilder("main")
	x := b.Input("x", model.Float32, 2, 3)
	y := b.Relu(x)
	b.Output("y", y)

	// Compile
	rt := New()
	exec, err := rt.Compile(b)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Close()

	// Prepare input data
	// Input: [[-1, 2, -3], [4, -5, 6]]
	input := []float32{-1, 2, -3, 4, -5, 6}

	// Run
	outputs, err := exec.Run(map[string]any{"x": input})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Check output
	result, ok := outputs["y"].([]float32)
	if !ok {
		t.Fatalf("expected []float32 output, got %T", outputs["y"])
	}

	// Expected: [[0, 2, 0], [4, 0, 6]]
	expected := []float32{0, 2, 0, 4, 0, 6}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %f, expected %f", i, result[i], v)
		}
	}
}

func TestRuntimeAddMul(t *testing.T) {
	skipIfNotMacOS(t)

	// Build model: y = (x + 1) * 2
	b := model.NewBuilder("main")
	x := b.Input("x", model.Float32, 4)

	one := b.Const("one", model.Float32, []int64{1}, []float32{1.0})
	two := b.Const("two", model.Float32, []int64{1}, []float32{2.0})

	added := b.Add(x, one)
	mulResult := b.Mul(added, two)
	b.Output("y", mulResult)

	// Compile
	rt := New()
	exec, err := rt.Compile(b)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Close()

	// Input: [1, 2, 3, 4]
	input := []float32{1, 2, 3, 4}

	// Run
	outputs, err := exec.Run(map[string]any{"x": input})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Expected: [(1+1)*2, (2+1)*2, (3+1)*2, (4+1)*2] = [4, 6, 8, 10]
	result, ok := outputs["y"].([]float32)
	if !ok {
		t.Fatalf("expected []float32 output, got %T", outputs["y"])
	}

	expected := []float32{4, 6, 8, 10}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %f, expected %f", i, result[i], v)
		}
	}
}

func TestRuntimeMatMul(t *testing.T) {
	skipIfNotMacOS(t)

	// Build model: z = matmul(x, y)
	// x: [2, 3], y: [3, 2] -> z: [2, 2]
	b := model.NewBuilder("main")
	x := b.Input("x", model.Float32, 2, 3)
	y := b.Input("y", model.Float32, 3, 2)
	z := b.MatMul(x, y)
	b.Output("z", z)

	// Compile
	rt := New()
	exec, err := rt.Compile(b)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Close()

	// x = [[1, 2, 3], [4, 5, 6]]
	// y = [[7, 8], [9, 10], [11, 12]]
	// z = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
	//   = [[58, 64], [139, 154]]
	inputX := []float32{1, 2, 3, 4, 5, 6}
	inputY := []float32{7, 8, 9, 10, 11, 12}

	// Run
	outputs, err := exec.Run(map[string]any{
		"x": inputX,
		"y": inputY,
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	result, ok := outputs["z"].([]float32)
	if !ok {
		t.Fatalf("expected []float32 output, got %T", outputs["z"])
	}

	expected := []float32{58, 64, 139, 154}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %f, expected %f", i, result[i], v)
		}
	}
}

func TestRuntimeWithComputeUnits(t *testing.T) {
	skipIfNotMacOS(t)

	// Build a simple model
	b := model.NewBuilder("main")
	x := b.Input("x", model.Float32, 4)
	y := b.Relu(x)
	b.Output("y", y)

	// Test CPU-only execution
	rt := New(WithComputeUnits(ComputeCPUOnly))
	exec, err := rt.Compile(b)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	defer exec.Close()

	input := []float32{-1, 2, -3, 4}
	outputs, err := exec.Run(map[string]any{"x": input})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	result := outputs["y"].([]float32)
	expected := []float32{0, 2, 0, 4}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %f, expected %f", i, result[i], v)
		}
	}
}

// Re-export compute units for convenience in tests.
const (
	ComputeAll       = bridge.ComputeAll
	ComputeCPUOnly   = bridge.ComputeCPUOnly
	ComputeCPUAndGPU = bridge.ComputeCPUAndGPU
	ComputeCPUAndANE = bridge.ComputeCPUAndANE
)
