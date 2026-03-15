package model

import (
	"testing"

	"github.com/gomlx/go-darwinml/proto/coreml/milspec"
	"google.golang.org/protobuf/proto"
)

func TestBuilderSimple(t *testing.T) {
	b := NewBuilder("main")

	// Create inputs
	x := b.Input("x", Float32, 2, 3)
	y := b.Input("y", Float32, 3, 4)

	// Matrix multiply
	z := b.MatMul(x, y)

	// Mark output
	b.Output("z", z)

	// Build the program
	program := b.Build()

	// Verify program structure
	if program.Version != 1 {
		t.Errorf("expected version 1, got %d", program.Version)
	}

	mainFunc, ok := program.Functions["main"]
	if !ok {
		t.Fatal("expected 'main' function")
	}

	if len(mainFunc.Inputs) != 2 {
		t.Errorf("expected 2 inputs, got %d", len(mainFunc.Inputs))
	}

	if mainFunc.Opset != "CoreML7" {
		t.Errorf("expected opset CoreML7, got %s", mainFunc.Opset)
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// MatMul generates 1 op + identity for output renaming
	// MatMul also adds 2 const ops for transpose_x and transpose_y
	if len(block.Operations) < 1 {
		t.Errorf("expected at least 1 operation, got %d", len(block.Operations))
	}

	// First non-identity op should be matmul
	foundMatmul := false
	for _, op := range block.Operations {
		if op.Type == "matmul" {
			foundMatmul = true
			break
		}
	}
	if !foundMatmul {
		t.Error("expected matmul operation in program")
	}

	if len(block.Outputs) != 1 {
		t.Errorf("expected 1 output, got %d", len(block.Outputs))
	}
}

func TestBuilderWithOps(t *testing.T) {
	b := NewBuilder("mlp")

	// Create a simple MLP: relu(matmul(x, w) + b)
	x := b.Input("x", Float32, 1, 4)
	w := b.Input("w", Float32, 4, 8)
	bias := b.Input("bias", Float32, 8)

	hidden := b.MatMul(x, w)
	added := b.Add(hidden, bias)
	out := b.Relu(added)

	b.Output("out", out)

	program := b.Build()

	// Verify operations
	mainFunc := program.Functions["mlp"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Verify key operations are present (identity ops may be added for renaming)
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	expectedOps := []string{"matmul", "add", "relu"}
	for _, exp := range expectedOps {
		if opTypeCount[exp] < 1 {
			t.Errorf("expected %s operation in program", exp)
		}
	}
}

func TestBuilderSerialization(t *testing.T) {
	b := NewBuilder("test")

	x := b.Input("x", Float32, 2, 2)
	y := b.Relu(x)
	b.Output("y", y)

	program := b.Build()

	// Verify we can serialize/deserialize
	data, err := proto.Marshal(program)
	if err != nil {
		t.Fatalf("failed to marshal program: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("serialized program is empty")
	}

	t.Logf("program serialized to %d bytes", len(data))
}

func TestBuilderConst(t *testing.T) {
	b := NewBuilder("const_test")

	x := b.Input("x", Float32, 2, 2)

	// Create a constant
	scale := b.Const("scale", Float32, []int64{1}, []float32{2.0})

	// Multiply by constant
	y := b.Mul(x, scale)
	b.Output("y", y)

	program := b.Build()

	mainFunc := program.Functions["const_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Find the mul operation (identity may be added for output renaming)
	var mulOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "mul" {
			mulOp = op
			break
		}
	}
	if mulOp == nil {
		t.Fatal("expected mul operation in program")
	}

	// Check that 'y' input is a constant value
	yArg := mulOp.Inputs["y"]
	if yArg == nil {
		t.Fatal("expected 'y' input in mul operation")
	}

	if len(yArg.Arguments) != 1 {
		t.Fatalf("expected 1 argument, got %d", len(yArg.Arguments))
	}

	// The constant should be embedded as a value, not a name reference
	if yArg.Arguments[0].GetValue() == nil {
		t.Error("expected constant to be embedded as value")
	}
}

func TestBuilderNormalization(t *testing.T) {
	b := NewBuilder("norm_test")

	// Create inputs
	x := b.Input("x", Float32, 2, 3, 4, 4) // [N, C, H, W]
	mean := b.Const("mean", Float32, []int64{3}, []float32{0.0, 0.0, 0.0})
	variance := b.Const("variance", Float32, []int64{3}, []float32{1.0, 1.0, 1.0})
	gamma := b.Const("gamma", Float32, []int64{3}, []float32{1.0, 1.0, 1.0})
	beta := b.Const("beta", Float32, []int64{3}, []float32{0.0, 0.0, 0.0})

	// Test BatchNorm
	bn := b.BatchNorm(x, mean, variance, gamma, beta, 1e-5)
	b.Output("bn_out", bn)

	// Test LayerNorm
	ln := b.LayerNorm(x, gamma, beta, []int64{1, 2, 3}, 1e-5)
	b.Output("ln_out", ln)

	// Test InstanceNorm
	in := b.InstanceNorm(x, gamma, beta, 1e-5)
	b.Output("in_out", in)

	program := b.Build()

	// Verify operations
	mainFunc := program.Functions["norm_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Count operation types
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	// Check normalization ops are present
	expectedOps := []string{"batch_norm", "layer_norm", "instance_norm"}
	for _, exp := range expectedOps {
		if opTypeCount[exp] < 1 {
			t.Errorf("expected %s operation in program", exp)
		}
	}
}
