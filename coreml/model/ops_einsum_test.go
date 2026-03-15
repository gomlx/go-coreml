package model

import (
	"testing"

	"github.com/gomlx/go-darwinml/proto/coreml/milspec"
)

func TestEinsumRank4(t *testing.T) {
	b := NewBuilder("einsum_rank4_test")

	// Create two rank-4 tensors for batched matrix multiplication
	// x: [2, 3, 4, 5] -> [B, C, H, W1]
	// y: [2, 5, 4, 6] -> [B, W1, H, W2]
	// Expected output: [2, 3, 4, 6] -> [B, C, H, W2]
	x := b.Input("x", Float32, 2, 3, 4, 5)
	y := b.Input("y", Float32, 2, 5, 4, 6)

	// Perform einsum with equation for batched matrix multiplication
	result := b.Einsum("nchw,nwhu->nchu", []*Value{x, y})

	b.Output("result", result)

	// Verify output shape
	expectedShape := []int64{2, 3, 4, 6}
	if len(result.shape) != len(expectedShape) {
		t.Fatalf("expected shape length %d, got %d", len(expectedShape), len(result.shape))
	}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}

	// Build and verify
	program := b.Build()
	mainFunc := program.Functions["einsum_rank4_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Find einsum operation
	var einsumOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "einsum" {
			einsumOp = op
			break
		}
	}

	if einsumOp == nil {
		t.Fatal("expected einsum operation in program")
	}

	// Verify equation argument
	equationArg := einsumOp.Inputs["equation"]
	if equationArg == nil {
		t.Fatal("expected 'equation' input in einsum operation")
	}

	// Verify values argument is a list with 2 bindings
	valuesArg := einsumOp.Inputs["values"]
	if valuesArg == nil {
		t.Fatal("expected 'values' input in einsum operation")
	}

	if len(valuesArg.Arguments) != 2 {
		t.Errorf("expected 2 values in einsum, got %d", len(valuesArg.Arguments))
	}
}

func TestEinsumRank3(t *testing.T) {
	b := NewBuilder("einsum_rank3_test")

	// Create two rank-3 tensors
	// x: [3, 4, 5] -> [C, H, W1]
	// y: [5, 4, 6] -> [W1, H, W2]
	// Expected output: [3, 4, 6] -> [C, H, W2]
	x := b.Input("x", Float32, 3, 4, 5)
	y := b.Input("y", Float32, 5, 4, 6)

	result := b.Einsum("chw,whr->chr", []*Value{x, y})

	// Verify output shape
	expectedShape := []int64{3, 4, 6}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}
}

func TestEinsumBroadcastBatch(t *testing.T) {
	b := NewBuilder("einsum_broadcast_batch")

	// Test broadcasting on batch dimension
	// x: [1, 3, 4, 5] -> B=1 should broadcast to B=2
	// y: [2, 5, 4, 6] -> B=2
	// Expected output: [2, 3, 4, 6]
	x := b.Input("x", Float32, 1, 3, 4, 5)
	y := b.Input("y", Float32, 2, 5, 4, 6)

	result := b.Einsum("nchw,nwhu->nchu", []*Value{x, y})

	expectedShape := []int64{2, 3, 4, 6}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}
}

func TestEinsumBroadcastHeight(t *testing.T) {
	b := NewBuilder("einsum_broadcast_height")

	// Test broadcasting on H dimension
	// x: [2, 3, 1, 5] -> H=1 should broadcast to H=4
	// y: [2, 5, 4, 6] -> H=4
	// Expected output: [2, 3, 4, 6]
	x := b.Input("x", Float32, 2, 3, 1, 5)
	y := b.Input("y", Float32, 2, 5, 4, 6)

	result := b.Einsum("nchw,nwhu->nchu", []*Value{x, y})

	expectedShape := []int64{2, 3, 4, 6}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}
}

func TestEinsumRank3BroadcastHeight(t *testing.T) {
	b := NewBuilder("einsum_rank3_broadcast")

	// Test broadcasting on H dimension for rank 3
	// x: [3, 1, 5] -> H=1 should broadcast to H=4
	// y: [5, 4, 6] -> H=4
	// Expected output: [3, 4, 6]
	x := b.Input("x", Float32, 3, 1, 5)
	y := b.Input("y", Float32, 5, 4, 6)

	result := b.Einsum("chw,whr->chr", []*Value{x, y})

	expectedShape := []int64{3, 4, 6}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}
}

func TestEinsumWithConstants(t *testing.T) {
	b := NewBuilder("einsum_with_const")

	// Mix variable and constant
	x := b.Input("x", Float32, 2, 3, 4, 5)
	// Create a constant tensor [2, 5, 4, 6]
	constData := make([]float32, 2*5*4*6)
	for i := range constData {
		constData[i] = float32(i) * 0.01
	}
	y := b.Const("y_const", Float32, []int64{2, 5, 4, 6}, constData)

	result := b.Einsum("nchw,nwhu->nchu", []*Value{x, y})

	// Verify output shape
	expectedShape := []int64{2, 3, 4, 6}
	for i, dim := range expectedShape {
		if result.shape[i] != dim {
			t.Errorf("shape[%d]: expected %d, got %d", i, dim, result.shape[i])
		}
	}

	// Build and verify constant is embedded
	program := b.Build()
	mainFunc := program.Functions["einsum_with_const"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	var einsumOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "einsum" {
			einsumOp = op
			break
		}
	}

	if einsumOp == nil {
		t.Fatal("expected einsum operation")
	}

	// Check that one of the values is embedded
	valuesArg := einsumOp.Inputs["values"]
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
