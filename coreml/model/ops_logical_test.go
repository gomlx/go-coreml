package model

import (
	"testing"
)

func TestLogicalOperations(t *testing.T) {
	b := NewBuilder("logical_test")

	// Create bool inputs
	x := b.Input("x", Bool, 4)
	y := b.Input("y", Bool, 4)

	// Test LogicalAnd
	andResult := b.LogicalAnd(x, y)
	b.Output("and_out", andResult)

	// Test LogicalOr
	orResult := b.LogicalOr(x, y)
	b.Output("or_out", orResult)

	// Test LogicalNot
	notResult := b.LogicalNot(x)
	b.Output("not_out", notResult)

	// Test LogicalXor
	xorResult := b.LogicalXor(x, y)
	b.Output("xor_out", xorResult)

	program := b.Build()

	// Verify operations
	mainFunc := program.Functions["logical_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Count operation types
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	// Check logical ops are present
	expectedOps := []string{"logical_and", "logical_or", "logical_not", "logical_xor"}
	for _, exp := range expectedOps {
		if opTypeCount[exp] < 1 {
			t.Errorf("expected %s operation in program", exp)
		}
	}
}

func TestRsqrt(t *testing.T) {
	b := NewBuilder("rsqrt_test")

	x := b.Input("x", Float32, 4)
	y := b.Rsqrt(x)
	b.Output("y", y)

	program := b.Build()

	mainFunc := program.Functions["rsqrt_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Find rsqrt operation
	foundRsqrt := false
	for _, op := range block.Operations {
		if op.Type == "rsqrt" {
			foundRsqrt = true
			break
		}
	}
	if !foundRsqrt {
		t.Error("expected rsqrt operation in program")
	}
}

func TestIsNanIsFinite(t *testing.T) {
	b := NewBuilder("isnan_isfinite_test")

	x := b.Input("x", Float32, 4)

	// Test IsNan
	isnanResult := b.IsNan(x)
	b.Output("isnan_out", isnanResult)

	// Test IsFinite
	isfiniteResult := b.IsFinite(x)
	b.Output("isfinite_out", isfiniteResult)

	program := b.Build()

	mainFunc := program.Functions["isnan_isfinite_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	// Count operation types
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	// IsNan is implemented as NotEqual(x, x) since NaN != NaN.
	if opTypeCount["not_equal"] < 1 {
		t.Error("expected not_equal operation in program (IsNan)")
	}

	// IsFinite is implemented as LogicalAnd(Equal(x,x), Equal(x*0, x*0)),
	// so we expect mul, equal, and logical_and ops.
	if opTypeCount["equal"] < 1 {
		t.Error("expected equal operation in program (IsFinite)")
	}
	if opTypeCount["logical_and"] < 1 {
		t.Error("expected logical_and operation in program (IsFinite)")
	}
}
