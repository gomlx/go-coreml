package model

import (
	"testing"
)

func TestClip(t *testing.T) {
	b := NewBuilder("main")

	// Create inputs
	x := b.Input("x", Float32, 2, 3)
	minVal := b.Const("min", Float32, []int64{}, []float32{-1.0})
	maxVal := b.Const("max", Float32, []int64{}, []float32{1.0})

	// Clamp x to [-1, 1]
	clamped := b.Clip(x, minVal, maxVal)

	// Mark output
	b.Output("clamped", clamped)

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

	if len(mainFunc.Inputs) != 1 {
		t.Errorf("expected 1 input, got %d", len(mainFunc.Inputs))
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Clip is implemented as Maximum(x, min) then Minimum(result, max),
	// so we expect maximum + minimum ops (not a single "clip" op).
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["maximum"] < 1 {
		t.Error("expected maximum operation in program (Clip lower bound)")
	}
	if opTypeCount["minimum"] < 1 {
		t.Error("expected minimum operation in program (Clip upper bound)")
	}
}

func TestCast(t *testing.T) {
	b := NewBuilder("main")

	// Create input
	x := b.Input("x", Float32, 2, 3)

	// Cast to Int32
	casted := b.Cast(x, Int32)

	// Mark output
	b.Output("casted", casted)

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

	if len(mainFunc.Inputs) != 1 {
		t.Errorf("expected 1 input, got %d", len(mainFunc.Inputs))
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Should have: 1 cast op + 1 output identity
	if len(block.Operations) < 2 {
		t.Errorf("expected at least 2 operations, got %d", len(block.Operations))
	}

	// Find the cast operation
	foundCast := false
	for _, op := range block.Operations {
		if op.Type == "cast" {
			foundCast = true
			// Verify inputs: x (tensor) and dtype (string constant)
			if len(op.Inputs) != 2 {
				t.Errorf("cast should have 2 inputs (x and dtype), got %d", len(op.Inputs))
			}
			// Verify x input exists
			if _, ok := op.Inputs["x"]; !ok {
				t.Error("cast should have 'x' input")
			}
			// Verify dtype input exists
			if _, ok := op.Inputs["dtype"]; !ok {
				t.Error("cast should have 'dtype' input")
			}
			break
		}
	}

	if !foundCast {
		t.Error("cast operation not found in program")
	}
}

func TestFloorFloat(t *testing.T) {
	b := NewBuilder("main")
	x := b.Input("x", Float32, 2, 3)
	y := b.Floor(x)
	b.Output("y", y)

	program := b.Build()
	block := program.Functions["main"].BlockSpecializations["CoreML7"]

	foundFloor := false
	for _, op := range block.Operations {
		if op.Type == "floor" {
			foundFloor = true
			break
		}
	}
	if !foundFloor {
		t.Error("floor operation not found for float input")
	}
}

func TestFloorIntegerPassthrough(t *testing.T) {
	b := NewBuilder("main")
	x := b.Input("x", Int32, 2, 3)
	y := b.Floor(x)
	b.Output("y", y)

	// Floor of int is identity — should produce no floor op
	program := b.Build()
	block := program.Functions["main"].BlockSpecializations["CoreML7"]

	for _, op := range block.Operations {
		if op.Type == "floor" {
			t.Error("floor operation should not be emitted for integer input")
		}
	}
}

func TestCeilIntegerPassthrough(t *testing.T) {
	b := NewBuilder("main")
	x := b.Input("x", Int32, 2, 3)
	y := b.Ceil(x)
	b.Output("y", y)

	program := b.Build()
	block := program.Functions["main"].BlockSpecializations["CoreML7"]

	for _, op := range block.Operations {
		if op.Type == "ceil" {
			t.Error("ceil operation should not be emitted for integer input")
		}
	}
}

func TestRoundIntegerPassthrough(t *testing.T) {
	b := NewBuilder("main")
	x := b.Input("x", Int32, 2, 3)
	y := b.Round(x)
	b.Output("y", y)

	program := b.Build()
	block := program.Functions["main"].BlockSpecializations["CoreML7"]

	for _, op := range block.Operations {
		if op.Type == "round" {
			t.Error("round operation should not be emitted for integer input")
		}
	}
}

func TestClipBroadcast(t *testing.T) {
	b := NewBuilder("main")

	// Create inputs with different shapes for broadcasting
	x := b.Input("x", Float32, 2, 3)
	minVal := b.Const("min", Float32, []int64{3}, []float32{-1.0, -2.0, -3.0})
	maxVal := b.Const("max", Float32, []int64{1}, []float32{1.0})

	// Clamp with broadcasting
	clamped := b.Clip(x, minVal, maxVal)

	// Mark output
	b.Output("clamped", clamped)

	// Build the program
	program := b.Build()

	// Verify program structure
	mainFunc, ok := program.Functions["main"]
	if !ok {
		t.Fatal("expected 'main' function")
	}

	block, ok := mainFunc.BlockSpecializations["CoreML7"]
	if !ok {
		t.Fatal("expected block specialization for CoreML7")
	}

	// Clip is implemented as Maximum + Minimum, verify both are present.
	opTypeCount := make(map[string]int)
	for _, op := range block.Operations {
		opTypeCount[op.Type]++
	}

	if opTypeCount["maximum"] < 1 {
		t.Error("expected maximum operation in program (Clip lower bound)")
	}
	if opTypeCount["minimum"] < 1 {
		t.Error("expected minimum operation in program (Clip upper bound)")
	}
}
