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

	// Should have: 1 clip op + 1 output identity (constants are inline, not separate operations)
	if len(block.Operations) < 2 {
		t.Errorf("expected at least 2 operations, got %d", len(block.Operations))
	}

	// Find the clip operation
	foundClip := false
	for _, op := range block.Operations {
		if op.Type == "clip" {
			foundClip = true
			// Verify inputs
			if len(op.Inputs) != 3 {
				t.Errorf("clip should have 3 inputs (x, min, max), got %d", len(op.Inputs))
			}
			break
		}
	}

	if !foundClip {
		t.Error("clip operation not found in program")
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

	// Find the clip operation
	foundClip := false
	for _, op := range block.Operations {
		if op.Type == "clip" {
			foundClip = true
			break
		}
	}

	if !foundClip {
		t.Error("clip operation not found in program")
	}
}
