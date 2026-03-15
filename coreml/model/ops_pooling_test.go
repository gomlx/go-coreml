package model

import (
	"testing"
)

func TestMaxPool(t *testing.T) {
	b := NewBuilder("maxpool_test")

	// Create input: [1, 3, 8, 8] (N=1, C=3, H=8, W=8)
	x := b.Input("x", Float32, 1, 3, 8, 8)

	// MaxPool with 2x2 kernel, stride 2, valid padding
	y := b.MaxPool(x, []int64{2, 2}, []int64{2, 2}, ConvPadValid, nil, nil)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape: [1, 3, 4, 4]
	if len(y.shape) != 4 {
		t.Errorf("expected 4 dimensions, got %d", len(y.shape))
	}
	expectedShape := []int64{1, 3, 4, 4}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is present
	mainFunc := program.Functions["maxpool_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundMaxPool := false
	for _, op := range block.Operations {
		if op.Type == "max_pool" {
			foundMaxPool = true
			break
		}
	}
	if !foundMaxPool {
		t.Error("expected max_pool operation in program")
	}
}

func TestAvgPool(t *testing.T) {
	b := NewBuilder("avgpool_test")

	// Create input: [1, 3, 8, 8]
	x := b.Input("x", Float32, 1, 3, 8, 8)

	// AvgPool with 2x2 kernel, stride 2, valid padding, exclude padding
	y := b.AvgPool(x, []int64{2, 2}, []int64{2, 2}, ConvPadValid, nil, nil, true)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape: [1, 3, 4, 4]
	expectedShape := []int64{1, 3, 4, 4}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is present
	mainFunc := program.Functions["avgpool_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundAvgPool := false
	for _, op := range block.Operations {
		if op.Type == "avg_pool" {
			foundAvgPool = true
			// Verify exclude_padding_from_average parameter exists
			if _, ok := op.Inputs["exclude_padding_from_average"]; !ok {
				t.Error("expected exclude_padding_from_average parameter in avg_pool operation")
			}
			break
		}
	}
	if !foundAvgPool {
		t.Error("expected avg_pool operation in program")
	}
}

func TestMaxPoolCustomPadding(t *testing.T) {
	b := NewBuilder("maxpool_custom_test")

	// Create input: [1, 3, 8, 8]
	x := b.Input("x", Float32, 1, 3, 8, 8)

	// MaxPool with custom padding
	padBefore := []int64{1, 1}
	padAfter := []int64{1, 1}
	y := b.MaxPool(x, []int64{3, 3}, []int64{1, 1}, ConvPadCustom, padBefore, padAfter)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape with padding: (8 + 1 + 1 - 3) / 1 + 1 = 8
	expectedShape := []int64{1, 3, 8, 8}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation
	mainFunc := program.Functions["maxpool_custom_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundMaxPool := false
	for _, op := range block.Operations {
		if op.Type == "max_pool" {
			foundMaxPool = true
			// Verify pad parameter exists
			if _, ok := op.Inputs["pad"]; !ok {
				t.Error("expected pad parameter in max_pool operation with custom padding")
			}
			break
		}
	}
	if !foundMaxPool {
		t.Error("expected max_pool operation in program")
	}
}

func TestGlobalAvgPool2D(t *testing.T) {
	b := NewBuilder("global_avgpool_test")

	// Create input: [1, 256, 7, 7]
	x := b.Input("x", Float32, 1, 256, 7, 7)

	// Global average pooling
	y := b.GlobalAvgPool2D(x)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape: [1, 256, 1, 1]
	expectedShape := []int64{1, 256, 1, 1}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is reduce_mean
	mainFunc := program.Functions["global_avgpool_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundReduceMean := false
	for _, op := range block.Operations {
		if op.Type == "reduce_mean" {
			foundReduceMean = true
			break
		}
	}
	if !foundReduceMean {
		t.Error("expected reduce_mean operation for global average pooling")
	}
}

func TestGlobalMaxPool2D(t *testing.T) {
	b := NewBuilder("global_maxpool_test")

	// Create input: [1, 256, 7, 7]
	x := b.Input("x", Float32, 1, 256, 7, 7)

	// Global max pooling
	y := b.GlobalMaxPool2D(x)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape: [1, 256, 1, 1]
	expectedShape := []int64{1, 256, 1, 1}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is reduce_max
	mainFunc := program.Functions["global_maxpool_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundReduceMax := false
	for _, op := range block.Operations {
		if op.Type == "reduce_max" {
			foundReduceMax = true
			break
		}
	}
	if !foundReduceMax {
		t.Error("expected reduce_max operation for global max pooling")
	}
}

func TestMaxPoolSamePadding(t *testing.T) {
	b := NewBuilder("maxpool_same_test")

	// Create input: [1, 3, 7, 7]
	x := b.Input("x", Float32, 1, 3, 7, 7)

	// MaxPool with same padding and stride 1
	y := b.MaxPool(x, []int64{3, 3}, []int64{1, 1}, ConvPadSame, nil, nil)

	b.Output("y", y)

	program := b.Build()

	// With SAME padding and stride 1, output should equal input
	// Output: (7 + 2 - 3) / 1 + 1 = 7
	expectedShape := []int64{1, 3, 7, 7}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation
	mainFunc := program.Functions["maxpool_same_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundMaxPool := false
	for _, op := range block.Operations {
		if op.Type == "max_pool" {
			foundMaxPool = true
			break
		}
	}
	if !foundMaxPool {
		t.Error("expected max_pool operation in program")
	}
}
