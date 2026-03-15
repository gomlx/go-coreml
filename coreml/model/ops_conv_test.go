package model

import (
	"testing"
)

func TestConv2D(t *testing.T) {
	b := NewBuilder("conv2d_test")

	// Create input: [1, 3, 32, 32] (N=1, C_in=3, H=32, W=32)
	x := b.Input("x", Float32, 1, 3, 32, 32)

	// Create weight: [16, 3, 3, 3] (C_out=16, C_in=3, kH=3, kW=3)
	weight := b.Input("weight", Float32, 16, 3, 3, 3)

	// Conv2D with 3x3 kernel, stride 1, valid padding
	y := b.Conv(x, weight, []int64{1, 1}, []int64{1, 1}, ConvPadValid, nil, nil, 1)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape: [1, 16, 30, 30]
	// Output size: (32 - 3) / 1 + 1 = 30
	if len(y.shape) != 4 {
		t.Errorf("expected 4 dimensions, got %d", len(y.shape))
	}
	expectedShape := []int64{1, 16, 30, 30}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is present
	mainFunc := program.Functions["conv2d_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConv := false
	for _, op := range block.Operations {
		if op.Type == "conv" {
			foundConv = true
			break
		}
	}
	if !foundConv {
		t.Error("expected conv operation in program")
	}
}

func TestConv2DWithStride(t *testing.T) {
	b := NewBuilder("conv2d_stride_test")

	// Create input: [1, 64, 56, 56]
	x := b.Input("x", Float32, 1, 64, 56, 56)

	// Create weight: [128, 64, 3, 3]
	weight := b.Input("weight", Float32, 128, 64, 3, 3)

	// Conv2D with 3x3 kernel, stride 2, valid padding
	y := b.Conv(x, weight, []int64{2, 2}, []int64{1, 1}, ConvPadValid, nil, nil, 1)

	b.Output("y", y)

	// Verify output shape: [1, 128, 27, 27]
	// Output size: (56 - 3) / 2 + 1 = 27
	expectedShape := []int64{1, 128, 27, 27}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}
}

func TestConv2DSamePadding(t *testing.T) {
	b := NewBuilder("conv2d_same_test")

	// Create input: [1, 3, 32, 32]
	x := b.Input("x", Float32, 1, 3, 32, 32)

	// Create weight: [16, 3, 3, 3]
	weight := b.Input("weight", Float32, 16, 3, 3, 3)

	// Conv2D with same padding, stride 1
	y := b.Conv(x, weight, []int64{1, 1}, []int64{1, 1}, ConvPadSame, nil, nil, 1)

	b.Output("y", y)

	program := b.Build()

	// With SAME padding and stride 1, output shape should equal input spatial dims
	expectedShape := []int64{1, 16, 32, 32}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify pad_type is "same"
	mainFunc := program.Functions["conv2d_same_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConv := false
	for _, op := range block.Operations {
		if op.Type == "conv" {
			foundConv = true
			break
		}
	}
	if !foundConv {
		t.Error("expected conv operation in program")
	}
}

func TestConv2DCustomPadding(t *testing.T) {
	b := NewBuilder("conv2d_custom_test")

	// Create input: [1, 3, 32, 32]
	x := b.Input("x", Float32, 1, 3, 32, 32)

	// Create weight: [16, 3, 5, 5]
	weight := b.Input("weight", Float32, 16, 3, 5, 5)

	// Conv2D with custom padding
	padBefore := []int64{2, 2}
	padAfter := []int64{2, 2}
	y := b.Conv(x, weight, []int64{1, 1}, []int64{1, 1}, ConvPadCustom, padBefore, padAfter, 1)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape with padding: (32 + 2 + 2 - 5) / 1 + 1 = 32
	expectedShape := []int64{1, 16, 32, 32}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation has pad parameter
	mainFunc := program.Functions["conv2d_custom_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConv := false
	for _, op := range block.Operations {
		if op.Type == "conv" {
			foundConv = true
			// Verify pad parameter exists
			if _, ok := op.Inputs["pad"]; !ok {
				t.Error("expected pad parameter in conv operation with custom padding")
			}
			break
		}
	}
	if !foundConv {
		t.Error("expected conv operation in program")
	}
}

func TestConv2DWithDilation(t *testing.T) {
	b := NewBuilder("conv2d_dilation_test")

	// Create input: [1, 3, 32, 32]
	x := b.Input("x", Float32, 1, 3, 32, 32)

	// Create weight: [16, 3, 3, 3]
	weight := b.Input("weight", Float32, 16, 3, 3, 3)

	// Conv2D with dilation=2, valid padding
	y := b.Conv(x, weight, []int64{1, 1}, []int64{2, 2}, ConvPadValid, nil, nil, 1)

	b.Output("y", y)

	// Verify output shape with dilation
	// Effective kernel size: 2*(3-1)+1 = 5
	// Output size: (32 - 5) / 1 + 1 = 28
	expectedShape := []int64{1, 16, 28, 28}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}
}

func TestConv2DGrouped(t *testing.T) {
	b := NewBuilder("conv2d_grouped_test")

	// Create input: [1, 32, 32, 32]
	x := b.Input("x", Float32, 1, 32, 32, 32)

	// Create weight for grouped conv: [64, 32/4, 3, 3] = [64, 8, 3, 3]
	// groups=4 means each group processes 32/4=8 input channels
	weight := b.Input("weight", Float32, 64, 8, 3, 3)

	// Grouped convolution with 4 groups
	y := b.Conv(x, weight, []int64{1, 1}, []int64{1, 1}, ConvPadSame, nil, nil, 4)

	b.Output("y", y)

	// Verify output shape
	expectedShape := []int64{1, 64, 32, 32}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}
}

func TestConvWithBias(t *testing.T) {
	b := NewBuilder("conv_bias_test")

	// Create input: [1, 3, 32, 32]
	x := b.Input("x", Float32, 1, 3, 32, 32)

	// Create weight: [16, 3, 3, 3]
	weight := b.Input("weight", Float32, 16, 3, 3, 3)

	// Create bias: [16]
	bias := b.Input("bias", Float32, 16)

	// Conv2D with bias
	y := b.ConvWithBias(x, weight, bias, []int64{1, 1}, []int64{1, 1}, ConvPadSame, nil, nil, 1)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape
	expectedShape := []int64{1, 16, 32, 32}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify both conv and add operations are present
	mainFunc := program.Functions["conv_bias_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConv := false
	foundAdd := false
	for _, op := range block.Operations {
		if op.Type == "conv" {
			foundConv = true
		}
		if op.Type == "add" {
			foundAdd = true
		}
	}
	if !foundConv {
		t.Error("expected conv operation in program")
	}
	if !foundAdd {
		t.Error("expected add operation for bias in program")
	}
}

func TestConvTranspose2D(t *testing.T) {
	b := NewBuilder("conv_transpose_test")

	// Create input: [1, 16, 16, 16]
	x := b.Input("x", Float32, 1, 16, 16, 16)

	// Create weight: [16, 32, 3, 3] (C_in=16, C_out=32, kH=3, kW=3)
	weight := b.Input("weight", Float32, 16, 32, 3, 3)

	// ConvTranspose with stride 2, valid padding
	y := b.ConvTranspose(x, weight, []int64{2, 2}, []int64{1, 1}, ConvPadValid, nil, nil, nil, 1)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape
	// Output size: (16-1)*2 + 3 = 33
	expectedShape := []int64{1, 32, 33, 33}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation is present
	mainFunc := program.Functions["conv_transpose_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConvTranspose := false
	for _, op := range block.Operations {
		if op.Type == "conv_transpose" {
			foundConvTranspose = true
			break
		}
	}
	if !foundConvTranspose {
		t.Error("expected conv_transpose operation in program")
	}
}

func TestConvTranspose2DSamePadding(t *testing.T) {
	b := NewBuilder("conv_transpose_same_test")

	// Create input: [1, 16, 16, 16]
	x := b.Input("x", Float32, 1, 16, 16, 16)

	// Create weight: [16, 32, 3, 3]
	weight := b.Input("weight", Float32, 16, 32, 3, 3)

	// ConvTranspose with same padding, stride 2
	y := b.ConvTranspose(x, weight, []int64{2, 2}, []int64{1, 1}, ConvPadSame, nil, nil, nil, 1)

	b.Output("y", y)

	// With SAME padding and stride 2, output = input * stride
	// Output size: 16 * 2 = 32
	expectedShape := []int64{1, 32, 32, 32}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}
}

func TestConvTranspose2DWithOutputPadding(t *testing.T) {
	b := NewBuilder("conv_transpose_output_pad_test")

	// Create input: [1, 16, 16, 16]
	x := b.Input("x", Float32, 1, 16, 16, 16)

	// Create weight: [16, 32, 4, 4]
	weight := b.Input("weight", Float32, 16, 32, 4, 4)

	// ConvTranspose with output padding
	outputPadding := []int64{1, 1}
	y := b.ConvTranspose(x, weight, []int64{2, 2}, []int64{1, 1}, ConvPadValid, nil, nil, outputPadding, 1)

	b.Output("y", y)

	program := b.Build()

	// Verify output shape with output padding
	// Output size: (16-1)*2 + 4 + 1 = 35
	expectedShape := []int64{1, 32, 35, 35}
	for i, dim := range y.shape {
		if dim != expectedShape[i] {
			t.Errorf("dimension %d: expected %d, got %d", i, expectedShape[i], dim)
		}
	}

	// Verify operation has output_padding parameter
	mainFunc := program.Functions["conv_transpose_output_pad_test"]
	block := mainFunc.BlockSpecializations["CoreML7"]

	foundConvTranspose := false
	for _, op := range block.Operations {
		if op.Type == "conv_transpose" {
			foundConvTranspose = true
			// Verify output_padding parameter exists
			if _, ok := op.Inputs["output_padding"]; !ok {
				t.Error("expected output_padding parameter in conv_transpose operation")
			}
			break
		}
	}
	if !foundConvTranspose {
		t.Error("expected conv_transpose operation in program")
	}
}
