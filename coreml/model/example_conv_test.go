package model

import (
	"fmt"
)

// ExampleBuilder_Conv demonstrates basic 2D convolution usage.
func ExampleBuilder_Conv() {
	b := NewBuilder("simple_conv")

	// Input image: batch=1, channels=3 (RGB), height=32, width=32
	x := b.Input("image", Float32, 1, 3, 32, 32)

	// Convolution weights: 16 output channels, 3 input channels, 3x3 kernel
	weights := b.Input("conv_weights", Float32, 16, 3, 3, 3)

	// Apply convolution with stride 1, no dilation, SAME padding
	features := b.Conv(x, weights,
		[]int64{1, 1},      // strides: [stride_h, stride_w]
		[]int64{1, 1},      // dilations: [dilation_h, dilation_w]
		ConvPadSame,        // padding type
		nil, nil,           // custom padding (not used with SAME)
		1,                  // groups (1 = standard convolution)
	)

	b.Output("features", features)

	program := b.Build()
	fmt.Printf("Output shape: %v\n", program.Functions["simple_conv"].BlockSpecializations["CoreML7"].Operations[0].Outputs[0].Type.GetTensorType().Rank)
	// Output: Output shape: 4
}

// ExampleBuilder_ConvWithBias demonstrates convolution with bias.
func ExampleBuilder_ConvWithBias() {
	b := NewBuilder("conv_with_bias")

	// Input: 1x3x32x32
	x := b.Input("image", Float32, 1, 3, 32, 32)

	// Weights: 16x3x3x3
	weights := b.Input("conv_weights", Float32, 16, 3, 3, 3)

	// Bias: 16 values (one per output channel)
	bias := b.Input("conv_bias", Float32, 16)

	// Apply convolution with bias
	output := b.ConvWithBias(x, weights, bias,
		[]int64{1, 1},    // strides
		[]int64{1, 1},    // dilations
		ConvPadSame,      // padding
		nil, nil,         // custom padding
		1,                // groups
	)

	b.Output("output", output)

	_ = b.Build()
	fmt.Printf("Conv with bias applied\n")
	// Output: Conv with bias applied
}

// ExampleBuilder_ConvTranspose demonstrates transposed convolution (upsampling).
func ExampleBuilder_ConvTranspose() {
	b := NewBuilder("upsample")

	// Input: 1x16x16x16 (small feature map)
	x := b.Input("features", Float32, 1, 16, 16, 16)

	// Transposed conv weights: [C_in, C_out, kH, kW] = [16, 32, 4, 4]
	weights := b.Input("deconv_weights", Float32, 16, 32, 4, 4)

	// Upsample by 2x using transposed convolution
	upsampled := b.ConvTranspose(x, weights,
		[]int64{2, 2},    // strides (controls upsampling factor)
		[]int64{1, 1},    // dilations
		ConvPadSame,      // padding
		nil, nil, nil,    // custom padding, output padding
		1,                // groups
	)

	b.Output("upsampled", upsampled)

	_ = b.Build()
	fmt.Printf("Upsampling factor: 2x\n")
	// Output: Upsampling factor: 2x
}

// ExampleBuilder_Conv_grouped demonstrates grouped convolution (depthwise-separable convolution building block).
func ExampleBuilder_Conv_grouped() {
	b := NewBuilder("grouped_conv")

	// Input: 1x32x32x32
	x := b.Input("features", Float32, 1, 32, 32, 32)

	// Grouped conv weights with 4 groups: [64, 32/4, 3, 3] = [64, 8, 3, 3]
	// Each group processes 32/4 = 8 input channels
	weights := b.Input("grouped_weights", Float32, 64, 8, 3, 3)

	// Apply grouped convolution
	output := b.Conv(x, weights,
		[]int64{1, 1},    // strides
		[]int64{1, 1},    // dilations
		ConvPadSame,      // padding
		nil, nil,         // custom padding
		4,                // groups = 4 (depthwise-like)
	)

	b.Output("output", output)

	_ = b.Build()
	fmt.Printf("Groups: 4\n")
	// Output: Groups: 4
}

// ExampleBuilder_Conv_customPadding demonstrates convolution with custom padding.
func ExampleBuilder_Conv_customPadding() {
	b := NewBuilder("custom_pad_conv")

	// Input: 1x3x32x32
	x := b.Input("image", Float32, 1, 3, 32, 32)

	// Weights: 16x3x5x5 (larger 5x5 kernel)
	weights := b.Input("conv_weights", Float32, 16, 3, 5, 5)

	// Custom padding: 2 pixels on each side to maintain spatial dimensions
	padBefore := []int64{2, 2}  // [pad_h_before, pad_w_before]
	padAfter := []int64{2, 2}   // [pad_h_after, pad_w_after]

	output := b.Conv(x, weights,
		[]int64{1, 1},     // strides
		[]int64{1, 1},     // dilations
		ConvPadCustom,     // custom padding mode
		padBefore,         // padding before
		padAfter,          // padding after
		1,                 // groups
	)

	b.Output("output", output)

	_ = b.Build()
	fmt.Printf("Padding: symmetric\n")
	// Output: Padding: symmetric
}
