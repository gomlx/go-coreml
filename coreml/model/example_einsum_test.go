package model_test

import (
	"fmt"

	"github.com/gomlx/go-darwinml/coreml/model"
)

// ExampleBuilder_Einsum demonstrates how to use the Einsum operation for
// batched matrix multiplication using Einstein summation notation.
//
// This example shows a rank-4 einsum operation that performs batched matrix
// multiplication on tensors with shape [B, C, H, W1] x [B, W1, H, W2] -> [B, C, H, W2].
func ExampleBuilder_Einsum() {
	// Create a new CoreML model builder
	b := model.NewBuilder("einsum_example")

	// Define input tensors
	// x: [2, 3, 4, 5] represents [Batch, Channels, Height, Width1]
	x := b.Input("x", model.Float32, 2, 3, 4, 5)

	// y: [2, 5, 4, 6] represents [Batch, Width1, Height, Width2]
	// Note: Width1 (5) matches between x and y, and Height (4) matches
	y := b.Input("y", model.Float32, 2, 5, 4, 6)

	// Perform einsum with equation "nchw,nwhu->nchu"
	// This performs batched matrix multiplication:
	// - 'n' is the batch dimension (shared across both inputs)
	// - First input has dimensions c, h, w (channels, height, width1)
	// - Second input has dimensions w, h, u (width1, height, width2)
	// - Output has dimensions c, h, u (channels, height, width2)
	result := b.Einsum("nchw,nwhu->nchu", []*model.Value{x, y})

	// Mark as output
	b.Output("result", result)

	// The output shape will be [2, 3, 4, 6]
	// which is [Batch, Channels, Height, Width2]
	fmt.Printf("Output shape: %v\n", result.Shape())

	// Output:
	// Output shape: [2 3 4 6]
}

// ExampleBuilder_Einsum_rank3 demonstrates einsum with rank-3 tensors.
//
// This example shows how to perform matrix multiplication on rank-3 tensors
// without batch dimensions.
func ExampleBuilder_Einsum_rank3() {
	b := model.NewBuilder("einsum_rank3_example")

	// Define rank-3 input tensors
	// x: [3, 4, 5] represents [Channels, Height, Width1]
	x := b.Input("x", model.Float32, 3, 4, 5)

	// y: [5, 4, 6] represents [Width1, Height, Width2]
	y := b.Input("y", model.Float32, 5, 4, 6)

	// Perform einsum with equation "chw,whr->chr"
	// This is similar to the rank-4 case but without batch dimension
	result := b.Einsum("chw,whr->chr", []*model.Value{x, y})

	b.Output("result", result)

	// The output shape will be [3, 4, 6]
	// which is [Channels, Height, Width2]
	fmt.Printf("Output shape: %v\n", result.Shape())

	// Output:
	// Output shape: [3 4 6]
}

// ExampleBuilder_Einsum_broadcasting demonstrates broadcasting in einsum.
//
// This example shows how einsum supports broadcasting on batch and height dimensions.
func ExampleBuilder_Einsum_broadcasting() {
	b := model.NewBuilder("einsum_broadcast_example")

	// x has batch size 1, which will broadcast to match y's batch size
	// x: [1, 3, 4, 5] with Batch=1
	x := b.Input("x", model.Float32, 1, 3, 4, 5)

	// y has batch size 2
	// y: [2, 5, 4, 6] with Batch=2
	y := b.Input("y", model.Float32, 2, 5, 4, 6)

	// Perform einsum - batch dimension broadcasts from 1 to 2
	result := b.Einsum("nchw,nwhu->nchu", []*model.Value{x, y})

	b.Output("result", result)

	// The output shape will be [2, 3, 4, 6]
	// The batch dimension was broadcast from 1 to 2
	fmt.Printf("Output shape: %v\n", result.Shape())

	// Output:
	// Output shape: [2 3 4 6]
}
