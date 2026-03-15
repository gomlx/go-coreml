package model_test

import (
	"fmt"

	"github.com/gomlx/go-darwinml/coreml/model"
)

// ExampleBuilder_Concat demonstrates concatenating multiple tensors along an axis.
func ExampleBuilder_Concat() {
	b := model.NewBuilder("concat_example")

	// Create three input tensors with different sizes along axis 1
	x := b.Input("x", model.Float32, 2, 3)  // [2, 3]
	y := b.Input("y", model.Float32, 2, 5)  // [2, 5]
	z := b.Input("z", model.Float32, 2, 2)  // [2, 2]

	// Concatenate along axis 1 (columns)
	// Result shape: [2, 10] (3 + 5 + 2 = 10)
	concat := b.Concat([]*model.Value{x, y, z}, 1)

	b.Output("output", concat)

	// Output shape should be [2, 10]
	fmt.Printf("Output shape: %v\n", concat.Shape())
	// Output: Output shape: [2 10]
}

// ExampleBuilder_Concat_negativeAxis shows using negative axis indices.
func ExampleBuilder_Concat_negativeAxis() {
	b := model.NewBuilder("concat_neg")

	x := b.Input("x", model.Float32, 2, 3, 4)
	y := b.Input("y", model.Float32, 2, 3, 5)

	// Concatenate along last axis using -1
	// Result shape: [2, 3, 9] (4 + 5 = 9)
	concat := b.Concat([]*model.Value{x, y}, -1)

	b.Output("output", concat)

	fmt.Printf("Output shape: %v\n", concat.Shape())
	// Output: Output shape: [2 3 9]
}

// ExampleBuilder_Concat_batchDimension shows concatenating along the batch dimension.
func ExampleBuilder_Concat_batchDimension() {
	b := model.NewBuilder("concat_batch")

	// Create tensors with different batch sizes
	batch1 := b.Input("batch1", model.Float32, 2, 3, 4)  // [2, 3, 4]
	batch2 := b.Input("batch2", model.Float32, 5, 3, 4)  // [5, 3, 4]

	// Concatenate along axis 0 (batch dimension)
	// Result shape: [7, 3, 4] (2 + 5 = 7)
	concat := b.Concat([]*model.Value{batch1, batch2}, 0)

	b.Output("output", concat)

	fmt.Printf("Output shape: %v\n", concat.Shape())
	// Output: Output shape: [7 3 4]
}

// ExampleBuilder_Concat_withConstants shows concatenating variables with constants.
func ExampleBuilder_Concat_withConstants() {
	b := model.NewBuilder("concat_const")

	// Input tensor
	x := b.Input("x", model.Float32, 2, 3)

	// Constant tensor
	constTensor := b.Const("padding", model.Float32, []int64{2, 2}, []float32{0, 0, 0, 0})

	// Concatenate along axis 1
	// Result shape: [2, 5] (3 + 2 = 5)
	concat := b.Concat([]*model.Value{x, constTensor}, 1)

	b.Output("output", concat)

	fmt.Printf("Output shape: %v\n", concat.Shape())
	// Output: Output shape: [2 5]
}
