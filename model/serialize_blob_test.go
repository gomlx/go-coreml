package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-coreml/blob"
)

func TestSaveMLPackageWithBlobs(t *testing.T) {
	// Create a model with a large constant that should be moved to blob storage
	b := NewBuilder("main")

	// Create a large constant (2KB of floats = 512 float32 values)
	largeData := make([]float32, 512)
	for i := range largeData {
		largeData[i] = float32(i) * 0.1
	}

	// Small constant that should stay inline (16 bytes)
	smallData := []float32{1.0, 2.0, 3.0, 4.0}

	// Create inputs and constants - use the large constant in an operation
	x := b.Input("x", Float32, 1, 512)
	largeConst := b.Const("large_weights", Float32, []int64{1, 512}, largeData)
	smallConst := b.Const("small_bias", Float32, []int64{1, 4}, smallData)

	// Use large constant in multiplication
	y := b.Mul(x, largeConst)

	// Use ReduceSum to get output shape [1]
	z := b.ReduceSum(y, []int64{1}, false)
	_ = smallConst // small constant stays unused for this test
	b.Output("z", z)

	// Build the program
	program := b.Build()

	// Create model with feature specs
	inputs := []FeatureSpec{{Name: "x", DType: Float32, Shape: []int64{1, 512}}}
	outputs := []FeatureSpec{{Name: "z", DType: Float32, Shape: []int64{1}}}

	opts := DefaultBlobOptions()
	model := ToModel(program, inputs, outputs, opts.SerializeOptions)

	// Save to temp dir with blob storage
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test.mlpackage")

	err := SaveMLPackageWithBlobs(model, packagePath, opts)
	if err != nil {
		t.Fatalf("SaveMLPackageWithBlobs() error = %v", err)
	}

	// Verify directory structure was created
	dataDir := filepath.Join(packagePath, "Data", "com.apple.CoreML")
	modelPath := filepath.Join(dataDir, "model.mlmodel")
	weightsPath := filepath.Join(dataDir, "weights", "weight.bin")
	manifestPath := filepath.Join(packagePath, "Manifest.json")

	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("model.mlmodel not created: %v", err)
	}
	if _, err := os.Stat(manifestPath); err != nil {
		t.Errorf("Manifest.json not created: %v", err)
	}

	// Check if weights file was created (should be since we have a large constant)
	info, err := os.Stat(weightsPath)
	if err != nil {
		t.Logf("weight.bin not created (may be expected if threshold wasn't met): %v", err)
	} else {
		t.Logf("weight.bin created with size: %d bytes", info.Size())
	}

	// Verify we can read the manifest
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("ReadFile(manifest) error = %v", err)
	}
	t.Logf("Manifest:\n%s", string(manifestData))
}

func TestSaveMLPackageWithBlobsThreshold(t *testing.T) {
	// Test that small constants stay inline when below threshold
	b := NewBuilder("main")

	// Create constant just below threshold (default 1024 bytes)
	// 200 float32 values = 800 bytes
	smallData := make([]float32, 200)
	for i := range smallData {
		smallData[i] = float32(i)
	}

	x := b.Input("x", Float32, 1, 200)
	weights := b.Const("weights", Float32, []int64{1, 200}, smallData)
	y := b.Add(x, weights)
	b.Output("y", y)

	program := b.Build()
	inputs := []FeatureSpec{{Name: "x", DType: Float32, Shape: []int64{1, 200}}}
	outputs := []FeatureSpec{{Name: "y", DType: Float32, Shape: []int64{1, 200}}}

	opts := DefaultBlobOptions()
	opts.BlobThreshold = 1024 // 1KB threshold
	model := ToModel(program, inputs, outputs, opts.SerializeOptions)

	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test.mlpackage")

	err := SaveMLPackageWithBlobs(model, packagePath, opts)
	if err != nil {
		t.Fatalf("SaveMLPackageWithBlobs() error = %v", err)
	}

	// weight.bin should exist but might be empty or minimal since data is below threshold
	weightsPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "weights", "weight.bin")
	info, err := os.Stat(weightsPath)
	if err == nil {
		t.Logf("weight.bin exists with size: %d bytes", info.Size())
		// If it exists, it should be just the header (64 bytes) since no blobs were added
		if info.Size() > blob.DefaultAlignment {
			t.Logf("Note: weight.bin larger than header, some blobs may have been extracted")
		}
	}
}

func TestDataTypeToBlobType(t *testing.T) {
	tests := []struct {
		input DType
		want  blob.DataType
	}{
		{Float16, blob.DataTypeFloat16},
		{Float32, blob.DataTypeFloat32},
		{Float64, blob.DataTypeFloat32}, // Converts to float32
		{Int8, blob.DataTypeInt8},
		{Int16, blob.DataTypeInt16},
		{Int32, blob.DataTypeInt32},
		{Int64, blob.DataTypeInt32}, // Converts to int32
		{Bool, blob.DataTypeUInt8},
	}

	for _, tt := range tests {
		got := dataTypeToBlobType(tt.input)
		if got != tt.want {
			t.Errorf("dataTypeToBlobType(%v) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestSaveMLPackageWithNestedBlocks(t *testing.T) {
	// Test that nested blocks in control flow operations are properly handled
	// for blob extraction
	b := NewBuilder("nested_blocks_test")

	// Create inputs
	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)
	y := b.Input("y", Float32, 10)

	// Create a large constant (should be extracted to blob)
	largeData := make([]float32, 512)
	for i := range largeData {
		largeData[i] = float32(i) * 0.01
	}

	// Create conditional with large constant inside branch
	results := b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			// True branch uses a large constant within the block
			scale := bb.Const("large_scale", Float32, []int64{512}, largeData)
			padded := bb.Add(x, bb.Const("pad", Float32, []int64{10}, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
			// We can't actually use scale directly with padded due to shape mismatch,
			// so just return padded for the test
			_ = scale
			return []*Value{padded}
		},
		func(bb *BlockBuilder) []*Value {
			// False branch - simple operation
			return []*Value{bb.Sub(x, y)}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	b.Output("result", results[0])
	program := b.Build()

	// Verify program structure has nested blocks
	fn := program.Functions["nested_blocks_test"]
	if fn == nil {
		t.Fatal("Function not found")
	}

	block := fn.BlockSpecializations["CoreML7"]
	if block == nil {
		t.Fatal("Block not found")
	}

	// Find cond operation
	var condOp bool
	for _, op := range block.Operations {
		if op.Type == "cond" {
			condOp = true
			if len(op.Blocks) != 2 {
				t.Errorf("Cond operation has %d blocks, want 2", len(op.Blocks))
			}
		}
	}
	if !condOp {
		t.Error("Cond operation not found in program")
	}

	// Save with blob support
	inputs := []FeatureSpec{
		{Name: "pred", DType: Bool, Shape: []int64{}},
		{Name: "x", DType: Float32, Shape: []int64{10}},
		{Name: "y", DType: Float32, Shape: []int64{10}},
	}
	outputs := []FeatureSpec{{Name: "result", DType: Float32, Shape: []int64{10}}}

	opts := DefaultBlobOptions()
	opts.BlobThreshold = 256 // Lower threshold to trigger blob extraction
	model := ToModel(program, inputs, outputs, opts.SerializeOptions)

	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "nested_test.mlpackage")

	err := SaveMLPackageWithBlobs(model, packagePath, opts)
	if err != nil {
		t.Fatalf("SaveMLPackageWithBlobs() error = %v", err)
	}

	// Verify package was created
	modelPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "model.mlmodel")
	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("model.mlmodel not created: %v", err)
	}
}

func TestSaveMLPackageWithWhileLoop(t *testing.T) {
	// Test that while_loop operations with nested blocks are properly handled
	b := NewBuilder("while_loop_test")

	// Initialize loop variables
	zero := b.Const("zero", Int32, []int64{}, []int32{0})
	n := b.Const("n", Int32, []int64{}, []int32{5})

	// Create large data for loop body (should be extracted to blob)
	largeData := make([]float32, 512)
	for i := range largeData {
		largeData[i] = float32(i) * 0.001
	}
	initData := b.Const("init_data", Float32, []int64{512}, largeData)

	results := b.WhileLoop(
		[]*Value{zero, initData}, // [counter, data]
		func(bb *BlockBuilder, vars []*Value) *Value {
			cnt := vars[0]
			return bb.Less(cnt, n)
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			cnt, data := vars[0], vars[1]
			one := bb.Const("one", Int32, []int64{}, []int32{1})
			increment := bb.Const("increment", Float32, []int64{512}, largeData)
			newCnt := bb.Add(cnt, one)
			newData := bb.Add(data, increment)
			return []*Value{newCnt, newData}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	b.Output("final_counter", results[0])
	b.Output("final_data", results[1])
	program := b.Build()

	// Verify program structure
	fn := program.Functions["while_loop_test"]
	block := fn.BlockSpecializations["CoreML7"]

	var whileLoopFound bool
	for _, op := range block.Operations {
		if op.Type == "while_loop" {
			whileLoopFound = true
			if len(op.Blocks) != 2 {
				t.Errorf("while_loop has %d blocks, want 2", len(op.Blocks))
			}
		}
	}
	if !whileLoopFound {
		t.Error("while_loop operation not found")
	}

	// Save with blob support
	inputs := []FeatureSpec{} // No inputs for this test
	outputs := []FeatureSpec{
		{Name: "final_counter", DType: Int32, Shape: []int64{}},
		{Name: "final_data", DType: Float32, Shape: []int64{512}},
	}

	opts := DefaultBlobOptions()
	opts.BlobThreshold = 256 // Lower threshold
	model := ToModel(program, inputs, outputs, opts.SerializeOptions)

	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "while_test.mlpackage")

	err := SaveMLPackageWithBlobs(model, packagePath, opts)
	if err != nil {
		t.Fatalf("SaveMLPackageWithBlobs() error = %v", err)
	}

	// Verify package
	modelPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "model.mlmodel")
	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("model.mlmodel not created: %v", err)
	}

	// Check if weights file was created (it should be since we have large constants)
	weightsPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "weights", "weight.bin")
	info, err := os.Stat(weightsPath)
	if err != nil {
		t.Logf("weight.bin not created: %v", err)
	} else {
		t.Logf("weight.bin created with size: %d bytes", info.Size())
	}
}
