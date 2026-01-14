package runtime

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-coreml/blob"
	"github.com/gomlx/go-coreml/model"
)

func TestBlobStorageE2E(t *testing.T) {
	// Create a model with large weights and verify blob storage structure
	b := model.NewBuilder("main")

	// Large weight matrix (4KB = 1024 float32 values)
	weightsData := make([]float32, 1024)
	for i := range weightsData {
		weightsData[i] = float32(i) * 0.001
	}

	// Input: [1, 1024], Weights: [1, 1024], Output: elementwise multiply then sum
	x := b.Input("x", model.Float32, 1, 1024)
	weights := b.Const("weights", model.Float32, []int64{1, 1024}, weightsData)
	product := b.Mul(x, weights)
	y := b.ReduceSum(product, []int64{0, 1}, false)
	b.Output("y", y)

	program := b.Build()
	inputs := []model.FeatureSpec{{Name: "x", DType: model.Float32, Shape: []int64{1, 1024}}}
	outputs := []model.FeatureSpec{{Name: "y", DType: model.Float32, Shape: []int64{}}}

	// Save with blob storage
	opts := model.DefaultBlobOptions()
	opts.BlobThreshold = 1024 // 1KB threshold - our weights are 4KB so should use blob
	coremlModel := model.ToModel(program, inputs, outputs, opts.SerializeOptions)

	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "blob_test.mlpackage")

	if err := model.SaveMLPackageWithBlobs(coremlModel, packagePath, opts); err != nil {
		t.Fatalf("SaveMLPackageWithBlobs() error = %v", err)
	}

	// Verify blob file structure
	blobPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "weights", "weight.bin")
	blobData, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatalf("ReadFile(weight.bin) error = %v", err)
	}

	// Check header
	if len(blobData) < 64 {
		t.Fatalf("blob file too small: %d bytes", len(blobData))
	}

	count := binary.LittleEndian.Uint32(blobData[0:4])
	version := binary.LittleEndian.Uint32(blobData[4:8])

	if version != blob.BlobVersion {
		t.Errorf("blob version = %d, want %d", version, blob.BlobVersion)
	}

	t.Logf("Blob file: %d bytes, %d entries, version %d", len(blobData), count, version)

	// The blob should contain our weight data (4KB + headers/alignment)
	expectedMinSize := 4096 + 64 + 64 // data + header + metadata
	if len(blobData) < expectedMinSize {
		t.Errorf("blob file too small for weight data: got %d, want >= %d", len(blobData), expectedMinSize)
	}

	// Verify metadata at offset 64 (first entry after header)
	if count >= 1 {
		sentinel := binary.LittleEndian.Uint32(blobData[64:68])
		if sentinel != blob.BlobMetadataSentinel {
			t.Errorf("metadata sentinel = %x, want %x", sentinel, blob.BlobMetadataSentinel)
		}

		dtype := binary.LittleEndian.Uint32(blobData[68:72])
		if blob.DataType(dtype) != blob.DataTypeFloat32 {
			t.Errorf("metadata dtype = %d, want %d", dtype, blob.DataTypeFloat32)
		}

		size := binary.LittleEndian.Uint64(blobData[72:80])
		expectedSize := uint64(1024 * 4) // 1024 floats * 4 bytes
		if size != expectedSize {
			t.Errorf("metadata size = %d, want %d", size, expectedSize)
		}

		t.Logf("First blob entry: dtype=%d, size=%d bytes", dtype, size)
	}
}
