package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	"github.com/gomlx/go-coreml/blob"
	"github.com/gomlx/go-coreml/proto/coreml/milspec"
	"github.com/gomlx/go-coreml/proto/coreml/spec"
	"google.golang.org/protobuf/proto"
)

// BlobSerializeOptions configures blob-based model serialization.
type BlobSerializeOptions struct {
	SerializeOptions

	// UseBlobStorage enables external weight storage in weight.bin.
	UseBlobStorage bool

	// BlobThreshold is the minimum tensor size (bytes) to use blob storage.
	// Tensors smaller than this are kept inline. Default: 1024 bytes.
	BlobThreshold int64

	// SharedWeightsPath, when set, causes SaveMLPackageWithBlobs to symlink
	// weight.bin to this path instead of writing a new one. A null blob writer
	// is used to compute correct protobuf offsets without performing I/O.
	// The file at SharedWeightsPath must have been written by a prior call
	// to SaveMLPackageWithBlobs for the same model (same weights, same order).
	SharedWeightsPath string
}

// DefaultBlobOptions returns default blob serialization options.
func DefaultBlobOptions() BlobSerializeOptions {
	return BlobSerializeOptions{
		SerializeOptions: DefaultOptions(),
		UseBlobStorage:   true,
		BlobThreshold:    1024, // 1KB threshold
	}
}

// SaveMLPackageWithBlobs saves a Model to an .mlpackage directory with blob storage.
// Large tensors are stored in an external weight.bin file for efficient loading.
//
// When opts.SharedWeightsPath is set, the weight.bin is symlinked from the
// shared location instead of being written anew. A null blob writer computes
// the correct protobuf offsets without disk I/O.
func SaveMLPackageWithBlobs(model *spec.Model, path string, opts BlobSerializeOptions) error {
	if !opts.UseBlobStorage {
		// Fall back to regular serialization
		return SaveMLPackage(model, path)
	}

	// Get the MIL program from the model
	mlProgram := model.GetMlProgram()
	if mlProgram == nil {
		// No MIL program, fall back to regular serialization
		return SaveMLPackage(model, path)
	}

	// Create directory structure
	dataDir := filepath.Join(path, "Data", "com.apple.CoreML")
	weightsDir := filepath.Join(dataDir, "weights")

	if err := createDirs(weightsDir); err != nil {
		return err
	}

	blobPath := filepath.Join(weightsDir, "weight.bin")

	var blobWriter *blob.Writer
	if opts.SharedWeightsPath != "" {
		// Shared weights: use a null writer for offset computation only,
		// then symlink to the existing weight.bin.
		blobWriter = blob.NewNullWriter()
	} else {
		var err error
		blobWriter, err = blob.NewWriter(blobPath)
		if err != nil {
			return fmt.Errorf("create blob writer: %w", err)
		}
	}

	// Walk program and extract large tensors to blob file
	if err := extractTensorsToBlob(mlProgram, blobWriter, opts.BlobThreshold); err != nil {
		blobWriter.Close()
		return fmt.Errorf("extract tensors to blob: %w", err)
	}

	// Close blob writer (no-op for null writer)
	if err := blobWriter.Close(); err != nil {
		return fmt.Errorf("close blob writer: %w", err)
	}

	// Create symlink for shared weights
	if opts.SharedWeightsPath != "" && blobWriter.EntryCount() > 0 {
		if err := os.Symlink(opts.SharedWeightsPath, blobPath); err != nil {
			return fmt.Errorf("symlink shared weights: %w", err)
		}
	}

	// Write model and manifest
	return saveModelAndManifest(model, path, dataDir, blobWriter.EntryCount() > 0)
}

// extractTensorsToBlob walks the program and replaces large ImmediateValue tensors
// with BlobFileValue references.
func extractTensorsToBlob(program *milspec.Program, w *blob.Writer, threshold int64) error {
	// Walk all functions in the program
	for _, fn := range program.GetFunctions() {
		if err := extractFromFunction(fn, w, threshold); err != nil {
			return err
		}
	}
	return nil
}

// extractFromFunction extracts tensors from a function's blocks and operations.
func extractFromFunction(fn *milspec.Function, w *blob.Writer, threshold int64) error {
	for _, block := range fn.GetBlockSpecializations() {
		if err := extractFromBlock(block, w, threshold); err != nil {
			return err
		}
	}
	return nil
}

// extractFromBlock extracts tensors from a block's operations.
func extractFromBlock(block *milspec.Block, w *blob.Writer, threshold int64) error {
	for _, op := range block.GetOperations() {
		if err := extractFromOperation(op, w, threshold); err != nil {
			return err
		}
	}
	return nil
}

// extractFromOperation extracts tensors from an operation's inputs and nested blocks.
func extractFromOperation(op *milspec.Operation, w *blob.Writer, threshold int64) error {
	// Extract from inputs
	for _, arg := range op.GetInputs() {
		for _, binding := range arg.GetArguments() {
			if val := binding.GetValue(); val != nil {
				if err := maybeExtractValue(val, w, threshold); err != nil {
					return err
				}
			}
		}
	}

	// Recursively extract from nested blocks (for control flow operations like cond, while_loop)
	for _, nestedBlock := range op.GetBlocks() {
		if err := extractFromBlock(nestedBlock, w, threshold); err != nil {
			return err
		}
	}

	return nil
}

// maybeExtractValue checks if a value should be extracted to blob storage.
func maybeExtractValue(val *milspec.Value, w *blob.Writer, threshold int64) error {
	imm := val.GetImmediateValue()
	if imm == nil {
		return nil // Not an immediate value
	}

	tensor := imm.GetTensor()
	if tensor == nil {
		return nil // Not a tensor
	}

	// Calculate tensor size
	data, dtype, size := extractTensorData(tensor, val.GetType())
	if size < threshold {
		return nil // Too small, keep inline
	}

	// Write to blob
	offset, err := w.AddBlob(dtype, data)
	if err != nil {
		return fmt.Errorf("add blob: %w", err)
	}

	// Replace ImmediateValue with BlobFileValue
	val.Value = &milspec.Value_BlobFileValue_{
		BlobFileValue: &milspec.Value_BlobFileValue{
			FileName: blob.DefaultBlobFilename,
			Offset:   offset,
		},
	}

	return nil
}

// dataTypeToBlobType converts a milspec.DataType to a blob.DataType.
func dataTypeToBlobType(dt milspec.DataType) blob.DataType {
	switch dt {
	case milspec.DataType_FLOAT16:
		return blob.DataTypeFloat16
	case milspec.DataType_FLOAT32:
		return blob.DataTypeFloat32
	case milspec.DataType_FLOAT64:
		return blob.DataTypeFloat32 // CoreML uses float32
	case milspec.DataType_INT8:
		return blob.DataTypeInt8
	case milspec.DataType_INT16:
		return blob.DataTypeInt16
	case milspec.DataType_INT32:
		return blob.DataTypeInt32
	case milspec.DataType_INT64:
		return blob.DataTypeInt32 // CoreML uses int32
	case milspec.DataType_BOOL:
		return blob.DataTypeUInt8
	default:
		return blob.DataTypeFloat32
	}
}

// extractTensorData extracts raw bytes from a TensorValue.
func extractTensorData(tensor *milspec.TensorValue, valType *milspec.ValueType) ([]byte, blob.DataType, int64) {
	// Get dtype from value type
	dtype := blob.DataTypeFloat32 // default
	if tt := valType.GetTensorType(); tt != nil {
		dtype = dataTypeToBlobType(tt.GetDataType())
	}

	// Extract data based on tensor type
	if floats := tensor.GetFloats(); floats != nil {
		data := floatsToBytes(floats.GetValues())
		return data, blob.DataTypeFloat32, int64(len(data))
	}

	if doubles := tensor.GetDoubles(); doubles != nil {
		data := doublesToBytes(doubles.GetValues())
		return data, blob.DataTypeFloat32, int64(len(data)) // CoreML uses float32
	}

	if ints := tensor.GetInts(); ints != nil {
		data := intsToBytes(ints.GetValues())
		return data, blob.DataTypeInt32, int64(len(data))
	}

	if longInts := tensor.GetLongInts(); longInts != nil {
		data := int64sToBytes(longInts.GetValues())
		return data, blob.DataTypeInt32, int64(len(data)) // CoreML uses int32
	}

	if bytes := tensor.GetBytes(); bytes != nil {
		data := bytes.GetValues()
		return data, dtype, int64(len(data))
	}

	return nil, dtype, 0
}

// Helper functions to convert typed slices to raw bytes

func floatsToBytes(vals []float32) []byte {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return data
}

func doublesToBytes(vals []float64) []byte {
	// Convert to float32 for CoreML
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(v)))
	}
	return data
}

func intsToBytes(vals []int32) []byte {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(v))
	}
	return data
}

func int64sToBytes(vals []int64) []byte {
	// Convert to int32 for CoreML
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(v))
	}
	return data
}

// createDirs creates the directory structure for an mlpackage.
func createDirs(weightsDir string) error {
	return os.MkdirAll(weightsDir, 0755)
}

// saveModelAndManifest writes the model.mlmodel and Manifest.json files.
func saveModelAndManifest(model *spec.Model, packagePath, dataDir string, hasWeights bool) error {
	// Write model.mlmodel
	modelPath := filepath.Join(dataDir, "model.mlmodel")
	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("marshal model: %w", err)
	}

	if err := os.WriteFile(modelPath, data, 0644); err != nil {
		return fmt.Errorf("write model: %w", err)
	}

	// Write Manifest.json
	modelUUID := uuid.New().String()

	itemEntries := map[string]any{
		modelUUID: map[string]any{
			"author":      "com.apple.CoreML",
			"description": "CoreML Model Specification",
			"name":        "model.mlmodel",
			"path":        "com.apple.CoreML/model.mlmodel",
		},
	}

	// Add weights entry if we have blob data
	if hasWeights {
		weightsUUID := uuid.New().String()
		itemEntries[weightsUUID] = map[string]any{
			"author":      "com.apple.CoreML",
			"description": "CoreML Model Weights",
			"name":        "weight.bin",
			"path":        "com.apple.CoreML/weights/weight.bin",
		}
	}

	manifest := map[string]any{
		"fileFormatVersion":   "1.0.0",
		"itemInfoEntries":     itemEntries,
		"rootModelIdentifier": modelUUID,
	}

	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}

	manifestPath := filepath.Join(packagePath, "Manifest.json")
	if err := os.WriteFile(manifestPath, manifestData, 0644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	return nil
}
