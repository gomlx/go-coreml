// Package blob implements CoreML's weight blob file format for storing
// large tensor data externally from the model protobuf.
//
// The blob format stores weights in a binary file (typically weight.bin)
// that can be memory-mapped for efficient loading. This is essential for
// large models where embedding weights inline in protobuf would be slow.
//
// File format:
//
//	[storage_header (64B)]
//	[blob_metadata_0 (64B)] [data_0 (64B aligned)]
//	[blob_metadata_1 (64B)] [data_1 (64B aligned)]
//	...
//
// Reference: https://github.com/apple/coremltools/blob/main/mlmodel/src/MILBlob/Blob/StorageFormat.hpp
package blob

const (
	// DefaultAlignment is the byte alignment for all sections in the blob file.
	DefaultAlignment = 64

	// BlobMetadataSentinel is a magic number used to validate blob metadata entries.
	BlobMetadataSentinel uint32 = 0xDEADBEEF

	// BlobVersion is the current blob file format version.
	BlobVersion uint32 = 2

	// DefaultBlobFilename is the standard name for the weight blob file.
	DefaultBlobFilename = "@model_path/weights/weight.bin"
)

// DataType represents the data type of a blob entry.
// These values must match coremltools' BlobDataType enum.
type DataType uint32

const (
	DataTypeFloat16 DataType = 1
	DataTypeFloat32 DataType = 2
	DataTypeUInt8   DataType = 3
	DataTypeInt8    DataType = 4
	DataTypeBFloat16 DataType = 5
	DataTypeInt16   DataType = 6
	DataTypeUInt16  DataType = 7
	DataTypeInt4    DataType = 8
	DataTypeUInt1   DataType = 9
	DataTypeUInt2   DataType = 10
	DataTypeUInt4   DataType = 11
	DataTypeUInt3   DataType = 12
	DataTypeUInt6   DataType = 13
	DataTypeInt32   DataType = 14
	DataTypeUInt32  DataType = 15
)

// StorageHeader is the file header for a blob storage file.
// It is always 64 bytes and appears at the start of the file.
type StorageHeader struct {
	Count    uint32   // Number of blob entries in the file
	Version  uint32   // Format version (always BlobVersion)
	Reserved [48]byte // Reserved for future use, must be zero
}

// BlobMetadata describes a single blob entry in the file.
// It is always 64 bytes and precedes the blob data.
type BlobMetadata struct {
	Sentinel          uint32   // Magic number (BlobMetadataSentinel)
	MilDType          uint32   // Data type (DataType enum)
	SizeInBytes       uint64   // Size of the blob data in bytes
	Offset            uint64   // Absolute file offset to the blob data
	PaddingSizeInBits uint64   // Unused bits for sub-byte types (0 for normal types)
	Reserved          [32]byte // Reserved for future use, must be zero
}


// alignTo returns the smallest multiple of alignment >= offset.
func alignTo(offset uint64, alignment uint64) uint64 {
	if alignment == 0 {
		return offset
	}
	remainder := offset % alignment
	if remainder == 0 {
		return offset
	}
	return offset + (alignment - remainder)
}
