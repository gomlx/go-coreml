# Spec: Weight Blob Support for Large Models

## Summary

Implement external weight storage using CoreML's blob file format. Currently all tensor data (weights, constants) is embedded inline in the protobuf using `ImmediateValue`. For large models, this is inefficient - the blob format stores weights in a separate binary file that can be memory-mapped.

## Problem

- Embedding gigabytes of weights inline in protobuf is slow and memory-intensive
- Protobuf parsing loads everything into memory at once
- Large models (LLMs, vision transformers) become impractical

## Solution

CoreML's MIL format supports `BlobFileValue` which references external binary files:

```protobuf
message Value {
    oneof value {
        ImmediateValue immediateValue = 3;
        BlobFileValue blobFileValue = 5;  // External storage
    }
}

message BlobFileValue {
    string fileName = 1;   // "@model_path/weights/weight.bin"
    uint64 offset = 2;     // Byte offset to blob_metadata
}
```

## Blob File Format

Based on coremltools' MILBlob implementation.

### File Layout

```
[storage_header (64B)]
[blob_metadata_0 (64B)] [data_0 (padded to 64B alignment)]
[blob_metadata_1 (64B)] [data_1 (padded to 64B alignment)]
...
```

### storage_header (64 bytes)

```go
type StorageHeader struct {
    Count    uint32   // Number of blob entries
    Version  uint32   // Always 2
    Reserved [48]byte // Zero padding
}
```

### blob_metadata (64 bytes)

```go
type BlobMetadata struct {
    Sentinel          uint32   // 0xDEADBEEF
    MilDType          uint32   // BlobDataType enum
    SizeInBytes       uint64   // Size of raw data
    Offset            uint64   // Absolute file offset to data
    PaddingSizeInBits uint64   // For sub-byte types (0 for normal types)
    Reserved          [32]byte // Zero padding
}
```

### BlobDataType Enum

| Type     | Code |
|----------|------|
| Float16  | 1    |
| Float32  | 2    |
| UInt8    | 3    |
| Int8     | 4    |
| BFloat16 | 5    |
| Int16    | 6    |
| UInt16   | 7    |
| Int32    | 14   |
| UInt32   | 15   |

### Constants

- `DefaultAlignment = 64`
- `BlobMetadataSentinel = 0xDEADBEEF`
- `BlobVersion = 2`

## Implementation Plan

### 1. New Package: `blob/`

**blob/format.go** - Data structures
```go
package blob

const (
    DefaultAlignment      = 64
    BlobMetadataSentinel  = 0xDEADBEEF
    BlobVersion           = 2
)

type DataType uint32

const (
    DataTypeFloat16 DataType = 1
    DataTypeFloat32 DataType = 2
    // ...
)

type StorageHeader struct {
    Count    uint32
    Version  uint32
    Reserved [48]byte
}

type BlobMetadata struct {
    Sentinel          uint32
    MilDType          uint32
    SizeInBytes       uint64
    Offset            uint64
    PaddingSizeInBits uint64
    Reserved          [32]byte
}
```

**blob/writer.go** - Write weight.bin files
```go
type Writer struct {
    file       *os.File
    offset     uint64
    entries    []BlobMetadata
}

func NewWriter(path string) (*Writer, error)
func (w *Writer) AddBlob(dtype DataType, data []byte) (offset uint64, err error)
func (w *Writer) Close() error
```

### 2. Update model/serialize.go

Add blob-aware serialization:

```go
type SerializeOptions struct {
    // ... existing fields ...

    // UseBlobStorage enables external weight storage
    UseBlobStorage bool
    // BlobThreshold is the minimum tensor size (bytes) to use blob storage
    BlobThreshold int64 // default: 1024
}

func SaveMLPackageWithBlobs(model *spec.Model, path string, opts SerializeOptions) error
```

### 3. Update model/builder.go

Track constants for potential blob export:

```go
type Builder struct {
    // ... existing fields ...

    // weights tracks constant tensors that may be exported to blobs
    weights []*weightEntry
}

type weightEntry struct {
    name    string
    dtype   DType
    data    []byte
    milVal  *milspec.Value
}
```

### 4. Integration

When `UseBlobStorage` is enabled:
1. During `SaveMLPackage`, scan for large constants
2. Write them to `weights/weight.bin` using blob format
3. Replace `ImmediateValue` with `BlobFileValue` referencing the blob
4. Update manifest to include weight file

## API Changes

### Option 1: Builder Option (Recommended)

```go
// Enable blob storage during model construction
b := model.NewBuilder("main", model.WithBlobStorage(true))

// Threshold can be configured
b := model.NewBuilder("main", model.WithBlobStorage(true), model.WithBlobThreshold(4096))
```

### Option 2: Serialize Option

```go
// Enable blob storage at serialization time
opts := model.SerializeOptions{
    UseBlobStorage: true,
    BlobThreshold:  1024,
}
model.SaveMLPackage(m, path, opts)
```

Recommend Option 1 since weights need to be tracked during construction.

## Testing

1. **Unit tests** for blob writer (format correctness)
2. **Round-trip test**: create model with large constant, save with blobs, load and verify
3. **Threshold test**: verify small constants stay inline
4. **Integration test**: verify CoreML can load blob-backed models

## File Changes

```
blob/
├── format.go      # NEW: Data structures and constants
├── writer.go      # NEW: Blob file writer
└── writer_test.go # NEW: Tests

model/
├── builder.go     # MODIFY: Add weight tracking, blob options
├── serialize.go   # MODIFY: Add blob-aware serialization
└── serialize_test.go # MODIFY: Add blob tests
```

## References

- [coremltools MIL.proto](https://github.com/apple/coremltools/blob/main/mlmodel/format/MIL.proto)
- [coremltools MILBlob StorageFormat.hpp](https://github.com/apple/coremltools/blob/main/mlmodel/src/MILBlob/Blob/StorageFormat.hpp)
- [coremltools MILBlob BlobDataType.hpp](https://github.com/apple/coremltools/blob/main/mlmodel/src/MILBlob/Blob/BlobDataType.hpp)
