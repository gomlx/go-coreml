package blob

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Writer writes blob data to a weight.bin file.
//
// Usage:
//
//	w, err := blob.NewWriter("weights/weight.bin")
//	if err != nil { ... }
//	defer w.Close()
//
//	offset, err := w.AddBlob(blob.DataTypeFloat32, weightData)
//	// Use offset in BlobFileValue.offset
type Writer struct {
	file    *os.File
	offset  uint64 // Current write position
	entries []blobEntry
}

type blobEntry struct {
	metadataOffset uint64
	dataOffset     uint64
	dtype          DataType
	data           []byte
}

// NewWriter creates a new blob writer that writes to the specified path.
func NewWriter(path string) (*Writer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create blob file: %w", err)
	}

	w := &Writer{
		file:    f,
		offset:  DefaultAlignment, // Start after the header (64 bytes)
		entries: nil,
	}

	return w, nil
}

// NewNullWriter creates a writer that computes offsets without writing to disk.
// Used when sharing weight.bin across multiple compilations — the protobuf is
// mutated to reference BlobFileValue offsets, but no data is written because
// the actual weight.bin is symlinked from a shared location.
func NewNullWriter() *Writer {
	return &Writer{
		offset:  DefaultAlignment,
		entries: nil,
		// file is nil — AddBlob tracks offsets but doesn't write
	}
}

// AddBlob adds a blob to the file and returns the metadata offset.
// This offset should be stored in BlobFileValue.offset in the protobuf.
func (w *Writer) AddBlob(dtype DataType, data []byte) (uint64, error) {
	// Metadata goes at current offset
	metadataOffset := w.offset

	// Data follows metadata (both 64-byte aligned)
	dataOffset := alignTo(metadataOffset+DefaultAlignment, DefaultAlignment)

	// Store entry for writing during Close()
	w.entries = append(w.entries, blobEntry{
		metadataOffset: metadataOffset,
		dataOffset:     dataOffset,
		dtype:          dtype,
		data:           data,
	})

	// Advance offset past data (aligned for next entry)
	w.offset = alignTo(dataOffset+uint64(len(data)), DefaultAlignment)

	return metadataOffset, nil
}

// Close finalizes the blob file by writing the header and all entries.
func (w *Writer) Close() error {
	if w.file == nil {
		return nil
	}

	// Write header at offset 0
	header := StorageHeader{
		Count:   uint32(len(w.entries)),
		Version: BlobVersion,
	}

	if err := w.writeStructAt(0, &header); err != nil {
		w.file.Close()
		return fmt.Errorf("write header: %w", err)
	}

	// Write each entry (metadata + data)
	for _, entry := range w.entries {
		// Create and write metadata
		metadata := BlobMetadata{
			Sentinel:          BlobMetadataSentinel,
			MilDType:          uint32(entry.dtype),
			SizeInBytes:       uint64(len(entry.data)),
			Offset:            entry.dataOffset,
			PaddingSizeInBits: 0,
		}

		if err := w.writeStructAt(int64(entry.metadataOffset), &metadata); err != nil {
			w.file.Close()
			return fmt.Errorf("write metadata at offset %d: %w", entry.metadataOffset, err)
		}

		// Write data
		if _, err := w.file.WriteAt(entry.data, int64(entry.dataOffset)); err != nil {
			w.file.Close()
			return fmt.Errorf("write data at offset %d: %w", entry.dataOffset, err)
		}
	}

	return w.file.Close()
}

// writeStructAt writes a struct at the specified offset using little-endian encoding.
func (w *Writer) writeStructAt(offset int64, data interface{}) error {
	if _, err := w.file.Seek(offset, io.SeekStart); err != nil {
		return err
	}
	return binary.Write(w.file, binary.LittleEndian, data)
}

// EntryCount returns the number of blob entries added.
func (w *Writer) EntryCount() int {
	return len(w.entries)
}
