package blob

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestWriter(t *testing.T) {
	// Create temp dir
	tmpDir := t.TempDir()
	blobPath := filepath.Join(tmpDir, "weight.bin")

	// Create writer
	w, err := NewWriter(blobPath)
	if err != nil {
		t.Fatalf("NewWriter() error = %v", err)
	}

	// Add some blobs
	data1 := make([]byte, 256)
	for i := range data1 {
		data1[i] = byte(i)
	}

	data2 := make([]byte, 512)
	for i := range data2 {
		data2[i] = byte(i * 2)
	}

	offset1, err := w.AddBlob(DataTypeFloat32, data1)
	if err != nil {
		t.Fatalf("AddBlob(1) error = %v", err)
	}

	offset2, err := w.AddBlob(DataTypeInt32, data2)
	if err != nil {
		t.Fatalf("AddBlob(2) error = %v", err)
	}

	// Close to write the file
	if err := w.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Verify offsets are different and aligned
	if offset1 == offset2 {
		t.Errorf("offsets should be different: offset1=%d, offset2=%d", offset1, offset2)
	}
	if offset1%DefaultAlignment != 0 {
		t.Errorf("offset1 should be aligned: %d", offset1)
	}
	if offset2%DefaultAlignment != 0 {
		t.Errorf("offset2 should be aligned: %d", offset2)
	}

	// Read and verify the file
	fileData, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	// Check header
	if len(fileData) < DefaultAlignment {
		t.Fatalf("file too small: %d bytes", len(fileData))
	}

	header := StorageHeader{
		Count:   binary.LittleEndian.Uint32(fileData[0:4]),
		Version: binary.LittleEndian.Uint32(fileData[4:8]),
	}

	if header.Count != 2 {
		t.Errorf("header.Count = %d, want 2", header.Count)
	}
	if header.Version != BlobVersion {
		t.Errorf("header.Version = %d, want %d", header.Version, BlobVersion)
	}

	// Check first metadata at offset1
	if int(offset1)+64 > len(fileData) {
		t.Fatalf("file too small for metadata1")
	}
	meta1Sentinel := binary.LittleEndian.Uint32(fileData[offset1 : offset1+4])
	if meta1Sentinel != BlobMetadataSentinel {
		t.Errorf("meta1.Sentinel = %x, want %x", meta1Sentinel, BlobMetadataSentinel)
	}

	meta1DType := binary.LittleEndian.Uint32(fileData[offset1+4 : offset1+8])
	if DataType(meta1DType) != DataTypeFloat32 {
		t.Errorf("meta1.DType = %d, want %d", meta1DType, DataTypeFloat32)
	}

	meta1Size := binary.LittleEndian.Uint64(fileData[offset1+8 : offset1+16])
	if meta1Size != uint64(len(data1)) {
		t.Errorf("meta1.Size = %d, want %d", meta1Size, len(data1))
	}

	t.Logf("File size: %d bytes", len(fileData))
	t.Logf("Offset1: %d, Offset2: %d", offset1, offset2)
}

func TestWriterEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	blobPath := filepath.Join(tmpDir, "weight.bin")

	w, err := NewWriter(blobPath)
	if err != nil {
		t.Fatalf("NewWriter() error = %v", err)
	}

	// Close without adding any blobs
	if err := w.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Verify header with count=0
	fileData, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	count := binary.LittleEndian.Uint32(fileData[0:4])
	if count != 0 {
		t.Errorf("count = %d, want 0", count)
	}
}

func TestAlignTo(t *testing.T) {
	tests := []struct {
		offset    uint64
		alignment uint64
		want      uint64
	}{
		{0, 64, 0},
		{1, 64, 64},
		{63, 64, 64},
		{64, 64, 64},
		{65, 64, 128},
		{128, 64, 128},
		{100, 0, 100}, // alignment=0 returns unchanged
	}

	for _, tt := range tests {
		got := alignTo(tt.offset, tt.alignment)
		if got != tt.want {
			t.Errorf("alignTo(%d, %d) = %d, want %d", tt.offset, tt.alignment, got, tt.want)
		}
	}
}
