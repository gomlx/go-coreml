package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestToModel(t *testing.T) {
	// Build a simple MIL program
	b := NewBuilder("main")
	x := b.Input("x", Float32, 2, 3)
	y := b.Relu(x)
	b.Output("y", y)
	program := b.Build()

	// Convert to CoreML Model
	inputs := []FeatureSpec{{
		Name:        "x",
		Description: "Input tensor",
		DType:       Float32,
		Shape:       []int64{2, 3},
	}}
	outputs := []FeatureSpec{{
		Name:        "y",
		Description: "Output tensor",
		DType:       Float32,
		Shape:       []int64{2, 3},
	}}

	model := ToModel(program, inputs, outputs, DefaultOptions())

	// Verify model structure
	if model.SpecificationVersion != SpecVersion {
		t.Errorf("expected spec version %d, got %d", SpecVersion, model.SpecificationVersion)
	}

	if model.Description == nil {
		t.Fatal("expected model description")
	}

	if len(model.Description.Input) != 1 {
		t.Errorf("expected 1 input, got %d", len(model.Description.Input))
	}

	if len(model.Description.Output) != 1 {
		t.Errorf("expected 1 output, got %d", len(model.Description.Output))
	}

	if model.GetMlProgram() == nil {
		t.Fatal("expected mlProgram to be set")
	}
}

func TestSaveMLPackage(t *testing.T) {
	// Build a simple MIL program
	b := NewBuilder("main")
	x := b.Input("x", Float32, 2, 3)
	y := b.Relu(x)
	b.Output("y", y)
	program := b.Build()

	// Convert to CoreML Model
	inputs := []FeatureSpec{{
		Name:        "x",
		Description: "Input tensor",
		DType:       Float32,
		Shape:       []int64{2, 3},
	}}
	outputs := []FeatureSpec{{
		Name:        "y",
		Description: "Output tensor",
		DType:       Float32,
		Shape:       []int64{2, 3},
	}}

	model := ToModel(program, inputs, outputs, DefaultOptions())

	// Save to temp directory
	tmpDir := t.TempDir()
	packagePath := filepath.Join(tmpDir, "test.mlpackage")

	if err := SaveMLPackage(model, packagePath); err != nil {
		t.Fatalf("SaveMLPackage failed: %v", err)
	}

	// Verify directory structure
	manifestPath := filepath.Join(packagePath, "Manifest.json")
	if _, err := os.Stat(manifestPath); err != nil {
		t.Errorf("Manifest.json not found: %v", err)
	}

	modelPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "model.mlmodel")
	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("model.mlmodel not found: %v", err)
	}

	t.Logf("MLPackage saved to %s", packagePath)
}
