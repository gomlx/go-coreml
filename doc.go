// Package gocoreml provides Go bindings to Apple's CoreML framework.
//
// This package enables running machine learning models on Apple's Neural Engine (ANE),
// Metal GPU, and CPU. It is designed to be used as a backend for GoMLX, providing
// high-performance inference on Apple Silicon.
//
// # Architecture
//
// The package is organized into several sub-packages:
//
//   - internal/bridge: Low-level cgo bindings to CoreML via Objective-C++
//   - model: High-level model building and management
//   - runtime: Model loading and execution
//   - ops: MIL (Model Intermediate Language) operation builders
//
// # Usage
//
// This package is primarily intended to be used through the GoMLX CoreML backend.
// Direct usage is also possible for loading and running pre-built CoreML models:
//
//	import "github.com/gomlx/go-coreml/runtime"
//
//	model, err := runtime.LoadModel("path/to/model.mlmodelc")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	// Create input tensor
//	input := runtime.NewTensor([]int64{1, 3, 224, 224}, runtime.Float32)
//	// ... fill input data ...
//
//	// Run inference
//	output, err := model.Predict(map[string]*runtime.Tensor{"input": input})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// # Requirements
//
//   - macOS 12.0+ (Monterey or later)
//   - Xcode Command Line Tools
//   - Go 1.21+
//
// # Compute Units
//
// CoreML can run on different compute units:
//
//   - All: Let CoreML decide (default, usually best performance)
//   - CPU Only: Force CPU-only execution
//   - CPU and GPU: Use CPU and Metal GPU
//   - CPU and ANE: Use CPU and Apple Neural Engine
//
// The Neural Engine (ANE) provides the best performance and power efficiency
// for supported operations on Apple Silicon.
package gocoreml
