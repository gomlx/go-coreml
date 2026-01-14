# go-coreml

Go bindings to Apple's CoreML framework for high-performance machine learning inference on Apple Silicon.

## Overview

go-coreml provides Go bindings to CoreML, enabling:

- Running ML models on Apple's Neural Engine (ANE)
- Metal GPU acceleration
- Programmatic model construction using MIL (Machine Learning Intermediate Language)
- Integration with [GoMLX](https://github.com/gomlx/gomlx) as a backend

## Status

**Alpha** - Core functionality is implemented but the API may change.

### Implemented

- [x] Low-level bridge to CoreML (tensor creation, model loading, inference)
- [x] Protobuf types generated from CoreML MIL.proto
- [x] MIL program builder with common operations (add, mul, matmul, conv2d, pooling, etc.)
- [x] Model serialization to .mlpackage format
- [x] Runtime for compiling and executing MIL programs
- [x] GoMLX backend integration
- [x] Weight blob support for large models

### Planned

- [ ] Performance benchmarks

## Requirements

- macOS 12.0+ (Monterey or later)
- Xcode (full installation for coremlcompiler)
- Go 1.21+

## Installation

```bash
go get github.com/gomlx/go-coreml
```

## Usage

### Building a MIL Program

```go
package main

import (
    "fmt"
    "github.com/gomlx/go-coreml/model"
    "github.com/gomlx/go-coreml/runtime"
)

func main() {
    // Build a simple model: y = relu(x)
    b := model.NewBuilder("main")
    x := b.Input("x", model.Float32, 2, 3)
    y := b.Relu(x)
    b.Output("y", y)

    // Compile and load
    rt := runtime.New()
    exec, err := rt.Compile(b)
    if err != nil {
        panic(err)
    }
    defer exec.Close()

    // Run inference
    input := []float32{-1, 2, -3, 4, -5, 6}
    outputs, err := exec.Run(map[string]interface{}{"x": input})
    if err != nil {
        panic(err)
    }

    result := outputs["y"].([]float32)
    fmt.Println("Output:", result)
    // Output: [0 2 0 4 0 6]
}
```

### Available Operations

The MIL builder supports these operations:

- **Element-wise**: Add, Sub, Mul, Div, Neg, Abs, Pow, Min, Max
- **Activations**: Relu, Sigmoid, Tanh, Softmax
- **Math**: Exp, Log, Sqrt, Sin, Cos, Erf
- **Linear Algebra**: MatMul, Einsum
- **Convolution**: Conv, ConvTranspose, MaxPool, AvgPool
- **Shape**: Reshape, Transpose, Concat, Gather, Pad, Slice
- **Reductions**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProduct, ArgMax, ArgMin
- **Comparison**: Equal, NotEqual, LessThan, LessOrEqual, GreaterThan, GreaterOrEqual
- **Other**: Where, Iota, Cast, Clamp

### Compute Unit Selection

Control which compute units are used:

```go
import "github.com/gomlx/go-coreml/internal/bridge"

// Use all available compute units (ANE + GPU + CPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeAll))

// CPU only (for debugging)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUOnly))

// CPU + GPU (skip Neural Engine)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndGPU))

// CPU + Neural Engine (skip GPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndANE))
```

### Saving Models with Blob Storage

For models with large weights (e.g., neural networks), use blob storage to keep weights in an external `weight.bin` file:

```go
import "github.com/gomlx/go-coreml/model"

// Build your model
b := model.NewBuilder("main")
// ... add operations with large weight constants ...
program := b.Build()

// Convert to CoreML model
inputs := []model.FeatureSpec{{Name: "x", DType: model.Float32, Shape: []int64{1, 512}}}
outputs := []model.FeatureSpec{{Name: "y", DType: model.Float32, Shape: []int64{1, 10}}}
opts := model.DefaultBlobOptions()
coremlModel := model.ToModel(program, inputs, outputs, opts.SerializeOptions)

// Save with blob storage (tensors > 1KB are stored externally)
err := model.SaveMLPackageWithBlobs(coremlModel, "model.mlpackage", opts)
```

This creates the standard CoreML package structure with weights in `Data/com.apple.CoreML/weights/weight.bin`.

## Project Structure

```
go-coreml/
|-- blob/                  # Weight blob storage
|   |-- format.go          # Blob file format structs
|   +-- writer.go          # Blob file writer
|-- gomlx/                 # GoMLX backend implementation
|   |-- backend.go         # Backend interface
|   |-- function.go        # Operation implementations
|   +-- executable.go      # Model execution
|-- internal/
|   +-- bridge/            # Low-level cgo bindings to CoreML
|       |-- bridge.h       # C-compatible function declarations
|       |-- bridge.m       # Objective-C implementation
|       +-- bridge.go      # cgo wrapper
|-- model/
|   |-- builder.go         # MIL program builder
|   |-- ops.go             # MIL operation implementations
|   |-- serialize.go       # Model serialization
|   +-- serialize_blob.go  # Blob-aware serialization
|-- runtime/
|   +-- runtime.go         # High-level compilation and execution
+-- proto/
    +-- coreml/
        |-- milspec/       # Generated Go types from MIL.proto
        |-- spec/          # Generated Go types from Model.proto
        +-- *.proto        # CoreML protobuf definitions
```

## Development

```bash
# Build
go build ./...

# Test
go test ./...

# Update protobuf files from coremltools
cd proto/coreml && ./update_protos.sh

# Regenerate Go code from protobufs
go generate ./...
```

## License

Apache 2.0 - see LICENSE file.

CoreML protobuf definitions are from [Apple's coremltools](https://github.com/apple/coremltools)
and are licensed under BSD-3-Clause.
