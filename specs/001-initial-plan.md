# Plan: go-coreml Backend for GoMLX

## Summary

Create a CoreML backend for GoMLX, analogous to how go-xla provides the XLA backend. This would enable GoMLX models to run on Apple's Neural Engine (ANE), Metal GPU, and CPU with Apple-optimized performance.

**Scope**:
- Inference-only, macOS
- Run GoMLX computation graphs on CoreML (no .mlmodel import needed)
- Goal is to beat XLA CPU on Apple Silicon

## Architecture Overview

GoMLX has a clean, pluggable backend architecture with four core interfaces:
- `Backend` - main entry point, device management
- `Builder` - graph construction (symbolic operations)
- `Executable` - compiled computation
- `DataInterface` - buffer/tensor management

A CoreML backend would implement these interfaces using CoreML's APIs.

## Two-Component Approach

### Component 1: go-coreml (Low-Level Bindings)

**Location**: New repository `github.com/gomlx/go-coreml`

**Purpose**: Go bindings to CoreML, similar to how go-xla wraps PJRT/StableHLO

**Implementation Strategy**:

1. **Protobuf Generation for MLModel**
   - Generate Go types from CoreML's `.proto` files (Model.proto, MIL.proto)
   - Enables programmatic model generation in pure Go
   - No cgo required for model construction

2. **Objective-C++ Bridge for Runtime**
   - Create thin Objective-C++ shim for CoreML runtime operations
   - Wrap `MLModel`, `MLMultiArray`, `MLShapedArray` types
   - Expose C-compatible functions callable via cgo

**Key Files**:
```
go-coreml/
├── proto/
│   └── coreml/          # Generated Go types from Apple's .proto files
├── internal/
│   └── bridge/
│       ├── bridge.h     # C-compatible function declarations
│       ├── bridge.mm    # Objective-C++ implementation
│       └── bridge.go    # cgo wrapper
├── model/
│   └── builder.go       # Programmatic model construction (MIL)
├── runtime/
│   ├── model.go         # MLModel wrapper
│   └── tensor.go        # MLShapedArray/MLMultiArray wrapper
└── ops/
    └── mil_ops.go       # MIL operation builders
```

### Component 2: GoMLX CoreML Backend

**Location**: `github.com/gomlx/gomlx/backends/coreml`

**Purpose**: Implement GoMLX's backend interfaces using go-coreml

**Key Types**:

```go
// Backend implementation
type Backend struct {
    device      DeviceType  // ANE, GPU, CPU, or All
    computeUnits ComputeUnits
}

// Builder implementation (wraps MIL program builder)
type Builder struct {
    backend *Backend
    program *gocoreml.MILProgram
    ops     []*gocoreml.MILOperation
}

// Executable implementation (wraps compiled MLModel)
type Executable struct {
    model       *gocoreml.Model
    inputNames  []string
    outputNames []string
}

// Node wrapping MIL values
type Node struct {
    value   *gocoreml.MILValue
    shape   shapes.Shape
    builder *Builder
}
```

## Operation Mapping

### Supported Operations (CoreML has 500+ ops)

| GoMLX Operation | MIL Operation | Notes |
|-----------------|---------------|-------|
| Add, Sub, Mul, Div | add, sub, mul, real_div | Direct mapping |
| MatMul | matmul | Direct mapping |
| Conv2D | conv | Requires weight format conversion |
| MaxPool, AvgPool | max_pool, avg_pool | Direct mapping |
| Relu, Sigmoid, Tanh | relu, sigmoid, tanh | Direct mapping |
| Softmax | softmax | Direct mapping |
| BatchNorm | batch_norm | Direct mapping |
| Reshape, Transpose | reshape, transpose | Direct mapping |
| Concat | concat | Direct mapping |
| Reduce* | reduce_* | Direct mapping |

### Operations Requiring Composite Implementation

| GoMLX Operation | Implementation Strategy |
|-----------------|------------------------|
| Custom einsum | Decompose to matmul/transpose |
| Scatter/Gather | Multiple MIL ops |
| Complex dtypes | Split real/imag channels |

### Unsupported Operations

- Distributed/SPMD operations (CoreML is single-device)
- Some advanced XLA-specific operations

## Execution Flow

```
GoMLX User Code
    ↓
graph.Exec() or context.Exec()
    ↓
CoreMLBuilder.Parameter(), Add(), MatMul(), etc.
    ↓  (builds MIL program)
CoreMLBuilder.Compile()
    ↓  (generates .mlmodel protobuf, loads via CoreML runtime)
CoreMLExecutable
    ↓
CoreMLExecutable.Execute(inputs)
    ↓  (MLModel.prediction())
Returns GoMLX Buffer (wrapping MLShapedArray)
```

## Implementation Phases

### Phase 1: go-coreml Foundation
- Create Objective-C++ bridge for CoreML runtime operations
- Implement MLShapedArray wrapper for tensor I/O
- Test: create simple model in Obj-C++, call from Go
- Validate we can execute basic ops via CoreML from Go

### Phase 2: MIL Program Builder
- Generate Go protobuf types from CoreML MIL.proto
- Implement MIL operation builders in pure Go
- Support basic ops: arithmetic, matmul, activations
- Generate valid MIL program, load via CoreML runtime

### Phase 3: GoMLX Backend Integration
- Implement `Backend`, `Builder`, `Executable` interfaces
- Map GoMLX operations to MIL operations
- Handle shape inference and type conversion
- Register backend with GoMLX

### Phase 4: Operation Completeness
- Implement remaining standard operations
- Add convolution, pooling, normalization
- Implement composite operations for unsupported ops
- Add comprehensive tests

### Phase 5: Optimization & Benchmarking
- Optimize memory management (buffer reuse)
- Add compute unit selection (ANE/GPU/CPU)
- Performance benchmarking vs XLA CPU backend
- Ensure we beat XLA CPU on common workloads

## Technical Challenges

1. **No Official C/C++ API**: CoreML only has Objective-C/Swift APIs. Requires Objective-C++ bridge with cgo.

2. **Graph Compilation Model**: CoreML compiles to .mlmodel files, not JIT like XLA. May need to cache compiled models.

3. **Tensor Format Differences**: CoreML uses NHWC or NCHW depending on operation. Need careful format handling.

4. **Dynamic Shapes**: CoreML supports dynamic shapes but requires explicit range specification. May limit flexibility.

5. **Platform Lock-in**: CoreML only runs on Apple platforms (macOS, iOS, tvOS, watchOS).

## Alternatives Considered

1. **ONNX Runtime with CoreML EP**: Could use ONNX as intermediate format. More mature but adds complexity.

2. **Metal Performance Shaders**: Lower-level but doesn't leverage ANE. Would be similar to writing a custom GPU backend.

3. **coremltools Python**: Generate models via Python subprocess. Works but adds Python dependency.

## Success Criteria

- [ ] GoMLX computation graphs compile to CoreML and execute
- [ ] 80%+ of GoMLX standard operations supported
- [ ] Performance beats XLA CPU backend on Apple Silicon
- [ ] ANE/Metal acceleration works (measurable via Instruments)

## Confirmed Requirements

- **Use case**: Inference only (no training/gradients needed)
- **Platform**: macOS only
- **Model source**: GoMLX graphs only (no .mlmodel import needed)
- **Performance goal**: Beat XLA CPU backend on Apple Silicon

## Critical Files to Reference

**GoMLX Backend Interfaces** (implement these):
- `gomlx/backends/backends.go` - Backend interface
- `gomlx/backends/builder.go` - Builder interface
- `gomlx/backends/executable.go` - Executable interface
- `gomlx/backends/data.go` - DataInterface
- `gomlx/backends/standard_ops.go` - Operations to implement

**Reference Implementations**:
- `gomlx/backends/xla/` - Full XLA backend (production reference)
- `gomlx/backends/simplego/` - Pure Go interpreter (simpler reference)

**CoreML Specification** (for protobuf generation):
- https://github.com/apple/coremltools/tree/main/mlmodel/format

## Next Steps

1. Create minimal Objective-C++ bridge proof-of-concept
2. Validate we can execute CoreML operations from Go (e.g., matmul)
3. Generate Go protobuf types from CoreML MIL.proto
4. Implement MIL program builder for graph construction
5. Create GoMLX backend wrapper implementing Backend/Builder/Executable
