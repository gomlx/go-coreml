# go-coreml Limitations

This document describes the known limitations, supported features, and workarounds for the go-coreml library. Understanding these constraints will help you determine if go-coreml is suitable for your use case.

## Platform Requirements

### Operating System

- **macOS only** - CoreML is an Apple-exclusive framework. This library only works on Darwin (macOS).
- **macOS 12.0+** (Monterey or later) required for CoreML MIL support.

### Build Requirements

- **CGO required** - The library uses cgo to interface with CoreML's Objective-C APIs.
- **Xcode** - Full Xcode installation is required (not just Command Line Tools) for `coremlcompiler`.
- **Go 1.21+** - Required Go version.

### Non-macOS Platforms

On non-macOS platforms (Linux, Windows), the package can be imported but:
- The backend will not be registered.
- Calling `New()` returns an error: "CoreML backend is only available on macOS (darwin)".
- Use a different backend (e.g., XLA) for cross-platform deployments.

---

## Supported Operations

### Basic Math Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Add | Supported | Element-wise with broadcasting |
| Sub | Supported | Element-wise with broadcasting |
| Mul | Supported | Element-wise with broadcasting |
| Div | Supported | Element-wise with broadcasting |
| Pow | Supported | Element-wise with broadcasting |
| Max | Supported | Element-wise maximum |
| Min | Supported | Element-wise minimum |
| Abs | Supported | |
| Neg | Supported | Implemented as multiplication by -1 |

### Unary Math Functions

| Operation | Status | Notes |
|-----------|--------|-------|
| Exp | Supported | |
| Expm1 | Supported | Implemented as exp(x) - 1 |
| Log | Supported | |
| Log1p | Supported | Implemented as log(x + 1) |
| Sqrt | Supported | |
| Rsqrt | Supported | 1/sqrt(x) with epsilon for stability |
| Floor | Supported | |
| Ceil | Supported | |
| Round | Supported | |
| Sign | Supported | |
| Erf | Supported | Error function |

### Trigonometric Functions

| Operation | Status | Notes |
|-----------|--------|-------|
| Cos | Supported | |
| Sin | Supported | |
| Tanh | Supported | |
| Acos | Supported | MIL builder only |
| Asin | Supported | MIL builder only |
| Atan | Supported | MIL builder only |
| Cosh | Supported | MIL builder only |
| Sinh | Supported | MIL builder only |

### Activation Functions

| Operation | Status | Notes |
|-----------|--------|-------|
| Logistic/Sigmoid | Supported | |
| Relu | Supported | MIL builder only |
| Softmax | Supported | MIL builder only |
| Gelu | Supported | MIL builder only (EXACT or TANH_APPROXIMATION) |
| Silu (Swish) | Supported | MIL builder only |
| LeakyRelu | Supported | MIL builder only |
| Elu | Supported | MIL builder only |
| Softplus | Supported | MIL builder only |

### Comparison Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Equal | Supported | Returns Bool |
| NotEqual | Supported | Returns Bool |
| LessThan | Supported | Returns Bool |
| LessOrEqual | Supported | Returns Bool |
| GreaterThan | Supported | Returns Bool |
| GreaterOrEqual | Supported | Returns Bool |

### Logical Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| LogicalAnd | Supported | MIL builder only |
| LogicalOr | Supported | MIL builder only |
| LogicalNot | Supported | MIL builder only |
| LogicalXor | Supported | MIL builder only |
| IsNan | Supported | MIL builder only |
| IsFinite | Supported | MIL builder only |

### Matrix Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Dot | Supported | 1D and 2D tensors |
| DotGeneral | Supported | Batched matrix multiplication with contracting/batch axes |
| MatMul | Supported | Standard matrix multiplication |
| Einsum | Partial | Limited to specific rank-3 and rank-4 patterns (see below) |

### Reduction Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| ReduceSum | Supported | |
| ReduceMax | Supported | |
| ReduceMin | Supported | |
| ReduceProduct | Supported | |
| ReduceMean | Supported | MIL builder only |
| ArgMin | Supported | |
| ArgMax | Supported | |

### Shape Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Reshape | Supported | |
| Transpose | Supported | |
| Squeeze | Supported | MIL builder only |
| ExpandDims | Supported | MIL builder only |

### Array Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Concatenate | Supported | |
| Slice | Supported | Static slicing |
| DynamicUpdateSlice | Supported | Via ScatterND |
| Pad | Partial | No interior padding support |
| Reverse | Supported | |
| Gather | Partial | Simple single-axis cases only |
| Iota | Supported | |
| Where | Supported | Element-wise conditional selection |
| Tile | Supported | MIL builder only |

### Convolution Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| Conv (2D) | Supported | NCHW layout |
| ConvTranspose | Supported | MIL builder only |
| ConvGeneral | Partial | See limitations below |

### Pooling Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| MaxPool | Supported | Spatial dimensions only |
| AvgPool | Supported | Spatial dimensions only |
| SumPool | Supported | Via AvgPool * window_size |
| MinPool | Supported | Via -MaxPool(-x) |
| ReduceWindow | Partial | See limitations below |
| GlobalAvgPool2D | Supported | MIL builder only |
| GlobalMaxPool2D | Supported | MIL builder only |

### Normalization Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| BatchNorm | Supported | MIL builder only |
| LayerNorm | Supported | MIL builder only |
| InstanceNorm | Supported | MIL builder only |
| L2Norm | Supported | MIL builder only |

### Type Conversion

| Operation | Status | Notes |
|-----------|--------|-------|
| ConvertDType/Cast | Supported | |
| Clip/Clamp | Supported | MIL builder only |

---

## Known Limitations

### Control Flow Operations (Not Implemented)

The following control flow operations are **not supported**:

#### While Loops

```go
// NOT SUPPORTED
func (f *Function) While(cond, body backends.Function, initialState ...backends.Value) ([]backends.Value, error)
```

**Why**: CoreML MIL does support `while_loop` via nested blocks, but go-coreml has not implemented building nested block structures.

**Workarounds**:
- Unroll the loop at graph construction time if the iteration count is known at compile time
- Use a different backend (e.g., XLA) for models that require dynamic loops

#### If Conditionals

```go
// NOT SUPPORTED
func (f *Function) If(pred backends.Value, trueBranch, falseBranch backends.Function) ([]backends.Value, error)
```

**Why**: Similar to While, CoreML MIL supports `cond` operations via nested blocks, but nested blocks are not implemented in go-coreml.

**Workarounds**:
- Use `Where()` for element-wise conditional selection (fully supported)
- Compute both branches and use `Where()` to select results based on condition
- Use a different backend for models requiring dynamic conditionals

#### Call (Sub-functions)

```go
// NOT SUPPORTED
func (f *Function) Call(fn backends.Function, inputs ...backends.Value) ([]backends.Value, error)
```

**Why**: Sub-function calls require nested block support.

#### Sort with Custom Comparators

```go
// NOT SUPPORTED
func (f *Function) Sort(comparator backends.Function, axis int, isStable bool, inputs ...backends.Value) ([]backends.Value, error)
```

**Why**: Custom comparators require nested block support for the comparison function.

### Convolution Limitations

#### Batch Group Count

```go
// NOT SUPPORTED: batchGroupCount > 1
if batchGroupCount > 1 {
    return nil, errors.Errorf("ConvGeneral: batchGroupCount > 1 is not supported by CoreML backend")
}
```

**Why**: CoreML's Conv operation only supports channel grouping, not batch grouping.

**Workaround**: Restructure your model to avoid batch grouping, or use a different backend.

#### Input Dilation (Base Dilation)

```go
// NOT SUPPORTED: inputDilations > 1
for i, d := range inputDilations {
    if d > 1 {
        return nil, errors.Errorf("ConvGeneral: input dilation ... is not supported by CoreML backend...")
    }
}
```

**Why**: CoreML Conv supports **kernel dilation** but not **input dilation** (also known as "base dilation" or "atrous inputs").

**Understanding the Difference**:
- **Kernel Dilation** (SUPPORTED): Spaces out the kernel/filter weights. A 3x3 kernel with dilation=2 acts like a 5x5 kernel with zeros inserted between weights. CoreML supports this via the `dilations` parameter.
- **Input Dilation** (NOT SUPPORTED): Inserts zeros between input elements before convolution. For example, input dilation=2 would double the input spatial dimensions by inserting zeros between each element.

Input dilation is commonly used in:
- Certain transposed convolution implementations
- Sub-pixel convolution for upsampling
- Some fractionally-strided convolution variants

**Workarounds**:
1. **Pre-process input manually**: Insert zeros into your input tensor before calling convolution. This can be done with reshape, concatenate with zeros, or scatter operations.
2. **Use a different backend**: The XLA backend supports input dilation natively.
3. **Restructure your model**: Consider alternative upsampling approaches like bilinear interpolation followed by regular convolution, or use `ConvTranspose` which handles upsampling internally.

### Pooling (ReduceWindow) Limitations

#### Window Dilations

```go
// NOT SUPPORTED: windowDilations > 1
if d > 1 {
    return nil, errors.Errorf("ReduceWindow: window dilations > 1 are not supported by CoreML backend pooling ops")
}
```

#### Base Dilations

```go
// NOT SUPPORTED: baseDilations > 1
if d > 1 {
    return nil, errors.Errorf("ReduceWindow: base dilations > 1 are not supported by CoreML backend")
}
```

#### Non-spatial Window Dimensions

CoreML pooling operates on spatial dimensions only. The window must have size 1 for batch and channel dimensions:

```go
if windowDimensions[0] != 1 || windowDimensions[1] != 1 {
    return nil, errors.Errorf("ReduceWindow: CoreML pooling only supports window size 1 for batch and channel dimensions")
}
```

#### Product Reduction

```go
// NOT SUPPORTED
case backends.ReduceOpProduct:
    return nil, errors.Errorf("ReduceWindow: ReduceOpProduct is not supported by CoreML backend")
```

### Gather Limitations

Only simple single-axis gather operations are supported:

```go
// SUPPORTED: Simple case where:
// - len(startIndexMap) == 1 (gathering along one axis)
// - len(collapsedSliceAxes) == 1 (collapsing that axis)
// - collapsedSliceAxes[0] == startIndexMap[0] (same axis)
// - sliceSizes[axis] == 1 for the gathered axis
```

Complex multi-axis gather operations return:
```go
return nil, errors.Wrapf(
    notimplemented.NotImplementedError,
    "complex Gather with multiple axes not yet supported for %q builder",
    BackendName)
```

**Workaround**: Decompose complex gathers into multiple simple gathers, or use a different backend.

### Pad Limitations

#### Interior Padding

```go
// NOT SUPPORTED
if hasInterior {
    return nil, errors.Wrapf(
        notimplemented.NotImplementedError,
        "Pad with interior padding not supported for %q builder", BackendName)
}
```

**Why**: CoreML's pad operation only supports edge padding, not interior (between-element) padding.

**Workaround**: Implement interior padding manually using scatter/gather operations, or use a different backend.

### Einsum Limitations

CoreML MIL einsum supports only specific equation patterns for batched matrix multiplication:

**Rank 4 inputs:**
- Equation: `"nchw,nwhu->nchu"` (and equivalent variations)
- Input 1: `[B, C, H, W1]`
- Input 2: `[B, W1, H, W2]`
- Output: `[B, C, H, W2]`

**Rank 3 inputs:**
- Equation: `"chw,whr->chr"` (and equivalent variations)
- Input 1: `[C, H, W1]`
- Input 2: `[W1, H, W2]`
- Output: `[C, H, W2]`

Other einsum patterns may not work or produce incorrect results.

**Workaround**: Use explicit transpose, reshape, and matmul operations for unsupported patterns.

---

## Data Type Support

### Supported Data Types

| DType | GoMLX Support | MIL Builder Support |
|-------|---------------|---------------------|
| Float32 | Yes | Yes |
| Float16 | Yes | Yes |
| Float64 | Yes | Yes |
| Int32 | Yes | Yes |
| Int16 | Yes | Yes |
| Int8 | Yes | Yes |
| Int64 | Yes | Yes |
| Bool | Yes | Yes |
| String | No | Yes (constants only) |

### Unsupported Data Types

- **Complex numbers** (Complex64, Complex128) - Not supported by CoreML
- **Unsigned integers** (UInt8, UInt16, UInt32, UInt64) - Not directly supported; use signed equivalents
- **BFloat16** - Not supported by CoreML MIL

---

## Sharding and Distribution

CoreML backend does **not** support:
- Sharding specifications
- Distributed execution
- Multi-device parallelism

```go
if len(shardings) != 0 {
    return errors.Errorf("sharding or distributed execution are not supported by CoreML backend")
}
```

---

## Best Practices

### 1. Check Operation Support Before Building Models

Before implementing a model, verify that all required operations are supported:

```go
import "github.com/gomlx/go-coreml/gomlx"

// Check if an operation is supported
if coreml.Capabilities.Operations[backends.OpTypeWhile] {
    // While is supported (it's not, this is just an example)
}
```

### 2. Use Where Instead of If

For element-wise conditionals, use `Where()`:

```go
// Instead of If (not supported):
// result = If(condition, trueValue, falseValue)

// Use Where (supported):
result = Where(condition, trueValue, falseValue)
```

### 3. Unroll Known-Length Loops

If your loop count is known at model construction time:

```go
// Instead of While (not supported):
// result = While(condition, body, initialState)

// Unroll the loop:
state := initialState
for i := 0; i < knownIterations; i++ {
    state = bodyStep(state)
}
result := state
```

### 4. Use Different Backend for Unsupported Operations

For models requiring unsupported operations, consider using a hybrid approach or a different backend entirely:

```go
// Set GOMLX_BACKEND=xla for full operation support
// Or use XLA backend programmatically for specific computations
```

### 5. Test on Target Platform

Always test your models on macOS before deployment, as the CoreML backend is only available there.

---

## Reporting Issues

If you encounter limitations not documented here, or find operations that should work but don't, please:

1. Check the [GitHub Issues](https://github.com/gomlx/go-coreml/issues) for existing reports
2. Open a new issue with:
   - The operation that failed
   - Input shapes and data types
   - Expected behavior
   - Actual error message or behavior
   - macOS and Xcode versions

---

## Future Work

The following features may be implemented in future versions:

- [ ] Nested block support for control flow (While, If, Call)
- [ ] Complex Gather with multiple axes
- [ ] Interior padding support
- [ ] Input dilation for convolutions
- [ ] Expanded einsum pattern support
- [ ] Performance benchmarks and tuning guide
