# go-coreml Full Backend Implementation Plan

## Goal
Make go-coreml a fully supported GoMLX backend capable of full model inference, including transformers like DeBERTa.

---

## Current State Analysis

### What go-coreml Already Has (Strong Foundation)
- **79/81 MIL operations** implemented in `model/ops.go`
- **Control flow**: While/If loops with nested block support
- **Core transformer ops**: MatMul, Softmax, LayerNorm, Gather, Transpose, Reshape
- **Activations**: GELU, ReLU, Tanh, Sigmoid, Silu
- **Reductions**: Sum, Mean, Max, Min, ArgMax
- **Convolution**: Conv2D, ConvTranspose, pooling operations

### What's Missing for Full Backend Parity

| Category | Missing Operations | MIL Support | Effort |
|----------|-------------------|-------------|--------|
| **Logical** | LogicalAnd/Or/Not/Xor | MIL has them | Low |
| **BatchNorm** | BatchNormForInference/Training | MIL has batch_norm | Medium |
| **Scatter** | ScatterSum/Max/Min | MIL ScatterND | Medium |
| **Misc** | Identity, BroadcastInDim, Clamp, DynamicSlice | MIL support | Low |
| **Bitwise** | BitwiseAnd/Or/Xor, Shifts | NO MIL support | High |
| **Complex** | Complex numbers, FFT | NO MIL support | Very High |
| **Control** | Call, Sort(custom comparator) | NO MIL equivalent | High |

---

## Implementation Phases

### Phase 1: Transformer-Critical Operations
**Goal**: Enable DeBERTa/BERT inference

**Operations to implement in `gomlx/function.go`:**

```go
// 1. Identity - trivial no-op
func (f *Function) Identity(x backends.Value) (backends.Value, error)

// 2. BroadcastInDim - map to MIL broadcast_to
func (f *Function) BroadcastInDim(x backends.Value, shape []int, broadcastDims []int) (backends.Value, error)

// 3. Clamp - map to MIL Clip
func (f *Function) Clamp(x, min, max backends.Value) (backends.Value, error)

// 4. Logical operations - wire existing MIL ops
func (f *Function) LogicalAnd(lhs, rhs backends.Value) (backends.Value, error)
func (f *Function) LogicalOr(lhs, rhs backends.Value) (backends.Value, error)
func (f *Function) LogicalNot(x backends.Value) (backends.Value, error)
func (f *Function) LogicalXor(lhs, rhs backends.Value) (backends.Value, error)

// 5. Special value checks - wire existing MIL ops
func (f *Function) IsFinite(x backends.Value) (backends.Value, error)
func (f *Function) IsNaN(x backends.Value) (backends.Value, error)

// 6. DynamicSlice - map to MIL SliceBySize
func (f *Function) DynamicSlice(operand backends.Value, startIndices []backends.Value, sliceSizes []int) (backends.Value, error)
```

**Files to modify:**
- `gomlx/function.go` - Add operation implementations
- `gomlx/capabilities.go` - Update supported operations
- `gomlx/function_test.go` - Add unit tests

### Phase 2: BatchNorm and Advanced Reductions
**Goal**: Support training and more model architectures

```go
// BatchNorm - map to MIL batch_norm
func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Value, epsilon float32, featureAxis int) (backends.Value, error)

// Compose for training (returns mean, variance, normalized)
func (f *Function) BatchNormForTraining(operand, scale, offset backends.Value, epsilon float32, featureAxis int) (normalized, batchMean, batchVar backends.Value, err error)

// Reduce logical - compose from type conversion + reduce
func (f *Function) ReduceLogicalAnd(x backends.Value, axes ...int) (backends.Value, error)
func (f *Function) ReduceLogicalOr(x backends.Value, axes ...int) (backends.Value, error)

// Remainder - use MIL floor_mod or compose
func (f *Function) Rem(lhs, rhs backends.Value) (backends.Value, error)
```

### Phase 3: Scatter Operations
**Goal**: Support gradient operations and advanced indexing

```go
// Scatter ops - map to MIL ScatterND with different modes
func (f *Function) ScatterSum(operand, indices, updates backends.Value, ...) (backends.Value, error)  // mode="add"
func (f *Function) ScatterMax(operand, indices, updates backends.Value, ...) (backends.Value, error)  // mode="max"
func (f *Function) ScatterMin(operand, indices, updates backends.Value, ...) (backends.Value, error)  // mode="min"

// SelectAndScatter - complex composition from ArgMax + Scatter
func (f *Function) SelectAndScatterMax(...) (backends.Value, error)
```

### Phase 4: TotalOrder Comparisons
**Goal**: Full numerical correctness

```go
// Compose from IsNaN/IsFinite + regular comparisons
func (f *Function) EqualTotalOrder(lhs, rhs backends.Value) (backends.Value, error)
func (f *Function) LessThanTotalOrder(lhs, rhs backends.Value) (backends.Value, error)
// etc.
```

### Phase 5: Documentation and Limitations
**Goal**: Document what's NOT supported

**Mark as unsupported (no MIL equivalent):**
- Bitwise operations (BitwiseAnd/Or/Xor/Not)
- Shift operations (ShiftLeft/Right)
- Complex numbers (Complex, Real, Imag, Conj)
- FFT operations
- Call (function calls) - must inline
- Sort with custom comparators
- Bitcast, BitCount, Clz

---

## Critical Files

| File | Purpose | Lines |
|------|---------|-------|
| `gomlx/function.go` | Main operation implementations | 2717 |
| `gomlx/capabilities.go` | Supported ops reporting | 103 |
| `model/ops.go` | MIL operation layer | 1865 |
| `gomlx/backends/standard_ops.go` | Reference interface | 635 |

---

## Testing Strategy

### 1. Unit Tests (per operation)
```go
// In gomlx/function_test.go
func TestLogicalAnd(t *testing.T) { /* test cases */ }
func TestBatchNormForInference(t *testing.T) { /* test cases */ }
```

### 2. Integration Test (DeBERTa inference)
```go
func TestDeBERTaInference(t *testing.T) {
    // Use gopeft/e2e/reranker model as test case
    // Compare outputs with XLA backend
}
```

### 3. Numerical Accuracy Tests
```go
func TestNumericalAccuracyVsXLA(t *testing.T) {
    // Run same operations on CoreML and XLA
    // Verify results match within tolerance
}
```

---

## Verification

1. **Build all packages:**
   ```bash
   cd ~/Documents/af/go-coreml && go build ./...
   ```

2. **Run unit tests:**
   ```bash
   cd ~/Documents/af/go-coreml && go test -v ./gomlx/...
   ```

3. **Run integration test with DeBERTa:**
   ```bash
   cd ~/Documents/af/gopeft && go test -v ./e2e/reranker/... -run TestDeBERTaCrossEncoderForward
   ```

4. **Benchmark CoreML vs XLA:**
   ```bash
   cd ~/Documents/af/gopeft && go test -bench=BenchmarkBackendComparison ./e2e/reranker/
   ```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MIL op behavior differs from XLA | Comprehensive numerical tests |
| Complex Gather patterns fail | Document supported patterns |
| Performance regression | Benchmark against XLA |
| Missing ops block models | Clear capability reporting |

---

## Summary

**Quick win:** Phase 1 enables transformer inference
**Key insight:** go-coreml already has 95% of needed MIL ops, just need to wire them to GoMLX interface
