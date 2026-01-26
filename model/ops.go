package model

import "fmt"

// This file contains MIL operation builders.
// MIL operations are documented at:
// https://apple.github.io/coremltools/docs-guides/source/ops-reference.html

// ConvPadType represents convolution/pooling padding type.
type ConvPadType int

const (
	// ConvPadValid means no padding (only valid positions).
	ConvPadValid ConvPadType = iota
	// ConvPadSame means output size equals input size (with stride=1).
	ConvPadSame
	// ConvPadCustom means custom padding specified by padBefore and padAfter.
	ConvPadCustom
)

// Add performs element-wise addition: z = x + y.
func (b *Builder) Add(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("add", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("add"), x.dtype, outShape)
}

// Sub performs element-wise subtraction: z = x - y.
func (b *Builder) Sub(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("sub", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("sub"), x.dtype, outShape)
}

// Mul performs element-wise multiplication: z = x * y.
func (b *Builder) Mul(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("mul", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("mul"), x.dtype, outShape)
}

// Div performs element-wise division: z = x / y.
func (b *Builder) Div(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("real_div", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("div"), x.dtype, outShape)
}

// MatMul performs matrix multiplication: z = x @ y.
// x: [..., M, K], y: [..., K, N] -> z: [..., M, N]
func (b *Builder) MatMul(x, y *Value) *Value {
	return b.MatMulTranspose(x, y, false, false)
}

// MatMulTranspose performs matrix multiplication with optional transposes.
// x: [..., M, K], y: [..., K, N] -> z: [..., M, N]
// If transposeX is true, x is transposed before multiplication.
// If transposeY is true, y is transposed before multiplication.
func (b *Builder) MatMulTranspose(x, y *Value, transposeX, transposeY bool) *Value {
	// Compute output shape for matmul
	xShape := x.shape
	yShape := y.shape

	// Adjust shapes based on transposes
	xM := xShape[len(xShape)-2]
	xK := xShape[len(xShape)-1]
	yK := yShape[len(yShape)-2]
	yN := yShape[len(yShape)-1]

	if transposeX {
		xM, xK = xK, xM
	}
	if transposeY {
		yK, yN = yN, yK
	}
	_ = xK // K dimension should match
	_ = yK

	outShape := make([]int64, len(xShape))
	copy(outShape, xShape[:len(xShape)-2])
	outShape[len(outShape)-2] = xM
	outShape[len(outShape)-1] = yN

	transposeXVal := b.Const(b.genName("transpose_x"), Bool, []int64{}, []bool{transposeX})
	transposeYVal := b.Const(b.genName("transpose_y"), Bool, []int64{}, []bool{transposeY})

	return b.addOp("matmul", map[string]*Value{
		"x":           x,
		"y":           y,
		"transpose_x": transposeXVal,
		"transpose_y": transposeYVal,
	}, b.genName("matmul"), x.dtype, outShape)
}

// Relu applies rectified linear unit: z = max(x, 0).
func (b *Builder) Relu(x *Value) *Value {
	return b.addOp("relu", map[string]*Value{
		"x": x,
	}, b.genName("relu"), x.dtype, x.shape)
}

// Sigmoid applies sigmoid activation: z = 1 / (1 + exp(-x)).
func (b *Builder) Sigmoid(x *Value) *Value {
	return b.addOp("sigmoid", map[string]*Value{
		"x": x,
	}, b.genName("sigmoid"), x.dtype, x.shape)
}

// Tanh applies hyperbolic tangent: z = tanh(x).
func (b *Builder) Tanh(x *Value) *Value {
	return b.addOp("tanh", map[string]*Value{
		"x": x,
	}, b.genName("tanh"), x.dtype, x.shape)
}

// Softmax applies softmax along the specified axis.
func (b *Builder) Softmax(x *Value, axis int) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	return b.addOp("softmax", map[string]*Value{
		"x":    x,
		"axis": axisVal,
	}, b.genName("softmax"), x.dtype, x.shape)
}

// Exp computes element-wise exponential: z = exp(x).
func (b *Builder) Exp(x *Value) *Value {
	return b.addOp("exp", map[string]*Value{
		"x": x,
	}, b.genName("exp"), x.dtype, x.shape)
}

// Log computes element-wise natural logarithm: z = log(x).
func (b *Builder) Log(x *Value) *Value {
	return b.addOp("log", map[string]*Value{
		"x": x,
	}, b.genName("log"), x.dtype, x.shape)
}

// Sqrt computes element-wise square root: z = sqrt(x).
func (b *Builder) Sqrt(x *Value) *Value {
	return b.addOp("sqrt", map[string]*Value{
		"x": x,
	}, b.genName("sqrt"), x.dtype, x.shape)
}

// Neg computes element-wise negation: z = -x.
func (b *Builder) Neg(x *Value) *Value {
	return b.addOp("neg", map[string]*Value{
		"x": x,
	}, b.genName("neg"), x.dtype, x.shape)
}

// Abs computes element-wise absolute value: z = |x|.
func (b *Builder) Abs(x *Value) *Value {
	return b.addOp("abs", map[string]*Value{
		"x": x,
	}, b.genName("abs"), x.dtype, x.shape)
}

// Pow performs element-wise power: z = x^y.
func (b *Builder) Pow(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("pow", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("pow"), x.dtype, outShape)
}

// Maximum computes element-wise maximum: z = max(x, y).
func (b *Builder) Maximum(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("maximum", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("maximum"), x.dtype, outShape)
}

// Minimum computes element-wise minimum: z = min(x, y).
func (b *Builder) Minimum(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("minimum", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("minimum"), x.dtype, outShape)
}

// Floor computes element-wise floor: z = floor(x).
func (b *Builder) Floor(x *Value) *Value {
	return b.addOp("floor", map[string]*Value{
		"x": x,
	}, b.genName("floor"), x.dtype, x.shape)
}

// Ceil computes element-wise ceiling: z = ceil(x).
func (b *Builder) Ceil(x *Value) *Value {
	return b.addOp("ceil", map[string]*Value{
		"x": x,
	}, b.genName("ceil"), x.dtype, x.shape)
}

// Round computes element-wise rounding: z = round(x).
func (b *Builder) Round(x *Value) *Value {
	return b.addOp("round", map[string]*Value{
		"x": x,
	}, b.genName("round"), x.dtype, x.shape)
}

// Sign computes element-wise sign: z = sign(x).
// Returns -1 for negative values, 0 for zero, and 1 for positive values.
func (b *Builder) Sign(x *Value) *Value {
	return b.addOp("sign", map[string]*Value{
		"x": x,
	}, b.genName("sign"), x.dtype, x.shape)
}

// Cos computes element-wise cosine: z = cos(x).
func (b *Builder) Cos(x *Value) *Value {
	return b.addOp("cos", map[string]*Value{
		"x": x,
	}, b.genName("cos"), x.dtype, x.shape)
}

// Sin computes element-wise sine: z = sin(x).
func (b *Builder) Sin(x *Value) *Value {
	return b.addOp("sin", map[string]*Value{
		"x": x,
	}, b.genName("sin"), x.dtype, x.shape)
}

// Acos computes element-wise arc cosine: z = acos(x).
func (b *Builder) Acos(x *Value) *Value {
	return b.addOp("acos", map[string]*Value{
		"x": x,
	}, b.genName("acos"), x.dtype, x.shape)
}

// Asin computes element-wise arc sine: z = asin(x).
func (b *Builder) Asin(x *Value) *Value {
	return b.addOp("asin", map[string]*Value{
		"x": x,
	}, b.genName("asin"), x.dtype, x.shape)
}

// Atan computes element-wise arc tangent: z = atan(x).
func (b *Builder) Atan(x *Value) *Value {
	return b.addOp("atan", map[string]*Value{
		"x": x,
	}, b.genName("atan"), x.dtype, x.shape)
}

// Cosh computes element-wise hyperbolic cosine: z = cosh(x).
func (b *Builder) Cosh(x *Value) *Value {
	return b.addOp("cosh", map[string]*Value{
		"x": x,
	}, b.genName("cosh"), x.dtype, x.shape)
}

// Sinh computes element-wise hyperbolic sine: z = sinh(x).
func (b *Builder) Sinh(x *Value) *Value {
	return b.addOp("sinh", map[string]*Value{
		"x": x,
	}, b.genName("sinh"), x.dtype, x.shape)
}

// Erf computes element-wise error function: z = erf(x).
func (b *Builder) Erf(x *Value) *Value {
	return b.addOp("erf", map[string]*Value{
		"x": x,
	}, b.genName("erf"), x.dtype, x.shape)
}

// Gelu computes Gaussian Error Linear Unit: x * Φ(x) where Φ is the cumulative distribution
// function of the standard normal distribution.
// mode should be "EXACT" or "TANH_APPROXIMATION".
func (b *Builder) Gelu(x *Value, mode string) *Value {
	modeVal := b.Const(b.genName("mode"), String, []int64{}, mode)
	return b.addOp("gelu", map[string]*Value{
		"x":    x,
		"mode": modeVal,
	}, b.genName("gelu"), x.dtype, x.shape)
}

// Silu computes Sigmoid Linear Unit (Swish): x * sigmoid(x).
func (b *Builder) Silu(x *Value) *Value {
	return b.addOp("silu", map[string]*Value{
		"x": x,
	}, b.genName("silu"), x.dtype, x.shape)
}

// LeakyRelu computes Leaky ReLU: max(x, alpha*x) where alpha is typically small (e.g., 0.01).
func (b *Builder) LeakyRelu(x *Value, alpha float32) *Value {
	alphaVal := b.Const(b.genName("alpha"), Float32, []int64{}, []float32{alpha})
	return b.addOp("leaky_relu", map[string]*Value{
		"x":     x,
		"alpha": alphaVal,
	}, b.genName("leaky_relu"), x.dtype, x.shape)
}

// Elu computes Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1).
func (b *Builder) Elu(x *Value, alpha float32) *Value {
	alphaVal := b.Const(b.genName("alpha"), Float32, []int64{}, []float32{alpha})
	return b.addOp("elu", map[string]*Value{
		"x":     x,
		"alpha": alphaVal,
	}, b.genName("elu"), x.dtype, x.shape)
}

// Softplus computes smooth approximation of ReLU: log(1 + exp(x)).
func (b *Builder) Softplus(x *Value) *Value {
	return b.addOp("softplus", map[string]*Value{
		"x": x,
	}, b.genName("softplus"), x.dtype, x.shape)
}

// Reshape changes the shape of a tensor.
func (b *Builder) Reshape(x *Value, shape []int64) *Value {
	shapeVal := b.Const(b.genName("shape"), Int32, []int64{int64(len(shape))}, toInt32Slice(shape))
	return b.addOp("reshape", map[string]*Value{
		"x":     x,
		"shape": shapeVal,
	}, b.genName("reshape"), x.dtype, shape)
}

// Transpose permutes the dimensions of a tensor.
func (b *Builder) Transpose(x *Value, perm []int64) *Value {
	permVal := b.Const(b.genName("perm"), Int32, []int64{int64(len(perm))}, toInt32Slice(perm))

	// Compute output shape
	outShape := make([]int64, len(perm))
	for i, p := range perm {
		outShape[i] = x.shape[p]
	}

	return b.addOp("transpose", map[string]*Value{
		"x":    x,
		"perm": permVal,
	}, b.genName("transpose"), x.dtype, outShape)
}

// ReduceSum computes sum along specified axes.
func (b *Builder) ReduceSum(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_sum", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_sum"), x.dtype, outShape)
}

// ReduceMean computes mean along specified axes.
func (b *Builder) ReduceMean(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_mean", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_mean"), x.dtype, outShape)
}

// ReduceMax computes max along specified axes.
func (b *Builder) ReduceMax(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_max", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_max"), x.dtype, outShape)
}

// ReduceMin computes min along specified axes.
func (b *Builder) ReduceMin(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_min", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_min"), x.dtype, outShape)
}

// ReduceProd computes product along specified axes.
func (b *Builder) ReduceProd(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_prod", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_prod"), x.dtype, outShape)
}

// ArgMax returns indices of maximum values along an axis.
func (b *Builder) ArgMax(x *Value, axis int64, keepDims bool) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	// ArgMax returns Int32 indices, not the input dtype
	outShape := computeReduceShape(x.shape, []int64{axis}, keepDims)

	return b.addOp("reduce_argmax", map[string]*Value{
		"x":         x,
		"axis":      axisVal,
		"keep_dims": keepVal,
	}, b.genName("argmax"), Int32, outShape)
}

// ArgMin returns indices of minimum values along an axis.
func (b *Builder) ArgMin(x *Value, axis int64, keepDims bool) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	// ArgMin returns Int32 indices, not the input dtype
	outShape := computeReduceShape(x.shape, []int64{axis}, keepDims)

	return b.addOp("reduce_argmin", map[string]*Value{
		"x":         x,
		"axis":      axisVal,
		"keep_dims": keepVal,
	}, b.genName("argmin"), Int32, outShape)
}

// Argsort returns indices that would sort the input tensor along the specified axis.
// x: Input tensor to sort.
// axis: Axis along which to sort. Must be non-negative and less than the rank of x.
// descending: If true, sort in descending order. If false, sort in ascending order.
// Output: Int32 tensor with the same shape as x, containing indices that would sort x.
//
// Example:
//
//	input = [3.1, 5.4, 32.9, 3.2]
//	axis = 0
//	descending = false
//	output = [0, 3, 1, 2]  // indices that would sort in ascending order
//
// Available in CoreML MIL for iOS 15+.
func (b *Builder) Argsort(x *Value, axis int64, descending bool) *Value {
	// Handle negative axis
	if axis < 0 {
		axis = int64(len(x.shape)) + axis
	}

	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	descendingVal := b.Const(b.genName("descending"), Bool, []int64{}, []bool{descending})

	// Argsort returns Int32 indices with the same shape as input
	return b.addOp("argsort", map[string]*Value{
		"x":          x,
		"axis":       axisVal,
		"descending": descendingVal,
	}, b.genName("argsort"), Int32, x.shape)
}

// TopK returns the top k values and their indices along the specified axis.
// x: Input tensor.
// k: Number of top elements to return (must be >= 1).
// axis: Axis along which to find top-k values. Must be non-negative and less than the rank of x.
// ascending: If true, returns the k smallest values. If false (default), returns the k largest values.
// Output: Two tensors - (values, indices) where:
//   - values: Tensor of dtype same as x, shape with axis dimension replaced by k
//   - indices: Int32 tensor with the same shape as values
//
// Example:
//
//	input = [3.1, 5.4, 32.9, 3.2, 77.0]
//	k = 2
//	axis = 0
//	ascending = false
//	values = [77.0, 32.9]
//	indices = [4, 2]
//
// Available in CoreML MIL for iOS 15+.
func (b *Builder) TopK(x *Value, k int64, axis int64, ascending bool) (*Value, *Value) {
	// Handle negative axis
	if axis < 0 {
		axis = int64(len(x.shape)) + axis
	}

	kVal := b.Const(b.genName("k"), Int32, []int64{}, []int32{int32(k)})
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	ascendingVal := b.Const(b.genName("ascending"), Bool, []int64{}, []bool{ascending})

	// Compute output shape: replace axis dimension with k
	outShape := make([]int64, len(x.shape))
	copy(outShape, x.shape)
	outShape[axis] = k

	// TopK returns two outputs: values and indices
	// We use addOpMultiOutput which creates a tuple output
	values := b.addOp("topk", map[string]*Value{
		"x":         x,
		"k":         kVal,
		"axis":      axisVal,
		"ascending": ascendingVal,
	}, b.genName("topk"), x.dtype, outShape)

	// Create a second output for indices using slice_by_index on the tuple
	// Actually, CoreML topk returns a tuple (values, indices), so we need to handle multi-output ops
	// For now, we'll return values directly and create indices separately
	// Note: This is a simplification - proper multi-output handling may need adjustment

	// For proper multi-output handling, we'd need to add support for tuple outputs
	// For now, we create a placeholder for indices
	indices := &Value{
		name:    b.genName("topk_indices"),
		dtype:   Int32,
		shape:   outShape,
		isConst: false,
	}

	return values, indices
}

// Equal performs element-wise equality comparison: z = (x == y).
// Returns Bool dtype.
func (b *Builder) Equal(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("equal"), Bool, outShape)
}

// NotEqual performs element-wise inequality comparison: z = (x != y).
// Returns Bool dtype.
func (b *Builder) NotEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("not_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("not_equal"), Bool, outShape)
}

// Less performs element-wise less-than comparison: z = (x < y).
// Returns Bool dtype.
func (b *Builder) Less(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("less", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("less"), Bool, outShape)
}

// LessEqual performs element-wise less-than-or-equal comparison: z = (x <= y).
// Returns Bool dtype.
func (b *Builder) LessEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("less_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("less_equal"), Bool, outShape)
}

// Greater performs element-wise greater-than comparison: z = (x > y).
// Returns Bool dtype.
func (b *Builder) Greater(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("greater", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("greater"), Bool, outShape)
}

// GreaterEqual performs element-wise greater-than-or-equal comparison: z = (x >= y).
// Returns Bool dtype.
func (b *Builder) GreaterEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("greater_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("greater_equal"), Bool, outShape)
}

// Select performs element-wise selection based on a condition.
// Returns a where cond is true, b where cond is false.
// cond must have Bool dtype, a and b must have matching dtypes.
func (b *Builder) Select(cond, a, bVal *Value) *Value {
	outShape := broadcastShape(a.shape, bVal.shape)
	return b.addOp("select", map[string]*Value{
		"cond": cond,
		"a":    a,
		"b":    bVal,
	}, b.genName("select"), a.dtype, outShape)
}

// Squeeze removes dimensions of size 1 from the tensor shape.
// If axes is empty or nil, all dimensions of size 1 are removed.
func (b *Builder) Squeeze(x *Value, axes []int64) *Value {
	// Compute output shape by removing specified axes
	outShape := make([]int64, 0)

	if len(axes) == 0 {
		// Squeeze all size-1 dimensions
		for _, dim := range x.shape {
			if dim != 1 {
				outShape = append(outShape, dim)
			}
		}
	} else {
		// Build set of axes to squeeze
		axisSet := make(map[int64]bool)
		for _, a := range axes {
			if a < 0 {
				a = int64(len(x.shape)) + a
			}
			axisSet[a] = true
		}

		// Remove specified axes
		for i, dim := range x.shape {
			if !axisSet[int64(i)] {
				outShape = append(outShape, dim)
			}
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("squeeze", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("squeeze"), x.dtype, outShape)
}

// ExpandDims adds dimensions of size 1 at specified axes.
func (b *Builder) ExpandDims(x *Value, axes []int64) *Value {
	// Compute output shape by inserting size-1 dimensions
	outRank := len(x.shape) + len(axes)
	outShape := make([]int64, outRank)

	// Normalize and build set of axes where we insert size-1 dims
	normalizedAxes := make(map[int64]bool)
	for _, a := range axes {
		if a < 0 {
			a = int64(outRank) + a
		}
		normalizedAxes[a] = true
	}

	// Build output shape
	srcIdx := 0
	for i := 0; i < outRank; i++ {
		if normalizedAxes[int64(i)] {
			outShape[i] = 1
		} else {
			outShape[i] = x.shape[srcIdx]
			srcIdx++
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("expand_dims", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("expand_dims"), x.dtype, outShape)
}

// SliceByIndex extracts a sub-tensor using start/end indices along each axis.
// begin: starting indices for each dimension (inclusive)
// end: ending indices for each dimension (exclusive)
// strides: step size for each dimension (nil or empty defaults to 1)
func (b *Builder) SliceByIndex(x *Value, begin, end, strides []int64) *Value {
	// Handle nil or empty strides (default to 1)
	if len(strides) == 0 {
		strides = make([]int64, len(begin))
		for i := range strides {
			strides[i] = 1
		}
	}

	// Compute output shape
	outShape := make([]int64, len(x.shape))
	for i := range outShape {
		start := begin[i]
		stop := end[i]
		stride := strides[i]
		if stride == 0 {
			stride = 1
		}
		// Handle negative indices
		if start < 0 {
			start = x.shape[i] + start
		}
		if stop < 0 {
			stop = x.shape[i] + stop
		}
		// Compute output dimension size
		outShape[i] = (stop - start + stride - 1) / stride
	}

	beginVal := b.Const(b.genName("begin"), Int32, []int64{int64(len(begin))}, toInt32Slice(begin))
	endVal := b.Const(b.genName("end"), Int32, []int64{int64(len(end))}, toInt32Slice(end))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))

	return b.addOp("slice_by_index", map[string]*Value{
		"x":      x,
		"begin":  beginVal,
		"end":    endVal,
		"stride": stridesVal,
	}, b.genName("slice"), x.dtype, outShape)
}

// Gather gathers values from x using indices along a specified axis.
// Output shape: x.shape[:axis] + indices.shape + x.shape[axis+1:]
func (b *Builder) Gather(x *Value, indices *Value, axis int64) *Value {
	// Handle negative axis
	if axis < 0 {
		axis = int64(len(x.shape)) + axis
	}

	// Compute output shape:
	// Replace x.shape[axis] with indices.shape
	outShape := make([]int64, 0, len(x.shape)-1+len(indices.shape))
	outShape = append(outShape, x.shape[:axis]...)
	outShape = append(outShape, indices.shape...)
	outShape = append(outShape, x.shape[axis+1:]...)

	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	// CoreML requires validate_indices parameter (typically set to false for performance)
	validateIndicesVal := b.Const(b.genName("validate_indices"), Bool, []int64{}, []bool{false})

	return b.addOp("gather", map[string]*Value{
		"x":                x,
		"indices":          indices,
		"axis":             axisVal,
		"validate_indices": validateIndicesVal,
	}, b.genName("gather"), x.dtype, outShape)
}

// GatherND gathers elements from x using multi-dimensional indices.
// x: Source tensor with shape [D0, D1, ..., DN-1, S0, S1, ...]
// indices: Index tensor with shape [I0, I1, ..., IK-1, N] where N is the number of dimensions to index into
// Output shape: [I0, I1, ..., IK-1] + x.shape[N:]
//
// Example:
//
//	x.shape = [4, 2, 3, 4]   (data tensor)
//	indices.shape = [6, 2]  (6 multi-indices, each of length 2)
//	output.shape = [6, 3, 4] (6 slices, each of shape [3, 4])
//
// The last dimension of indices specifies coordinates into the first N dimensions of x.
func (b *Builder) GatherND(x *Value, indices *Value) *Value {
	// Output shape computation:
	// indices.shape = [..., N] where N = indices.shape[-1]
	// output.shape = indices.shape[:-1] + x.shape[N:]
	indicesRank := len(indices.shape)
	indexDepth := indices.shape[indicesRank-1] // N: number of dimensions to index into

	// Output shape: batch dims from indices + remaining dims from x
	outShape := make([]int64, 0, indicesRank-1+len(x.shape)-int(indexDepth))
	outShape = append(outShape, indices.shape[:indicesRank-1]...)
	outShape = append(outShape, x.shape[indexDepth:]...)

	// CoreML requires validate_indices parameter (typically set to false for performance)
	validateIndicesVal := b.Const(b.genName("validate_indices"), Bool, []int64{}, []bool{false})

	return b.addOp("gather_nd", map[string]*Value{
		"x":                x,
		"indices":          indices,
		"validate_indices": validateIndicesVal,
	}, b.genName("gather_nd"), x.dtype, outShape)
}

// GatherAlongAxis gathers elements along a specified axis where indices has the same rank as x.
// Similar to PyTorch's torch.gather operation.
// x: Source tensor with shape [D0, D1, ..., DN-1]
// indices: Index tensor with same rank as x, shape matches except at the gather axis
// axis: Axis along which to gather
// Output shape: same as indices.shape
//
// For axis=0:
//
//	output[i,j,...,k] = x[indices[i,j,...,k], j, ..., k]
func (b *Builder) GatherAlongAxis(x *Value, indices *Value, axis int64) *Value {
	// Handle negative axis
	if axis < 0 {
		axis = int64(len(x.shape)) + axis
	}

	// Output shape is same as indices shape
	outShape := make([]int64, len(indices.shape))
	copy(outShape, indices.shape)

	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})

	return b.addOp("gather_along_axis", map[string]*Value{
		"x":       x,
		"indices": indices,
		"axis":    axisVal,
	}, b.genName("gather_along_axis"), x.dtype, outShape)
}

// BatchNorm applies batch normalization to the input tensor.
// x: Input tensor with shape [N, C, *D] where N is batch size, C is channels, *D are spatial dimensions (rank 3-5).
// mean: Channel-wise mean with shape [C].
// variance: Channel-wise variance with shape [C].
// gamma: Optional scale parameter with shape [C]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape [C]. If nil, defaults to all zeros.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) BatchNorm(x, mean, variance, gamma, beta *Value, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":        x,
		"mean":     mean,
		"variance": variance,
		"epsilon":  epsilonVal,
	}

	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("batch_norm", inputs, b.genName("batch_norm"), x.dtype, x.shape)
}

// LayerNorm applies layer normalization to the input tensor.
// x: Input tensor of any shape.
// gamma: Optional scale parameter with shape matching x.shape[axes]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape matching x.shape[axes]. If nil, defaults to all zeros.
// axes: Dimensions along which to perform normalization. If nil or empty, normalizes over all axes.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) LayerNorm(x, gamma, beta *Value, axes []int64, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}

	if len(axes) > 0 {
		axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
		inputs["axes"] = axesVal
	}
	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("layer_norm", inputs, b.genName("layer_norm"), x.dtype, x.shape)
}

// InstanceNorm applies instance normalization to the input tensor.
// x: Input tensor with shape [N, C, *D] where N is batch size, C is channels, *D are spatial dimensions (rank 3-4).
// gamma: Optional scale parameter with shape [C]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape [C]. If nil, defaults to all zeros.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) InstanceNorm(x, gamma, beta *Value, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}

	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("instance_norm", inputs, b.genName("instance_norm"), x.dtype, x.shape)
}

// MaxPool applies max pooling operation.
// x: input tensor with shape [N, C, H, W] for 2D pooling
// kernelSize: size of pooling window for each spatial dimension
// strides: stride for each spatial dimension
// padType: padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore, padAfter: custom padding (only used if padType == ConvPadCustom)
func (b *Builder) MaxPool(x *Value, kernelSize, strides []int64, padType ConvPadType, padBefore, padAfter []int64) *Value {
	// Compute output shape
	// For 2D: input [N, C, H, W] -> output [N, C, H_out, W_out]
	outShape := make([]int64, len(x.shape))
	copy(outShape[:2], x.shape[:2]) // Copy N, C dimensions

	// Compute spatial dimensions based on padding type
	for i := 0; i < len(kernelSize); i++ {
		spatialIdx := 2 + i
		inputSize := x.shape[spatialIdx]
		kernelSz := kernelSize[i]
		stride := strides[i]

		var padTotal int64
		switch padType {
		case ConvPadValid:
			padTotal = 0
		case ConvPadSame:
			// Output size equals input size when stride=1
			padTotal = (kernelSz - 1)
		case ConvPadCustom:
			padTotal = padBefore[i] + padAfter[i]
		}

		outShape[spatialIdx] = (inputSize + padTotal - kernelSz) / stride + 1
	}

	kernelVal := b.Const(b.genName("kernel_sizes"), Int32, []int64{int64(len(kernelSize))}, toInt32Slice(kernelSize))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))

	inputs := map[string]*Value{
		"x":            x,
		"kernel_sizes": kernelVal,
		"strides":      stridesVal,
	}

	// Add padding parameters based on type
	// CoreML MIL requires "pad" parameter for pooling operations
	switch padType {
	case ConvPadValid:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "valid")
		inputs["pad_type"] = padTypeVal
		// Still need to provide pad parameter with zeros
		padSpec := make([]int32, 2*len(kernelSize))
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	case ConvPadSame:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "same")
		inputs["pad_type"] = padTypeVal
		// For same padding, CoreML calculates pad values automatically
		padSpec := make([]int32, 2*len(kernelSize))
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	case ConvPadCustom:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "custom")
		inputs["pad_type"] = padTypeVal
		// CoreML MIL uses a single "pad" parameter with format [before_0, after_0, before_1, after_1, ...]
		padSpec := make([]int32, 2*len(padBefore))
		for i := range padBefore {
			padSpec[2*i] = int32(padBefore[i])
			padSpec[2*i+1] = int32(padAfter[i])
		}
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	}

	// ceil_mode is required by CoreML MIL
	ceilModeVal := b.Const(b.genName("ceil_mode"), Bool, []int64{}, []bool{false})
	inputs["ceil_mode"] = ceilModeVal

	return b.addOp("max_pool", inputs, b.genName("max_pool"), x.dtype, outShape)
}

// AvgPool applies average pooling operation.
// x: input tensor with shape [N, C, H, W] for 2D pooling
// kernelSize: size of pooling window for each spatial dimension
// strides: stride for each spatial dimension
// padType: padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore, padAfter: custom padding (only used if padType == ConvPadCustom)
// excludePaddingFromAverage: if true, exclude padding values from average calculation
func (b *Builder) AvgPool(x *Value, kernelSize, strides []int64, padType ConvPadType, padBefore, padAfter []int64, excludePaddingFromAverage bool) *Value {
	// Compute output shape (same as MaxPool)
	outShape := make([]int64, len(x.shape))
	copy(outShape[:2], x.shape[:2]) // Copy N, C dimensions

	for i := 0; i < len(kernelSize); i++ {
		spatialIdx := 2 + i
		inputSize := x.shape[spatialIdx]
		kernelSz := kernelSize[i]
		stride := strides[i]

		var padTotal int64
		switch padType {
		case ConvPadValid:
			padTotal = 0
		case ConvPadSame:
			padTotal = (kernelSz - 1)
		case ConvPadCustom:
			padTotal = padBefore[i] + padAfter[i]
		}

		outShape[spatialIdx] = (inputSize + padTotal - kernelSz) / stride + 1
	}

	kernelVal := b.Const(b.genName("kernel_sizes"), Int32, []int64{int64(len(kernelSize))}, toInt32Slice(kernelSize))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	excludeVal := b.Const(b.genName("exclude_padding"), Bool, []int64{}, []bool{excludePaddingFromAverage})

	inputs := map[string]*Value{
		"x":                            x,
		"kernel_sizes":                 kernelVal,
		"strides":                      stridesVal,
		"exclude_padding_from_average": excludeVal,
	}

	// Add padding parameters based on type
	switch padType {
	case ConvPadValid:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "valid")
		inputs["pad_type"] = padTypeVal
		// Still need to provide pad parameter with zeros
		padSpec := make([]int32, 2*len(kernelSize))
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	case ConvPadSame:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "same")
		inputs["pad_type"] = padTypeVal
		// Still need to provide pad parameter with zeros
		padSpec := make([]int32, 2*len(kernelSize))
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	case ConvPadCustom:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "custom")
		inputs["pad_type"] = padTypeVal
		// Interleave padding values
		padSpec := make([]int32, 2*len(padBefore))
		for i := range padBefore {
			padSpec[2*i] = int32(padBefore[i])
			padSpec[2*i+1] = int32(padAfter[i])
		}
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	}

	// ceil_mode is required by CoreML MIL
	ceilModeVal := b.Const(b.genName("ceil_mode"), Bool, []int64{}, []bool{false})
	inputs["ceil_mode"] = ceilModeVal

	return b.addOp("avg_pool", inputs, b.genName("avg_pool"), x.dtype, outShape)
}

// GlobalAvgPool2D applies global average pooling over spatial dimensions (H, W).
// For NCHW input, reduces over dimensions 2 and 3.
// Output has shape [N, C, 1, 1] with keepDims=true.
func (b *Builder) GlobalAvgPool2D(x *Value) *Value {
	// Reduce over H, W dimensions (axes 2, 3 for NCHW)
	return b.ReduceMean(x, []int64{2, 3}, true)
}

// GlobalMaxPool2D applies global max pooling over spatial dimensions (H, W).
// For NCHW input, reduces over dimensions 2 and 3.
// Output has shape [N, C, 1, 1] with keepDims=true.
func (b *Builder) GlobalMaxPool2D(x *Value) *Value {
	// Reduce over H, W dimensions (axes 2, 3 for NCHW)
	return b.ReduceMax(x, []int64{2, 3}, true)
}

// Tile repeats a tensor along each axis by the specified repetition factors.
// reps: number of repetitions for each dimension
// Output shape: [x.shape[i] * reps[i] for i in range(rank)]
func (b *Builder) Tile(x *Value, reps []int64) *Value {
	// Compute output shape
	outShape := make([]int64, len(x.shape))
	for i := range outShape {
		outShape[i] = x.shape[i] * reps[i]
	}

	repsVal := b.Const(b.genName("reps"), Int32, []int64{int64(len(reps))}, toInt32Slice(reps))

	return b.addOp("tile", map[string]*Value{
		"x":    x,
		"reps": repsVal,
	}, b.genName("tile"), x.dtype, outShape)
}

// PadMode specifies the padding mode for Pad operation.
type PadMode int

const (
	// PadConstant fills padded values with a constant value.
	PadConstant PadMode = iota
	// PadReflect reflects values at the boundaries (mirroring without repeating edge values).
	PadReflect
	// PadReplicate replicates edge values.
	PadReplicate
)

// Pad adds padding to a tensor.
// padBefore: number of values to pad before each dimension
// padAfter: number of values to pad after each dimension
// mode: padding mode (constant, reflect, or replicate)
// constantValue: value to use for constant padding (ignored for other modes)
// Output shape: [x.shape[i] + padBefore[i] + padAfter[i] for i in range(rank)]
func (b *Builder) Pad(x *Value, padBefore, padAfter []int64, mode PadMode, constantValue float32) *Value {
	// Compute output shape
	outShape := make([]int64, len(x.shape))
	for i := range outShape {
		outShape[i] = x.shape[i] + padBefore[i] + padAfter[i]
	}

	// Create pad specification: [before_0, after_0, before_1, after_1, ...]
	padSpec := make([]int32, 2*len(padBefore))
	for i := range padBefore {
		padSpec[2*i] = int32(padBefore[i])
		padSpec[2*i+1] = int32(padAfter[i])
	}
	padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)

	// Convert mode to string
	var modeStr string
	switch mode {
	case PadConstant:
		modeStr = "constant"
	case PadReflect:
		modeStr = "reflect"
	case PadReplicate:
		modeStr = "replicate"
	default:
		modeStr = "constant"
	}
	modeVal := b.Const(b.genName("mode"), String, []int64{}, modeStr)

	// Constant value (only used for constant mode)
	constVal := b.Const(b.genName("constant_val"), Float32, []int64{}, []float32{constantValue})

	return b.addOp("pad", map[string]*Value{
		"x":            x,
		"pad":          padVal,
		"mode":         modeVal,
		"constant_val": constVal,
	}, b.genName("pad"), x.dtype, outShape)
}

// Reverse reverses a tensor along specified axes.
// axes: axes along which to reverse (empty or nil reverses all axes)
// Output shape: same as input shape
func (b *Builder) Reverse(x *Value, axes []int64) *Value {
	// If axes is empty, reverse along all axes
	if len(axes) == 0 {
		axes = make([]int64, len(x.shape))
		for i := range axes {
			axes[i] = int64(i)
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("reverse", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("reverse"), x.dtype, x.shape)
}

// Rsqrt computes element-wise reciprocal square root: z = 1/sqrt(x + epsilon).
// epsilon: Small constant added for numerical stability (typically 1e-12).
func (b *Builder) Rsqrt(x *Value) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{1e-12})

	return b.addOp("rsqrt", map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}, b.genName("rsqrt"), x.dtype, x.shape)
}

// LogicalAnd performs element-wise logical AND: z = x && y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalAnd(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("logical_and", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("logical_and"), Bool, outShape)
}

// LogicalOr performs element-wise logical OR: z = x || y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalOr(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("logical_or", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("logical_or"), Bool, outShape)
}

// LogicalNot performs element-wise logical NOT: z = !x.
// Input must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalNot(x *Value) *Value {
	return b.addOp("logical_not", map[string]*Value{
		"x": x,
	}, b.genName("logical_not"), Bool, x.shape)
}

// LogicalXor performs element-wise logical XOR: z = x ^ y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalXor(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("logical_xor", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("logical_xor"), Bool, outShape)
}

// IsNan checks element-wise if values are NaN.
// Returns Bool dtype indicating which elements are NaN.
// Implementation: NaN is the only value not equal to itself, so isnan(x) = (x != x)
func (b *Builder) IsNan(x *Value) *Value {
	// NaN != NaN, so we use not_equal(x, x)
	return b.NotEqual(x, x)
}

// IsFinite checks element-wise if values are finite (not NaN or Inf).
// Returns Bool dtype indicating which elements are finite.
// Implementation: A value is finite if it's not NaN and Inf * 0 doesn't produce NaN.
// Since Inf * 0 = NaN and finite * 0 = 0, we check: !isnan(x) && !isnan(x * 0)
func (b *Builder) IsFinite(x *Value) *Value {
	// Create zero constant
	zero := b.Const(b.genName("zero"), x.dtype, []int64{}, []float32{0.0})

	// x * 0: will be NaN for Inf, 0 for finite, NaN for NaN
	xTimesZero := b.Mul(x, zero)

	// Check if x is not NaN: x == x
	xNotNan := b.Equal(x, x)

	// Check if x*0 is not NaN: (x*0) == (x*0)
	xTimesZeroNotNan := b.Equal(xTimesZero, xTimesZero)

	// isfinite = !isnan(x) && !isnan(x*0)
	return b.LogicalAnd(xNotNan, xTimesZeroNotNan)
}

// Range1D generates a 1D tensor of values from start to end (exclusive) with given step.
// All parameters (start, end, step) are scalar values.
// Returns a 1D tensor with dtype matching start.
// Output size is ceil((end - start) / step).
//
// Note: This function can infer output size when all inputs are Int32 constants with scalar shape.
func (b *Builder) Range1D(start, end, step *Value) *Value {
	// Output is 1D, size depends on (end - start) / step
	// Try to compute size if all inputs are Int32 scalar constants
	var outputSize int64 = -1

	if start.isConst && end.isConst && step.isConst && start.dtype == Int32 {
		// Check if all are scalars (empty or single-element shape)
		if (len(start.shape) == 0 || (len(start.shape) == 1 && start.shape[0] == 1)) &&
			(len(end.shape) == 0 || (len(end.shape) == 1 && end.shape[0] == 1)) &&
			(len(step.shape) == 0 || (len(step.shape) == 1 && step.shape[0] == 1)) {

			// Extract values from the TensorValue immediate
			if start.constVal != nil && end.constVal != nil && step.constVal != nil {
				if startImm := start.constVal.GetImmediateValue(); startImm != nil {
					if endImm := end.constVal.GetImmediateValue(); endImm != nil {
						if stepImm := step.constVal.GetImmediateValue(); stepImm != nil {
							if startTensor := startImm.GetTensor(); startTensor != nil {
								if endTensor := endImm.GetTensor(); endTensor != nil {
									if stepTensor := stepImm.GetTensor(); stepTensor != nil {
										if startVals := startTensor.GetInts(); startVals != nil && len(startVals.Values) > 0 {
											if endVals := endTensor.GetInts(); endVals != nil && len(endVals.Values) > 0 {
												if stepVals := stepTensor.GetInts(); stepVals != nil && len(stepVals.Values) > 0 {
													startVal := startVals.Values[0]
													endVal := endVals.Values[0]
													stepVal := stepVals.Values[0]
													if stepVal > 0 && endVal > startVal {
														outputSize = int64((endVal-startVal+stepVal-1) / stepVal)
													} else if stepVal < 0 && endVal < startVal {
														outputSize = int64((startVal-endVal-stepVal-1) / -stepVal)
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	outShape := []int64{outputSize}

	return b.addOp("range_1d", map[string]*Value{
		"start": start,
		"end":   end,
		"step":  step,
	}, b.genName("range_1d"), start.dtype, outShape)
}

// Conv performs 2D convolution on input tensor x with filter weights.
// x: Input tensor in NCHW format [batch, channels_in, height, width]
// weight: Filter tensor [channels_out, channels_in/groups, kernel_height, kernel_width]
// strides: Stride for each spatial dimension [stride_h, stride_w]
// dilations: Dilation for each spatial dimension [dilation_h, dilation_w]
// padType: Padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore: Padding before each spatial dimension [pad_h_before, pad_w_before] (used only if padType is ConvPadCustom)
// padAfter: Padding after each spatial dimension [pad_h_after, pad_w_after] (used only if padType is ConvPadCustom)
// groups: Number of groups for grouped convolution (1 for standard convolution)
func (b *Builder) Conv(x, weight *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter []int64, groups int64) *Value {
	// Input shape: [N, C_in, H, W]
	// Weight shape: [C_out, C_in/groups, kH, kW]
	N := x.shape[0]
	Cout := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]
	inH := x.shape[2]
	inW := x.shape[3]

	// Default strides and dilations if not provided
	if len(strides) == 0 {
		strides = []int64{1, 1}
	}
	if len(dilations) == 0 {
		dilations = []int64{1, 1}
	}

	// Compute output spatial dimensions based on padding type
	var outH, outW int64
	var padTypeStr string

	switch padType {
	case ConvPadValid:
		padTypeStr = "valid"
		// Output size with no padding
		outH = (inH - dilations[0]*(kH-1) - 1) / strides[0] + 1
		outW = (inW - dilations[1]*(kW-1) - 1) / strides[1] + 1

	case ConvPadSame:
		padTypeStr = "same"
		// Output size preserves input dimensions (accounting for stride)
		outH = (inH + strides[0] - 1) / strides[0]
		outW = (inW + strides[1] - 1) / strides[1]

	case ConvPadCustom:
		padTypeStr = "custom"
		// Compute output size with custom padding
		if len(padBefore) == 0 {
			padBefore = []int64{0, 0}
		}
		if len(padAfter) == 0 {
			padAfter = []int64{0, 0}
		}
		paddedH := inH + padBefore[0] + padAfter[0]
		paddedW := inW + padBefore[1] + padAfter[1]
		outH = (paddedH - dilations[0]*(kH-1) - 1) / strides[0] + 1
		outW = (paddedW - dilations[1]*(kW-1) - 1) / strides[1] + 1
	}

	outShape := []int64{N, Cout, outH, outW}

	// Build operation inputs
	inputs := map[string]*Value{
		"x":      x,
		"weight": weight,
	}

	// Add parameters
	inputs["strides"] = b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	inputs["dilations"] = b.Const(b.genName("dilations"), Int32, []int64{int64(len(dilations))}, toInt32Slice(dilations))
	inputs["groups"] = b.Const(b.genName("groups"), Int32, []int64{}, []int32{int32(groups)})
	inputs["pad_type"] = b.Const(b.genName("pad_type"), String, []int64{}, padTypeStr)

	if padType == ConvPadCustom {
		// Flatten padding into [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
		pad := []int64{padBefore[0], padAfter[0], padBefore[1], padAfter[1]}
		inputs["pad"] = b.Const(b.genName("pad"), Int32, []int64{4}, toInt32Slice(pad))
	}

	return b.addOp("conv", inputs, b.genName("conv"), x.dtype, outShape)
}

// ConvTranspose performs 2D transposed convolution (also known as deconvolution).
// x: Input tensor in NCHW format [batch, channels_in, height, width]
// weight: Filter tensor [channels_in, channels_out/groups, kernel_height, kernel_width]
// strides: Stride for each spatial dimension [stride_h, stride_w]
// dilations: Dilation for each spatial dimension [dilation_h, dilation_w]
// padType: Padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore: Padding before each spatial dimension [pad_h_before, pad_w_before] (used only if padType is ConvPadCustom)
// padAfter: Padding after each spatial dimension [pad_h_after, pad_w_after] (used only if padType is ConvPadCustom)
// outputPadding: Additional padding added to output [output_pad_h, output_pad_w]
// groups: Number of groups for grouped convolution (1 for standard convolution)
func (b *Builder) ConvTranspose(x, weight *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter, outputPadding []int64, groups int64) *Value {
	// Input shape: [N, C_in, H, W]
	// Weight shape: [C_in, C_out/groups, kH, kW]
	N := x.shape[0]
	Cout := weight.shape[1] * groups
	kH := weight.shape[2]
	kW := weight.shape[3]
	inH := x.shape[2]
	inW := x.shape[3]

	// Default strides and dilations if not provided
	if len(strides) == 0 {
		strides = []int64{1, 1}
	}
	if len(dilations) == 0 {
		dilations = []int64{1, 1}
	}
	if len(outputPadding) == 0 {
		outputPadding = []int64{0, 0}
	}

	// Compute output spatial dimensions based on padding type
	var outH, outW int64
	var padTypeStr string

	switch padType {
	case ConvPadValid:
		padTypeStr = "valid"
		// Transposed convolution output size with no padding
		outH = (inH-1)*strides[0] + dilations[0]*(kH-1) + 1 + outputPadding[0]
		outW = (inW-1)*strides[1] + dilations[1]*(kW-1) + 1 + outputPadding[1]

	case ConvPadSame:
		padTypeStr = "same"
		// Output size preserves input dimensions (accounting for stride)
		outH = inH * strides[0]
		outW = inW * strides[1]

	case ConvPadCustom:
		padTypeStr = "custom"
		// Compute output size with custom padding
		if len(padBefore) == 0 {
			padBefore = []int64{0, 0}
		}
		if len(padAfter) == 0 {
			padAfter = []int64{0, 0}
		}
		outH = (inH-1)*strides[0] + dilations[0]*(kH-1) + 1 - padBefore[0] - padAfter[0] + outputPadding[0]
		outW = (inW-1)*strides[1] + dilations[1]*(kW-1) + 1 - padBefore[1] - padAfter[1] + outputPadding[1]
	}

	outShape := []int64{N, Cout, outH, outW}

	// Build operation inputs
	inputs := map[string]*Value{
		"x":      x,
		"weight": weight,
	}

	// Add parameters
	inputs["strides"] = b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	inputs["dilations"] = b.Const(b.genName("dilations"), Int32, []int64{int64(len(dilations))}, toInt32Slice(dilations))
	inputs["groups"] = b.Const(b.genName("groups"), Int32, []int64{}, []int32{int32(groups)})
	inputs["pad_type"] = b.Const(b.genName("pad_type"), String, []int64{}, padTypeStr)

	if padType == ConvPadCustom {
		// Flatten padding into [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
		pad := []int64{padBefore[0], padAfter[0], padBefore[1], padAfter[1]}
		inputs["pad"] = b.Const(b.genName("pad"), Int32, []int64{4}, toInt32Slice(pad))
	}

	if outputPadding[0] != 0 || outputPadding[1] != 0 {
		inputs["output_padding"] = b.Const(b.genName("output_padding"), Int32, []int64{int64(len(outputPadding))}, toInt32Slice(outputPadding))
	}

	return b.addOp("conv_transpose", inputs, b.genName("conv_transpose"), x.dtype, outShape)
}

// ConvWithBias performs 2D convolution with bias addition.
// This is a convenience function that combines Conv and bias addition.
// bias: Bias tensor [channels_out] to add to each output channel
// Other parameters are the same as Conv.
func (b *Builder) ConvWithBias(x, weight, bias *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter []int64, groups int64) *Value {
	// Perform convolution
	conv := b.Conv(x, weight, strides, dilations, padType, padBefore, padAfter, groups)

	// Reshape bias for broadcasting: [C_out] -> [1, C_out, 1, 1]
	biasShape := []int64{1, bias.shape[0], 1, 1}
	biasReshaped := b.Reshape(bias, biasShape)

	// Add bias to convolution output
	return b.Add(conv, biasReshaped)
}

// Concat concatenates a list of tensors along a specified axis.
// values: List of tensors to concatenate. All must have the same shape except along the concat axis.
// axis: Axis along which to concatenate. Must be in range [-rank, rank).
// Output shape: same as input shapes, except dimension along axis is the sum of input dimensions.
// Returns nil and sets builder error if called with no inputs.
func (b *Builder) Concat(values []*Value, axis int64) *Value {
	if len(values) == 0 {
		b.setErr(fmt.Errorf("concat requires at least one input tensor"))
		return nil
	}

	// Get first tensor's properties
	firstValue := values[0]
	dtype := firstValue.dtype
	rank := len(firstValue.shape)

	// Normalize negative axis
	if axis < 0 {
		axis = int64(rank) + axis
	}

	// Compute output shape: sum dimensions along concat axis
	outShape := make([]int64, rank)
	copy(outShape, firstValue.shape)

	// Sum the concat axis dimension across all inputs
	concatDim := int64(0)
	for _, v := range values {
		concatDim += v.shape[axis]
	}
	outShape[axis] = concatDim

	// Create axis constant
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})

	// Create interleave constant (default to false for standard concatenation)
	interleaveVal := b.Const(b.genName("interleave"), Bool, []int64{}, []bool{false})

	// Use addOpWithListArg to handle the list of values
	return b.addOpWithListArg("concat",
		map[string]*Value{
			"axis":       axisVal,
			"interleave": interleaveVal,
		}, // scalar inputs
		map[string][]*Value{"values": values}, // list inputs
		b.genName("concat"),
		dtype,
		outShape)
}

// Clip clamps values to the range [minVal, maxVal].
// x: Input tensor to clamp.
// minVal: Minimum value (can be a tensor or scalar).
// maxVal: Maximum value (can be a tensor or scalar).
// Output: x clamped to [minVal, maxVal].
// Implementation: clamp(x, min, max) = minimum(maximum(x, min), max)
func (b *Builder) Clip(x, minVal, maxVal *Value) *Value {
	// First apply the lower bound: max(x, minVal)
	lowerBounded := b.Maximum(x, minVal)
	// Then apply the upper bound: min(lowerBounded, maxVal)
	return b.Minimum(lowerBounded, maxVal)
}

// Cast converts a tensor to a different dtype.
// x: Input tensor to convert.
// dtype: Target data type.
// Output: x converted to dtype.
func (b *Builder) Cast(x *Value, dtype DType) *Value {
	// CoreML MIL cast operation requires dtype as a string parameter
	dtypeStr := dtypeToString(dtype)
	dtypeVal := b.Const(b.genName("dtype"), String, []int64{}, dtypeStr)
	return b.addOp("cast", map[string]*Value{
		"x":     x,
		"dtype": dtypeVal,
	}, b.genName("cast"), dtype, x.shape)
}

// dtypeToString converts a DType to its string representation for MIL operations.
func dtypeToString(dtype DType) string {
	switch dtype {
	case Float32:
		return "fp32"
	case Float16:
		return "fp16"
	case Float64:
		return "fp64"
	case Int32:
		return "int32"
	case Int16:
		return "int16"
	case Int8:
		return "int8"
	case Int64:
		return "int64"
	case Bool:
		return "bool"
	default:
		return "fp32"
	}
}

// L2Norm computes L2 normalization along specified axes.
// x: Input tensor.
// axes: Axes along which to compute L2 norm. If nil or empty, normalizes over all axes.
// epsilon: Small constant added to norm for numerical stability (typically 1e-12).
// Output: x / (L2_norm(x, axes) + epsilon), same shape as input.
func (b *Builder) L2Norm(x *Value, axes []int64, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}

	if len(axes) > 0 {
		axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
		inputs["axes"] = axesVal
	}

	return b.addOp("l2_norm", inputs, b.genName("l2_norm"), x.dtype, x.shape)
}

// Linear performs a fused matrix multiplication and bias addition: y = x @ weight^T + bias.
// This is more efficient than separate MatMul and Add operations.
// x: Input tensor with shape [..., in_features].
// weight: Weight matrix with shape [out_features, in_features].
// bias: Bias vector with shape [out_features]. Can be nil for no bias.
// Output: [..., out_features]
func (b *Builder) Linear(x, weight, bias *Value) *Value {
	// Compute output shape
	// x: [..., in_features], weight: [out_features, in_features]
	// output: [..., out_features]
	xShape := x.shape
	outFeatures := weight.shape[0]

	outShape := make([]int64, len(xShape))
	copy(outShape, xShape)
	outShape[len(outShape)-1] = outFeatures

	inputs := map[string]*Value{
		"x":      x,
		"weight": weight,
	}

	if bias != nil {
		inputs["bias"] = bias
	}

	return b.addOp("linear", inputs, b.genName("linear"), x.dtype, outShape)
}

// SliceBySize extracts a sub-tensor using dynamic start indices and fixed sizes.
// This is the MIL operation that supports dynamic start indices (runtime values).
// x: Input tensor to slice.
// begin: Start indices for each dimension (can be runtime values, shape [rank]).
// size: Size of the slice for each dimension (must be constants).
// Output shape: size (the slice dimensions).
func (b *Builder) SliceBySize(x *Value, begin *Value, size []int64) *Value {
	// Create size constant
	sizeVal := b.Const(b.genName("size"), Int32, []int64{int64(len(size))}, toInt32Slice(size))

	// Output shape is just the size parameter
	outShape := make([]int64, len(size))
	copy(outShape, size)

	return b.addOp("slice_by_size", map[string]*Value{
		"x":     x,
		"begin": begin,
		"size":  sizeVal,
	}, b.genName("slice_by_size"), x.dtype, outShape)
}

// ScatterND scatters updates into a tensor using multi-dimensional indices.
// This is a more general scatter operation that can update multiple dimensions.
// data: Input tensor to update.
// indices: Multi-dimensional indices [batch_dims..., index_depth].
// updates: Values to scatter.
// mode: Scatter mode ("update", "add", "sub", "mul", "div", "max", "min").
// Output: Updated tensor with same shape as data.
func (b *Builder) ScatterND(data, indices, updates *Value, mode string) *Value {
	modeVal := b.Const(b.genName("mode"), String, []int64{}, mode)
	// CoreML requires validate_indices parameter (typically set to false for performance)
	validateIndicesVal := b.Const(b.genName("validate_indices"), Bool, []int64{}, []bool{false})

	return b.addOp("scatter_nd", map[string]*Value{
		"data":             data,
		"indices":          indices,
		"updates":          updates,
		"mode":             modeVal,
		"validate_indices": validateIndicesVal,
	}, b.genName("scatter_nd"), data.dtype, data.shape)
}

// Einsum performs tensor multiplication using einsum notation.
// This operation is available in CoreML MIL for iOS 15+.
//
// CoreML MIL einsum supports a limited set of equation patterns, specifically for
// multiplying matrices on dimensions -1 and -3, treating other dimensions as batch.
// Broadcasting is supported along batch dimensions.
//
// Supported equation patterns:
//
// Rank 4 inputs:
//   - Equation: "nchw,nwhu->nchu" (and equivalent variations)
//   - Input 1: [B, C, H, W1]
//   - Input 2: [B, W1, H, W2]
//   - Output:  [B, C, H, W2]
//   - Broadcasting: If B or H is 1 in one input, it broadcasts to match the other
//
// Rank 3 inputs:
//   - Equation: "chw,whr->chr" (and equivalent variations)
//   - Input 1: [C, H, W1]
//   - Input 2: [W1, H, W2]
//   - Output:  [C, H, W2]
//   - Broadcasting: If H is 1 in one input, it broadcasts to match the other
//
// equation: Einstein summation notation string (e.g., "nchw,nwhu->nchu")
// values: Tuple of two input tensors (rank 3 or 4)
//
// Returns: Result tensor with shape determined by the equation.
// Returns nil and sets builder error if not given exactly 2 inputs.
func (b *Builder) Einsum(equation string, values []*Value) *Value {
	if len(values) != 2 {
		b.setErr(fmt.Errorf("einsum requires exactly 2 input tensors, got %d", len(values)))
		return nil
	}

	x := values[0]
	y := values[1]

	// Compute output shape based on equation and input shapes
	// For the supported patterns, the output has the same rank as inputs
	outShape := computeEinsumOutputShape(equation, x.shape, y.shape)

	// Create equation constant
	equationVal := b.Const(b.genName("equation"), String, []int64{}, equation)

	// Use addOpWithListArg to handle the list of values
	return b.addOpWithListArg("einsum",
		map[string]*Value{"equation": equationVal}, // scalar inputs
		map[string][]*Value{"values": values},      // list inputs
		b.genName("einsum"),
		x.dtype,
		outShape)
}

// Helper functions

func toInt32Slice(s []int64) []int32 {
	result := make([]int32, len(s))
	for i, v := range s {
		result[i] = int32(v)
	}
	return result
}

func broadcastShape(a, b []int64) []int64 {
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}

	result := make([]int64, maxLen)
	for i := 0; i < maxLen; i++ {
		ai := int64(1)
		bi := int64(1)

		if i < len(a) {
			ai = a[len(a)-1-i]
		}
		if i < len(b) {
			bi = b[len(b)-1-i]
		}

		if ai == 1 {
			result[maxLen-1-i] = bi
		} else if bi == 1 {
			result[maxLen-1-i] = ai
		} else if ai == bi {
			result[maxLen-1-i] = ai
		} else {
			// Incompatible shapes - return larger
			if ai > bi {
				result[maxLen-1-i] = ai
			} else {
				result[maxLen-1-i] = bi
			}
		}
	}
	return result
}

func computeReduceShape(shape []int64, axes []int64, keepDims bool) []int64 {
	axisSet := make(map[int64]bool)
	for _, a := range axes {
		if a < 0 {
			a = int64(len(shape)) + a
		}
		axisSet[a] = true
	}

	if keepDims {
		result := make([]int64, len(shape))
		for i, dim := range shape {
			if axisSet[int64(i)] {
				result[i] = 1
			} else {
				result[i] = dim
			}
		}
		return result
	}

	var result []int64
	for i, dim := range shape {
		if !axisSet[int64(i)] {
			result = append(result, dim)
		}
	}
	if len(result) == 0 {
		return []int64{} // Scalar
	}
	return result
}

// computeEinsumOutputShape computes the output shape for einsum operation.
// CoreML MIL einsum supports limited patterns for batched matrix multiplication.
// This function handles the supported patterns and computes the output shape.
func computeEinsumOutputShape(equation string, xShape, yShape []int64) []int64 {
	// For the supported CoreML patterns, the output shape follows this logic:
	// Rank 4: [B, C, H, W1] x [B, W1, H, W2] -> [B, C, H, W2]
	// Rank 3: [C, H, W1] x [W1, H, W2] -> [C, H, W2]
	//
	// The general pattern is:
	// - Batch dimensions (if present) are preserved with broadcasting
	// - First input contributes dimension at position 1 (C)
	// - Both inputs share dimension at position -2 (H)
	// - Last dimension of output comes from second input (W2)

	rank := len(xShape)
	outShape := make([]int64, rank)

	if rank == 4 {
		// Rank 4: [B, C, H, W1] x [B, W1, H, W2] -> [B, C, H, W2]
		// Broadcast batch dimension
		if xShape[0] == 1 {
			outShape[0] = yShape[0]
		} else {
			outShape[0] = xShape[0]
		}
		// C from first input
		outShape[1] = xShape[1]
		// H with broadcasting
		if xShape[2] == 1 {
			outShape[2] = yShape[2]
		} else {
			outShape[2] = xShape[2]
		}
		// W2 from second input
		outShape[3] = yShape[3]
	} else if rank == 3 {
		// Rank 3: [C, H, W1] x [W1, H, W2] -> [C, H, W2]
		// C from first input
		outShape[0] = xShape[0]
		// H with broadcasting
		if xShape[1] == 1 {
			outShape[1] = yShape[1]
		} else {
			outShape[1] = xShape[1]
		}
		// W2 from second input
		outShape[2] = yShape[2]
	} else {
		// Unsupported rank - return -1 to indicate unknown shape
		// This will be caught at runtime when CoreML validates the operation
		outShape = make([]int64, rank)
		for i := range outShape {
			outShape[i] = -1
		}
	}

	return outShape
}
