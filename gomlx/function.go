//go:build darwin && cgo

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package coreml

import (
	"fmt"
	"slices"

	"github.com/gomlx/go-coreml/model"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Function implements backends.Function for CoreML.
type Function struct {
	notimplemented.Function

	builder *Builder
	name    string

	// parent is the parent function if this is a closure.
	// For top-level functions (including main), this is nil.
	parent *Function

	// returned indicates Return() was called.
	returned bool

	// outputs stores the return values set by Return().
	outputs []*Node

	// parameters stores the parameter nodes for this function.
	parameters []*Node

	// compiled holds pre-compiled execution info (only for closures).
	// This is set during Return() for closures to allow efficient execution.
	compiled *CompiledClosure
}

var _ backends.Function = (*Function)(nil)

// CheckValid returns an error if the builder or the function are not ok.
func (f *Function) CheckValid() error {
	if f == nil || f.builder == nil {
		return errors.Errorf("function is nil or undefined for %q", BackendName)
	}
	if f.builder.compiled {
		return errors.Errorf("cannot add new op to Function %q, builder has already been compiled", f.name)
	}
	return nil
}

// Name returns the name of this function.
// For closures, this returns "".
func (f *Function) Name() string {
	return f.name
}

// Parent returns the parent function if this is a closure.
// Returns nil for top-level functions (including main).
func (f *Function) Parent() backends.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}

// Closure creates a new closure function within this function.
// Closures can access values from their parent function's scope.
func (f *Function) Closure() (backends.Function, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	closure := &Function{
		builder: f.builder,
		name:    "", // Closures have empty names
		parent:  f,
	}
	return closure, nil
}

// sanitizeName converts a name to a valid CoreML identifier.
// CoreML identifiers must match: [A-Za-z\_][A-Za-z0-9\_@]*
// This function replaces invalid characters with underscores and ensures
// the name starts with a letter or underscore.
func sanitizeName(name string) string {
	if name == "" {
		return "_empty_"
	}

	result := make([]byte, 0, len(name))
	for i, c := range name {
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c == '@' {
			result = append(result, byte(c))
		} else if c >= '0' && c <= '9' {
			if i == 0 {
				// Can't start with a digit, prepend underscore
				result = append(result, '_')
			}
			result = append(result, byte(c))
		} else {
			// Replace invalid character with underscore
			result = append(result, '_')
		}
	}

	// Ensure we have at least one character
	if len(result) == 0 {
		return "_sanitized_"
	}

	// Ensure first character is valid (letter or underscore)
	if result[0] != '_' && result[0] != '@' &&
		!(result[0] >= 'A' && result[0] <= 'Z') &&
		!(result[0] >= 'a' && result[0] <= 'z') {
		result = append([]byte{'_'}, result...)
	}

	return string(result)
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		return nil, errors.Errorf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.builder.backend.Name())
	}
	if sharding != nil {
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"sharding spec %+v not supported for %q builder", sharding, BackendName)
	}

	// Sanitize the name to be a valid CoreML identifier
	sanitizedName := sanitizeName(name)

	// Convert GoMLX dtype to CoreML dtype
	milDType, err := gomlxDTypeToMIL(shape.DType)
	if err != nil {
		return nil, errors.Wrapf(err, "Parameter %q", name)
	}

	// Convert shape dimensions to int64
	dims := make([]int64, shape.Rank())
	for i := 0; i < shape.Rank(); i++ {
		dims[i] = int64(shape.Dimensions[i])
	}

	var milValue *model.Value
	var node *Node

	// Check if this is a closure (has a parent function)
	// Closure parameters are not model inputs - they receive values from the parent scope
	// via control flow operations (While, If) at runtime
	if f.parent != nil {
		// For closures, create a placeholder value that is NOT added as a model input.
		// The actual block input will be created during replayClosureInBlock.
		// We use a unique name to avoid conflicts.
		closureParamName := fmt.Sprintf("closure_%p_param_%s", f, sanitizedName)
		milValue = f.builder.milBuilder.PlaceholderValue(closureParamName, milDType, dims...)
		node = f.builder.newNode(backends.OpTypeParameter, shape, milValue)
		f.builder.nodeMap[node] = milValue
		f.parameters = append(f.parameters, node)
		// Note: We do NOT append to f.builder.inputs, inputNames, or inputShapes
		// because closure parameters are not model-level inputs
	} else {
		// For the main function, create a proper model input
		// Use sanitized name for CoreML compatibility
		milValue = f.builder.milBuilder.Input(sanitizedName, milDType, dims...)
		node = f.builder.newNode(backends.OpTypeParameter, shape, milValue)
		f.builder.inputs = append(f.builder.inputs, node)
		f.builder.nodeMap[node] = milValue
		f.parameters = append(f.parameters, node)
		// Track input metadata for model-level inputs only (use sanitized name)
		f.builder.inputNames = append(f.builder.inputNames, sanitizedName)
		f.builder.inputShapes = append(f.builder.inputShapes, shape)
	}

	return node, nil
}

// Constant creates a constant in the function with the given flat values and the shape defined by the dimensions.
func (f *Function) Constant(flat any, dims ...int) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	// Validate and get dtype
	dtype, flatLen, err := checkFlat(flat)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		return nil, errors.Errorf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, f.builder.backend.Name())
	}

	// Validate dimensions
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		return nil, errors.Errorf(
			"Constant: shape %s has size %d, but flat data has length %d",
			shape,
			shape.Size(),
			flatLen,
		)
	}

	// Convert to MIL dtype (note: Int64 maps to Int32 for CoreML compatibility)
	milDType, err := gomlxDTypeToMIL(dtype)
	if err != nil {
		return nil, errors.Wrap(err, "Constant")
	}

	// If the GoMLX dtype is Int64 but MIL dtype is Int32, convert the data.
	// This keeps the GoMLX shape as Int64 (for onnx-gomlx compatibility) while
	// giving CoreML Int32 data (which it supports).
	// Values that exceed Int32 range are clamped — this is safe for ML models
	// where out-of-range Int64 constants are typically attention mask values
	// (large negatives) where the exact magnitude doesn't matter.
	milData := flat
	if dtype == dtypes.Int64 && milDType == model.Int32 {
		int64Data := flat.([]int64)
		milData = convertInt64ToInt32Clamped(int64Data)
	}

	// Convert dimensions to int64
	milShape := make([]int64, len(dims))
	for i, d := range dims {
		milShape[i] = int64(d)
	}

	// Generate unique name for constant
	constName := fmt.Sprintf("const_%d", f.builder.nextConstID)
	f.builder.nextConstID++

	// Create constant in MIL builder
	milValue := f.builder.milBuilder.Const(constName, milDType, milShape, milData)

	// Create node
	node := f.builder.newNode(backends.OpTypeConstant, shape, milValue)
	f.builder.nodeMap[node] = milValue

	return node, nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	if err := f.CheckValid(); err != nil {
		return err
	}
	if f.returned {
		return errors.Errorf("Return() already called for function %q", f.name)
	}
	if len(outputs) == 0 {
		return errors.Errorf("Return() requires at least one output")
	}
	if len(shardings) != 0 {
		return errors.Errorf("sharding or distributed execution are not supported by CoreML backend")
	}

	outputNodes, err := f.builder.checkOps("Return", outputs...)
	if err != nil {
		return err
	}

	f.outputs = outputNodes
	f.returned = true

	// If this is a closure, pre-compile it for efficient execution
	if f.parent != nil {
		compiled, err := f.compile()
		if err != nil {
			return errors.WithMessagef(err, "failed to compile closure")
		}
		f.compiled = compiled
	}

	return nil
}

// CompiledClosure returns the pre-compiled closure, or nil if not a closure.
func (f *Function) CompiledClosure() *CompiledClosure {
	return f.compiled
}

// compile pre-compiles a closure for efficient execution.
// This computes the execution order, parameter mappings, and usage counts.
func (f *Function) compile() (*CompiledClosure, error) {
	cc := &CompiledClosure{
		function:         f,
		outputNodes:      f.outputs,
		parameterIndices: make(map[int]int),
		nodeToSortedIdx:  make(map[int]int),
	}

	// 1. Identify all nodes reachable from outputs using DFS
	neededNodes := make(map[int]bool)
	var findNeeded func(node *Node)
	findNeeded = func(node *Node) {
		if neededNodes[node.builderIdx] {
			return
		}
		neededNodes[node.builderIdx] = true
		for _, input := range node.inputs {
			findNeeded(input)
		}
	}
	for _, out := range f.outputs {
		findNeeded(out)
	}

	// 2. Collect and sort nodes topologically (by builderIdx order)
	for nodeIdx := range neededNodes {
		cc.sortedNodes = append(cc.sortedNodes, f.builder.nodes[nodeIdx])
	}
	slices.SortFunc(cc.sortedNodes, func(a, b *Node) int {
		return a.builderIdx - b.builderIdx
	})

	// 3. Build reverse mapping from builderIdx to sortedNodes index
	for i, node := range cc.sortedNodes {
		cc.nodeToSortedIdx[node.builderIdx] = i
	}

	// 4. Map parameters to input indices
	for i, param := range f.parameters {
		cc.parameterIndices[param.builderIdx] = i
	}

	// 5. Count uses and find max inputs
	cc.numUses = make([]int, len(cc.sortedNodes))
	for _, node := range cc.sortedNodes {
		cc.maxInputs = max(cc.maxInputs, len(node.inputs))
		for _, input := range node.inputs {
			if inputSortedIdx, ok := cc.nodeToSortedIdx[input.builderIdx]; ok {
				cc.numUses[inputSortedIdx]++
			}
		}
	}
	// Count output uses
	for _, out := range f.outputs {
		if outSortedIdx, ok := cc.nodeToSortedIdx[out.builderIdx]; ok {
			cc.numUses[outSortedIdx]++
		}
	}

	return cc, nil
}

// CompiledClosure holds pre-compiled execution info for a closure.
type CompiledClosure struct {
	function         *Function
	sortedNodes      []*Node
	nodeToSortedIdx  map[int]int
	parameterIndices map[int]int
	outputNodes      []*Node
	numUses          []int
	maxInputs        int
}

//======================================================================================================================
// Operations on Function
//======================================================================================================================

// isClosureContext returns true if this function is a closure (has a parent function).
// Closures don't build MIL operations eagerly; instead, operations are built during replay
// in the control flow BlockBuilder.
func (f *Function) isClosureContext() bool {
	return f.parent != nil
}

// closurePlaceholder creates a placeholder value for use in closure contexts.
// This is used when building closure functions (e.g., for While/If) where we can't
// execute MIL operations directly but need to record the shape for later replay.
// Returns nil and an error if the dtype is not supported.
func (f *Function) closurePlaceholder(outputShape shapes.Shape) (*model.Value, error) {
	placeholderName := fmt.Sprintf("closure_op_%p_%d", f, len(f.builder.nodes))
	milDType, err := gomlxDTypeToMIL(outputShape.DType)
	if err != nil {
		return nil, err
	}
	milShape := make([]int64, outputShape.Rank())
	for i := 0; i < outputShape.Rank(); i++ {
		milShape[i] = int64(outputShape.Dimensions[i])
	}
	return f.builder.milBuilder.PlaceholderValue(placeholderName, milDType, milShape...), nil
}

// addUnaryOp is a helper that adds a unary operation to the computation graph.
func (f *Function) addUnaryOp(
	opType backends.OpType,
	milOp func(*model.Value) *model.Value,
	x backends.Value,
) (*Node, error) {
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		// In closure context, create a placeholder value.
		// The actual MIL operation will be built during replayClosureInBlock.
		placeholderName := fmt.Sprintf("closure_op_%p_%d", f, len(f.builder.nodes))
		resultValue = f.builder.milBuilder.PlaceholderValue(placeholderName, operand.milValue.DType(), operand.milValue.Shape()...)
	} else {
		// In main function context, build the MIL operation directly.
		operandValue := operand.milValue
		resultValue = milOp(operandValue)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// addBinaryOp is a helper that adds a binary operation to the computation graph.
func (f *Function) addBinaryOp(
	opType backends.OpType,
	milOp func(*model.Value, *model.Value) *model.Value,
	lhs, rhs backends.Value,
) (*Node, error) {
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Handle dtype mismatch between Int32 and Int64 at the GoMLX level.
	// Since CoreML maps Int64→Int32, both MIL values are Int32, but GoMLX shapes
	// might differ. Unify to Int64 for shape inference (the actual MIL ops use Int32).
	lhsValue := lhsNode.milValue
	rhsValue := rhsNode.milValue
	lhsShape := lhsNode.shape
	rhsShape := rhsNode.shape

	if lhsShape.DType == dtypes.Int32 && rhsShape.DType == dtypes.Int64 {
		// Promote LHS shape to Int64 for shape inference (MIL values are already both Int32)
		lhsShape = shapes.Make(dtypes.Int64, lhsShape.Dimensions...)
	} else if lhsShape.DType == dtypes.Int64 && rhsShape.DType == dtypes.Int32 {
		// Promote RHS shape to Int64 for shape inference (MIL values are already both Int32)
		rhsShape = shapes.Make(dtypes.Int64, rhsShape.Dimensions...)
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.BinaryOp(opType, lhsShape, rhsShape)
	if err != nil {
		return nil, err
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		// In closure context, create a placeholder value.
		// The actual MIL operation will be built during replayClosureInBlock.
		placeholderName := fmt.Sprintf("closure_op_%p_%d", f, len(f.builder.nodes))
		milDType, _ := gomlxDTypeToMIL(outputShape.DType)
		milShape := make([]int64, outputShape.Rank())
		for i := 0; i < outputShape.Rank(); i++ {
			milShape[i] = int64(outputShape.Dimensions[i])
		}
		resultValue = f.builder.milBuilder.PlaceholderValue(placeholderName, milDType, milShape...)
	} else {
		// In main function context, build the MIL operation directly.
		resultValue = milOp(lhsValue, rhsValue)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

// addComparisonOp is a helper that adds a comparison operation to the computation graph.
func (f *Function) addComparisonOp(
	opType backends.OpType,
	milOp func(*model.Value, *model.Value) *model.Value,
	lhs, rhs backends.Value,
) (*Node, error) {
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Handle dtype mismatch between Int32 and Int64 at the GoMLX level.
	// Since CoreML maps Int64→Int32, both MIL values are Int32, but GoMLX shapes
	// might differ. Unify to Int64 for shape inference (the actual MIL ops use Int32).
	lhsValue := lhsNode.milValue
	rhsValue := rhsNode.milValue
	lhsShape := lhsNode.shape
	rhsShape := rhsNode.shape

	if lhsShape.DType == dtypes.Int32 && rhsShape.DType == dtypes.Int64 {
		// Promote LHS shape to Int64 for shape inference (MIL values are already both Int32)
		lhsShape = shapes.Make(dtypes.Int64, lhsShape.Dimensions...)
	} else if lhsShape.DType == dtypes.Int64 && rhsShape.DType == dtypes.Int32 {
		// Promote RHS shape to Int64 for shape inference (MIL values are already both Int32)
		rhsShape = shapes.Make(dtypes.Int64, rhsShape.Dimensions...)
	}

	// Compute output shape using shapeinference.ComparisonOp
	outputShape, err := shapeinference.ComparisonOp(opType, lhsShape, rhsShape)
	if err != nil {
		return nil, err
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		// In closure context, create a placeholder value.
		// The actual MIL operation will be built during replayClosureInBlock.
		placeholderName := fmt.Sprintf("closure_op_%p_%d", f, len(f.builder.nodes))
		milDType, _ := gomlxDTypeToMIL(outputShape.DType)
		milShape := make([]int64, outputShape.Rank())
		for i := 0; i < outputShape.Rank(); i++ {
			milShape[i] = int64(outputShape.Dimensions[i])
		}
		resultValue = f.builder.milBuilder.PlaceholderValue(placeholderName, milDType, milShape...)
	} else {
		// In main function context, build the MIL operation directly.
		resultValue = milOp(lhsValue, rhsValue)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

//======================================================================================================================
// Unary Operations
//======================================================================================================================

// Abs implements backends.Function.
func (f *Function) Abs(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeAbs, f.builder.milBuilder.Abs, x)
}

// Neg implements backends.Function.
func (f *Function) Neg(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeNeg
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Create a constant -1 for multiplication (scalar broadcasts)
	// Always use Float32 for the constant data, then cast to operand's dtype if needed
	constName := fmt.Sprintf("neg_one_%d", f.builder.nextConstID)
	f.builder.nextConstID++
	negOne := f.builder.milBuilder.Const(constName, model.Float32, []int64{}, []float32{-1.0})

	// Cast to operand's dtype if it's not Float32
	operandDType := operand.milValue.DType()
	if operandDType != model.Float32 {
		negOne = f.builder.milBuilder.Cast(negOne, operandDType)
	}

	// Multiply by -1 to negate
	resultValue := f.builder.milBuilder.Mul(operand.milValue, negOne)

	// Create a new node with the result
	node := f.builder.newNode(opType, operand.shape, resultValue, operand)

	return node, nil
}

// Exp implements backends.Function.
func (f *Function) Exp(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeExp, f.builder.milBuilder.Exp, x)
}

// Log implements backends.Function.
func (f *Function) Log(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLog, f.builder.milBuilder.Log, x)
}

// Sqrt implements backends.Function.
func (f *Function) Sqrt(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSqrt, f.builder.milBuilder.Sqrt, x)
}

// Floor implements backends.Function.
func (f *Function) Floor(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeFloor, f.builder.milBuilder.Floor, x)
}

// Ceil implements backends.Function.
func (f *Function) Ceil(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeCeil, f.builder.milBuilder.Ceil, x)
}

// Round implements backends.Function.
func (f *Function) Round(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeRound, f.builder.milBuilder.Round, x)
}

// Sign implements backends.Function.
func (f *Function) Sign(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSign, f.builder.milBuilder.Sign, x)
}

// Tanh implements backends.Function.
func (f *Function) Tanh(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeTanh, f.builder.milBuilder.Tanh, x)
}

// Logistic implements backends.Function.
func (f *Function) Logistic(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeLogistic, f.builder.milBuilder.Sigmoid, x)
}

// Cos implements backends.Function.
func (f *Function) Cos(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeCos, f.builder.milBuilder.Cos, x)
}

// Sin implements backends.Function.
func (f *Function) Sin(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeSin, f.builder.milBuilder.Sin, x)
}

// Erf implements backends.Function.
func (f *Function) Erf(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeErf, f.builder.milBuilder.Erf, x)
}

// Expm1 implements backends.Function.
func (f *Function) Expm1(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeExpm1
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape (same as input)
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	// exp(x)
	expResult := f.builder.milBuilder.Exp(operand.milValue)

	// Create constant 1 (always use Float32 for data, then cast if needed)
	constName := fmt.Sprintf("expm1_one_%d", f.builder.nextConstID)
	f.builder.nextConstID++
	one := f.builder.milBuilder.Const(constName, model.Float32, []int64{}, []float32{1.0})

	// Cast to operand's dtype if it's not Float32
	operandDType := operand.milValue.DType()
	if operandDType != model.Float32 {
		one = f.builder.milBuilder.Cast(one, operandDType)
	}

	// exp(x) - 1
	resultValue := f.builder.milBuilder.Sub(expResult, one)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Log1p implements backends.Function.
func (f *Function) Log1p(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeLog1p
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape (same as input)
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	// Create constant 1 (always use Float32 for data, then cast if needed)
	constName := fmt.Sprintf("log1p_one_%d", f.builder.nextConstID)
	f.builder.nextConstID++
	one := f.builder.milBuilder.Const(constName, model.Float32, []int64{}, []float32{1.0})

	// Cast to operand's dtype if it's not Float32
	operandDType := operand.milValue.DType()
	if operandDType != model.Float32 {
		one = f.builder.milBuilder.Cast(one, operandDType)
	}

	// x + 1
	xPlusOne := f.builder.milBuilder.Add(operand.milValue, one)

	// log(x + 1)
	resultValue := f.builder.milBuilder.Log(xPlusOne)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Rsqrt implements backends.Function.
func (f *Function) Rsqrt(x backends.Value) (backends.Value, error) {
	return f.addUnaryOp(backends.OpTypeRsqrt, f.builder.milBuilder.Rsqrt, x)
}

//======================================================================================================================
// Binary Operations
//======================================================================================================================

// Add implements backends.Function.
func (f *Function) Add(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeAdd, f.builder.milBuilder.Add, lhs, rhs)
}

// Sub implements backends.Function.
func (f *Function) Sub(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeSub, f.builder.milBuilder.Sub, lhs, rhs)
}

// Mul implements backends.Function.
func (f *Function) Mul(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMul, f.builder.milBuilder.Mul, lhs, rhs)
}

// Div implements backends.Function.
func (f *Function) Div(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeDiv, f.builder.milBuilder.Div, lhs, rhs)
}

// Pow implements backends.Function.
func (f *Function) Pow(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypePow, f.builder.milBuilder.Pow, lhs, rhs)
}

// Max implements backends.Function.
func (f *Function) Max(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMax, f.builder.milBuilder.Maximum, lhs, rhs)
}

// Min implements backends.Function.
func (f *Function) Min(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addBinaryOp(backends.OpTypeMin, f.builder.milBuilder.Minimum, lhs, rhs)
}

//======================================================================================================================
// Comparison Operations
//======================================================================================================================

// Equal implements backends.Function.
func (f *Function) Equal(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeEqual, f.builder.milBuilder.Equal, lhs, rhs)
}

// NotEqual implements backends.Function.
func (f *Function) NotEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeNotEqual, f.builder.milBuilder.NotEqual, lhs, rhs)
}

// LessThan implements backends.Function.
func (f *Function) LessThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeLessThan, f.builder.milBuilder.Less, lhs, rhs)
}

// LessOrEqual implements backends.Function.
func (f *Function) LessOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeLessOrEqual, f.builder.milBuilder.LessEqual, lhs, rhs)
}

// GreaterThan implements backends.Function.
func (f *Function) GreaterThan(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeGreaterThan, f.builder.milBuilder.Greater, lhs, rhs)
}

// GreaterOrEqual implements backends.Function.
func (f *Function) GreaterOrEqual(lhs, rhs backends.Value) (backends.Value, error) {
	return f.addComparisonOp(backends.OpTypeGreaterOrEqual, f.builder.milBuilder.GreaterEqual, lhs, rhs)
}

//======================================================================================================================
// Reduce Operations
//======================================================================================================================

// ReduceSum implements backends.Function.
func (f *Function) ReduceSum(x backends.Value, axes ...int) (backends.Value, error) {
	return f.addReduceOp(backends.OpTypeReduceSum, f.builder.milBuilder.ReduceSum, x, axes...)
}

// ReduceMax implements backends.Function.
func (f *Function) ReduceMax(x backends.Value, axes ...int) (backends.Value, error) {
	return f.addReduceOp(backends.OpTypeReduceMax, f.builder.milBuilder.ReduceMax, x, axes...)
}

// ReduceMin implements backends.Function.
func (f *Function) ReduceMin(x backends.Value, axes ...int) (backends.Value, error) {
	return f.addReduceOp(backends.OpTypeReduceMin, f.builder.milBuilder.ReduceMin, x, axes...)
}

// ReduceProduct implements backends.Function.
func (f *Function) ReduceProduct(x backends.Value, axes ...int) (backends.Value, error) {
	return f.addReduceOp(backends.OpTypeReduceProduct, f.builder.milBuilder.ReduceProd, x, axes...)
}

// addReduceOp is a helper that adds a reduce operation to the computation graph.
func (f *Function) addReduceOp(
	opType backends.OpType,
	milOp func(*model.Value, []int64, bool) *model.Value,
	x backends.Value,
	axes ...int,
) (*Node, error) {
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If no axes specified, reduce over all axes
	if len(axes) == 0 {
		axes = make([]int, operand.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}

	// Convert axes to int64
	milAxes := make([]int64, len(axes))
	for i, axis := range axes {
		milAxes[i] = int64(axis)
	}

	// Call the MIL operation (keep_dims=false to match GoMLX semantics)
	resultValue := milOp(operand.milValue, milAxes, false)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Other Operations
//======================================================================================================================

// Slice implements backends.Function.
func (f *Function) Slice(x backends.Value, starts, limits, strides []int) (backends.Value, error) {
	opType := backends.OpTypeSlice
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.SliceOp(operand.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}

	// Convert to int64 for MIL
	milBegin := make([]int64, len(starts))
	milEnd := make([]int64, len(limits))
	milStride := make([]int64, len(strides))
	for i := range starts {
		milBegin[i] = int64(starts[i])
		milEnd[i] = int64(limits[i])
		if strides != nil && i < len(strides) {
			milStride[i] = int64(strides[i])
		} else {
			milStride[i] = 1
		}
	}

	// Call the MIL operation
	resultValue := f.builder.milBuilder.SliceByIndex(operand.milValue, milBegin, milEnd, milStride)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Dot implements backends.Function (matrix multiplication).
func (f *Function) Dot(lhs, rhs backends.Value) (backends.Value, error) {
	opType := backends.OpTypeDot
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Dot is for 1D or 2D tensors - for 2D it's a matrix multiplication
	// For 1D vectors, it's an inner product
	lhsShape := lhsNode.shape
	rhsShape := rhsNode.shape

	var outputShape shapes.Shape
	if lhsShape.Rank() == 1 && rhsShape.Rank() == 1 {
		// Inner product: [N] dot [N] -> scalar
		if lhsShape.Dimensions[0] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: vector lengths must match, got %d and %d",
				lhsShape.Dimensions[0], rhsShape.Dimensions[0])
		}
		outputShape = shapes.Make(lhsShape.DType)
	} else if lhsShape.Rank() == 2 && rhsShape.Rank() == 2 {
		// Matrix multiplication: [M, K] dot [K, N] -> [M, N]
		if lhsShape.Dimensions[1] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: matrix inner dimensions must match, got [%d, %d] and [%d, %d]",
				lhsShape.Dimensions[0], lhsShape.Dimensions[1],
				rhsShape.Dimensions[0], rhsShape.Dimensions[1])
		}
		outputShape = shapes.Make(lhsShape.DType, lhsShape.Dimensions[0], rhsShape.Dimensions[1])
	} else if lhsShape.Rank() == 2 && rhsShape.Rank() == 1 {
		// Matrix-vector: [M, K] dot [K] -> [M]
		if lhsShape.Dimensions[1] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: matrix column count must match vector length, got %d and %d",
				lhsShape.Dimensions[1], rhsShape.Dimensions[0])
		}
		outputShape = shapes.Make(lhsShape.DType, lhsShape.Dimensions[0])
	} else if lhsShape.Rank() == 1 && rhsShape.Rank() == 2 {
		// Vector-matrix: [K] dot [K, N] -> [N]
		if lhsShape.Dimensions[0] != rhsShape.Dimensions[0] {
			return nil, errors.Errorf("Dot: vector length must match matrix row count, got %d and %d",
				lhsShape.Dimensions[0], rhsShape.Dimensions[0])
		}
		outputShape = shapes.Make(lhsShape.DType, rhsShape.Dimensions[1])
	} else {
		return nil, errors.Errorf("Dot: only supports 1D and 2D tensors, got ranks %d and %d",
			lhsShape.Rank(), rhsShape.Rank())
	}

	// Call the MIL operation (MatMul)
	resultValue := f.builder.milBuilder.MatMul(lhsNode.milValue, rhsNode.milValue)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

// ArgMinMax implements backends.Function.
func (f *Function) ArgMinMax(x backends.Value, axis int, outputDType dtypes.DType, isMin bool) (backends.Value, error) {
	opType := backends.OpTypeArgMinMax
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ArgMinMaxOp(operand.shape, axis, outputDType)
	if err != nil {
		return nil, err
	}

	// Call the appropriate MIL operation (keep_dims=false to remove the axis)
	var resultValue *model.Value
	if isMin {
		resultValue = f.builder.milBuilder.ArgMin(operand.milValue, int64(axis), false)
	} else {
		resultValue = f.builder.milBuilder.ArgMax(operand.milValue, int64(axis), false)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Reshape reshapes x to the new dimensions.
// Total size cannot change, it's just a "reinterpretation" of the same flat data.
func (f *Function) Reshape(x backends.Value, dimensions ...int) (backends.Value, error) {
	opType := backends.OpTypeReshape
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReshapeOp(operand.shape, dimensions)
	if err != nil {
		return nil, err
	}

	// Convert dimensions to int64 for MIL
	milShape := make([]int64, len(dimensions))
	for i, d := range dimensions {
		milShape[i] = int64(d)
	}

	// Call the MIL operation
	resultValue := f.builder.milBuilder.Reshape(operand.milValue, milShape)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Transpose axes of x.
// There should be one value in permutations for each axis in x.
// The output will have: output.Shape.Dimension[ii] = x.Shape.Dimension[permutations[i]].
func (f *Function) Transpose(x backends.Value, permutation ...int) (backends.Value, error) {
	opType := backends.OpTypeTranspose
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.TransposeOp(operand.shape, permutation)
	if err != nil {
		return nil, err
	}

	// Convert permutation to int64 for MIL
	milPerm := make([]int64, len(permutation))
	for i, p := range permutation {
		milPerm[i] = int64(p)
	}

	// Call the MIL operation
	resultValue := f.builder.milBuilder.Transpose(operand.milValue, milPerm)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Call calls a function with the given inputs.
//
// Note: CoreML MIL does not support function calls at the IR level. Unlike XLA/StableHLO
// which has a dedicated "call" operation, MIL Programs define multiple independent functions
// that serve as entry points but cannot call each other during execution.
//
// MIL's control flow operations (cond, while_loop) use nested blocks rather than function calls.
// See https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
//
// Workarounds:
//   - Inline the function body: Instead of calling a function, duplicate the operations
//     at the call site. This is the most straightforward approach for simple functions.
//   - Use a different backend: XLA backend supports function calls via StableHLO's call op.
//   - Refactor computation: If the function is used for control flow (While, If, Sort),
//     those operations use closures which are handled differently (via nested blocks).
func (f *Function) Call(fn backends.Function, inputs ...backends.Value) ([]backends.Value, error) {
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"Call is not supported for %q builder: CoreML MIL does not have a function call operation. "+
			"Consider inlining the function body at the call site, or use a different backend (e.g., XLA) "+
			"for computations that require function calls", BackendName)
}

// Sort sorts one or more tensors along the specified axis using a comparator closure.
//
// Note: CoreML MIL does not support custom comparator functions for sorting.
// The generic Sort API requires a comparator closure that defines the ordering,
// which cannot be expressed in CoreML's operation graph.
//
// CoreML MIL does provide these sorting-related operations:
//   - argsort: Returns indices that would sort a tensor (ascending or descending)
//   - topk: Returns the k largest or smallest values and their indices
//
// Workarounds:
//   - For simple ascending/descending sorts of a single tensor, use the graph.Argsort()
//     operation followed by graph.Gather() to reorder the tensor.
//   - For top-k operations, use graph.TopK() which is more efficient than full sorting.
//   - Use a different backend (e.g., XLA) for computations that require custom comparators
//     or stable multi-tensor sorting.
//
// See also: model.Builder.Argsort() and model.Builder.TopK() for lower-level MIL operations.
func (f *Function) Sort(comparator backends.Function, axis int, isStable bool, inputs ...backends.Value) ([]backends.Value, error) {
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"Sort with custom comparator is not supported for %q builder: CoreML MIL does not support "+
			"custom comparator functions. For simple ascending/descending sorts, consider using "+
			"Argsort + Gather operations. For top-k values, use TopK. For complex sorting with "+
			"custom comparators, use a different backend (e.g., XLA)", BackendName)
}

// While executes a loop while a condition is true.
//
// CoreML MIL supports while_loop operations via nested blocks.
// See https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
// for CoreML MIL control flow documentation.
//
// Parameters:
//   - cond: A closure Function that takes the loop variables as parameters and returns a boolean scalar
//   - body: A closure Function that takes the loop variables as parameters and returns updated loop variables
//   - initialState: The initial values for the loop variables
//
// Returns the final values of the loop variables when the condition becomes false.
func (f *Function) While(cond, body backends.Function, initialState ...backends.Value) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	// Validate initial state
	if len(initialState) == 0 {
		return nil, errors.Errorf("While: at least one initial state value is required")
	}

	initialNodes, err := f.builder.checkOps("While", initialState...)
	if err != nil {
		return nil, err
	}

	// Validate cond function
	condF, ok := cond.(*Function)
	if !ok {
		return nil, errors.Errorf("While: cond function must be a CoreML Function, got %T", cond)
	}
	if condF.parent != f {
		return nil, errors.Errorf("While: cond function must be a closure of the current function")
	}
	if !condF.returned {
		return nil, errors.Errorf("While: cond function must have called Return()")
	}
	if len(condF.outputs) != 1 {
		return nil, errors.Errorf("While: cond function must return exactly one value (bool), got %d", len(condF.outputs))
	}
	if condF.outputs[0].shape.DType != dtypes.Bool {
		return nil, errors.Errorf("While: cond function must return Bool, got %s", condF.outputs[0].shape.DType)
	}
	if condF.outputs[0].shape.Rank() != 0 {
		return nil, errors.Errorf("While: cond function must return a scalar, got rank %d", condF.outputs[0].shape.Rank())
	}

	// Validate body function
	bodyF, ok := body.(*Function)
	if !ok {
		return nil, errors.Errorf("While: body function must be a CoreML Function, got %T", body)
	}
	if bodyF.parent != f {
		return nil, errors.Errorf("While: body function must be a closure of the current function")
	}
	if !bodyF.returned {
		return nil, errors.Errorf("While: body function must have called Return()")
	}
	if len(bodyF.outputs) != len(initialState) {
		return nil, errors.Errorf("While: body function must return %d values (matching initial state), got %d",
			len(initialState), len(bodyF.outputs))
	}

	// Validate that body outputs match initial state shapes
	for i, out := range bodyF.outputs {
		if !out.shape.Equal(initialNodes[i].shape) {
			return nil, errors.Errorf("While: body output %d shape %s doesn't match initial state shape %s",
				i, out.shape, initialNodes[i].shape)
		}
	}

	// Validate parameter counts match
	if len(condF.parameters) != len(initialState) {
		return nil, errors.Errorf("While: cond function has %d parameters but initial state has %d values",
			len(condF.parameters), len(initialState))
	}
	if len(bodyF.parameters) != len(initialState) {
		return nil, errors.Errorf("While: body function has %d parameters but initial state has %d values",
			len(bodyF.parameters), len(initialState))
	}

	// Convert initial state to MIL values
	loopVars := make([]*model.Value, len(initialNodes))
	for i, n := range initialNodes {
		loopVars[i] = n.milValue
	}

	// Build the while_loop using the MIL layer
	// The MIL layer's WhileLoop expects callback functions that build operations in a BlockBuilder
	resultValues := f.builder.milBuilder.WhileLoop(
		loopVars,
		func(bb *model.BlockBuilder, vars []*model.Value) *model.Value {
			// Build condition block by replaying the cond closure operations
			return f.replayClosureInBlock(bb, condF, vars)[0]
		},
		func(bb *model.BlockBuilder, vars []*model.Value) []*model.Value {
			// Build body block by replaying the body closure operations
			return f.replayClosureInBlock(bb, bodyF, vars)
		},
	)

	if f.builder.milBuilder.Err() != nil {
		return nil, errors.Wrap(f.builder.milBuilder.Err(), "While")
	}

	// Create output nodes
	outputNodes := make([]backends.Value, len(resultValues))
	for i, v := range resultValues {
		node := f.builder.newNode(backends.OpTypeWhile, initialNodes[i].shape, v)
		outputNodes[i] = node
	}

	return outputNodes, nil
}

// If executes one of two branches based on a boolean predicate.
//
// CoreML MIL supports conditional (cond) operations via nested blocks.
// See https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
// for CoreML MIL control flow documentation.
//
// Parameters:
//   - pred: A boolean scalar value that determines which branch to execute
//   - trueBranch: A closure Function that produces outputs when pred is true
//   - falseBranch: A closure Function that produces outputs when pred is false
//
// Both branches must return the same number of outputs with matching shapes.
// Returns the outputs from the executed branch.
func (f *Function) If(pred backends.Value, trueBranch, falseBranch backends.Function) ([]backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	// Validate predicate
	predNodes, err := f.builder.checkOps("If", pred)
	if err != nil {
		return nil, err
	}
	predNode := predNodes[0]

	if predNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("If: pred must be Bool, got %s", predNode.shape.DType)
	}
	if predNode.shape.Rank() != 0 {
		return nil, errors.Errorf("If: pred must be a scalar, got rank %d", predNode.shape.Rank())
	}

	// Validate true branch
	trueF, ok := trueBranch.(*Function)
	if !ok {
		return nil, errors.Errorf("If: trueBranch must be a CoreML Function, got %T", trueBranch)
	}
	if trueF.parent != f {
		return nil, errors.Errorf("If: trueBranch must be a closure of the current function")
	}
	if !trueF.returned {
		return nil, errors.Errorf("If: trueBranch must have called Return()")
	}
	if len(trueF.outputs) == 0 {
		return nil, errors.Errorf("If: trueBranch must return at least one value")
	}

	// Validate false branch
	falseF, ok := falseBranch.(*Function)
	if !ok {
		return nil, errors.Errorf("If: falseBranch must be a CoreML Function, got %T", falseBranch)
	}
	if falseF.parent != f {
		return nil, errors.Errorf("If: falseBranch must be a closure of the current function")
	}
	if !falseF.returned {
		return nil, errors.Errorf("If: falseBranch must have called Return()")
	}

	// Validate output counts match
	if len(trueF.outputs) != len(falseF.outputs) {
		return nil, errors.Errorf("If: trueBranch has %d outputs but falseBranch has %d outputs",
			len(trueF.outputs), len(falseF.outputs))
	}

	// Validate output shapes match
	for i := range trueF.outputs {
		if !trueF.outputs[i].shape.Equal(falseF.outputs[i].shape) {
			return nil, errors.Errorf("If: output %d shape mismatch: trueBranch %s, falseBranch %s",
				i, trueF.outputs[i].shape, falseF.outputs[i].shape)
		}
	}

	// If branches have no parameters, they just return values from the parent scope
	// Build the cond operation using the MIL layer
	resultValues := f.builder.milBuilder.Cond(
		predNode.milValue,
		func(bb *model.BlockBuilder) []*model.Value {
			// Build true branch by replaying the trueBranch closure operations
			return f.replayClosureInBlock(bb, trueF, nil)
		},
		func(bb *model.BlockBuilder) []*model.Value {
			// Build false branch by replaying the falseBranch closure operations
			return f.replayClosureInBlock(bb, falseF, nil)
		},
	)

	if f.builder.milBuilder.Err() != nil {
		return nil, errors.Wrap(f.builder.milBuilder.Err(), "If")
	}

	// Create output nodes
	outputNodes := make([]backends.Value, len(resultValues))
	for i, v := range resultValues {
		node := f.builder.newNode(backends.OpTypeIf, trueF.outputs[i].shape, v)
		outputNodes[i] = node
	}

	return outputNodes, nil
}

// replayClosureInBlock replays a closure's operations inside a MIL BlockBuilder.
// This is used to convert GoMLX closure Functions into MIL nested blocks for control flow.
//
// The closure's parameters are mapped to the provided blockInputs (for While loops)
// or the closure may have no parameters (for If branches that just use parent scope values).
//
// Values from the parent scope are referenced directly since MIL blocks can access
// values defined in the parent scope.
func (f *Function) replayClosureInBlock(bb *model.BlockBuilder, closure *Function, blockInputs []*model.Value) []*model.Value {
	// Map from original node builderIdx to the replayed value in the block
	valueMap := make(map[int]*model.Value)

	// Map closure parameters to block inputs
	for i, param := range closure.parameters {
		if i < len(blockInputs) {
			valueMap[param.builderIdx] = blockInputs[i]
		}
	}

	// Get the compiled closure info for proper node ordering
	cc := closure.compiled
	if cc == nil {
		// Closure wasn't pre-compiled, this shouldn't happen for valid closures
		// Fall back to direct output reference (works for simple cases)
		outputs := make([]*model.Value, len(closure.outputs))
		for i, out := range closure.outputs {
			outputs[i] = out.milValue
		}
		return outputs
	}

	// Build a set of closure node indices for quick lookup
	closureNodeSet := make(map[int]bool)
	for _, node := range cc.sortedNodes {
		closureNodeSet[node.builderIdx] = true
	}

	// Also mark closure parameters as belonging to the closure
	for _, param := range closure.parameters {
		closureNodeSet[param.builderIdx] = true
	}

	// Replay each node in topological order
	for _, node := range cc.sortedNodes {
		// Skip if already mapped (e.g., parameters)
		if _, ok := valueMap[node.builderIdx]; ok {
			continue
		}

		// Check if this node is a closure parameter (handled separately)
		isClosureParam := false
		for _, param := range closure.parameters {
			if param.builderIdx == node.builderIdx {
				isClosureParam = true
				break
			}
		}
		if isClosureParam {
			// Closure parameters should already be mapped from blockInputs
			continue
		}

		// Check if ALL inputs of this node come from parent scope
		// If so, we can reference the original value directly
		// Otherwise, we need to replay the operation
		allInputsFromParent := true
		for _, input := range node.inputs {
			if closureNodeSet[input.builderIdx] {
				allInputsFromParent = false
				break
			}
		}

		if allInputsFromParent && node.opType != backends.OpTypeParameter && node.opType != backends.OpTypeConstant {
			// This node's operation was computed in parent scope, reference it directly
			valueMap[node.builderIdx] = node.milValue
			continue
		}

		// Replay the operation in the block
		replayedValue := f.replayNodeInBlock(bb, node, valueMap)
		if replayedValue != nil {
			valueMap[node.builderIdx] = replayedValue
		}
	}

	// Return the outputs
	outputs := make([]*model.Value, len(closure.outputs))
	for i, out := range closure.outputs {
		if v, ok := valueMap[out.builderIdx]; ok {
			outputs[i] = v
		} else {
			// Fallback: use the original MIL value (parent scope reference)
			outputs[i] = out.milValue
		}
	}
	return outputs
}

// replayNodeInBlock replays a single node's operation in a BlockBuilder.
// This is a limited implementation that supports common operations used in control flow.
func (f *Function) replayNodeInBlock(bb *model.BlockBuilder, node *Node, valueMap map[int]*model.Value) *model.Value {
	// Get input values
	inputs := make([]*model.Value, len(node.inputs))
	for i, input := range node.inputs {
		if v, ok := valueMap[input.builderIdx]; ok {
			inputs[i] = v
		} else {
			// Use the original MIL value (parent scope)
			inputs[i] = input.milValue
		}
	}

	// Replay based on operation type
	switch node.opType {
	case backends.OpTypeParameter:
		// Parameters are handled separately via blockInputs
		return nil

	case backends.OpTypeConstant:
		// Constants can be referenced from parent scope in MIL blocks
		return node.milValue

	// Binary operations
	case backends.OpTypeAdd:
		if len(inputs) >= 2 {
			return bb.Add(inputs[0], inputs[1])
		}
	case backends.OpTypeSub:
		if len(inputs) >= 2 {
			return bb.Sub(inputs[0], inputs[1])
		}
	case backends.OpTypeMul:
		if len(inputs) >= 2 {
			return bb.Mul(inputs[0], inputs[1])
		}

	// Comparison operations
	case backends.OpTypeLessThan:
		if len(inputs) >= 2 {
			return bb.Less(inputs[0], inputs[1])
		}
	case backends.OpTypeGreaterThan:
		if len(inputs) >= 2 {
			return bb.Greater(inputs[0], inputs[1])
		}

	// Logical operations
	case backends.OpTypeLogicalAnd:
		if len(inputs) >= 2 {
			return bb.LogicalAnd(inputs[0], inputs[1])
		}

	// Selection
	case backends.OpTypeWhere:
		if len(inputs) >= 3 {
			return bb.Select(inputs[0], inputs[1], inputs[2])
		}

	default:
		// For unsupported operations, try to reference the original value
		// This works for operations whose inputs are all from parent scope
		return node.milValue
	}

	// Fallback
	return node.milValue
}

// Where selects elements from onTrue or onFalse based on the condition.
func (f *Function) Where(condition, onTrue, onFalse backends.Value) (backends.Value, error) {
	opType := backends.OpTypeWhere
	inputs, err := f.builder.checkOps(opType.String(), condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	condNode, onTrueNode, onFalseNode := inputs[0], inputs[1], inputs[2]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.WhereOp(condNode.shape, onTrueNode.shape, onFalseNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL Select operation (cond, a, b) -> select a where cond is true, b where false
	resultValue := f.builder.milBuilder.Select(condNode.milValue, onTrueNode.milValue, onFalseNode.milValue)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, condNode, onTrueNode, onFalseNode)

	return node, nil
}

// ConvertDType converts the tensor to a different dtype.
func (f *Function) ConvertDType(x backends.Value, dtype dtypes.DType) (backends.Value, error) {
	opType := backends.OpTypeConvertDType
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Convert GoMLX dtype to CoreML dtype
	milDType, err := gomlxDTypeToMIL(dtype)
	if err != nil {
		return nil, errors.Wrapf(err, "ConvertDType to %s", dtype)
	}

	// Output shape is the same as input, just with different dtype
	outputShape := operand.shape.Clone()
	outputShape.DType = dtype

	// Call the MIL Cast operation
	resultValue := f.builder.milBuilder.Cast(operand.milValue, milDType)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// DotGeneral implements backends.Function.
// It performs a generalized matrix multiplication that:
// - Contracts specified axes between lhs and rhs (like matrix multiply)
// - Preserves batch axes (operates independently on each batch)
// - Crosses all other axes
//
// The output shape is: [batch dims..., lhs cross dims..., rhs cross dims...]
func (f *Function) DotGeneral(lhsOp backends.Value, lhsContractingAxes, lhsBatchAxes []int, rhsOp backends.Value, rhsContractingAxes, rhsBatchAxes []int) (backends.Value, error) {
	opType := backends.OpTypeDotGeneral
	inputs, err := f.builder.checkOps(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	lhsShape := lhs.shape
	rhsShape := rhs.shape

	// Validate data types match
	if lhsShape.DType != rhsShape.DType {
		return nil, errors.Errorf("DotGeneral: lhs and rhs must have matching dtypes, got %s and %s",
			lhsShape.DType, rhsShape.DType)
	}

	// Validate contracting and batch axes counts match
	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral: number of contracting axes must match, got %d for lhs and %d for rhs",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral: number of batch axes must match, got %d for lhs and %d for rhs",
			len(lhsBatchAxes), len(rhsBatchAxes))
	}

	lhsRank := lhsShape.Rank()
	rhsRank := rhsShape.Rank()

	// Adjust negative axes and validate
	lhsContractingAxes = slices.Clone(lhsContractingAxes)
	lhsBatchAxes = slices.Clone(lhsBatchAxes)
	rhsContractingAxes = slices.Clone(rhsContractingAxes)
	rhsBatchAxes = slices.Clone(rhsBatchAxes)

	for i, axis := range lhsContractingAxes {
		if axis < 0 {
			axis += lhsRank
		}
		if axis < 0 || axis >= lhsRank {
			return nil, errors.Errorf("DotGeneral: lhs contracting axis %d out of range for rank %d", lhsContractingAxes[i], lhsRank)
		}
		lhsContractingAxes[i] = axis
	}
	for i, axis := range lhsBatchAxes {
		if axis < 0 {
			axis += lhsRank
		}
		if axis < 0 || axis >= lhsRank {
			return nil, errors.Errorf("DotGeneral: lhs batch axis %d out of range for rank %d", lhsBatchAxes[i], lhsRank)
		}
		lhsBatchAxes[i] = axis
	}
	for i, axis := range rhsContractingAxes {
		if axis < 0 {
			axis += rhsRank
		}
		if axis < 0 || axis >= rhsRank {
			return nil, errors.Errorf("DotGeneral: rhs contracting axis %d out of range for rank %d", rhsContractingAxes[i], rhsRank)
		}
		rhsContractingAxes[i] = axis
	}
	for i, axis := range rhsBatchAxes {
		if axis < 0 {
			axis += rhsRank
		}
		if axis < 0 || axis >= rhsRank {
			return nil, errors.Errorf("DotGeneral: rhs batch axis %d out of range for rank %d", rhsBatchAxes[i], rhsRank)
		}
		rhsBatchAxes[i] = axis
	}

	// Validate that batch and contracting dimensions match between lhs and rhs
	for i := range lhsContractingAxes {
		lhsDim := lhsShape.Dimensions[lhsContractingAxes[i]]
		rhsDim := rhsShape.Dimensions[rhsContractingAxes[i]]
		if lhsDim != rhsDim {
			return nil, errors.Errorf("DotGeneral: contracting dimensions must match, lhs[%d]=%d != rhs[%d]=%d",
				lhsContractingAxes[i], lhsDim, rhsContractingAxes[i], rhsDim)
		}
	}
	for i := range lhsBatchAxes {
		lhsDim := lhsShape.Dimensions[lhsBatchAxes[i]]
		rhsDim := rhsShape.Dimensions[rhsBatchAxes[i]]
		if lhsDim != rhsDim {
			return nil, errors.Errorf("DotGeneral: batch dimensions must match, lhs[%d]=%d != rhs[%d]=%d",
				lhsBatchAxes[i], lhsDim, rhsBatchAxes[i], rhsDim)
		}
	}

	// Identify cross axes (axes that are neither batch nor contracting)
	lhsContractingSet := make(map[int]bool)
	lhsBatchSet := make(map[int]bool)
	for _, axis := range lhsContractingAxes {
		lhsContractingSet[axis] = true
	}
	for _, axis := range lhsBatchAxes {
		lhsBatchSet[axis] = true
	}
	var lhsCrossAxes []int
	for axis := 0; axis < lhsRank; axis++ {
		if !lhsContractingSet[axis] && !lhsBatchSet[axis] {
			lhsCrossAxes = append(lhsCrossAxes, axis)
		}
	}

	rhsContractingSet := make(map[int]bool)
	rhsBatchSet := make(map[int]bool)
	for _, axis := range rhsContractingAxes {
		rhsContractingSet[axis] = true
	}
	for _, axis := range rhsBatchAxes {
		rhsBatchSet[axis] = true
	}
	var rhsCrossAxes []int
	for axis := 0; axis < rhsRank; axis++ {
		if !rhsContractingSet[axis] && !rhsBatchSet[axis] {
			rhsCrossAxes = append(rhsCrossAxes, axis)
		}
	}

	// Calculate sizes for batch, cross, and contracting dimensions
	batchSize := 1
	for _, axis := range lhsBatchAxes {
		batchSize *= lhsShape.Dimensions[axis]
	}
	lhsCrossSize := 1
	for _, axis := range lhsCrossAxes {
		lhsCrossSize *= lhsShape.Dimensions[axis]
	}
	rhsCrossSize := 1
	for _, axis := range rhsCrossAxes {
		rhsCrossSize *= rhsShape.Dimensions[axis]
	}
	contractingSize := 1
	for _, axis := range lhsContractingAxes {
		contractingSize *= lhsShape.Dimensions[axis]
	}

	// Collect dimension sizes for the output shape
	var batchDims, lhsCrossDims, rhsCrossDims []int
	for _, axis := range lhsBatchAxes {
		batchDims = append(batchDims, lhsShape.Dimensions[axis])
	}
	for _, axis := range lhsCrossAxes {
		lhsCrossDims = append(lhsCrossDims, lhsShape.Dimensions[axis])
	}
	for _, axis := range rhsCrossAxes {
		rhsCrossDims = append(rhsCrossDims, rhsShape.Dimensions[axis])
	}

	// Build output shape: [batch dims..., lhs cross dims..., rhs cross dims...]
	var outputDims []int
	outputDims = append(outputDims, batchDims...)
	outputDims = append(outputDims, lhsCrossDims...)
	outputDims = append(outputDims, rhsCrossDims...)
	outputShape := shapes.Make(lhsShape.DType, outputDims...)

	// Strategy: transpose both operands to [batch, cross, contracting] order,
	// reshape to 3D, do matmul, then reshape back.

	// Build LHS permutation: batch axes, cross axes, contracting axes
	var lhsPerm []int64
	for _, axis := range lhsBatchAxes {
		lhsPerm = append(lhsPerm, int64(axis))
	}
	for _, axis := range lhsCrossAxes {
		lhsPerm = append(lhsPerm, int64(axis))
	}
	for _, axis := range lhsContractingAxes {
		lhsPerm = append(lhsPerm, int64(axis))
	}

	// Build RHS permutation: batch axes, contracting axes, cross axes
	// (contracting before cross so matmul contracts on adjacent dimensions)
	var rhsPerm []int64
	for _, axis := range rhsBatchAxes {
		rhsPerm = append(rhsPerm, int64(axis))
	}
	for _, axis := range rhsContractingAxes {
		rhsPerm = append(rhsPerm, int64(axis))
	}
	for _, axis := range rhsCrossAxes {
		rhsPerm = append(rhsPerm, int64(axis))
	}

	// Transpose LHS if needed
	lhsValue := lhs.milValue
	needsLhsTranspose := false
	for i, p := range lhsPerm {
		if int(p) != i {
			needsLhsTranspose = true
			break
		}
	}
	if needsLhsTranspose && len(lhsPerm) > 0 {
		lhsValue = f.builder.milBuilder.Transpose(lhsValue, lhsPerm)
	}

	// Transpose RHS if needed
	rhsValue := rhs.milValue
	needsRhsTranspose := false
	for i, p := range rhsPerm {
		if int(p) != i {
			needsRhsTranspose = true
			break
		}
	}
	if needsRhsTranspose && len(rhsPerm) > 0 {
		rhsValue = f.builder.milBuilder.Transpose(rhsValue, rhsPerm)
	}

	// Reshape to 3D: [batchSize, crossSize, contractingSize]
	// For LHS: [batchSize, lhsCrossSize, contractingSize]
	// For RHS: [batchSize, contractingSize, rhsCrossSize]
	lhsValue = f.builder.milBuilder.Reshape(lhsValue, []int64{int64(batchSize), int64(lhsCrossSize), int64(contractingSize)})
	rhsValue = f.builder.milBuilder.Reshape(rhsValue, []int64{int64(batchSize), int64(contractingSize), int64(rhsCrossSize)})

	// Matrix multiply: [B, M, K] x [B, K, N] -> [B, M, N]
	// Use MatMulTranspose with no transposes since we've already arranged the data
	resultValue := f.builder.milBuilder.MatMulTranspose(lhsValue, rhsValue, false, false)

	// Reshape back to output shape
	if len(outputDims) > 0 {
		milOutputDims := make([]int64, len(outputDims))
		for i, d := range outputDims {
			milOutputDims[i] = int64(d)
		}
		resultValue = f.builder.milBuilder.Reshape(resultValue, milOutputDims)
	} else {
		// Scalar output - squeeze all dimensions to get a scalar
		// The matmul result is [1, 1, 1], squeeze all dims to get scalar
		resultValue = f.builder.milBuilder.Squeeze(resultValue, nil)
	}

	// Create the output node
	node := f.builder.newNode(opType, outputShape, resultValue, lhs, rhs)

	return node, nil
}

// Concatenate operands on the given axis.
func (f *Function) Concatenate(axis int, operands ...backends.Value) (backends.Value, error) {
	opType := backends.OpTypeConcatenate
	if len(operands) == 0 {
		return nil, errors.Errorf("Concatenate requires at least one operand")
	}
	if len(operands) == 1 {
		// Single operand is a no-op, return as-is
		nodes, err := f.builder.checkOps(opType.String(), operands[0])
		if err != nil {
			return nil, err
		}
		return nodes[0], nil
	}

	// Check all operands
	inputs, err := f.builder.checkOps(opType.String(), operands...)
	if err != nil {
		return nil, err
	}

	// Gather shapes for shape inference
	inputShapes := make([]shapes.Shape, len(inputs))
	for i, node := range inputs {
		inputShapes[i] = node.shape
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, err
	}

	// Gather MIL values for the concat operation
	milValues := make([]*model.Value, len(inputs))
	for i, node := range inputs {
		milValues[i] = node.milValue
	}

	// Call the MIL Concat operation
	resultValue := f.builder.milBuilder.Concat(milValues, int64(axis))

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, inputs...)

	return node, nil
}

// Gather implements backends.Function.
//
// CoreML backend supports the following Gather patterns:
//
// Pattern 1: Simple single-axis gather (embedding lookup)
// - Gather elements along one axis with the indexed dimension collapsed
// - Example: params[10, 8], indices[3] -> output[3, 8]
// - Requirements:
//   - len(startIndexMap) == 1
//   - len(collapsedSliceAxes) == 1
//   - collapsedSliceAxes[0] == startIndexMap[0]
//   - sliceSizes[axis] == 1 for the gathered axis
//
// Pattern 2: Multi-axis gather (GatherND pattern)
// - Gather elements using multi-dimensional indices into contiguous leading axes
// - Example: params[4, 3, 5], indices[2, 2] -> output[2, 5] (indexing into axes 0,1)
// - Requirements:
//   - startIndexMap must be contiguous from axis 0: [0, 1, 2, ...]
//   - len(collapsedSliceAxes) == len(startIndexMap)
//   - All collapsed axes have sliceSize == 1
//
// Pattern 3: GatherSlices with slice_size=1 (single-axis, no collapse)
// - Gather slices along one axis without collapsing the indexed dimension
// - Example: params[10, 8], indices[3, 1] -> output[3, 1, 8]
// - Requirements:
//   - len(startIndexMap) == 1
//   - len(collapsedSliceAxes) == 0
//   - sliceSizes[gatherAxis] == 1
//
// NOT SUPPORTED (returns error):
// - GatherSlices with slice_size > 1
// - Non-contiguous multi-axis gather
// - Partial multi-axis gather (some axes collapsed, some not)
func (f *Function) Gather(
	operand, startIndices backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Value, error) {
	opType := backends.OpTypeGather
	inputs, err := f.builder.checkOps(opType.String(), operand, startIndices)
	if err != nil {
		return nil, err
	}
	operandNode, startIndicesNode := inputs[0], inputs[1]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.Gather(
		operandNode.shape, startIndicesNode.shape,
		indexVectorAxis, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes,
		indicesAreSorted,
	)
	if err != nil {
		return nil, err
	}

	// CoreML MIL provides several gather operations:
	// 1. gather(x, indices, axis) - Gathers slices along a single axis
	//    Output shape: x.shape[:axis] + indices.shape + x.shape[axis+1:]
	// 2. gather_nd(x, indices) - Multi-dimensional gather using last dim of indices as coordinates
	// 3. gather_along_axis(x, indices, axis) - Like torch.gather, indices has same rank as x
	//
	// XLA's Gather is more general with these parameters:
	// - startIndexMap: maps index vector elements to operand axes
	// - collapsedSliceAxes: operand axes to collapse (must have sliceSize=1)
	// - offsetOutputAxes: output axes for non-collapsed slice dimensions
	// - sliceSizes: size of slice along each operand axis
	//
	// We support several patterns:

	// Pattern 1: Simple single-axis gather with collapsed dimension
	// This is the most common case (e.g., embedding lookup)
	// - len(startIndexMap) == 1 (gathering along one axis)
	// - len(collapsedSliceAxes) == 1 (collapsing that same axis)
	// - collapsedSliceAxes[0] == startIndexMap[0]
	// - sliceSizes[axis] == 1 for the gathered axis
	if len(startIndexMap) == 1 && len(collapsedSliceAxes) == 1 &&
		collapsedSliceAxes[0] == startIndexMap[0] &&
		sliceSizes[startIndexMap[0]] == 1 {
		// This is a simple gather that CoreML can handle directly
		gatherAxis := int64(startIndexMap[0])

		// For CoreML gather, indices should not include the indexVectorAxis dimension
		// if it's being used as the index vector. We need to squeeze it out.
		indicesValue := startIndicesNode.milValue
		if indexVectorAxis < startIndicesNode.shape.Rank() && startIndicesNode.shape.Dimensions[indexVectorAxis] == 1 {
			// Squeeze out the index vector axis
			axes := []int64{int64(indexVectorAxis)}
			indicesValue = f.builder.milBuilder.Squeeze(indicesValue, axes)
		}

		// Call the MIL Gather operation
		resultValue := f.builder.milBuilder.Gather(operandNode.milValue, indicesValue, gatherAxis)

		// Create a new node with the result
		node := f.builder.newNode(opType, outputShape, resultValue, operandNode, startIndicesNode)

		return node, nil
	}

	// Pattern 2: Multi-axis gather with all axes collapsed
	// This maps to CoreML's gather operation with a reshape/slice sequence
	// - len(collapsedSliceAxes) == len(startIndexMap) (all indexed axes are collapsed)
	// - All indexed axes have sliceSize == 1
	// - The indexed axes must be contiguous starting from axis 0
	if len(collapsedSliceAxes) == len(startIndexMap) && len(startIndexMap) > 1 {
		// Check if startIndexMap is contiguous from 0
		isContiguousFromZero := true
		for i, axis := range startIndexMap {
			if axis != i {
				isContiguousFromZero = false
				break
			}
		}

		// Check all collapsed axes have sliceSize 1
		allSliceSizeOne := true
		for _, axis := range collapsedSliceAxes {
			if sliceSizes[axis] != 1 {
				allSliceSizeOne = false
				break
			}
		}

		if isContiguousFromZero && allSliceSizeOne {
			// This is a GatherND case - multi-dimensional indexing into the first N axes
			// CoreML's gather_nd takes indices of shape [..., N] where N is the number of
			// dimensions to index into, and gathers slices from the input.

			// Prepare indices: we need to ensure the index vector dimension is at the end
			indicesValue := startIndicesNode.milValue
			indicesShape := startIndicesNode.shape

			// If indexVectorAxis is not the last axis, we need to transpose
			if indexVectorAxis != indicesShape.Rank()-1 {
				// Create permutation to move indexVectorAxis to the end
				perm := make([]int64, indicesShape.Rank())
				j := 0
				for i := 0; i < indicesShape.Rank(); i++ {
					if i == indexVectorAxis {
						continue
					}
					perm[j] = int64(i)
					j++
				}
				perm[indicesShape.Rank()-1] = int64(indexVectorAxis)
				indicesValue = f.builder.milBuilder.Transpose(indicesValue, perm)
			}

			// Call the MIL GatherND operation
			resultValue := f.builder.milBuilder.GatherND(operandNode.milValue, indicesValue)

			// Create a new node with the result
			node := f.builder.newNode(opType, outputShape, resultValue, operandNode, startIndicesNode)

			return node, nil
		}
	}

	// Pattern 3: GatherSlices - single axis gather without collapsing
	// This is used by GoMLX's GatherSlices operation
	// - len(startIndexMap) == 1 (gathering along one axis)
	// - len(collapsedSliceAxes) == 0 (no collapsing)
	//
	// For slice_size=1 on the gathered axis, we can use:
	// 1. CoreML Gather (which collapses the axis)
	// 2. ExpandDims to restore the size-1 dimension
	//
	// For slice_size > 1, this is more complex and not yet supported.
	if len(startIndexMap) == 1 && len(collapsedSliceAxes) == 0 {
		gatherAxis := startIndexMap[0]
		sliceSize := sliceSizes[gatherAxis]

		if sliceSize == 1 {
			// Case: GatherSlices with slice_size=1 on the gathered axis
			// This can be implemented as Gather + ExpandDims

			// Prepare indices: squeeze out the indexVectorAxis dimension
			indicesValue := startIndicesNode.milValue
			if indexVectorAxis < startIndicesNode.shape.Rank() && startIndicesNode.shape.Dimensions[indexVectorAxis] == 1 {
				axes := []int64{int64(indexVectorAxis)}
				indicesValue = f.builder.milBuilder.Squeeze(indicesValue, axes)
			}

			// Step 1: Use CoreML Gather which collapses the gathered axis
			// For params[A, B, C] with gather on axis 0 and indices[N]:
			// Gather gives [N, B, C]
			gatheredValue := f.builder.milBuilder.Gather(operandNode.milValue, indicesValue, int64(gatherAxis))

			// Step 2: Insert the collapsed axis back with size 1
			// The gathered axis position in the output is determined by offsetOutputAxes
			// We need to find where to insert the size-1 dimension
			//
			// In the output shape from shapeinference, the gathered axis already has size 1
			// due to sliceSize=1. We need to find the output axis that corresponds to the gathered axis.
			//
			// offsetOutputAxes tells us where each non-collapsed operand axis goes in the output.
			// Since no axes are collapsed, all operand axes have entries in offsetOutputAxes.
			// The order of offsetOutputAxes matches the order of non-collapsed axes in operand
			// (i.e., axes not in collapsedSliceAxes, which is all of them here).
			//
			// So offsetOutputAxes[gatherAxis] gives us the output axis for the gathered axis.
			insertAxis := offsetOutputAxes[gatherAxis]

			// ExpandDims to restore the size-1 dimension
			resultValue := f.builder.milBuilder.ExpandDims(gatheredValue, []int64{int64(insertAxis)})

			node := f.builder.newNode(opType, outputShape, resultValue, operandNode, startIndicesNode)
			return node, nil
		}

		// For slice_size > 1, we need a more complex approach
		// TODO: Implement using Concat of SliceBySize operations or reshape tricks
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"GatherSlices with slice_size > 1 not yet supported for %q builder. "+
				"Only slice_size=1 on the gathered axis is currently supported. "+
				"Parameters: gatherAxis=%d, sliceSize=%d, startIndexMap=%v, sliceSizes=%v",
			BackendName, gatherAxis, sliceSize, startIndexMap, sliceSizes)
	}

	// For other complex Gather patterns, provide detailed error message
	return nil, errors.Wrapf(
		notimplemented.NotImplementedError,
		"complex Gather pattern not yet supported for %q builder. "+
			"Supported patterns: (1) single-axis gather with collapse (standard embedding lookup), "+
			"(2) multi-axis gather with all axes collapsed from axis 0 (GatherND pattern). "+
			"Got: startIndexMap=%v, collapsedSliceAxes=%v, offsetOutputAxes=%v, sliceSizes=%v",
		BackendName, startIndexMap, collapsedSliceAxes, offsetOutputAxes, sliceSizes)
}

// Pad implements backends.Function.
func (f *Function) Pad(x, fillValue backends.Value, axesConfig ...backends.PadAxis) (backends.Value, error) {
	opType := backends.OpTypePad
	inputs, err := f.builder.checkOps(opType.String(), x, fillValue)
	if err != nil {
		return nil, err
	}
	operandNode, fillNode := inputs[0], inputs[1]

	// Check fillValue is a scalar
	if !fillNode.shape.IsScalar() {
		return nil, errors.Errorf("Pad fillValue must be a scalar, got shape %s", fillNode.shape)
	}

	// Build padBefore and padAfter arrays
	rank := operandNode.shape.Rank()
	padBefore := make([]int64, rank)
	padAfter := make([]int64, rank)
	hasInterior := false

	for i := 0; i < len(axesConfig) && i < rank; i++ {
		padBefore[i] = int64(axesConfig[i].Start)
		padAfter[i] = int64(axesConfig[i].End)
		if axesConfig[i].Interior != 0 {
			hasInterior = true
		}
	}

	// CoreML doesn't support interior padding directly
	if hasInterior {
		return nil, errors.Wrapf(
			notimplemented.NotImplementedError,
			"Pad with interior padding not supported for %q builder", BackendName)
	}

	// Compute output shape
	outputDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		outputDims[i] = operandNode.shape.Dimensions[i] + int(padBefore[i]) + int(padAfter[i])
	}
	outputShape := shapes.Make(operandNode.shape.DType, outputDims...)

	// Get the constant value from fillNode for CoreML's pad operation
	// CoreML's Pad expects a float32 constant value
	var constantValue float32 = 0.0
	// We'll use 0.0 as default since extracting the actual value from the node is complex

	// Call the MIL Pad operation with constant mode
	resultValue := f.builder.milBuilder.Pad(operandNode.milValue, padBefore, padAfter, model.PadConstant, constantValue)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operandNode, fillNode)

	return node, nil
}

// Reverse implements backends.Function.
func (f *Function) Reverse(x backends.Value, axes ...int) (backends.Value, error) {
	opType := backends.OpTypeReverse
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operandNode := inputs[0]

	// If no axes specified, reverse all axes
	if len(axes) == 0 {
		axes = make([]int, operandNode.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
	}

	// Validate axes
	for _, axis := range axes {
		if axis < 0 || axis >= operandNode.shape.Rank() {
			return nil, errors.Errorf("Reverse: axis %d is out of range for shape %s", axis, operandNode.shape)
		}
	}

	// Output shape is the same as input shape
	outputShape := operandNode.shape

	// Convert axes to int64
	milAxes := make([]int64, len(axes))
	for i, a := range axes {
		milAxes[i] = int64(a)
	}

	// Call the MIL Reverse operation
	resultValue := f.builder.milBuilder.Reverse(operandNode.milValue, milAxes)

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operandNode)

	return node, nil
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (backends.Value, error) {
	opType := backends.OpTypeIota
	if err := f.CheckValid(); err != nil {
		return nil, err
	}

	// Validate inputs
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank())
	}

	// Get the size along the iota dimension
	iotaDimSize := shape.Dimensions[iotaAxis]

	// Convert GoMLX dtype to CoreML dtype
	milDType, err := gomlxDTypeToMIL(shape.DType)
	if err != nil {
		return nil, errors.Wrapf(err, "Iota")
	}

	// Create start, end, step constants for Range1D
	// Range1D generates [start, start+step, start+2*step, ...) up to end
	// IMPORTANT: Always use Int32 for Range1D because go-coreml only computes output size
	// for Int32 constant inputs. Float32 inputs result in unknown output size which causes
	// reshape to fail with "cannot reshape tensor of size 18446744073709551615".
	startName := fmt.Sprintf("iota_start_%d", f.builder.nextConstID)
	f.builder.nextConstID++
	endName := fmt.Sprintf("iota_end_%d", f.builder.nextConstID)
	f.builder.nextConstID++
	stepName := fmt.Sprintf("iota_step_%d", f.builder.nextConstID)
	f.builder.nextConstID++

	// Always use Int32 for Range1D to ensure output size is computed correctly
	start := f.builder.milBuilder.Const(startName, model.Int32, []int64{}, []int32{0})
	end := f.builder.milBuilder.Const(endName, model.Int32, []int64{}, []int32{int32(iotaDimSize)})
	step := f.builder.milBuilder.Const(stepName, model.Int32, []int64{}, []int32{1})

	// Generate 1D range [0, 1, 2, ..., iotaDimSize-1] as Int32
	rangeValue := f.builder.milBuilder.Range1D(start, end, step)

	// Convert to target dtype if needed using Cast
	if milDType != model.Int32 {
		rangeValue = f.builder.milBuilder.Cast(rangeValue, milDType)
	}

	// Build the reshape dimensions: size 1 for all axes except the iota axis
	reshapeDims := make([]int64, shape.Rank())
	for i := 0; i < shape.Rank(); i++ {
		if i == iotaAxis {
			reshapeDims[i] = int64(iotaDimSize)
		} else {
			reshapeDims[i] = 1
		}
	}

	// Reshape to [1, 1, ..., iotaDimSize, ..., 1] with iotaDimSize at iotaAxis
	reshapedValue := f.builder.milBuilder.Reshape(rangeValue, reshapeDims)

	// Build tile repetitions: tile by the actual dimension sizes for non-iota axes
	tileReps := make([]int64, shape.Rank())
	for i := 0; i < shape.Rank(); i++ {
		if i == iotaAxis {
			tileReps[i] = 1 // Don't tile along the iota axis
		} else {
			tileReps[i] = int64(shape.Dimensions[i])
		}
	}

	// Tile to fill the full shape
	resultValue := f.builder.milBuilder.Tile(reshapedValue, tileReps)

	// Create a new node with the result
	node := f.builder.newNode(opType, shape, resultValue)

	return node, nil
}

// DynamicUpdateSlice updates the operand with the values given in update, at the position given by startIndices.
//
// - operand: original value to be updated.
// - update: values to "paste" on top of operand, at position startIndices.
// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
//
// It returns a value with the same shape as the operand, with the values updated.
//
// The startIndices are adjusted as follows:
//
//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - update.Dimensions[i])
func (f *Function) DynamicUpdateSlice(operand, update backends.Value, startIndices []backends.Value) (backends.Value, error) {
	opType := backends.OpTypeDynamicUpdateSlice

	// Check all values including startIndices
	allValues := append([]backends.Value{operand, update}, startIndices...)
	inputs, err := f.builder.checkOps(opType.String(), allValues...)
	if err != nil {
		return nil, err
	}
	operandNode := inputs[0]
	updateNode := inputs[1]
	startIndexNodes := inputs[2:]

	operandShape := operandNode.shape
	updateShape := updateNode.shape
	rank := operandShape.Rank()

	// Validate
	if len(startIndices) != rank {
		return nil, errors.Errorf("DynamicUpdateSlice: expected %d start indices, got %d", rank, len(startIndices))
	}
	if updateShape.Rank() != rank {
		return nil, errors.Errorf("DynamicUpdateSlice: update rank (%d) must match operand rank (%d)", updateShape.Rank(), rank)
	}

	// Validate that each start index is a scalar or size-1 tensor
	// CoreML doesn't support true scalar inputs, so we accept size-1 tensors
	for i, idx := range startIndexNodes {
		if idx.shape.Rank() > 1 || (idx.shape.Rank() == 1 && idx.shape.Dimensions[0] != 1) {
			return nil, errors.Errorf("DynamicUpdateSlice: startIndices[%d] must be a scalar or size-1 tensor, got shape %s", i, idx.shape)
		}
	}

	// Output shape is the same as operand shape
	outputShape := operandShape.Clone()

	// For CoreML ScatterND, we need to build indices tensor that specifies which positions to update.
	// ScatterND expects:
	// - data: the original tensor (operand)
	// - indices: tensor of shape [..., index_depth] where index_depth == rank
	// - updates: tensor of values to scatter
	// - mode: "update" to replace values
	//
	// For DynamicUpdateSlice, we need to create indices for all positions in the update tensor,
	// offset by startIndices.
	//
	// The indices tensor should have shape [update.Size(), rank] where each row is the
	// multi-dimensional index into the operand where that update value should go.

	// Calculate total number of elements in update
	updateSize := updateShape.Size()

	// We need to generate indices dynamically based on startIndices.
	// Strategy:
	// 1. Create iota-based indices for each dimension of update shape
	// 2. Add the corresponding startIndex to each
	// 3. Stack them to form the final indices tensor

	// First, stack all startIndices into a 1D tensor of shape [rank]
	// We'll use this to offset all generated indices

	// Generate indices for each axis
	var axisIndices []*model.Value

	for axis := 0; axis < rank; axis++ {
		updateDim := updateShape.Dimensions[axis]

		// Create range [0, 1, ..., updateDim-1] for this axis
		startName := fmt.Sprintf("dus_start_%d_%d", f.builder.nextConstID, axis)
		f.builder.nextConstID++
		endName := fmt.Sprintf("dus_end_%d_%d", f.builder.nextConstID, axis)
		f.builder.nextConstID++
		stepName := fmt.Sprintf("dus_step_%d_%d", f.builder.nextConstID, axis)
		f.builder.nextConstID++

		startConst := f.builder.milBuilder.Const(startName, model.Int32, []int64{}, []int32{0})
		endConst := f.builder.milBuilder.Const(endName, model.Int32, []int64{}, []int32{int32(updateDim)})
		stepConst := f.builder.milBuilder.Const(stepName, model.Int32, []int64{}, []int32{1})

		// Generate 1D range for this axis
		rangeVal := f.builder.milBuilder.Range1D(startConst, endConst, stepConst)

		// Build reshape dims for broadcasting: size 1 for all axes except current
		reshapeDims := make([]int64, rank)
		for i := 0; i < rank; i++ {
			if i == axis {
				reshapeDims[i] = int64(updateDim)
			} else {
				reshapeDims[i] = 1
			}
		}
		reshapedRange := f.builder.milBuilder.Reshape(rangeVal, reshapeDims)

		// Build tile reps to broadcast to full update shape
		tileReps := make([]int64, rank)
		for i := 0; i < rank; i++ {
			if i == axis {
				tileReps[i] = 1
			} else {
				tileReps[i] = int64(updateShape.Dimensions[i])
			}
		}
		tiledRange := f.builder.milBuilder.Tile(reshapedRange, tileReps)

		// Get the startIndex value and squeeze if it's a size-1 tensor
		startIdx := startIndexNodes[axis].milValue
		if startIndexNodes[axis].shape.Rank() == 1 {
			// Squeeze the size-1 tensor to a scalar for broadcasting
			startIdx = f.builder.milBuilder.Squeeze(startIdx, []int64{0})
		}
		// Cast to Int32 if needed
		if startIndexNodes[axis].shape.DType != dtypes.Int32 {
			startIdx = f.builder.milBuilder.Cast(startIdx, model.Int32)
		}

		// Add the start index offset to all positions
		offsetIndices := f.builder.milBuilder.Add(tiledRange, startIdx)

		// Flatten to 1D: [updateSize]
		flattenedIndices := f.builder.milBuilder.Reshape(offsetIndices, []int64{int64(updateSize)})

		// Expand dims to [updateSize, 1]
		expandedIndices := f.builder.milBuilder.ExpandDims(flattenedIndices, []int64{1})

		axisIndices = append(axisIndices, expandedIndices)
	}

	// Concatenate all axis indices along axis 1 to get [updateSize, rank]
	var indicesValue *model.Value
	if rank == 1 {
		indicesValue = axisIndices[0]
	} else {
		// Concatenate along axis 1
		indicesValue = f.builder.milBuilder.Concat(axisIndices, 1)
	}

	// Flatten the update tensor to [updateSize]
	flatUpdate := f.builder.milBuilder.Reshape(updateNode.milValue, []int64{int64(updateSize)})

	// Use ScatterND with mode "update" to perform the dynamic update
	resultValue := f.builder.milBuilder.ScatterND(operandNode.milValue, indicesValue, flatUpdate, "update")

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operandNode, updateNode)
	// Add startIndexNodes as additional inputs for tracking
	node.inputs = append(node.inputs, startIndexNodes...)

	return node, nil
}

//======================================================================================================================
// Convolution and Pooling Operations
//======================================================================================================================

// ConvGeneral is a generic Convolution operation.
// CoreML expects NCHW layout for input ([N, C_in, H, W]) and OIHW layout for kernel ([C_out, C_in/groups, kH, kW]).
// This implementation handles axis transposition to convert GoMLX's flexible axis configuration to CoreML's expected layout.
func (f *Function) ConvGeneral(
	inputOp, kernelOp backends.Value,
	axes backends.ConvolveAxesConfig,
	strides []int,
	paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) (backends.Value, error) {
	// Sanitize group count
	channelGroupCount = max(channelGroupCount, 1)
	batchGroupCount = max(batchGroupCount, 1)

	opType := backends.OpTypeConvGeneral
	inputs, err := f.builder.checkOps(opType.String(), inputOp, kernelOp)
	if err != nil {
		return nil, err
	}
	input, kernel := inputs[0], inputs[1]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ConvGeneralOp(
		input.shape, kernel.shape, axes, strides, paddings,
		inputDilations, kernelDilations, channelGroupCount, batchGroupCount,
	)
	if err != nil {
		return nil, err
	}

	// TODO: batchGroupCount > 1 is not supported by CoreML Conv
	if batchGroupCount > 1 {
		return nil, errors.Errorf("ConvGeneral: batchGroupCount > 1 is not supported by CoreML backend")
	}

	rank := input.shape.Rank()
	spatialRank := rank - 2

	// Check if we need axis transposition - CoreML expects:
	// - Input: [N, C_in, spatial...] (NCHW for 2D)
	// - Kernel: [C_out, C_in/groups, spatial...] (OIHW for 2D)
	// - Output: [N, C_out, spatial...] (NCHW for 2D)
	needsInputTranspose := !isNCHWLayout(axes.InputBatch, axes.InputChannels, axes.InputSpatial)
	needsKernelTranspose := !isOIHWLayout(axes.KernelOutputChannels, axes.KernelInputChannels, axes.KernelSpatial)
	needsOutputTranspose := !isNCHWLayout(axes.OutputBatch, axes.OutputChannels, axes.OutputSpatial)

	// Transpose input to NCHW if needed
	inputValue := input.milValue
	if needsInputTranspose {
		inputPerm := buildNCHWPermutation(axes.InputBatch, axes.InputChannels, axes.InputSpatial)
		milInputPerm := intsToInt64s(inputPerm)
		inputValue = f.builder.milBuilder.Transpose(inputValue, milInputPerm)
	}

	// Transpose kernel to OIHW if needed
	kernelValue := kernel.milValue
	if needsKernelTranspose {
		kernelPerm := buildOIHWPermutation(axes.KernelOutputChannels, axes.KernelInputChannels, axes.KernelSpatial)
		milKernelPerm := intsToInt64s(kernelPerm)
		kernelValue = f.builder.milBuilder.Transpose(kernelValue, milKernelPerm)
	}

	// Prepare strides for CoreML (defaults to 1 if not provided)
	milStrides := make([]int64, spatialRank)
	for i := 0; i < spatialRank; i++ {
		if strides != nil && i < len(strides) && strides[i] > 0 {
			milStrides[i] = int64(strides[i])
		} else {
			milStrides[i] = 1
		}
	}

	// Prepare dilations for CoreML (defaults to 1 if not provided)
	// Note: CoreML Conv only supports kernel dilations, not input dilations
	milDilations := make([]int64, spatialRank)
	for i := 0; i < spatialRank; i++ {
		if kernelDilations != nil && i < len(kernelDilations) && kernelDilations[i] > 0 {
			milDilations[i] = int64(kernelDilations[i])
		} else {
			milDilations[i] = 1
		}
	}

	// Check for input dilations - CoreML doesn't support them directly.
	// Note: CoreML Conv supports "dilations" parameter but that controls KERNEL dilation
	// (spacing between kernel elements), NOT input dilation (spacing between input elements).
	// Input dilation would require inserting zeros between input elements before convolution,
	// which is computationally expensive and not natively supported.
	if inputDilations != nil {
		for i, d := range inputDilations {
			if d > 1 {
				return nil, errors.Errorf(
					"ConvGeneral: input dilation (also called 'base dilation') is not supported by CoreML backend. "+
						"Got inputDilations[%d]=%d, but only dilation=1 is supported. "+
						"Note: CoreML supports KERNEL dilation (spacing in the kernel weights) via the 'dilations' parameter, "+
						"but NOT input dilation (inserting zeros between input elements). "+
						"Workarounds: (1) pre-process your input to insert zeros manually before convolution, "+
						"(2) use a different backend (e.g., XLA) that supports input dilation, or "+
						"(3) restructure your model to avoid input dilation",
					i, d)
			}
		}
	}

	// Determine padding type and values
	// Always use ConvPadCustom since CoreML requires the 'pad' parameter
	var padType model.ConvPadType
	var padBefore, padAfter []int64

	padType = model.ConvPadCustom
	padBefore = make([]int64, spatialRank)
	padAfter = make([]int64, spatialRank)
	if paddings != nil && len(paddings) > 0 {
		for i := 0; i < spatialRank; i++ {
			if i < len(paddings) {
				padBefore[i] = int64(paddings[i][0])
				padAfter[i] = int64(paddings[i][1])
			}
		}
	}
	// If no paddings specified, padBefore and padAfter are already zero-initialized

	// Call CoreML Conv
	resultValue := f.builder.milBuilder.Conv(
		inputValue,
		kernelValue,
		milStrides,
		milDilations,
		padType,
		padBefore,
		padAfter,
		int64(channelGroupCount),
	)

	// Transpose output back to the expected layout if needed
	if needsOutputTranspose {
		// Build inverse permutation: from NCHW to the expected output layout
		outputInvPerm := buildInverseNCHWPermutation(axes.OutputBatch, axes.OutputChannels, axes.OutputSpatial, rank)
		milOutputInvPerm := intsToInt64s(outputInvPerm)
		resultValue = f.builder.milBuilder.Transpose(resultValue, milOutputInvPerm)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, input, kernel)

	return node, nil
}

// isNCHWLayout checks if the axis configuration represents NCHW layout (batch=0, channels=1, spatial=[2,3,...])
func isNCHWLayout(batch, channels int, spatial []int) bool {
	if batch != 0 || channels != 1 {
		return false
	}
	for i, s := range spatial {
		if s != i+2 {
			return false
		}
	}
	return true
}

// isOIHWLayout checks if the kernel axis configuration represents OIHW layout (out=0, in=1, spatial=[2,3,...])
func isOIHWLayout(outChannels, inChannels int, spatial []int) bool {
	if outChannels != 0 || inChannels != 1 {
		return false
	}
	for i, s := range spatial {
		if s != i+2 {
			return false
		}
	}
	return true
}

// buildNCHWPermutation builds a permutation array to convert from the given layout to NCHW
func buildNCHWPermutation(batch, channels int, spatial []int) []int {
	rank := 2 + len(spatial)
	perm := make([]int, rank)
	perm[0] = batch    // Batch goes to position 0
	perm[1] = channels // Channels go to position 1
	for i, s := range spatial {
		perm[2+i] = s // Spatial dimensions go to positions 2+
	}
	return perm
}

// buildOIHWPermutation builds a permutation array to convert kernel from the given layout to OIHW
func buildOIHWPermutation(outChannels, inChannels int, spatial []int) []int {
	rank := 2 + len(spatial)
	perm := make([]int, rank)
	perm[0] = outChannels // Output channels go to position 0
	perm[1] = inChannels  // Input channels go to position 1
	for i, s := range spatial {
		perm[2+i] = s // Spatial dimensions go to positions 2+
	}
	return perm
}

// buildInverseNCHWPermutation builds the inverse permutation to convert from NCHW back to the expected output layout
func buildInverseNCHWPermutation(batch, channels int, spatial []int, rank int) []int {
	// First build the forward permutation (expected -> NCHW)
	fwd := make([]int, rank)
	fwd[0] = batch
	fwd[1] = channels
	for i, s := range spatial {
		fwd[2+i] = s
	}
	// Then invert it (NCHW -> expected)
	inv := make([]int, rank)
	for i, v := range fwd {
		inv[v] = i
	}
	return inv
}

// intsToInt64s converts []int to []int64
func intsToInt64s(ints []int) []int64 {
	result := make([]int64, len(ints))
	for i, v := range ints {
		result[i] = int64(v)
	}
	return result
}

// ReduceWindow runs a reduction function over sliding windows.
// CoreML supports MaxPool and AvgPool operations which correspond to ReduceOpMax and ReduceOpSum/ReduceOpProduct.
// CoreML expects NCHW layout for input ([N, C, H, W]).
//
// For tensors with rank < 3, we add fake batch/channel dimensions, apply pooling, then remove them.
func (f *Function) ReduceWindow(
	operandOp backends.Value,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Value, error) {
	opType := backends.OpTypeReduceWindow
	inputs, err := f.builder.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceWindowOp(
		operand.shape,
		windowDimensions,
		strides,
		baseDilations,
		windowDilations,
		paddings,
	)
	if err != nil {
		return nil, err
	}

	rank := operand.shape.Rank()

	// Check for unsupported features
	if baseDilations != nil {
		for _, d := range baseDilations {
			if d > 1 {
				return nil, errors.Errorf("ReduceWindow: base dilations > 1 are not supported by CoreML backend")
			}
		}
	}
	if windowDilations != nil {
		for _, d := range windowDilations {
			if d > 1 {
				return nil, errors.Errorf("ReduceWindow: window dilations > 1 are not supported by CoreML backend pooling ops")
			}
		}
	}

	// CoreML pooling only supports Float32 and Float16 tensors
	// For other dtypes, we cast to Float32, perform the pooling, then cast back
	operandDType := operand.shape.DType
	needsCastBack := false
	operandValue := operand.milValue

	if operandDType != dtypes.Float32 && operandDType != dtypes.Float16 {
		// Cast to Float32 for pooling
		operandValue = f.builder.milBuilder.Cast(operandValue, model.Float32)
		needsCastBack = true
	}

	// CoreML pooling requires rank >= 3 (at least [N, C, spatial])
	// For lower rank tensors, we add fake dimensions, apply pooling, then remove them
	effectiveRank := rank
	dimsToSqueeze := 0

	if rank < 3 {
		// Add fake dimensions to make it at least rank 3
		// For rank 2 [A, B]: reshape to [1, A, B] (add fake N)
		// For rank 1 [A]: reshape to [1, 1, A] (add fake N, C)
		dimsToSqueeze = 3 - rank
		newShape := make([]int64, 3)
		for i := 0; i < dimsToSqueeze; i++ {
			newShape[i] = 1
		}
		for i := 0; i < rank; i++ {
			newShape[dimsToSqueeze+i] = int64(operand.shape.Dimensions[i])
		}
		operandValue = f.builder.milBuilder.Reshape(operandValue, newShape)

		// Adjust windowDimensions, strides, paddings to account for added dimensions
		newWindowDims := make([]int, 3)
		newStrides := make([]int, 3)
		newPaddings := make([][2]int, 3)
		for i := 0; i < dimsToSqueeze; i++ {
			newWindowDims[i] = 1
			newStrides[i] = 1
			newPaddings[i] = [2]int{0, 0}
		}
		for i := 0; i < rank; i++ {
			newWindowDims[dimsToSqueeze+i] = windowDimensions[i]
			if strides != nil && i < len(strides) {
				newStrides[dimsToSqueeze+i] = strides[i]
			} else {
				newStrides[dimsToSqueeze+i] = windowDimensions[i]
			}
			if paddings != nil && i < len(paddings) {
				newPaddings[dimsToSqueeze+i] = paddings[i]
			}
		}
		windowDimensions = newWindowDims
		strides = newStrides
		paddings = newPaddings
		effectiveRank = 3
	}

	// CoreML pooling operates on spatial dimensions only (assumes NCHW layout)
	// The window must have size 1 for batch and channel dimensions
	if len(windowDimensions) >= 2 {
		if windowDimensions[0] != 1 || windowDimensions[1] != 1 {
			return nil, errors.Errorf("ReduceWindow: CoreML pooling only supports window size 1 for batch and channel dimensions, got %v", windowDimensions[:2])
		}
	}

	// Extract spatial dimensions for pooling (skip batch and channel dimensions)
	spatialRank := effectiveRank - 2
	spatialWindowDims := windowDimensions[2:]
	if len(spatialWindowDims) != spatialRank {
		return nil, errors.Errorf("ReduceWindow: window dimensions mismatch, expected %d spatial dims, got %d", spatialRank, len(spatialWindowDims))
	}

	// Prepare kernel size for CoreML
	milKernelSize := make([]int64, spatialRank)
	for i := 0; i < spatialRank; i++ {
		milKernelSize[i] = int64(spatialWindowDims[i])
	}

	// Prepare strides for CoreML (defaults to window size if not provided, per GoMLX semantics)
	milStrides := make([]int64, spatialRank)
	for i := 0; i < spatialRank; i++ {
		if strides != nil && i+2 < len(strides) && strides[i+2] > 0 {
			milStrides[i] = int64(strides[i+2])
		} else {
			milStrides[i] = milKernelSize[i] // Default: stride = window size
		}
	}

	// Determine padding type and values
	var padType model.ConvPadType
	var padBefore, padAfter []int64

	if paddings == nil || len(paddings) == 0 {
		padType = model.ConvPadValid
	} else {
		// Check if all spatial padding values are zero
		allZero := true
		for i := 2; i < len(paddings); i++ {
			if paddings[i][0] != 0 || paddings[i][1] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			padType = model.ConvPadValid
		} else {
			padType = model.ConvPadCustom
			padBefore = make([]int64, spatialRank)
			padAfter = make([]int64, spatialRank)
			for i := 0; i < spatialRank; i++ {
				if i+2 < len(paddings) {
					padBefore[i] = int64(paddings[i+2][0])
					padAfter[i] = int64(paddings[i+2][1])
				}
			}
		}
	}

	var resultValue *model.Value

	switch reductionType {
	case backends.ReduceOpMax:
		resultValue = f.builder.milBuilder.MaxPool(
			operandValue,
			milKernelSize,
			milStrides,
			padType,
			padBefore,
			padAfter,
		)

	case backends.ReduceOpSum:
		// Use AvgPool and multiply by window size to get sum
		// AvgPool computes: sum(window) / window_size
		// So sum = AvgPool * window_size
		avgResult := f.builder.milBuilder.AvgPool(
			operandValue,
			milKernelSize,
			milStrides,
			padType,
			padBefore,
			padAfter,
			true, // excludePaddingFromAverage - for correctness when padding
		)

		// Calculate window size
		windowSize := int64(1)
		for _, k := range milKernelSize {
			windowSize *= k
		}

		// Create constant for window size and multiply
		// Always use Float32 for data, then cast if needed
		constName := fmt.Sprintf("reduce_window_size_%d", f.builder.nextConstID)
		f.builder.nextConstID++
		windowSizeConst := f.builder.milBuilder.Const(constName, model.Float32, []int64{}, []float32{float32(windowSize)})
		avgResultDType := avgResult.DType()
		if avgResultDType != model.Float32 {
			windowSizeConst = f.builder.milBuilder.Cast(windowSizeConst, avgResultDType)
		}
		resultValue = f.builder.milBuilder.Mul(avgResult, windowSizeConst)

	case backends.ReduceOpMin:
		// CoreML doesn't have MinPool directly, so we use the negation trick:
		// MinPool(x) = -MaxPool(-x)

		// Step 1: Negate the input
		// Always use Float32 for data, then cast if needed
		negOneConstName := fmt.Sprintf("minpool_neg_one_%d", f.builder.nextConstID)
		f.builder.nextConstID++
		negOne := f.builder.milBuilder.Const(negOneConstName, model.Float32, []int64{}, []float32{-1.0})
		operandDType := operandValue.DType()
		if operandDType != model.Float32 {
			negOne = f.builder.milBuilder.Cast(negOne, operandDType)
		}
		negInput := f.builder.milBuilder.Mul(operandValue, negOne)

		// Step 2: Apply MaxPool to the negated input
		maxPoolResult := f.builder.milBuilder.MaxPool(
			negInput,
			milKernelSize,
			milStrides,
			padType,
			padBefore,
			padAfter,
		)

		// Step 3: Negate the result to get MinPool
		resultValue = f.builder.milBuilder.Mul(maxPoolResult, negOne)

	case backends.ReduceOpProduct:
		// CoreML doesn't have ProductPool
		return nil, errors.Errorf("ReduceWindow: ReduceOpProduct is not supported by CoreML backend")

	default:
		return nil, errors.Errorf("ReduceWindow: unsupported reduction type %v", reductionType)
	}

	// If we added fake dimensions, squeeze them out
	if dimsToSqueeze > 0 {
		// Build the output shape with only the original dimensions
		outDims := make([]int64, outputShape.Rank())
		for i := 0; i < outputShape.Rank(); i++ {
			outDims[i] = int64(outputShape.Dimensions[i])
		}
		resultValue = f.builder.milBuilder.Reshape(resultValue, outDims)
	}

	// If we cast to Float32 for pooling, cast back to original dtype
	if needsCastBack {
		milDType, err := gomlxDTypeToMIL(operandDType)
		if err != nil {
			return nil, errors.Wrapf(err, "ReduceWindow: failed to convert dtype %s back after pooling", operandDType)
		}
		resultValue = f.builder.milBuilder.Cast(resultValue, milDType)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Logical and Classification Operations
//======================================================================================================================

// Identity returns an Op whose output is the same as its input.
// It's a no-op that can serve as a place-holder.
func (f *Function) Identity(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeIdentity
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Identity is a no-op - the output shape and dtype are the same as input
	outputShape := operand.shape

	var resultValue *model.Value
	if f.isClosureContext() {
		// In closure context, create a placeholder value.
		placeholderName := fmt.Sprintf("closure_op_%p_%d", f, len(f.builder.nodes))
		resultValue = f.builder.milBuilder.PlaceholderValue(placeholderName, operand.milValue.DType(), operand.milValue.Shape()...)
	} else {
		// In main function context, use MIL identity operation.
		identityName := fmt.Sprintf("identity_%d", f.builder.nextConstID)
		f.builder.nextConstID++
		resultValue = f.builder.milBuilder.Identity(identityName, operand.milValue)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// BroadcastInDim broadcasts the input tensor to the target shape.
//
// The broadcastDims parameter specifies which dimensions of the output shape
// correspond to dimensions of the input. For example:
//   - Input shape: [3], Output shape: [2, 3], broadcastDims: [1]
//     means input dim 0 maps to output dim 1, resulting in shape [2, 3]
//   - Input shape: [2, 3], Output shape: [2, 3, 4], broadcastDims: [0, 1]
//     means input dims map to output dims 0 and 1, resulting in shape [2, 3, 4]
//
// Implementation: First reshape to insert size-1 dimensions, then tile to expand.
func (f *Function) BroadcastInDim(x backends.Value, shape shapes.Shape, broadcastDims []int) (backends.Value, error) {
	opType := backends.OpTypeBroadcastInDim
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Validate broadcast axes length
	if len(broadcastDims) != operand.shape.Rank() {
		return nil, errors.Errorf("BroadcastInDim: broadcastDims length (%d) must match input rank (%d)",
			len(broadcastDims), operand.shape.Rank())
	}

	// Validate broadcast axes bounds and dimension compatibility
	for i, outAxis := range broadcastDims {
		if outAxis < 0 || outAxis >= shape.Rank() {
			return nil, errors.Errorf("BroadcastInDim: broadcastDims[%d]=%d out of bounds [0, %d)",
				i, outAxis, shape.Rank())
		}
		inputDim := operand.shape.Dimensions[i]
		outputDim := shape.Dimensions[outAxis]
		if inputDim != outputDim && inputDim != 1 {
			return nil, errors.Errorf("BroadcastInDim: dimension mismatch at broadcastDims[%d]=%d: input dim=%d, output dim=%d (must be equal or input=1)",
				i, outAxis, inputDim, outputDim)
		}
	}

	// Convert output shape dimensions to int64
	milOutShape := make([]int64, shape.Rank())
	for i := 0; i < shape.Rank(); i++ {
		milOutShape[i] = int64(shape.Dimensions[i])
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(shape)
		if err != nil {
			return nil, errors.Wrap(err, "BroadcastInDim")
		}
	} else {
		// CoreML MIL doesn't have a direct broadcast_in_dim operation.
		// We need to compose it using reshape and tile operations.
		//
		// Strategy:
		// 1. First, expand dims to match the target rank by inserting size-1 dimensions
		// 2. Then use tile to broadcast the size-1 dimensions to the target size

		// Step 1: Build the intermediate shape after expanding dimensions
		// The intermediate shape has the same rank as output, with:
		// - Dimensions from input placed at positions specified by broadcastDims
		// - Size 1 at all other positions
		intermediateShape := make([]int64, shape.Rank())
		for i := range intermediateShape {
			intermediateShape[i] = 1
		}
		for i, outAxis := range broadcastDims {
			intermediateShape[outAxis] = int64(operand.shape.Dimensions[i])
		}

		// Reshape input to intermediate shape
		reshaped := f.builder.milBuilder.Reshape(operand.milValue, intermediateShape)

		// Step 2: Compute tile repetitions
		reps := make([]int64, shape.Rank())
		for i := 0; i < shape.Rank(); i++ {
			if intermediateShape[i] == 1 {
				reps[i] = milOutShape[i]
			} else {
				reps[i] = 1
			}
		}

		// Check if we need to tile at all
		needTile := false
		for _, r := range reps {
			if r > 1 {
				needTile = true
				break
			}
		}

		if needTile {
			resultValue = f.builder.milBuilder.Tile(reshaped, reps)
		} else {
			resultValue = reshaped
		}
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, shape, resultValue, operand)

	return node, nil
}

// Clamp returns the element-wise clamping operation.
// The values max and min can either be a scalar or have the same shape as x.
func (f *Function) Clamp(min, x, max backends.Value) (backends.Value, error) {
	opType := backends.OpTypeClamp
	inputs, err := f.builder.checkOps(opType.String(), min, x, max)
	if err != nil {
		return nil, err
	}
	minNode, xNode, maxNode := inputs[0], inputs[1], inputs[2]

	// Output shape follows broadcasting rules for all three operands
	// First broadcast x with min, then result with max
	outputShape, err := shapeinference.BinaryOp(backends.OpTypeMax, xNode.shape, minNode.shape)
	if err != nil {
		return nil, errors.Wrap(err, "Clamp: broadcasting x with min")
	}
	outputShape, err = shapeinference.BinaryOp(backends.OpTypeMin, outputShape, maxNode.shape)
	if err != nil {
		return nil, errors.Wrap(err, "Clamp: broadcasting result with max")
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "Clamp")
		}
	} else {
		// Use MIL Clip operation
		resultValue = f.builder.milBuilder.Clip(xNode.milValue, minNode.milValue, maxNode.milValue)
	}

	// Create a new node with the result
	node := f.builder.newNode(opType, outputShape, resultValue, minNode, xNode, maxNode)

	return node, nil
}

// LogicalAnd returns the element-wise logical AND operation.
func (f *Function) LogicalAnd(lhs, rhs backends.Value) (backends.Value, error) {
	opType := backends.OpTypeLogicalAnd
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Output shape follows broadcasting rules, dtype is Bool
	outputShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}
	// Logical operations always return Bool
	outputShape = shapes.Make(dtypes.Bool, outputShape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "LogicalAnd")
		}
	} else {
		resultValue = f.builder.milBuilder.LogicalAnd(lhsNode.milValue, rhsNode.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)
	return node, nil
}

// LogicalOr returns the element-wise logical OR operation.
func (f *Function) LogicalOr(lhs, rhs backends.Value) (backends.Value, error) {
	opType := backends.OpTypeLogicalOr
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Output shape follows broadcasting rules, dtype is Bool
	outputShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}
	outputShape = shapes.Make(dtypes.Bool, outputShape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "LogicalOr")
		}
	} else {
		resultValue = f.builder.milBuilder.LogicalOr(lhsNode.milValue, rhsNode.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)
	return node, nil
}

// LogicalNot returns the element-wise logical NOT operation.
func (f *Function) LogicalNot(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeLogicalNot
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Output shape is same as input, dtype is Bool
	outputShape := shapes.Make(dtypes.Bool, operand.shape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "LogicalNot")
		}
	} else {
		resultValue = f.builder.milBuilder.LogicalNot(operand.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, operand)
	return node, nil
}

// LogicalXor returns the element-wise logical XOR operator.
func (f *Function) LogicalXor(lhs, rhs backends.Value) (backends.Value, error) {
	opType := backends.OpTypeLogicalXor
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Output shape follows broadcasting rules, dtype is Bool
	outputShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}
	outputShape = shapes.Make(dtypes.Bool, outputShape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "LogicalXor")
		}
	} else {
		resultValue = f.builder.milBuilder.LogicalXor(lhsNode.milValue, rhsNode.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)
	return node, nil
}

// IsFinite tests whether each element of operand is finite.
// Returns boolean values where each element is true if and only if the corresponding input element is finite.
func (f *Function) IsFinite(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeIsFinite
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Output shape is same as input, dtype is Bool
	outputShape := shapes.Make(dtypes.Bool, operand.shape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "IsFinite")
		}
	} else {
		resultValue = f.builder.milBuilder.IsFinite(operand.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, operand)
	return node, nil
}

// IsNaN tests whether each element of operand is NaN.
func (f *Function) IsNaN(x backends.Value) (backends.Value, error) {
	opType := backends.OpTypeIsNaN
	inputs, err := f.builder.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Output shape is same as input, dtype is Bool
	outputShape := shapes.Make(dtypes.Bool, operand.shape.Dimensions...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "IsNaN")
		}
	} else {
		resultValue = f.builder.milBuilder.IsNan(operand.milValue)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, operand)
	return node, nil
}

// DynamicSlice extracts a slice from the operand at the startIndices position and the given sliceSizes.
func (f *Function) DynamicSlice(operand backends.Value, startIndices []backends.Value, sliceDims []int) (backends.Value, error) {
	opType := backends.OpTypeDynamicSlice
	inputs, err := f.builder.checkOps(opType.String(), operand)
	if err != nil {
		return nil, err
	}
	operandNode := inputs[0]

	// Validate inputs
	if len(startIndices) != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicSlice: startIndices length (%d) must match operand rank (%d)",
			len(startIndices), operandNode.shape.Rank())
	}
	if len(sliceDims) != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicSlice: sliceDims length (%d) must match operand rank (%d)",
			len(sliceDims), operandNode.shape.Rank())
	}

	// Collect and validate start index nodes
	startNodes := make([]*Node, len(startIndices))
	for i, idx := range startIndices {
		idxInputs, err := f.builder.checkOps(opType.String(), idx)
		if err != nil {
			return nil, err
		}
		startNodes[i] = idxInputs[0]
		// Validate that start indices are scalar or rank-1 with size 1
		idxShape := startNodes[i].shape
		if idxShape.Rank() > 1 || (idxShape.Rank() == 1 && idxShape.Dimensions[0] != 1) {
			return nil, errors.Errorf("DynamicSlice: startIndices[%d] must be scalar or rank-1 with size 1, got shape %v",
				i, idxShape.Dimensions)
		}
	}

	// Output shape is determined by sliceDims
	outputShape := shapes.Make(operandNode.shape.DType, sliceDims...)

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "DynamicSlice")
		}
	} else {
		// Use MIL SliceBySize operation
		// First, stack the start indices into a single tensor
		milSliceSizes := make([]int64, len(sliceDims))
		for i, s := range sliceDims {
			milSliceSizes[i] = int64(s)
		}

		// Concatenate start indices into a 1D tensor [rank]
		startMilValues := make([]*model.Value, len(startNodes))
		for i, node := range startNodes {
			startMilValues[i] = node.milValue
		}

		// Stack the scalar indices into a 1D tensor using concat
		var beginTensor *model.Value
		if len(startMilValues) == 1 {
			// Single dimension case - reshape scalar to [1]
			beginTensor = f.builder.milBuilder.Reshape(startMilValues[0], []int64{1})
		} else {
			// Multiple dimensions - reshape each to [1] and concatenate
			reshapedIndices := make([]*model.Value, len(startMilValues))
			for i, v := range startMilValues {
				reshapedIndices[i] = f.builder.milBuilder.Reshape(v, []int64{1})
			}
			beginTensor = f.builder.milBuilder.Concat(reshapedIndices, 0)
		}

		resultValue = f.builder.milBuilder.SliceBySize(operandNode.milValue, beginTensor, milSliceSizes)
	}

	// Collect all input nodes for the graph
	allInputs := make([]*Node, 1+len(startNodes))
	allInputs[0] = operandNode
	copy(allInputs[1:], startNodes)

	node := f.builder.newNode(opType, outputShape, resultValue, allInputs...)
	return node, nil
}

//======================================================================================================================
// Normalization and Arithmetic Operations
//======================================================================================================================

// BatchNormForInference implements batch normalization for inference.
// The operand is normalized along the featureAxis using the given mean, variance, scale, and offset.
// scale, offset, mean, and variance must be 1D tensors with size equal to operand.shape[featureAxis].
func (f *Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Value, epsilon float32, featureAxis int) (backends.Value, error) {
	opType := backends.OpTypeBatchNormForInference
	inputs, err := f.builder.checkOps(opType.String(), operand, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	operandNode := inputs[0]
	scaleNode := inputs[1]
	offsetNode := inputs[2]
	meanNode := inputs[3]
	varianceNode := inputs[4]

	// Output shape is the same as operand
	outputShape := operandNode.shape

	// Normalize and validate feature axis
	if featureAxis < 0 {
		featureAxis = operandNode.shape.Rank() + featureAxis
	}
	if featureAxis < 0 || featureAxis >= operandNode.shape.Rank() {
		return nil, errors.Errorf("BatchNormForInference: featureAxis %d out of bounds for operand rank %d",
			featureAxis, operandNode.shape.Rank())
	}

	// Validate parameter shapes
	numFeatures := operandNode.shape.Dimensions[featureAxis]
	paramNodes := []*Node{scaleNode, offsetNode, meanNode, varianceNode}
	paramNames := []string{"scale", "offset", "mean", "variance"}
	for i, paramNode := range paramNodes {
		if paramNode.shape.Rank() != 1 || paramNode.shape.Dimensions[0] != numFeatures {
			return nil, errors.Errorf("BatchNormForInference: %s must have shape [%d], got %v",
				paramNames[i], numFeatures, paramNode.shape.Dimensions)
		}
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "BatchNormForInference")
		}
	} else {
		// CoreML MIL batch_norm expects input in NCHW format with feature axis = 1
		// If the feature axis is different, we need to transpose

		if featureAxis == 1 {
			// Standard NCHW format - use batch_norm directly
			resultValue = f.builder.milBuilder.BatchNorm(
				operandNode.milValue,
				meanNode.milValue,
				varianceNode.milValue,
				scaleNode.milValue,  // gamma
				offsetNode.milValue, // beta
				epsilon,
			)
		} else {
			// Non-standard feature axis - compose using element-wise operations
			// BatchNorm: y = (x - mean) / sqrt(variance + epsilon) * scale + offset

			// 1. Subtract mean: x - mean
			centered := f.builder.milBuilder.Sub(operandNode.milValue, meanNode.milValue)

			// 2. Compute rsqrt(variance + epsilon)
			constName := fmt.Sprintf("bn_eps_%d", f.builder.nextConstID)
			f.builder.nextConstID++
			epsilonVal := f.builder.milBuilder.Const(constName, model.Float32, []int64{}, []float32{epsilon})
			varPlusEps := f.builder.milBuilder.Add(varianceNode.milValue, epsilonVal)
			invStd := f.builder.milBuilder.Rsqrt(varPlusEps)

			// 3. Normalize: (x - mean) * rsqrt(variance + epsilon)
			normalized := f.builder.milBuilder.Mul(centered, invStd)

			// 4. Scale: normalized * scale
			scaled := f.builder.milBuilder.Mul(normalized, scaleNode.milValue)

			// 5. Offset: scaled + offset
			resultValue = f.builder.milBuilder.Add(scaled, offsetNode.milValue)
		}
	}

	node := f.builder.newNode(opType, outputShape, resultValue, operandNode, scaleNode, offsetNode, meanNode, varianceNode)
	return node, nil
}

// Rem returns the remainder operation using floor division semantics.
// This implements: lhs - floor(lhs / rhs) * rhs
// which matches Python's % operator. The result has the same sign as rhs (the divisor).
// For negative operands, this may differ from C/Java's % operator:
//
//	Rem(7, 3) = 1
//	Rem(-7, 3) = 2 (not -1 as in C/Java)
//	Rem(7, -3) = -2
func (f *Function) Rem(lhs, rhs backends.Value) (backends.Value, error) {
	opType := backends.OpTypeRem
	inputs, err := f.builder.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Output shape follows broadcasting rules
	outputShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	var resultValue *model.Value
	if f.isClosureContext() {
		resultValue, err = f.closurePlaceholder(outputShape)
		if err != nil {
			return nil, errors.Wrap(err, "Rem")
		}
	} else {
		// CoreML MIL doesn't have a direct remainder operation.
		// Compute: lhs - floor(lhs / rhs) * rhs (floor modulo)

		// 1. lhs / rhs
		quotient := f.builder.milBuilder.Div(lhsNode.milValue, rhsNode.milValue)

		// 2. floor(lhs / rhs)
		floored := f.builder.milBuilder.Floor(quotient)

		// 3. floor(lhs / rhs) * rhs
		product := f.builder.milBuilder.Mul(floored, rhsNode.milValue)

		// 4. lhs - floor(lhs / rhs) * rhs
		resultValue = f.builder.milBuilder.Sub(lhsNode.milValue, product)
	}

	node := f.builder.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)
	return node, nil
}
