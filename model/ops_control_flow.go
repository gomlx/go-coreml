package model

import (
	"fmt"

	"github.com/gomlx/go-coreml/proto/coreml/milspec"
)

// BlockBuilder builds a nested block within an operation.
// It is similar to Builder but creates a block rather than a full program.
// Blocks are used for control flow operations like cond and while_loop.
type BlockBuilder struct {
	parent     *Builder
	inputs     []*Value
	outputs    []string
	operations []*milspec.Operation
	values     map[string]*Value
	nextID     int
	err        error
}

// NewBlockBuilder creates a new block builder within a parent builder.
// The block inherits the parent's opset but has its own scope for operations.
func (b *Builder) NewBlockBuilder() *BlockBuilder {
	return &BlockBuilder{
		parent: b,
		values: make(map[string]*Value),
		nextID: 0,
	}
}

// Err returns the first error encountered during block building, if any.
func (bb *BlockBuilder) Err() error {
	return bb.err
}

// setErr records the first error encountered.
func (bb *BlockBuilder) setErr(err error) {
	if bb.err == nil {
		bb.err = err
	}
}

// genName generates a unique name for intermediate values within this block.
func (bb *BlockBuilder) genName(prefix string) string {
	name := fmt.Sprintf("%s_blk_%d", prefix, bb.nextID)
	bb.nextID++
	return name
}

// BlockInput adds an input to this block and returns a Value that references it.
// Block inputs are used to receive values passed into the block from the parent scope.
// For while_loop, these are the loop variables.
// For cond, blocks typically don't have explicit inputs.
func (bb *BlockBuilder) BlockInput(name string, dtype DType, shape []int64) *Value {
	v := &Value{
		name:    name,
		dtype:   dtype,
		shape:   shape,
		builder: bb.parent,
	}
	bb.inputs = append(bb.inputs, v)
	bb.values[name] = v
	return v
}

// BlockOutput marks a value as an output of this block.
func (bb *BlockBuilder) BlockOutput(name string, v *Value) {
	// If the name matches the internal value name, use it directly
	if name == v.name {
		bb.outputs = append(bb.outputs, name)
		return
	}

	// Otherwise, add an identity operation to rename the output
	renamed := bb.Identity(name, v)
	bb.outputs = append(bb.outputs, renamed.name)
}

// Const creates a constant tensor value within this block.
func (bb *BlockBuilder) Const(name string, dtype DType, shape []int64, data interface{}) *Value {
	val := createValue(dtype, shape, data)
	v := &Value{
		name:     name,
		dtype:    dtype,
		shape:    shape,
		builder:  bb.parent,
		isConst:  true,
		constVal: val,
	}
	bb.values[name] = v
	return v
}

// Identity creates an identity operation that copies a value with a new name.
func (bb *BlockBuilder) Identity(name string, x *Value) *Value {
	return bb.addOp("identity", map[string]*Value{
		"x": x,
	}, name, x.dtype, x.shape)
}

// addOp adds an operation to this block and returns the output value.
func (bb *BlockBuilder) addOp(opType string, inputs map[string]*Value, outputName string, outputDtype DType, outputShape []int64) *Value {
	// Build input arguments
	opInputs := make(map[string]*milspec.Argument)
	for name, v := range inputs {
		if v.isConst {
			opInputs[name] = &milspec.Argument{
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: v.constVal},
				}},
			}
		} else {
			opInputs[name] = &milspec.Argument{
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: v.name},
				}},
			}
		}
	}

	// Build output type
	tensorType := &milspec.TensorType{
		DataType:   outputDtype,
		Rank:       int64(len(outputShape)),
		Dimensions: make([]*milspec.Dimension, len(outputShape)),
	}
	for i, dim := range outputShape {
		tensorType.Dimensions[i] = &milspec.Dimension{
			Dimension: &milspec.Dimension_Constant{
				Constant: &milspec.Dimension_ConstantDimension{Size: uint64(dim)},
			},
		}
	}

	op := &milspec.Operation{
		Type:   opType,
		Inputs: opInputs,
		Outputs: []*milspec.NamedValueType{{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: tensorType},
			},
		}},
	}

	bb.operations = append(bb.operations, op)

	v := &Value{
		name:    outputName,
		dtype:   outputDtype,
		shape:   outputShape,
		builder: bb.parent,
	}
	bb.values[outputName] = v
	return v
}

// Build constructs the MIL Block from this builder.
func (bb *BlockBuilder) Build() *milspec.Block {
	// Build block inputs
	inputs := make([]*milspec.NamedValueType, len(bb.inputs))
	for i, v := range bb.inputs {
		tensorType := &milspec.TensorType{
			DataType:   v.dtype,
			Rank:       int64(len(v.shape)),
			Dimensions: make([]*milspec.Dimension, len(v.shape)),
		}
		for j, dim := range v.shape {
			tensorType.Dimensions[j] = &milspec.Dimension{
				Dimension: &milspec.Dimension_Constant{
					Constant: &milspec.Dimension_ConstantDimension{Size: uint64(dim)},
				},
			}
		}
		inputs[i] = &milspec.NamedValueType{
			Name: v.name,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: tensorType},
			},
		}
	}

	return &milspec.Block{
		Inputs:     inputs,
		Outputs:    bb.outputs,
		Operations: bb.operations,
	}
}

// GetValue retrieves a value by name from the block's scope.
// If not found in block scope, it checks the parent scope.
func (bb *BlockBuilder) GetValue(name string) (*Value, bool) {
	if v, ok := bb.values[name]; ok {
		return v, true
	}
	if v, ok := bb.parent.values[name]; ok {
		return v, true
	}
	return nil, false
}

// =============================================================================
// Operations within BlockBuilder - mirroring Builder operations
// =============================================================================

// Add performs element-wise addition within this block.
func (bb *BlockBuilder) Add(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("add", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("add"), x.dtype, outShape)
}

// Sub performs element-wise subtraction within this block.
func (bb *BlockBuilder) Sub(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("sub", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("sub"), x.dtype, outShape)
}

// Mul performs element-wise multiplication within this block.
func (bb *BlockBuilder) Mul(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("mul", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("mul"), x.dtype, outShape)
}

// Less performs element-wise less-than comparison within this block.
func (bb *BlockBuilder) Less(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("less", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("less"), Bool, outShape)
}

// Greater performs element-wise greater-than comparison within this block.
func (bb *BlockBuilder) Greater(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("greater", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("greater"), Bool, outShape)
}

// Select performs element-wise selection based on a condition within this block.
func (bb *BlockBuilder) Select(cond, a, bVal *Value) *Value {
	outShape := broadcastShape(a.shape, bVal.shape)
	return bb.addOp("select", map[string]*Value{
		"cond": cond,
		"a":    a,
		"b":    bVal,
	}, bb.genName("select"), a.dtype, outShape)
}

// LogicalAnd performs element-wise logical AND within this block.
func (bb *BlockBuilder) LogicalAnd(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return bb.addOp("logical_and", map[string]*Value{
		"x": x,
		"y": y,
	}, bb.genName("logical_and"), Bool, outShape)
}

// =============================================================================
// Control Flow Operations for the main Builder
// =============================================================================

// CondFunc is a function type for building blocks in conditional operations.
// It receives a BlockBuilder and should add operations to produce outputs.
type CondFunc func(bb *BlockBuilder) []*Value

// LoopCondFunc is a function type for building the condition block of a while loop.
// It receives the loop variables and should return a boolean scalar value.
type LoopCondFunc func(bb *BlockBuilder, loopVars []*Value) *Value

// LoopBodyFunc is a function type for building the body block of a while loop.
// It receives the loop variables and should return updated loop variables.
type LoopBodyFunc func(bb *BlockBuilder, loopVars []*Value) []*Value

// Cond executes one of two branches based on a boolean predicate.
//
// pred: A scalar boolean tensor (shape []) that determines which branch to execute.
// trueFn: Function that builds operations for the true branch.
// falseFn: Function that builds operations for the false branch.
//
// Both functions must return the same number of outputs with matching types.
// Returns the outputs from the executed branch.
//
// Example:
//
//	pred := b.Input("pred", model.Bool)
//	x := b.Input("x", model.Float32, 10)
//	y := b.Input("y", model.Float32, 10)
//	result := b.Cond(pred,
//	    func(bb *model.BlockBuilder) []*model.Value {
//	        return []*model.Value{bb.Add(x, y)}
//	    },
//	    func(bb *model.BlockBuilder) []*model.Value {
//	        return []*model.Value{bb.Mul(x, y)}
//	    },
//	)
func (b *Builder) Cond(pred *Value, trueFn, falseFn CondFunc) []*Value {
	// Validate pred is a boolean scalar
	if pred.dtype != Bool {
		b.setErr(fmt.Errorf("cond: pred must be Bool type, got %v", pred.dtype))
		return nil
	}
	if len(pred.shape) != 0 {
		b.setErr(fmt.Errorf("cond: pred must be a scalar (shape []), got shape %v", pred.shape))
		return nil
	}

	// Build true branch block
	trueBlock := b.NewBlockBuilder()
	trueOutputs := trueFn(trueBlock)
	if trueBlock.Err() != nil {
		b.setErr(fmt.Errorf("cond true branch: %w", trueBlock.Err()))
		return nil
	}

	// Mark outputs in true block
	for i, v := range trueOutputs {
		trueBlock.BlockOutput(fmt.Sprintf("cond_true_out_%d", i), v)
	}

	// Build false branch block
	falseBlock := b.NewBlockBuilder()
	falseOutputs := falseFn(falseBlock)
	if falseBlock.Err() != nil {
		b.setErr(fmt.Errorf("cond false branch: %w", falseBlock.Err()))
		return nil
	}

	// Mark outputs in false block
	for i, v := range falseOutputs {
		falseBlock.BlockOutput(fmt.Sprintf("cond_false_out_%d", i), v)
	}

	// Validate that both branches have the same number of outputs
	if len(trueOutputs) != len(falseOutputs) {
		b.setErr(fmt.Errorf("cond: true branch has %d outputs but false branch has %d outputs",
			len(trueOutputs), len(falseOutputs)))
		return nil
	}

	// Validate output types match
	for i := range trueOutputs {
		if trueOutputs[i].dtype != falseOutputs[i].dtype {
			b.setErr(fmt.Errorf("cond: output %d type mismatch: true=%v, false=%v",
				i, trueOutputs[i].dtype, falseOutputs[i].dtype))
			return nil
		}
		if len(trueOutputs[i].shape) != len(falseOutputs[i].shape) {
			b.setErr(fmt.Errorf("cond: output %d shape rank mismatch: true=%v, false=%v",
				i, trueOutputs[i].shape, falseOutputs[i].shape))
			return nil
		}
		for j := range trueOutputs[i].shape {
			if trueOutputs[i].shape[j] != falseOutputs[i].shape[j] {
				b.setErr(fmt.Errorf("cond: output %d shape mismatch: true=%v, false=%v",
					i, trueOutputs[i].shape, falseOutputs[i].shape))
				return nil
			}
		}
	}

	// Build the cond operation with nested blocks
	opInputs := map[string]*milspec.Argument{
		"pred": {
			Arguments: []*milspec.Argument_Binding{{
				Binding: &milspec.Argument_Binding_Name{Name: pred.name},
			}},
		},
	}

	// Build output types (same for both branches)
	outputs := make([]*milspec.NamedValueType, len(trueOutputs))
	for i, v := range trueOutputs {
		tensorType := &milspec.TensorType{
			DataType:   v.dtype,
			Rank:       int64(len(v.shape)),
			Dimensions: make([]*milspec.Dimension, len(v.shape)),
		}
		for j, dim := range v.shape {
			tensorType.Dimensions[j] = &milspec.Dimension{
				Dimension: &milspec.Dimension_Constant{
					Constant: &milspec.Dimension_ConstantDimension{Size: uint64(dim)},
				},
			}
		}
		outputName := b.genName(fmt.Sprintf("cond_out_%d", i))
		outputs[i] = &milspec.NamedValueType{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: tensorType},
			},
		}
	}

	op := &milspec.Operation{
		Type:    "cond",
		Inputs:  opInputs,
		Outputs: outputs,
		Blocks: []*milspec.Block{
			trueBlock.Build(),
			falseBlock.Build(),
		},
	}

	b.operations = append(b.operations, op)

	// Create output values
	resultValues := make([]*Value, len(trueOutputs))
	for i := range trueOutputs {
		v := &Value{
			name:    outputs[i].Name,
			dtype:   trueOutputs[i].dtype,
			shape:   trueOutputs[i].shape,
			builder: b,
		}
		b.values[v.name] = v
		resultValues[i] = v
	}

	return resultValues
}

// WhileLoop executes a body repeatedly while a condition is true.
//
// loopVars: Initial values for the loop variables.
// condFn: Function that takes loop variables and returns a boolean condition.
// bodyFn: Function that takes loop variables and returns updated loop variables.
//
// The condition function must return a boolean scalar.
// The body function must return the same number of outputs as loopVars with matching types.
// Returns the final values of the loop variables when the condition becomes false.
//
// Example:
//
//	// Sum numbers from 0 to n
//	n := b.Const("n", model.Int32, []int64{}, []int32{10})
//	zero := b.Const("zero", model.Int32, []int64{}, []int32{0})
//	loopVars := []*model.Value{zero, zero} // i, sum
//	results := b.WhileLoop(loopVars,
//	    func(bb *model.BlockBuilder, vars []*model.Value) *model.Value {
//	        i := vars[0]
//	        return bb.Less(i, n)
//	    },
//	    func(bb *model.BlockBuilder, vars []*model.Value) []*model.Value {
//	        i, sum := vars[0], vars[1]
//	        one := bb.Const("one", model.Int32, []int64{}, []int32{1})
//	        newI := bb.Add(i, one)
//	        newSum := bb.Add(sum, i)
//	        return []*model.Value{newI, newSum}
//	    },
//	)
func (b *Builder) WhileLoop(loopVars []*Value, condFn LoopCondFunc, bodyFn LoopBodyFunc) []*Value {
	if len(loopVars) == 0 {
		b.setErr(fmt.Errorf("while_loop: at least one loop variable is required"))
		return nil
	}

	// Build condition block
	condBlock := b.NewBlockBuilder()

	// Create block inputs for loop variables in condition block
	condVars := make([]*Value, len(loopVars))
	for i, v := range loopVars {
		condVars[i] = condBlock.BlockInput(fmt.Sprintf("cond_var_%d", i), v.dtype, v.shape)
	}

	// Execute condition function
	condResult := condFn(condBlock, condVars)
	if condBlock.Err() != nil {
		b.setErr(fmt.Errorf("while_loop condition: %w", condBlock.Err()))
		return nil
	}

	// Validate condition result is boolean scalar
	if condResult == nil {
		b.setErr(fmt.Errorf("while_loop: condition function must return a value"))
		return nil
	}
	if condResult.dtype != Bool {
		b.setErr(fmt.Errorf("while_loop: condition must return Bool type, got %v", condResult.dtype))
		return nil
	}
	if len(condResult.shape) != 0 {
		b.setErr(fmt.Errorf("while_loop: condition must return a scalar (shape []), got shape %v", condResult.shape))
		return nil
	}

	// Mark the condition result as the output of the condition block
	condBlock.BlockOutput("cond_result", condResult)

	// Build body block
	bodyBlock := b.NewBlockBuilder()

	// Create block inputs for loop variables in body block
	bodyVars := make([]*Value, len(loopVars))
	for i, v := range loopVars {
		bodyVars[i] = bodyBlock.BlockInput(fmt.Sprintf("body_var_%d", i), v.dtype, v.shape)
	}

	// Execute body function
	bodyResults := bodyFn(bodyBlock, bodyVars)
	if bodyBlock.Err() != nil {
		b.setErr(fmt.Errorf("while_loop body: %w", bodyBlock.Err()))
		return nil
	}

	// Validate body outputs match loop variables
	if len(bodyResults) != len(loopVars) {
		b.setErr(fmt.Errorf("while_loop: body must return %d values (matching loopVars), got %d",
			len(loopVars), len(bodyResults)))
		return nil
	}

	for i := range loopVars {
		if bodyResults[i].dtype != loopVars[i].dtype {
			b.setErr(fmt.Errorf("while_loop: body output %d type mismatch: expected %v, got %v",
				i, loopVars[i].dtype, bodyResults[i].dtype))
			return nil
		}
		if len(bodyResults[i].shape) != len(loopVars[i].shape) {
			b.setErr(fmt.Errorf("while_loop: body output %d shape rank mismatch: expected %v, got %v",
				i, loopVars[i].shape, bodyResults[i].shape))
			return nil
		}
		for j := range loopVars[i].shape {
			if bodyResults[i].shape[j] != loopVars[i].shape[j] {
				b.setErr(fmt.Errorf("while_loop: body output %d shape mismatch: expected %v, got %v",
					i, loopVars[i].shape, bodyResults[i].shape))
				return nil
			}
		}
	}

	// Mark body outputs
	for i, v := range bodyResults {
		bodyBlock.BlockOutput(fmt.Sprintf("body_out_%d", i), v)
	}

	// Build the while_loop operation with nested blocks
	// Loop variables are passed as inputs to the operation
	loopVarBindings := make([]*milspec.Argument_Binding, len(loopVars))
	for i, v := range loopVars {
		if v.isConst {
			loopVarBindings[i] = &milspec.Argument_Binding{
				Binding: &milspec.Argument_Binding_Value{Value: v.constVal},
			}
		} else {
			loopVarBindings[i] = &milspec.Argument_Binding{
				Binding: &milspec.Argument_Binding_Name{Name: v.name},
			}
		}
	}

	opInputs := map[string]*milspec.Argument{
		"loop_vars": {
			Arguments: loopVarBindings,
		},
	}

	// Build output types (same as loop variables)
	outputs := make([]*milspec.NamedValueType, len(loopVars))
	for i, v := range loopVars {
		tensorType := &milspec.TensorType{
			DataType:   v.dtype,
			Rank:       int64(len(v.shape)),
			Dimensions: make([]*milspec.Dimension, len(v.shape)),
		}
		for j, dim := range v.shape {
			tensorType.Dimensions[j] = &milspec.Dimension{
				Dimension: &milspec.Dimension_Constant{
					Constant: &milspec.Dimension_ConstantDimension{Size: uint64(dim)},
				},
			}
		}
		outputName := b.genName(fmt.Sprintf("while_out_%d", i))
		outputs[i] = &milspec.NamedValueType{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: tensorType},
			},
		}
	}

	op := &milspec.Operation{
		Type:    "while_loop",
		Inputs:  opInputs,
		Outputs: outputs,
		Blocks: []*milspec.Block{
			condBlock.Build(),
			bodyBlock.Build(),
		},
	}

	b.operations = append(b.operations, op)

	// Create output values
	resultValues := make([]*Value, len(loopVars))
	for i := range loopVars {
		v := &Value{
			name:    outputs[i].Name,
			dtype:   loopVars[i].dtype,
			shape:   loopVars[i].shape,
			builder: b,
		}
		b.values[v.name] = v
		resultValues[i] = v
	}

	return resultValues
}
