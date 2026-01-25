// Package model provides a fluent API for constructing CoreML MIL programs.
//
// MIL (Machine Learning Intermediate Language) is CoreML's graph representation.
// This package enables programmatic construction of MIL programs that can be
// compiled and executed on Apple devices using CoreML.
//
// Example usage:
//
//	b := model.NewBuilder("main")
//	x := b.Input("x", model.Float32, 2, 3)
//	y := b.Input("y", model.Float32, 3, 4)
//	z := b.MatMul(x, y)
//	b.Output("z", z)
//	program := b.Build()
package model

import (
	"fmt"

	"github.com/gomlx/go-coreml/proto/coreml/milspec"
)

// DType represents a data type for tensors.
type DType = milspec.DataType

// Common data type constants.
const (
	Float16 = milspec.DataType_FLOAT16
	Float32 = milspec.DataType_FLOAT32
	Float64 = milspec.DataType_FLOAT64
	Int8    = milspec.DataType_INT8
	Int16   = milspec.DataType_INT16
	Int32   = milspec.DataType_INT32
	Int64   = milspec.DataType_INT64
	Bool    = milspec.DataType_BOOL
	String  = milspec.DataType_STRING
)

// Value represents a named value in the MIL graph.
// It can be an input, an operation output, or a constant.
type Value struct {
	name     string
	dtype    DType
	shape    []int64
	builder  *Builder
	isConst  bool
	constVal *milspec.Value
}

// Name returns the value's name.
func (v *Value) Name() string {
	return v.name
}

// Shape returns the value's shape.
func (v *Value) Shape() []int64 {
	return v.shape
}

// DType returns the value's data type.
func (v *Value) DType() DType {
	return v.dtype
}

// IsConst returns true if this value is a constant.
func (v *Value) IsConst() bool {
	return v.isConst
}

// Builder constructs MIL programs.
type Builder struct {
	name       string
	opset      string
	inputs     []*Value
	outputs    []string
	operations []*milspec.Operation
	values     map[string]*Value
	nextID     int
	err        error // first error encountered during building
}

// Err returns the first error encountered during building, if any.
// Callers should check this after constructing a graph to ensure
// all operations were valid.
func (b *Builder) Err() error {
	return b.err
}

// setErr records the first error encountered.
func (b *Builder) setErr(err error) {
	if b.err == nil {
		b.err = err
	}
}

// NewBuilder creates a new MIL program builder.
// The name is used as the function name in the program.
func NewBuilder(name string) *Builder {
	return &Builder{
		name:   name,
		opset:  "CoreML7", // Default opset
		values: make(map[string]*Value),
	}
}

// SetOpset sets the operation set version (e.g., "CoreML5", "CoreML6", "CoreML7").
func (b *Builder) SetOpset(opset string) *Builder {
	b.opset = opset
	return b
}

// genName generates a unique name for intermediate values.
func (b *Builder) genName(prefix string) string {
	name := fmt.Sprintf("%s_%d", prefix, b.nextID)
	b.nextID++
	return name
}

// Input adds an input to the program.
func (b *Builder) Input(name string, dtype DType, shape ...int64) *Value {
	v := &Value{
		name:    name,
		dtype:   dtype,
		shape:   shape,
		builder: b,
	}
	b.inputs = append(b.inputs, v)
	b.values[name] = v
	return v
}

// PlaceholderValue creates a Value that is NOT added as a model input.
// This is used for closure parameters which receive values via block inputs
// in control flow operations, not from model inputs.
func (b *Builder) PlaceholderValue(name string, dtype DType, shape ...int64) *Value {
	v := &Value{
		name:    name,
		dtype:   dtype,
		shape:   shape,
		builder: b,
	}
	// Note: intentionally NOT added to b.inputs
	b.values[name] = v
	return v
}

// Output marks a value as an output of the program.
// The output will be named with the given name, which can be different
// from the value's internal name.
func (b *Builder) Output(name string, v *Value) {
	// If the name matches the internal value name, use it directly
	if name == v.name {
		b.outputs = append(b.outputs, name)
		return
	}

	// Otherwise, add an identity operation to rename the output
	renamed := b.Identity(name, v)
	b.outputs = append(b.outputs, renamed.name)
}

// Identity creates an identity operation that copies a value with a new name.
// This is useful for renaming outputs.
func (b *Builder) Identity(name string, x *Value) *Value {
	return b.addOp("identity", map[string]*Value{
		"x": x,
	}, name, x.dtype, x.shape)
}

// Const creates a constant tensor value.
func (b *Builder) Const(name string, dtype DType, shape []int64, data interface{}) *Value {
	val := createValue(dtype, shape, data)
	v := &Value{
		name:     name,
		dtype:    dtype,
		shape:    shape,
		builder:  b,
		isConst:  true,
		constVal: val,
	}
	b.values[name] = v
	return v
}

// createValue creates a MIL Value from Go data.
func createValue(dtype DType, shape []int64, data interface{}) *milspec.Value {
	tensorType := &milspec.TensorType{
		DataType:   dtype,
		Rank:       int64(len(shape)),
		Dimensions: make([]*milspec.Dimension, len(shape)),
	}
	for i, dim := range shape {
		tensorType.Dimensions[i] = &milspec.Dimension{
			Dimension: &milspec.Dimension_Constant{
				Constant: &milspec.Dimension_ConstantDimension{Size: uint64(dim)},
			},
		}
	}

	var tensorVal *milspec.TensorValue
	switch d := data.(type) {
	case []float32:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Floats{
				Floats: &milspec.TensorValue_RepeatedFloats{Values: d},
			},
		}
	case []float64:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Doubles{
				Doubles: &milspec.TensorValue_RepeatedDoubles{Values: d},
			},
		}
	case []int32:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Ints{
				Ints: &milspec.TensorValue_RepeatedInts{Values: d},
			},
		}
	case []int64:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_LongInts{
				LongInts: &milspec.TensorValue_RepeatedLongInts{Values: d},
			},
		}
	case []bool:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Bools{
				Bools: &milspec.TensorValue_RepeatedBools{Values: d},
			},
		}
	case string:
		// Handle string as a scalar string value
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Strings{
				Strings: &milspec.TensorValue_RepeatedStrings{Values: []string{d}},
			},
		}
	case []string:
		tensorVal = &milspec.TensorValue{
			Value: &milspec.TensorValue_Strings{
				Strings: &milspec.TensorValue_RepeatedStrings{Values: d},
			},
		}
	}

	return &milspec.Value{
		Type: &milspec.ValueType{
			Type: &milspec.ValueType_TensorType{TensorType: tensorType},
		},
		Value: &milspec.Value_ImmediateValue_{
			ImmediateValue: &milspec.Value_ImmediateValue{
				Value: &milspec.Value_ImmediateValue_Tensor{Tensor: tensorVal},
			},
		},
	}
}

// addOpWithListArg adds an operation to the builder with support for list arguments.
// listArgs maps parameter names to slices of Values (for operations like concat).
func (b *Builder) addOpWithListArg(opType string, inputs map[string]*Value, listArgs map[string][]*Value, outputName string, outputDtype DType, outputShape []int64) *Value {
	// Build input arguments
	opInputs := make(map[string]*milspec.Argument)

	// Handle regular scalar inputs
	for name, v := range inputs {
		if v.isConst {
			// Constant value - embed the value directly
			opInputs[name] = &milspec.Argument{
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: v.constVal},
				}},
			}
		} else {
			// Reference to another value
			opInputs[name] = &milspec.Argument{
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: v.name},
				}},
			}
		}
	}

	// Handle list arguments (for concat, etc.)
	for name, values := range listArgs {
		bindings := make([]*milspec.Argument_Binding, len(values))
		for i, v := range values {
			if v.isConst {
				bindings[i] = &milspec.Argument_Binding{
					Binding: &milspec.Argument_Binding_Value{Value: v.constVal},
				}
			} else {
				bindings[i] = &milspec.Argument_Binding{
					Binding: &milspec.Argument_Binding_Name{Name: v.name},
				}
			}
		}
		opInputs[name] = &milspec.Argument{
			Arguments: bindings,
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

	b.operations = append(b.operations, op)

	v := &Value{
		name:    outputName,
		dtype:   outputDtype,
		shape:   outputShape,
		builder: b,
	}
	b.values[outputName] = v
	return v
}

// addOp adds an operation to the builder and returns the output value.
func (b *Builder) addOp(opType string, inputs map[string]*Value, outputName string, outputDtype DType, outputShape []int64) *Value {
	// Build input arguments
	opInputs := make(map[string]*milspec.Argument)
	for name, v := range inputs {
		if v.isConst {
			// Constant value - embed the value directly
			opInputs[name] = &milspec.Argument{
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: v.constVal},
				}},
			}
		} else {
			// Reference to another value
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

	b.operations = append(b.operations, op)

	v := &Value{
		name:    outputName,
		dtype:   outputDtype,
		shape:   outputShape,
		builder: b,
	}
	b.values[outputName] = v
	return v
}

// InputSpecs returns the input feature specifications.
func (b *Builder) InputSpecs() []FeatureSpec {
	specs := make([]FeatureSpec, len(b.inputs))
	for i, v := range b.inputs {
		specs[i] = FeatureSpec{
			Name:  v.name,
			DType: v.dtype,
			Shape: v.shape,
		}
	}
	return specs
}

// OutputSpecs returns the output feature specifications.
func (b *Builder) OutputSpecs() []FeatureSpec {
	specs := make([]FeatureSpec, len(b.outputs))
	for i, name := range b.outputs {
		v := b.values[name]
		specs[i] = FeatureSpec{
			Name:  name,
			DType: v.dtype,
			Shape: v.shape,
		}
	}
	return specs
}

// Program is an alias for the MIL Program type.
type Program = milspec.Program

// Build constructs the final MIL Program.
func (b *Builder) Build() *Program {
	// Build function inputs
	inputs := make([]*milspec.NamedValueType, len(b.inputs))
	for i, v := range b.inputs {
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

	// Build the block
	block := &milspec.Block{
		Outputs:    b.outputs,
		Operations: b.operations,
	}

	// Build the function
	function := &milspec.Function{
		Inputs: inputs,
		Opset:  b.opset,
		BlockSpecializations: map[string]*milspec.Block{
			b.opset: block,
		},
	}

	// Build the program
	program := &milspec.Program{
		Version: 1,
		Functions: map[string]*milspec.Function{
			b.name: function,
		},
	}

	return program
}
