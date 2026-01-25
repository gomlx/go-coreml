package model

import (
	"testing"

	"github.com/gomlx/go-coreml/proto/coreml/milspec"
)

func TestBlockBuilder_BasicOps(t *testing.T) {
	b := NewBuilder("test")

	// Create a block builder
	bb := b.NewBlockBuilder()

	// Create block inputs
	x := bb.BlockInput("x", Float32, []int64{2, 3})
	y := bb.BlockInput("y", Float32, []int64{2, 3})

	// Test basic operations within the block
	sum := bb.Add(x, y)
	if sum.dtype != Float32 {
		t.Errorf("Add dtype = %v, want %v", sum.dtype, Float32)
	}
	if len(sum.shape) != 2 || sum.shape[0] != 2 || sum.shape[1] != 3 {
		t.Errorf("Add shape = %v, want [2, 3]", sum.shape)
	}

	diff := bb.Sub(x, y)
	if diff.dtype != Float32 {
		t.Errorf("Sub dtype = %v, want %v", diff.dtype, Float32)
	}

	prod := bb.Mul(x, y)
	if prod.dtype != Float32 {
		t.Errorf("Mul dtype = %v, want %v", prod.dtype, Float32)
	}

	// Mark output
	bb.BlockOutput("result", sum)

	// Build the block
	block := bb.Build()

	// Verify block structure
	if len(block.Inputs) != 2 {
		t.Errorf("Block inputs count = %d, want 2", len(block.Inputs))
	}
	if len(block.Outputs) != 1 {
		t.Errorf("Block outputs count = %d, want 1", len(block.Outputs))
	}
	if block.Outputs[0] != "result" {
		t.Errorf("Block output name = %q, want %q", block.Outputs[0], "result")
	}
}

func TestBlockBuilder_ComparisonOps(t *testing.T) {
	b := NewBuilder("test")
	bb := b.NewBlockBuilder()

	x := bb.BlockInput("x", Float32, []int64{3})
	y := bb.BlockInput("y", Float32, []int64{3})

	// Test comparison operations
	less := bb.Less(x, y)
	if less.dtype != Bool {
		t.Errorf("Less dtype = %v, want %v", less.dtype, Bool)
	}

	greater := bb.Greater(x, y)
	if greater.dtype != Bool {
		t.Errorf("Greater dtype = %v, want %v", greater.dtype, Bool)
	}
}

func TestBlockBuilder_Const(t *testing.T) {
	b := NewBuilder("test")
	bb := b.NewBlockBuilder()

	// Create a constant within the block
	c := bb.Const("const_val", Float32, []int64{}, []float32{1.0})

	if c.dtype != Float32 {
		t.Errorf("Const dtype = %v, want %v", c.dtype, Float32)
	}
	if len(c.shape) != 0 {
		t.Errorf("Const shape = %v, want []", c.shape)
	}
	if !c.isConst {
		t.Error("Const isConst = false, want true")
	}
}

func TestCond_BasicBranching(t *testing.T) {
	b := NewBuilder("cond_test")

	// Create inputs
	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)
	y := b.Input("y", Float32, 10)

	// Create conditional
	results := b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			// True branch: add x and y
			sum := bb.Add(x, y)
			return []*Value{sum}
		},
		func(bb *BlockBuilder) []*Value {
			// False branch: multiply x and y
			prod := bb.Mul(x, y)
			return []*Value{prod}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	if len(results) != 1 {
		t.Fatalf("Cond results count = %d, want 1", len(results))
	}

	if results[0].dtype != Float32 {
		t.Errorf("Result dtype = %v, want %v", results[0].dtype, Float32)
	}
	if len(results[0].shape) != 1 || results[0].shape[0] != 10 {
		t.Errorf("Result shape = %v, want [10]", results[0].shape)
	}

	// Build the program and check the structure
	b.Output("result", results[0])
	program := b.Build()

	// Verify the program has the cond operation
	fn := program.Functions["cond_test"]
	if fn == nil {
		t.Fatal("Function not found in program")
	}

	block := fn.BlockSpecializations["CoreML7"]
	if block == nil {
		t.Fatal("Block not found in function")
	}

	// Find the cond operation
	var condOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "cond" {
			condOp = op
			break
		}
	}

	if condOp == nil {
		t.Fatal("Cond operation not found in block")
	}

	// Verify nested blocks
	if len(condOp.Blocks) != 2 {
		t.Errorf("Cond operation has %d blocks, want 2", len(condOp.Blocks))
	}
}

func TestCond_MultipleOutputs(t *testing.T) {
	b := NewBuilder("cond_multi_out")

	pred := b.Input("pred", Bool)
	a := b.Input("a", Float32, 5)
	bVal := b.Input("b", Float32, 5)

	results := b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			// Return both sum and diff
			sum := bb.Add(a, bVal)
			diff := bb.Sub(a, bVal)
			return []*Value{sum, diff}
		},
		func(bb *BlockBuilder) []*Value {
			// Return both prod and the constant
			prod := bb.Mul(a, bVal)
			one := bb.Const("one", Float32, []int64{5}, []float32{1, 1, 1, 1, 1})
			return []*Value{prod, one}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	if len(results) != 2 {
		t.Fatalf("Cond results count = %d, want 2", len(results))
	}
}

func TestCond_ErrorOnTypeMismatch(t *testing.T) {
	b := NewBuilder("cond_type_error")

	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)
	y := b.Input("y", Int32, 10)

	_ = b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{y} // Type mismatch!
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for type mismatch, got nil")
	}
}

func TestCond_ErrorOnShapeMismatch(t *testing.T) {
	b := NewBuilder("cond_shape_error")

	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)
	y := b.Input("y", Float32, 5) // Shape mismatch!

	_ = b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{y}
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for shape mismatch, got nil")
	}
}

func TestCond_ErrorOnOutputCountMismatch(t *testing.T) {
	b := NewBuilder("cond_count_error")

	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)
	y := b.Input("y", Float32, 10)

	_ = b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			return []*Value{x, y} // Two outputs
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{x} // One output!
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for output count mismatch, got nil")
	}
}

func TestCond_ErrorOnNonBoolPred(t *testing.T) {
	b := NewBuilder("cond_pred_error")

	pred := b.Input("pred", Float32) // Not Bool!
	x := b.Input("x", Float32, 10)

	_ = b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for non-Bool predicate, got nil")
	}
}

func TestCond_ErrorOnNonScalarPred(t *testing.T) {
	b := NewBuilder("cond_scalar_error")

	pred := b.Input("pred", Bool, 10) // Not scalar!
	x := b.Input("x", Float32, 10)

	_ = b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for non-scalar predicate, got nil")
	}
}

func TestWhileLoop_BasicLoop(t *testing.T) {
	b := NewBuilder("while_test")

	// Create initial loop variables: i=0, sum=0
	zero := b.Const("zero", Int32, []int64{}, []int32{0})
	n := b.Const("n", Int32, []int64{}, []int32{10})

	// Loop: while i < n: sum += i; i++
	results := b.WhileLoop(
		[]*Value{zero, zero}, // [i, sum]
		func(bb *BlockBuilder, vars []*Value) *Value {
			i := vars[0]
			return bb.Less(i, n) // Continue while i < n
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			i, sum := vars[0], vars[1]
			one := bb.Const("one", Int32, []int64{}, []int32{1})
			newI := bb.Add(i, one)
			newSum := bb.Add(sum, i)
			return []*Value{newI, newSum}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	if len(results) != 2 {
		t.Fatalf("WhileLoop results count = %d, want 2", len(results))
	}

	// Both outputs should be Int32 scalar
	for i, r := range results {
		if r.dtype != Int32 {
			t.Errorf("Result[%d] dtype = %v, want %v", i, r.dtype, Int32)
		}
		if len(r.shape) != 0 {
			t.Errorf("Result[%d] shape = %v, want []", i, r.shape)
		}
	}

	// Build the program and check the structure
	b.Output("final_i", results[0])
	b.Output("final_sum", results[1])
	program := b.Build()

	// Verify the program has the while_loop operation
	fn := program.Functions["while_test"]
	block := fn.BlockSpecializations["CoreML7"]

	var whileOp *milspec.Operation
	for _, op := range block.Operations {
		if op.Type == "while_loop" {
			whileOp = op
			break
		}
	}

	if whileOp == nil {
		t.Fatal("WhileLoop operation not found in block")
	}

	// Verify nested blocks (cond and body)
	if len(whileOp.Blocks) != 2 {
		t.Errorf("WhileLoop operation has %d blocks, want 2", len(whileOp.Blocks))
	}
}

func TestWhileLoop_ErrorOnNonBoolCondition(t *testing.T) {
	b := NewBuilder("while_cond_error")

	zero := b.Const("zero", Int32, []int64{}, []int32{0})

	_ = b.WhileLoop(
		[]*Value{zero},
		func(bb *BlockBuilder, vars []*Value) *Value {
			// Return non-Bool value
			return vars[0]
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			return vars
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for non-Bool condition, got nil")
	}
}

func TestWhileLoop_ErrorOnNonScalarCondition(t *testing.T) {
	b := NewBuilder("while_scalar_error")

	zeros := b.Const("zeros", Int32, []int64{5}, []int32{0, 0, 0, 0, 0})
	five := b.Const("five", Int32, []int64{5}, []int32{5, 5, 5, 5, 5})

	_ = b.WhileLoop(
		[]*Value{zeros},
		func(bb *BlockBuilder, vars []*Value) *Value {
			// Return non-scalar Bool
			return bb.Less(vars[0], five)
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			return vars
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for non-scalar condition, got nil")
	}
}

func TestWhileLoop_ErrorOnBodyOutputCountMismatch(t *testing.T) {
	b := NewBuilder("while_count_error")

	zero := b.Const("zero", Int32, []int64{}, []int32{0})
	n := b.Const("n", Int32, []int64{}, []int32{10})

	_ = b.WhileLoop(
		[]*Value{zero, zero}, // Two loop vars
		func(bb *BlockBuilder, vars []*Value) *Value {
			return bb.Less(vars[0], n)
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			// Only return one value instead of two!
			return []*Value{vars[0]}
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for body output count mismatch, got nil")
	}
}

func TestWhileLoop_ErrorOnBodyOutputTypeMismatch(t *testing.T) {
	b := NewBuilder("while_type_error")

	zero := b.Const("zero", Int32, []int64{}, []int32{0})
	n := b.Const("n", Int32, []int64{}, []int32{10})

	_ = b.WhileLoop(
		[]*Value{zero},
		func(bb *BlockBuilder, vars []*Value) *Value {
			return bb.Less(vars[0], n)
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			// Return wrong type (Bool instead of Int32)
			return []*Value{bb.Less(vars[0], n)}
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for body output type mismatch, got nil")
	}
}

func TestWhileLoop_ErrorOnEmptyLoopVars(t *testing.T) {
	b := NewBuilder("while_empty_error")

	_ = b.WhileLoop(
		[]*Value{}, // Empty loop vars
		func(bb *BlockBuilder, vars []*Value) *Value {
			return bb.Const("cond", Bool, []int64{}, []bool{true})
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			return vars
		},
	)

	if b.Err() == nil {
		t.Error("Expected error for empty loop vars, got nil")
	}
}

func TestBlockBuilder_GetValue(t *testing.T) {
	b := NewBuilder("test")

	// Create a value in the parent scope
	parentVal := b.Input("parent_input", Float32, 5)

	bb := b.NewBlockBuilder()
	blockVal := bb.BlockInput("block_input", Float32, []int64{5})

	// Should find block-local value
	v, found := bb.GetValue("block_input")
	if !found || v != blockVal {
		t.Error("GetValue failed to find block-local value")
	}

	// Should find parent value
	v, found = bb.GetValue("parent_input")
	if !found || v != parentVal {
		t.Error("GetValue failed to find parent value")
	}

	// Should not find non-existent value
	_, found = bb.GetValue("nonexistent")
	if found {
		t.Error("GetValue found non-existent value")
	}
}

func TestBlockBuilder_Select(t *testing.T) {
	b := NewBuilder("test")
	bb := b.NewBlockBuilder()

	cond := bb.BlockInput("cond", Bool, []int64{3})
	a := bb.BlockInput("a", Float32, []int64{3})
	bVal := bb.BlockInput("b", Float32, []int64{3})

	result := bb.Select(cond, a, bVal)

	if result.dtype != Float32 {
		t.Errorf("Select dtype = %v, want %v", result.dtype, Float32)
	}
	if len(result.shape) != 1 || result.shape[0] != 3 {
		t.Errorf("Select shape = %v, want [3]", result.shape)
	}
}

func TestBlockBuilder_LogicalAnd(t *testing.T) {
	b := NewBuilder("test")
	bb := b.NewBlockBuilder()

	x := bb.BlockInput("x", Bool, []int64{3})
	y := bb.BlockInput("y", Bool, []int64{3})

	result := bb.LogicalAnd(x, y)

	if result.dtype != Bool {
		t.Errorf("LogicalAnd dtype = %v, want %v", result.dtype, Bool)
	}
	if len(result.shape) != 1 || result.shape[0] != 3 {
		t.Errorf("LogicalAnd shape = %v, want [3]", result.shape)
	}
}

// TestCondWithParentScopeValues tests that values from parent scope can be used in branches
func TestCondWithParentScopeValues(t *testing.T) {
	b := NewBuilder("cond_parent_scope")

	pred := b.Input("pred", Bool)
	x := b.Input("x", Float32, 10)

	// Constant in parent scope
	scale := b.Const("scale", Float32, []int64{}, []float32{2.0})

	results := b.Cond(pred,
		func(bb *BlockBuilder) []*Value {
			// Use parent scope value
			scaled := bb.Mul(x, scale)
			return []*Value{scaled}
		},
		func(bb *BlockBuilder) []*Value {
			return []*Value{x}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	if len(results) != 1 {
		t.Fatalf("Cond results count = %d, want 1", len(results))
	}
}

// TestWhileLoopWithTensorLoopVars tests while loop with tensor loop variables
func TestWhileLoopWithTensorLoopVars(t *testing.T) {
	b := NewBuilder("while_tensor_vars")

	// Initialize loop variable as a tensor
	zeros := b.Const("zeros", Float32, []int64{5}, []float32{0, 0, 0, 0, 0})
	one := b.Const("one", Float32, []int64{}, []float32{1.0})
	counter := b.Const("counter", Int32, []int64{}, []int32{0})
	maxIter := b.Const("max_iter", Int32, []int64{}, []int32{5})

	results := b.WhileLoop(
		[]*Value{counter, zeros}, // [counter, tensor]
		func(bb *BlockBuilder, vars []*Value) *Value {
			cnt := vars[0]
			return bb.Less(cnt, maxIter)
		},
		func(bb *BlockBuilder, vars []*Value) []*Value {
			cnt, tensor := vars[0], vars[1]
			oneInt := bb.Const("one_int", Int32, []int64{}, []int32{1})
			newCnt := bb.Add(cnt, oneInt)
			newTensor := bb.Add(tensor, one)
			return []*Value{newCnt, newTensor}
		},
	)

	if b.Err() != nil {
		t.Fatalf("Builder error: %v", b.Err())
	}

	if len(results) != 2 {
		t.Fatalf("WhileLoop results count = %d, want 2", len(results))
	}

	// First result should be Int32 scalar
	if results[0].dtype != Int32 {
		t.Errorf("Result[0] dtype = %v, want %v", results[0].dtype, Int32)
	}

	// Second result should be Float32 tensor [5]
	if results[1].dtype != Float32 {
		t.Errorf("Result[1] dtype = %v, want %v", results[1].dtype, Float32)
	}
	if len(results[1].shape) != 1 || results[1].shape[0] != 5 {
		t.Errorf("Result[1] shape = %v, want [5]", results[1].shape)
	}
}
