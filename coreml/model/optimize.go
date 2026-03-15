package model

import (
	"sort"

	"github.com/gomlx/go-darwinml/proto/coreml/milspec"
)

// optimizeHighRankOps applies optimization passes to eliminate operations
// with rank > 5, which CoreML's runtime cannot handle.
//
// Four passes run in sequence:
//  1. Decompose reshape(rank≥6)→transpose→reshape(rank≤5) patterns (window attention)
//  2. Collapse consecutive reshapes where the intermediate has rank > 5
//  3. Fuse reshape(rank>5)→tile→reshape into rank≤5 equivalents
//  4. Replace remaining high-rank reshapes that only insert 1-dims with expand_dims
func (b *Builder) optimizeHighRankOps() {
	b.expandHighRankReshapes()

	for b.collapseConsecutiveReshapes() {
	}

	b.fuseReshapeTileReshape()

	b.replaceHighRankReshapeWithExpandDims()
}

// expandHighRankReshapes detects reshape(rank≥6)→transpose→reshape(rank≤5) patterns
// and decomposes them into rank-4 operations that CoreML can handle.
//
// CoreML's runtime limits reshape to rank ≤ 5. Models like Florence-2's DaViT
// vision encoder use window attention that creates rank-6 intermediates:
//
//	[batch, h, w, c] → reshape [batch, h_win, win_h, w_win, win_w, c] (rank 6)
//
// This pass implements Apple's expand_high_rank_reshape_and_transpose algorithm
// from coremltools, decomposing these patterns into equivalent rank-4 operations.
func (b *Builder) expandHighRankReshapes() {
	consumers := buildConsumerMap(b.operations)
	outputSet := buildOutputSet(b.outputs)

	type match struct {
		reshapeIdx     int
		transposeIdx   int
		lastReshapeIdx int
	}
	var matches []match

	for i, op := range b.operations {
		if op.Type != "reshape" {
			continue
		}
		reshapeShape := getOpOutputShape(op)
		if len(reshapeShape) < 6 {
			continue
		}
		reshapeOutName := op.Outputs[0].Name
		if outputSet[reshapeOutName] {
			continue
		}
		reshapeConsumers := consumers[reshapeOutName]
		if len(reshapeConsumers) != 1 {
			continue
		}
		transposeIdx := reshapeConsumers[0]
		transposeOp := b.operations[transposeIdx]
		if transposeOp.Type != "transpose" {
			continue
		}
		transposeOutName := transposeOp.Outputs[0].Name
		if outputSet[transposeOutName] {
			continue
		}
		transposeConsumers := consumers[transposeOutName]
		if len(transposeConsumers) != 1 {
			continue
		}
		lastReshapeIdx := transposeConsumers[0]
		lastReshapeOp := b.operations[lastReshapeIdx]
		if lastReshapeOp.Type != "reshape" {
			continue
		}
		lastReshapeShape := getOpOutputShape(lastReshapeOp)
		if len(lastReshapeShape) > 5 {
			continue
		}

		matches = append(matches, match{
			reshapeIdx:     i,
			transposeIdx:   transposeIdx,
			lastReshapeIdx: lastReshapeIdx,
		})
	}

	if len(matches) == 0 {
		return
	}

	removeSet := make(map[int]bool)
	insertMap := make(map[int][]*milspec.Operation)

	for _, m := range matches {
		reshapeOp := b.operations[m.reshapeIdx]
		transposeOp := b.operations[m.transposeIdx]
		lastReshapeOp := b.operations[m.lastReshapeIdx]

		inputName := getOpInputName(reshapeOp, "x")
		highRankShape := getOpOutputShape(reshapeOp)
		perm := getInlineInt32sAsPerm(transposeOp)
		finalShape := getOpOutputShape(lastReshapeOp)
		finalOutName := lastReshapeOp.Outputs[0].Name
		dtype := getOpOutputDType(reshapeOp)

		newOps := b.decomposeReshapeTranspose(inputName, highRankShape, perm, finalShape, finalOutName, dtype)

		removeSet[m.reshapeIdx] = true
		removeSet[m.transposeIdx] = true
		removeSet[m.lastReshapeIdx] = true
		insertMap[m.reshapeIdx] = newOps
	}

	b.operations = rebuildOps(b.operations, removeSet, insertMap)

	for _, ops := range insertMap {
		for _, op := range ops {
			b.registerOpValue(op)
		}
	}
}

// collapseConsecutiveReshapes finds reshape(rank>5) operations whose single
// consumer is another reshape, and collapses them by bypassing the high-rank
// intermediate. Returns true if any changes were made.
func (b *Builder) collapseConsecutiveReshapes() bool {
	consumers := buildConsumerMap(b.operations)
	outputSet := buildOutputSet(b.outputs)

	changed := false
	removeSet := make(map[int]bool)

	for i, op := range b.operations {
		if op.Type != "reshape" || removeSet[i] {
			continue
		}
		shape := getOpOutputShape(op)
		if len(shape) <= 5 {
			continue
		}
		outName := op.Outputs[0].Name
		if outputSet[outName] {
			continue
		}
		cons := consumers[outName]
		if len(cons) != 1 {
			continue
		}
		consOp := b.operations[cons[0]]
		if consOp.Type != "reshape" {
			continue
		}

		// Collapse: rewrite consumer reshape's input to our input.
		inputName := getOpInputName(op, "x")
		consOp.Inputs["x"] = &milspec.Argument{
			Arguments: []*milspec.Argument_Binding{{
				Binding: &milspec.Argument_Binding_Name{Name: inputName},
			}},
		}
		removeSet[i] = true
		changed = true
	}

	if !changed {
		return false
	}

	var newOps []*milspec.Operation
	for i, op := range b.operations {
		if !removeSet[i] {
			newOps = append(newOps, op)
		}
	}
	b.operations = newOps
	return true
}

// fuseReshapeTileReshape fuses reshape(rank>5)→tile→reshape(rank≤5) chains
// into equivalent rank≤5 operations by merging consecutive singleton dimensions.
//
// For example:
//
//	reshape [1,24,512] → [1,1,1,24,1,512] → tile [R0,R1,R2,1,R4,1] → reshape [...]
//
// becomes:
//
//	reshape [1,24,512] → [1,24,1,512] → tile [R0*R1*R2,1,R4,1] → reshape [...]
func (b *Builder) fuseReshapeTileReshape() {
	consumers := buildConsumerMap(b.operations)
	outputSet := buildOutputSet(b.outputs)

	removeSet := make(map[int]bool)
	insertMap := make(map[int][]*milspec.Operation)

	for i, op := range b.operations {
		if op.Type != "reshape" || removeSet[i] {
			continue
		}
		shape := getOpOutputShape(op)
		if len(shape) <= 5 {
			continue
		}
		outName := op.Outputs[0].Name
		if outputSet[outName] {
			continue
		}
		cons := consumers[outName]
		if len(cons) != 1 {
			continue
		}

		tileIdx := cons[0]
		tileOp := b.operations[tileIdx]
		if tileOp.Type != "tile" {
			continue
		}
		tileOutName := tileOp.Outputs[0].Name
		if outputSet[tileOutName] {
			continue
		}
		tileCons := consumers[tileOutName]
		if len(tileCons) != 1 {
			continue
		}

		lastReshapeIdx := tileCons[0]
		lastReshapeOp := b.operations[lastReshapeIdx]
		if lastReshapeOp.Type != "reshape" {
			continue
		}
		lastReshapeShape := getOpOutputShape(lastReshapeOp)
		if len(lastReshapeShape) > 5 {
			continue
		}

		reps := getInlineInt64s(tileOp, "reps")
		if reps == nil || len(reps) != len(shape) {
			continue
		}

		mergedShape, mergedReps := mergeSingletonDims(shape, reps)
		if len(mergedShape) > 5 {
			continue
		}

		inputName := getOpInputName(op, "x")
		dtype := getOpOutputDType(op)
		finalOutName := lastReshapeOp.Outputs[0].Name

		var newOps []*milspec.Operation

		// Merged reshape (rank ≤ 5).
		reshapeName := b.genName("opt_reshape")
		newOps = append(newOps, makeReshapeOp(inputName, reshapeName, mergedShape, dtype))

		// Compute tile output shape.
		tileOutShape := make([]int64, len(mergedShape))
		for j := range mergedShape {
			tileOutShape[j] = mergedShape[j] * mergedReps[j]
		}

		// Merged tile (rank ≤ 5).
		tileName := b.genName("opt_tile")
		newOps = append(newOps, makeTileOp(reshapeName, tileName, mergedReps, tileOutShape, dtype))

		// Final reshape to original output shape.
		newOps = append(newOps, makeReshapeOp(tileName, finalOutName, lastReshapeShape, dtype))

		removeSet[i] = true
		removeSet[tileIdx] = true
		removeSet[lastReshapeIdx] = true
		insertMap[i] = newOps
	}

	if len(removeSet) == 0 {
		return
	}

	b.operations = rebuildOps(b.operations, removeSet, insertMap)

	for _, ops := range insertMap {
		for _, op := range ops {
			b.registerOpValue(op)
		}
	}
}

// replaceHighRankReshapeWithExpandDims replaces remaining reshape operations
// that produce rank > 5 with expand_dims, when the reshape is purely inserting
// 1-dimensions. This is a fallback for patterns not caught by earlier passes.
func (b *Builder) replaceHighRankReshapeWithExpandDims() {
	shapeMap := make(map[string][]int64)
	for _, v := range b.inputs {
		shapeMap[v.name] = v.shape
	}
	for _, op := range b.operations {
		if len(op.Outputs) > 0 {
			shapeMap[op.Outputs[0].Name] = getOpOutputShape(op)
		}
	}

	for i, op := range b.operations {
		if op.Type != "reshape" {
			continue
		}
		outShape := getOpOutputShape(op)
		if len(outShape) <= 5 {
			continue
		}

		inputName := getOpInputName(op, "x")
		inputShape := shapeMap[inputName]
		if inputShape == nil {
			continue
		}

		axes := findInsertedAxes(inputShape, outShape)
		if axes == nil {
			continue
		}

		b.operations[i] = makeExpandDimsOp(inputName, op.Outputs[0].Name, axes, outShape, getOpOutputDType(op))
	}
}

// mergeSingletonDims merges consecutive size-1 dimensions in a shape,
// multiplying the corresponding tile reps. This reduces rank while
// preserving the mathematical equivalence of the reshape+tile combination.
//
// Only consecutive dims where shape[i]=1 are merged. Non-1 dims are kept
// as-is since merging them would change the tile's data layout semantics.
func mergeSingletonDims(shape, reps []int64) (mergedShape, mergedReps []int64) {
	i := 0
	for i < len(shape) {
		if shape[i] != 1 {
			mergedShape = append(mergedShape, shape[i])
			mergedReps = append(mergedReps, reps[i])
			i++
		} else {
			// Merge consecutive size-1 dims.
			repProd := int64(1)
			for i < len(shape) && shape[i] == 1 {
				repProd *= reps[i]
				i++
			}
			mergedShape = append(mergedShape, 1)
			mergedReps = append(mergedReps, repProd)
		}
	}
	return
}

// findInsertedAxes checks if targetShape is srcShape with 1-dimensions inserted.
// Returns the insertion axes, or nil if not a pure insertion.
func findInsertedAxes(srcShape, targetShape []int64) []int64 {
	if len(targetShape) <= len(srcShape) {
		return nil
	}

	var axes []int64
	srcIdx := 0
	for tgtIdx := 0; tgtIdx < len(targetShape); tgtIdx++ {
		if srcIdx < len(srcShape) && targetShape[tgtIdx] == srcShape[srcIdx] {
			srcIdx++
		} else if targetShape[tgtIdx] == 1 {
			axes = append(axes, int64(tgtIdx))
		} else {
			return nil
		}
	}
	if srcIdx != len(srcShape) {
		return nil
	}
	return axes
}

// decomposeReshapeTranspose decomposes a reshape(rank≥6)→transpose→reshape(rank≤5)
// pattern into rank-4 operations following Apple's coremltools algorithm.
func (b *Builder) decomposeReshapeTranspose(
	inputName string,
	highRankShape []int64,
	perm []int,
	finalShape []int64,
	finalOutName string,
	dtype DType,
) []*milspec.Operation {
	groupAxes := groupConsecutiveAxes(perm)

	groupShape := make([]int64, len(groupAxes))
	for i, axes := range groupAxes {
		prod := int64(1)
		for _, a := range axes {
			prod *= highRankShape[a]
		}
		groupShape[i] = prod
	}

	startGroupAxis := make([]int, len(groupAxes))
	for i, axes := range groupAxes {
		startGroupAxis[i] = axes[0]
	}

	groupAxisOrder := argsortInts(startGroupAxis)

	mergedShape := make([]int64, len(groupShape))
	for i, idx := range groupAxisOrder {
		mergedShape[i] = groupShape[idx]
	}

	sortedStartGroupAxis := make([]int, len(startGroupAxis))
	copy(sortedStartGroupAxis, startGroupAxis)
	sort.Ints(sortedStartGroupAxis)

	mergedPerm := make([]int, len(startGroupAxis))
	for i, v := range startGroupAxis {
		for j, sv := range sortedStartGroupAxis {
			if sv == v {
				mergedPerm[i] = j
				break
			}
		}
	}

	rank := len(mergedPerm)
	var ops []*milspec.Operation
	currentInput := inputName

	if rank < 6 {
		intermName := b.genName("opt_reshape")
		ops = append(ops, makeReshapeOp(currentInput, intermName, mergedShape, dtype))
		currentInput = intermName

		intermName2 := b.genName("opt_transpose")
		ops = append(ops, makeTransposeOp(currentInput, intermName2, intSliceToInt64(mergedPerm), mergedShape, dtype))
		currentInput = intermName2
	} else {
		leadingDim := int64(1)
		memo := make(map[int]bool)

		for i := range rank {
			axis := mergedPerm[i]
			dim := mergedShape[axis]
			memo[axis] = true

			reshapeShape := []int64{
				leadingDim,
				getProd(0, axis, mergedShape, memo),
				dim,
				getProd(axis+1, rank, mergedShape, memo),
			}

			intermName := b.genName("opt_reshape")
			ops = append(ops, makeReshapeOp(currentInput, intermName, reshapeShape, dtype))
			currentInput = intermName

			intermName2 := b.genName("opt_transpose")
			ops = append(ops, makeTransposeOp(currentInput, intermName2, []int64{0, 2, 1, 3}, reshapeShape, dtype))
			currentInput = intermName2

			leadingDim *= dim
		}
	}

	ops = append(ops, makeReshapeOp(currentInput, finalOutName, finalShape, dtype))

	return ops
}

// --- Helpers ---

// buildConsumerMap builds a map from value name to indices of consuming operations.
func buildConsumerMap(operations []*milspec.Operation) map[string][]int {
	consumers := make(map[string][]int)
	for i, op := range operations {
		for _, arg := range op.Inputs {
			for _, binding := range arg.Arguments {
				if name := binding.GetName(); name != "" {
					consumers[name] = append(consumers[name], i)
				}
			}
		}
	}
	return consumers
}

// buildOutputSet builds a set of model output names.
func buildOutputSet(outputs []string) map[string]bool {
	set := make(map[string]bool, len(outputs))
	for _, name := range outputs {
		set[name] = true
	}
	return set
}

// rebuildOps reconstructs the operations list, removing ops in removeSet
// and inserting replacement ops from insertMap at their original positions.
func rebuildOps(operations []*milspec.Operation, removeSet map[int]bool, insertMap map[int][]*milspec.Operation) []*milspec.Operation {
	var newOps []*milspec.Operation
	for i, op := range operations {
		if rOps, ok := insertMap[i]; ok {
			newOps = append(newOps, rOps...)
		}
		if removeSet[i] {
			continue
		}
		newOps = append(newOps, op)
	}
	return newOps
}

// registerOpValue adds an operation's output to b.values.
func (b *Builder) registerOpValue(op *milspec.Operation) {
	outName := op.Outputs[0].Name
	outShape := getOpOutputShape(op)
	outDType := getOpOutputDType(op)
	b.values[outName] = &Value{
		name:    outName,
		dtype:   outDType,
		shape:   outShape,
		builder: b,
	}
}

// groupConsecutiveAxes groups consecutive axes in a permutation.
// E.g., perm [0,1,3,4,2,5] → [[0,1],[3,4],[2],[5]].
func groupConsecutiveAxes(perm []int) [][]int {
	if len(perm) == 0 {
		return nil
	}

	var groups [][]int
	current := []int{perm[0]}

	for i := 1; i < len(perm); i++ {
		if perm[i] == perm[i-1]+1 {
			current = append(current, perm[i])
		} else {
			groups = append(groups, current)
			current = []int{perm[i]}
		}
	}
	groups = append(groups, current)

	return groups
}

// getProd computes the product of shape elements from start to end (exclusive),
// skipping indices in the skip set.
func getProd(start, end int, shape []int64, skip map[int]bool) int64 {
	prod := int64(1)
	for i := start; i < end; i++ {
		if skip[i] {
			continue
		}
		prod *= shape[i]
	}
	return prod
}

// argsortInts returns indices that would sort the input slice.
func argsortInts(a []int) []int {
	indices := make([]int, len(a))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return a[indices[i]] < a[indices[j]]
	})
	return indices
}

// intSliceToInt64 converts []int to []int64.
func intSliceToInt64(a []int) []int64 {
	result := make([]int64, len(a))
	for i, v := range a {
		result[i] = int64(v)
	}
	return result
}

// getOpOutputShape extracts the output shape from an operation's first output.
func getOpOutputShape(op *milspec.Operation) []int64 {
	if len(op.Outputs) == 0 {
		return nil
	}
	tt := op.Outputs[0].Type.GetTensorType()
	if tt == nil {
		return nil
	}
	shape := make([]int64, len(tt.Dimensions))
	for i, dim := range tt.Dimensions {
		if c := dim.GetConstant(); c != nil {
			shape[i] = int64(c.Size)
		}
	}
	return shape
}

// getOpOutputDType extracts the output dtype from an operation's first output.
func getOpOutputDType(op *milspec.Operation) DType {
	if len(op.Outputs) == 0 {
		return Float32
	}
	tt := op.Outputs[0].Type.GetTensorType()
	if tt == nil {
		return Float32
	}
	return tt.DataType
}

// getOpInputName extracts the name reference from an operation's input argument.
func getOpInputName(op *milspec.Operation, paramName string) string {
	arg := op.Inputs[paramName]
	if arg == nil || len(arg.Arguments) == 0 {
		return ""
	}
	return arg.Arguments[0].GetName()
}

// getInlineInt32sAsPerm extracts the "perm" parameter from a transpose op
// as an int slice. The perm is stored as an inline Int32 constant.
func getInlineInt32sAsPerm(op *milspec.Operation) []int {
	arg := op.Inputs["perm"]
	if arg == nil || len(arg.Arguments) == 0 {
		return nil
	}
	val := arg.Arguments[0].GetValue()
	if val == nil {
		return nil
	}
	imm := val.GetImmediateValue()
	if imm == nil {
		return nil
	}
	tensor := imm.GetTensor()
	if tensor == nil {
		return nil
	}
	ints := tensor.GetInts()
	if ints == nil {
		return nil
	}
	result := make([]int, len(ints.Values))
	for i, v := range ints.Values {
		result[i] = int(v)
	}
	return result
}

// getInlineInt64s extracts an inline Int32 constant argument as []int64.
func getInlineInt64s(op *milspec.Operation, paramName string) []int64 {
	arg := op.Inputs[paramName]
	if arg == nil || len(arg.Arguments) == 0 {
		return nil
	}
	val := arg.Arguments[0].GetValue()
	if val == nil {
		return nil
	}
	imm := val.GetImmediateValue()
	if imm == nil {
		return nil
	}
	tensor := imm.GetTensor()
	if tensor == nil {
		return nil
	}
	ints := tensor.GetInts()
	if ints == nil {
		return nil
	}
	result := make([]int64, len(ints.Values))
	for i, v := range ints.Values {
		result[i] = int64(v)
	}
	return result
}

// makeReshapeOp creates a MIL reshape operation in protobuf form.
func makeReshapeOp(inputName, outputName string, shape []int64, dtype DType) *milspec.Operation {
	shapeConst := createValue(Int32, []int64{int64(len(shape))}, toInt32Slice(shape))
	outType := makeTensorType(dtype, shape)

	return &milspec.Operation{
		Type: "reshape",
		Inputs: map[string]*milspec.Argument{
			"x": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: inputName},
				}},
			},
			"shape": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: shapeConst},
				}},
			},
		},
		Outputs: []*milspec.NamedValueType{{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: outType},
			},
		}},
	}
}

// makeTransposeOp creates a MIL transpose operation in protobuf form.
func makeTransposeOp(inputName, outputName string, perm []int64, inputShape []int64, dtype DType) *milspec.Operation {
	permConst := createValue(Int32, []int64{int64(len(perm))}, toInt32Slice(perm))

	outShape := make([]int64, len(perm))
	for i, p := range perm {
		outShape[i] = inputShape[p]
	}

	outType := makeTensorType(dtype, outShape)

	return &milspec.Operation{
		Type: "transpose",
		Inputs: map[string]*milspec.Argument{
			"x": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: inputName},
				}},
			},
			"perm": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: permConst},
				}},
			},
		},
		Outputs: []*milspec.NamedValueType{{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: outType},
			},
		}},
	}
}

// makeTileOp creates a MIL tile operation in protobuf form.
func makeTileOp(inputName, outputName string, reps []int64, outShape []int64, dtype DType) *milspec.Operation {
	repsConst := createValue(Int32, []int64{int64(len(reps))}, toInt32Slice(reps))
	outType := makeTensorType(dtype, outShape)

	return &milspec.Operation{
		Type: "tile",
		Inputs: map[string]*milspec.Argument{
			"x": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: inputName},
				}},
			},
			"reps": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: repsConst},
				}},
			},
		},
		Outputs: []*milspec.NamedValueType{{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: outType},
			},
		}},
	}
}

// makeExpandDimsOp creates a MIL expand_dims operation in protobuf form.
func makeExpandDimsOp(inputName, outputName string, axes []int64, outShape []int64, dtype DType) *milspec.Operation {
	axesConst := createValue(Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	outType := makeTensorType(dtype, outShape)

	return &milspec.Operation{
		Type: "expand_dims",
		Inputs: map[string]*milspec.Argument{
			"x": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Name{Name: inputName},
				}},
			},
			"axes": {
				Arguments: []*milspec.Argument_Binding{{
					Binding: &milspec.Argument_Binding_Value{Value: axesConst},
				}},
			},
		},
		Outputs: []*milspec.NamedValueType{{
			Name: outputName,
			Type: &milspec.ValueType{
				Type: &milspec.ValueType_TensorType{TensorType: outType},
			},
		}},
	}
}

// makeTensorType creates a TensorType with constant dimensions.
func makeTensorType(dtype DType, shape []int64) *milspec.TensorType {
	dims := make([]*milspec.Dimension, len(shape))
	for i, s := range shape {
		dims[i] = &milspec.Dimension{
			Dimension: &milspec.Dimension_Constant{
				Constant: &milspec.Dimension_ConstantDimension{Size: uint64(s)},
			},
		}
	}
	return &milspec.TensorType{
		DataType:   dtype,
		Rank:       int64(len(shape)),
		Dimensions: dims,
	}
}
