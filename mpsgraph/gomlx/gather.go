// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import (
	"slices"

	"github.com/gomlx/go-darwinml/mpsgraph/gomlx/internal/bridge"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Gather implements the XLA-style Gather operation.
//
// Strategy: decompose the general XLA Gather into MPSGraph operations.
//
// For the common embedding-lookup pattern (single indexed axis, gather full slices),
// we use MPSGraph's gatherAlongAxis. For the general case, we decompose into
// reshape + gatherND + reshape.
func (f *Function) Gather(
	operand, startIndices backends.Value,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Value, error) {
	operandNode, err := f.resolveNode(operand)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: operand")
	}
	indicesNode, err := f.resolveNode(startIndices)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: startIndices")
	}

	outShape, err := shapeinference.Gather(
		operandNode.shape, indicesNode.shape,
		indexVectorAxis, offsetOutputAxes, collapsedSliceAxes,
		startIndexMap, sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, errors.Wrap(err, "Gather")
	}

	// Try fast path: simple embedding lookup along a single axis.
	if result, ok := f.gatherEmbeddingLookup(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape); ok {
		return result, nil
	}

	// General path: decompose XLA Gather into MPSGraph operations.
	return f.gatherGeneral(operandNode, indicesNode, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes, outShape)
}

// gatherEmbeddingLookup handles the common pattern: gather full slices along one axis.
// Pattern: startIndexMap = [axis], sliceSizes[axis] = 1, collapsedSliceAxes = [axis],
// and all other sliceSizes are the full dimension.
func (f *Function) gatherEmbeddingLookup(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (backends.Value, bool) {
	// Must have exactly one axis in startIndexMap.
	if len(startIndexMap) != 1 {
		return nil, false
	}
	gatherAxis := startIndexMap[0]

	// sliceSizes[gatherAxis] must be 1.
	if sliceSizes[gatherAxis] != 1 {
		return nil, false
	}

	// gatherAxis must be in collapsedSliceAxes.
	if !slices.Contains(collapsedSliceAxes, gatherAxis) {
		return nil, false
	}

	// All other sliceSizes must be the full dimension.
	for i, sz := range sliceSizes {
		if i == gatherAxis {
			continue
		}
		if sz != operandNode.shape.Dimensions[i] {
			return nil, false
		}
	}

	// offsetOutputAxes must be contiguous ascending.
	for i := 1; i < len(offsetOutputAxes); i++ {
		if offsetOutputAxes[i] != offsetOutputAxes[i-1]+1 {
			return nil, false
		}
	}

	// indexVectorAxis should be at the end (or startIndices.Rank() for implicit size-1 axis).
	idxRank := indicesNode.shape.Rank()
	if indexVectorAxis != idxRank-1 && indexVectorAxis != idxRank {
		return nil, false
	}

	// Remove the indexVector axis (size 1) from indices to get batch dims.
	idxTensor := indicesNode.tensor
	var err error

	// Compute batch dimensions (all index dims except indexVectorAxis).
	batchDims := make([]int64, 0, idxRank)
	for i, d := range indicesNode.shape.Dimensions {
		if i != indexVectorAxis || indexVectorAxis >= idxRank {
			batchDims = append(batchDims, int64(d))
		}
	}
	if len(batchDims) == 0 {
		batchDims = []int64{1} // Scalar index becomes [1].
	}

	if indexVectorAxis < idxRank {
		// Remove the indexVector axis (size 1) by reshaping.
		idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
		if err != nil {
			return nil, false
		}
	}

	// MPSGraph's gatherAlongAxis requires indices to have the same rank as operand.
	// We need to reshape indices to match operand rank, inserting size-1 for non-gather axes,
	// then broadcast to match the full gather output shape for gatherAlongAxis.
	operandRank := operandNode.shape.Rank()

	// Build the shape for indices matching operand rank: batch dims along gatherAxis,
	// size-1 for all other axes.
	idxFullShape := make([]int64, operandRank)
	for i := range operandRank {
		if i == gatherAxis {
			// Flatten all batch dims into this axis.
			batchSize := int64(1)
			for _, d := range batchDims {
				batchSize *= d
			}
			idxFullShape[i] = batchSize
		} else {
			idxFullShape[i] = 1
		}
	}
	idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
	if err != nil {
		return nil, false
	}

	// Broadcast indices to match the gatherAlongAxis output shape:
	// same as operand but with gatherAxis replaced by the batch size.
	broadcastShape := make([]int64, operandRank)
	for i, d := range operandNode.shape.Dimensions {
		if i == gatherAxis {
			broadcastShape[i] = idxFullShape[i]
		} else {
			broadcastShape[i] = int64(d)
		}
	}
	idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
	if err != nil {
		return nil, false
	}

	// Use gatherAlongAxis — indices and operand now have the same rank.
	result, err := f.ctx().GatherAlongAxis(operandNode.tensor, idxTensor, gatherAxis)
	if err != nil {
		return nil, false
	}

	// After gatherAlongAxis, the result has the gathered dimension at gatherAxis.
	// For the output, batch dims should come first. If gatherAxis > 0, we need
	// to transpose the gatherAxis to position 0, then reshape to output shape.
	if gatherAxis > 0 {
		// Build permutation: move gatherAxis to front, keep rest in order.
		perm := make([]int, operandRank)
		perm[0] = gatherAxis
		idx := 1
		for i := range operandRank {
			if i != gatherAxis {
				perm[idx] = i
				idx++
			}
		}
		result, err = f.ctx().Transpose(result, perm)
		if err != nil {
			return nil, false
		}
	}

	// Reshape to the expected output shape.
	outDims := toInt64Slice(outShape.Dimensions)
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, false
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, true
}

// gatherGeneral implements the full XLA Gather semantics by decomposing into
// flatten indices → gatherND → reshape output.
func (f *Function) gatherGeneral(
	operandNode, indicesNode *graphNode,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	outShape shapes.Shape,
) (backends.Value, error) {
	operandShape := operandNode.shape
	indicesShape := indicesNode.shape
	operandRank := operandShape.Rank()

	// Step 1: Normalize indexVectorAxis to trailing position.
	idxTensor, batchSize, err := f.normalizeIndexVectorAxis(
		indicesNode.tensor, indicesShape, indexVectorAxis, "Gather")
	if err != nil {
		return nil, err
	}

	indexVectorSize := len(startIndexMap)

	// Step 2: If startIndexMap is a contiguous prefix [0,1,...,k-1], we can use gatherND directly.
	// Otherwise, we need to remap indices.
	isContiguousPrefix := true
	for i, v := range startIndexMap {
		if v != i {
			isContiguousPrefix = false
			break
		}
	}

	// Step 3: Check if all slice sizes beyond the indexed axes are the full dimension
	// and all indexed axes have slice size 1 (with collapsed).
	allSlicesFullOrCollapsed := true
	collapsedSet := make(map[int]bool)
	for _, a := range collapsedSliceAxes {
		collapsedSet[a] = true
	}
	indexedSet := make(map[int]bool)
	for _, a := range startIndexMap {
		indexedSet[a] = true
	}
	for i := range operandRank {
		if collapsedSet[i] {
			if sliceSizes[i] != 1 {
				allSlicesFullOrCollapsed = false
				break
			}
		} else if !indexedSet[i] {
			if sliceSizes[i] != operandShape.Dimensions[i] {
				allSlicesFullOrCollapsed = false
				break
			}
		}
	}

	if isContiguousPrefix && allSlicesFullOrCollapsed && len(collapsedSliceAxes) == len(startIndexMap) {
		// Optimal path: gatherND with contiguous index prefix.
		// Reshape indices to [batchSize, indexVectorSize] for gatherND.
		// indexVectorAxis is already in trailing position after the transpose above.
		idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
		if err != nil {
			return nil, errors.Wrap(err, "Gather: reshape indices for gatherND")
		}

		result, err := f.ctx().GatherND(operandNode.tensor, idxTensor, 0)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: gatherND")
		}

		// Reshape to output shape.
		outDims := toInt64Slice(outShape.Dimensions)
		result, err = f.ctx().Reshape(result, outDims)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: reshape output")
		}
		return &graphNode{tensor: result, shape: outShape, owner: f}, nil
	}

	// Fallback: decompose into per-element gather using gatherND with index remapping.
	// This handles the most general case by building the full index tensor.

	// Build remapped indices: for each batch element, create the full operand index.
	// The startIndexMap tells us which operand axes are indexed by each position in the index vector.
	// Non-indexed axes start at 0 with full slice.

	// For non-contiguous startIndexMap, we need to rearrange the index columns.
	// Reshape indices to [batchSize, indexVectorSize].
	// indexVectorAxis is already in trailing position after the transpose above.
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape indices")
	}

	// If startIndexMap is not [0, 1, ..., k-1], we need to build a full-rank index tensor
	// by inserting zeros for non-indexed axes and rearranging.
	// For now, handle the case where startIndexMap maps to a contiguous set of axes
	// (even if not starting at 0) by transposing the operand first.

	// General strategy: transpose operand so that startIndexMap axes come first,
	// then use gatherND on the transposed operand.
	perm := make([]int, operandRank)
	copy(perm, startIndexMap)
	idx := len(startIndexMap)
	for i := range operandRank {
		if !indexedSet[i] {
			perm[idx] = i
			idx++
		}
	}

	operandTensor := operandNode.tensor
	needsTranspose := false
	for i, v := range perm {
		if v != i {
			needsTranspose = true
			break
		}
	}
	if needsTranspose {
		operandTensor, err = f.ctx().Transpose(operandTensor, perm)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: transpose operand")
		}
	}

	// Now the first len(startIndexMap) axes of operand are the indexed ones.
	// Use gatherND with the indices.
	result, err := f.ctx().GatherND(operandTensor, idxTensor, 0)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: gatherND general")
	}

	// Post-GatherND: trim non-collapsed, non-indexed axes where sliceSizes < full dimension.
	// The GatherND result has shape [batchSize, <remaining transposed operand dims>].
	// The remaining axes correspond to perm[len(startIndexMap):].
	needsSlice := false
	for k := len(startIndexMap); k < operandRank; k++ {
		origAxis := perm[k]
		if !collapsedSet[origAxis] && sliceSizes[origAxis] < operandShape.Dimensions[origAxis] {
			needsSlice = true
			break
		}
	}
	if needsSlice {
		// Build starts/ends/strides for the slice.
		// Axis 0 is the batch dimension, kept fully.
		sliceRank := 1 + (operandRank - len(startIndexMap))
		starts := make([]int64, sliceRank)
		ends := make([]int64, sliceRank)
		strides := make([]int64, sliceRank)
		starts[0] = 0
		ends[0] = int64(batchSize)
		strides[0] = 1
		for k := len(startIndexMap); k < operandRank; k++ {
			si := 1 + k - len(startIndexMap)
			origAxis := perm[k]
			starts[si] = 0
			strides[si] = 1
			if collapsedSet[origAxis] {
				ends[si] = 1
			} else {
				ends[si] = int64(sliceSizes[origAxis])
			}
		}
		result, err = f.ctx().Slice(result, starts, ends, strides)
		if err != nil {
			return nil, errors.Wrap(err, "Gather: post-gatherND slice")
		}
	}

	// Reshape to output shape.
	outDims := toInt64Slice(outShape.Dimensions)
	result, err = f.ctx().Reshape(result, outDims)
	if err != nil {
		return nil, errors.Wrap(err, "Gather: reshape output general")
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

// normalizeIndexVectorAxis handles the common pattern of ensuring the indexVectorAxis
// is expanded (if implicit) and transposed to the trailing position.
// Returns the (possibly modified) tensor and the computed batchSize.
func (f *Function) normalizeIndexVectorAxis(
	idxTensor bridge.Tensor, indicesShape shapes.Shape,
	indexVectorAxis int, opName string,
) (bridge.Tensor, int, error) {
	var err error

	// If indexVectorAxis == rank, there's an implicit axis of size 1.
	if indexVectorAxis == indicesShape.Rank() {
		newDims := append(toInt64Slice(indicesShape.Dimensions), 1)
		idxTensor, err = f.ctx().Reshape(idxTensor, newDims)
		if err != nil {
			return nil, 0, errors.Wrap(err, opName+": reshape indices for implicit axis")
		}
	}

	// Compute batchSize (product of all dims except indexVectorAxis).
	batchSize := 1
	for i := range indicesShape.Rank() {
		if i != indexVectorAxis || indexVectorAxis == indicesShape.Rank() {
			batchSize *= indicesShape.Dimensions[i]
		}
	}

	// Transpose indexVectorAxis to trailing position if needed.
	effectiveRank := indicesShape.Rank()
	if indexVectorAxis == indicesShape.Rank() {
		effectiveRank = indicesShape.Rank() + 1
	}
	if indexVectorAxis != effectiveRank-1 {
		perm := make([]int, effectiveRank)
		pi := 0
		for i := 0; i < effectiveRank; i++ {
			if i != indexVectorAxis {
				perm[pi] = i
				pi++
			}
		}
		perm[effectiveRank-1] = indexVectorAxis
		idxTensor, err = f.ctx().Transpose(idxTensor, perm)
		if err != nil {
			return nil, 0, errors.Wrap(err, opName+": transposing index vector axis to trailing position")
		}
	}

	return idxTensor, batchSize, nil
}

// ===========================================================================
// Scatter Operations
// ===========================================================================

// scatterImpl is the common implementation for ScatterSum, ScatterMax, ScatterMin.
func (f *Function) scatterImpl(
	opName string,
	operandOp, scatterIndicesOp, updatesOp backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
	scatterMode int,
) (backends.Value, error) {
	operandNode, err := f.resolveNode(operandOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": operand")
	}
	indicesNode, err := f.resolveNode(scatterIndicesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": indices")
	}
	updatesNode, err := f.resolveNode(updatesOp)
	if err != nil {
		return nil, errors.Wrap(err, opName+": updates")
	}

	outShape, err := shapeinference.ScatterOp(
		operandNode.shape, indicesNode.shape, updatesNode.shape,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes,
		scatterAxesToOperandAxes)
	if err != nil {
		return nil, errors.Wrap(err, opName)
	}

	// Try simple scatter along a single axis (common case for embedding gradients).
	if len(scatterAxesToOperandAxes) == 1 && len(insertedWindowAxes) == 1 {
		axis := scatterAxesToOperandAxes[0]

		// Flatten indices to remove indexVectorAxis, yielding batch dimensions.
		idxTensor := indicesNode.tensor
		idxRank := indicesNode.shape.Rank()
		batchDims := make([]int64, 0, idxRank)
		for i, d := range indicesNode.shape.Dimensions {
			if i != indexVectorAxis || indexVectorAxis >= idxRank {
				batchDims = append(batchDims, int64(d))
			}
		}
		if len(batchDims) == 0 {
			batchDims = []int64{1}
		}
		if indexVectorAxis < idxRank {
			idxTensor, err = f.ctx().Reshape(idxTensor, batchDims)
			if err != nil {
				return nil, errors.Wrap(err, opName+": reshape indices")
			}
		}

		// MPSGraph's scatterAlongAxis requires indices rank == updates rank.
		// Reshape indices to match updates rank: batch dims go into the scatter axis,
		// size-1 for all other axes, then broadcast to updates shape.
		updatesRank := updatesNode.shape.Rank()
		batchSize := int64(1)
		for _, d := range batchDims {
			batchSize *= d
		}
		idxFullShape := make([]int64, updatesRank)
		for i := range updatesRank {
			if i == axis {
				idxFullShape[i] = batchSize
			} else {
				idxFullShape[i] = 1
			}
		}
		idxTensor, err = f.ctx().Reshape(idxTensor, idxFullShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": reshape indices to updates rank")
		}

		// Broadcast to match updates shape.
		broadcastShape := make([]int64, updatesRank)
		for i, d := range updatesNode.shape.Dimensions {
			if i == axis {
				broadcastShape[i] = batchSize
			} else {
				broadcastShape[i] = int64(d)
			}
		}
		idxTensor, err = f.ctx().BroadcastTo(idxTensor, broadcastShape)
		if err != nil {
			return nil, errors.Wrap(err, opName+": broadcast indices")
		}

		result, err := f.ctx().ScatterAlongAxis(operandNode.tensor, idxTensor, updatesNode.tensor, axis, scatterMode)
		if err != nil {
			return nil, errors.Wrap(err, opName+": scatter along axis")
		}
		return &graphNode{tensor: result, shape: outShape, owner: f}, nil
	}

	// General case: use scatterND.
	// Reshape indices for scatterND.
	indicesShape := indicesNode.shape
	indexVectorSize := len(scatterAxesToOperandAxes)

	idxTensor, batchSize, err := f.normalizeIndexVectorAxis(
		indicesNode.tensor, indicesShape, indexVectorAxis, opName)
	if err != nil {
		return nil, err
	}

	// indexVectorAxis is now in trailing position; reshape to [batchSize, indexVectorSize].
	idxTensor, err = f.ctx().Reshape(idxTensor, []int64{int64(batchSize), int64(indexVectorSize)})
	if err != nil {
		return nil, errors.Wrap(err, opName+": reshape indices for scatterND")
	}

	// Transpose operand so scatterAxesToOperandAxes axes come first,
	// matching the index columns.
	operandRank := operandNode.shape.Rank()
	scatterIndexedSet := make(map[int]bool)
	for _, a := range scatterAxesToOperandAxes {
		scatterIndexedSet[a] = true
	}
	scatterPerm := make([]int, operandRank)
	copy(scatterPerm, scatterAxesToOperandAxes)
	si := len(scatterAxesToOperandAxes)
	for i := range operandRank {
		if !scatterIndexedSet[i] {
			scatterPerm[si] = i
			si++
		}
	}

	operandTensor := operandNode.tensor
	needsScatterTranspose := false
	for i, v := range scatterPerm {
		if v != i {
			needsScatterTranspose = true
			break
		}
	}
	if needsScatterTranspose {
		operandTensor, err = f.ctx().Transpose(operandTensor, scatterPerm)
		if err != nil {
			return nil, errors.Wrap(err, opName+": transpose operand for scatterND")
		}
	}

	// For scatterND, determine the output shape from the (possibly transposed) operand.
	transposedDims := make([]int64, operandRank)
	for i, p := range scatterPerm {
		transposedDims[i] = int64(operandNode.shape.Dimensions[p])
	}

	result, err := f.ctx().ScatterND(operandTensor, idxTensor, updatesNode.tensor, transposedDims, scatterMode)
	if err != nil {
		return nil, errors.Wrap(err, opName+": scatterND")
	}

	// Transpose result back if we transposed the operand.
	if needsScatterTranspose {
		inversePerm := make([]int, operandRank)
		for i, v := range scatterPerm {
			inversePerm[v] = i
		}
		result, err = f.ctx().Transpose(result, inversePerm)
		if err != nil {
			return nil, errors.Wrap(err, opName+": inverse transpose after scatterND")
		}
	}

	return &graphNode{tensor: result, shape: outShape, owner: f}, nil
}

func (f *Function) ScatterSum(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterSum", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeAdd)
}

func (f *Function) ScatterMax(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterMax", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMax)
}

func (f *Function) ScatterMin(
	operand, scatterIndices, updates backends.Value,
	indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
	indicesAreSorted, uniqueIndices bool,
) (backends.Value, error) {
	return f.scatterImpl("ScatterMin", operand, scatterIndices, updates,
		indexVectorAxis, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes,
		indicesAreSorted, uniqueIndices, bridge.ScatterModeMin)
}
