package model

import (
	"testing"
)

func TestExpandRank5Passthrough(t *testing.T) {
	// Rank-5 reshape should NOT be transformed.
	b := NewBuilder("test")

	x := b.Input("x", Float32, 2, 3, 4)
	r := b.Reshape(x, []int64{1, 2, 3, 2, 2}) // rank 5 — should be left alone
	tr := b.Transpose(r, []int64{0, 2, 1, 3, 4})
	out := b.Reshape(tr, []int64{2, 3, 4})
	b.Output("out", out)

	program := b.Build()

	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// Count reshape and transpose ops.
	reshapeCount := 0
	transposeCount := 0
	for _, op := range block.Operations {
		switch op.Type {
		case "reshape":
			reshapeCount++
		case "transpose":
			transposeCount++
		}
	}

	// Should be unchanged: 2 reshapes + 1 transpose.
	if reshapeCount != 2 {
		t.Errorf("expected 2 reshape ops (no transformation), got %d", reshapeCount)
	}
	if transposeCount != 1 {
		t.Errorf("expected 1 transpose op (no transformation), got %d", transposeCount)
	}
}

func TestExpandRank6ReshapeTranspose(t *testing.T) {
	// Rank-6 reshape→transpose→reshape(rank≤5) should be decomposed.
	b := NewBuilder("test")

	// Simulate: [6] → reshape [1,1,2,1,3,1] (rank 6) → transpose → reshape [6]
	x := b.Input("x", Float32, 6)
	r := b.Reshape(x, []int64{1, 1, 2, 1, 3, 1}) // rank 6
	tr := b.Transpose(r, []int64{0, 1, 3, 2, 4, 5})
	out := b.Reshape(tr, []int64{6})
	b.Output("out", out)

	program := b.Build()

	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// All operations should have rank ≤ 5.
	for _, op := range block.Operations {
		if op.Type == "reshape" || op.Type == "transpose" {
			shape := getOpOutputShape(op)
			if len(shape) > 5 {
				t.Errorf("op %s (%s) has rank %d > 5, shape=%v",
					op.Outputs[0].Name, op.Type, len(shape), shape)
			}
		}
	}

	// The original rank-6 reshape should be gone.
	for _, op := range block.Operations {
		if op.Type == "reshape" {
			shape := getOpOutputShape(op)
			if len(shape) == 6 {
				t.Error("rank-6 reshape still present after optimization")
			}
		}
	}
}

func TestExpandWindowAttentionPattern(t *testing.T) {
	// Simulate DaViT's exact window attention pattern:
	// [1,14,14,192] → reshape [1,2,7,2,7,192] → transpose [0,1,3,2,4,5] → reshape [1,4,7,7,192]
	b := NewBuilder("test")

	x := b.Input("x", Float32, 1, 14, 14, 192)
	r := b.Reshape(x, []int64{1, 2, 7, 2, 7, 192}) // rank 6
	tr := b.Transpose(r, []int64{0, 1, 3, 2, 4, 5})
	out := b.Reshape(tr, []int64{1, 4, 7, 7, 192})
	b.Output("out", out)

	program := b.Build()

	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// All operations should have rank ≤ 5.
	for _, op := range block.Operations {
		if op.Type == "reshape" || op.Type == "transpose" {
			shape := getOpOutputShape(op)
			if len(shape) > 5 {
				t.Errorf("op %s (%s) has rank %d > 5, shape=%v",
					op.Outputs[0].Name, op.Type, len(shape), shape)
			}
		}
	}

	// Verify final output name is preserved.
	if len(block.Outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(block.Outputs))
	}

	// The output should reference a valid value that was produced.
	outputName := block.Outputs[0]
	found := false
	for _, op := range block.Operations {
		if len(op.Outputs) > 0 && op.Outputs[0].Name == outputName {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("output %q not produced by any operation", outputName)
	}

	// Verify final reshape produces the correct shape [1,4,7,7,192].
	var lastReshapeShape []int64
	for _, op := range block.Operations {
		if op.Type == "reshape" {
			lastReshapeShape = getOpOutputShape(op)
		}
	}
	expectedShape := []int64{1, 4, 7, 7, 192}
	if len(lastReshapeShape) != len(expectedShape) {
		t.Fatalf("final reshape shape rank %d, want %d", len(lastReshapeShape), len(expectedShape))
	}
	for i, v := range expectedShape {
		if lastReshapeShape[i] != v {
			t.Errorf("final reshape shape[%d] = %d, want %d", i, lastReshapeShape[i], v)
		}
	}
}

func TestGroupConsecutiveAxes(t *testing.T) {
	tests := []struct {
		name   string
		perm   []int
		groups [][]int
	}{
		{
			name:   "identity",
			perm:   []int{0, 1, 2},
			groups: [][]int{{0, 1, 2}},
		},
		{
			name:   "single swap",
			perm:   []int{1, 0},
			groups: [][]int{{1}, {0}},
		},
		{
			name:   "DaViT pattern",
			perm:   []int{0, 1, 3, 2, 4, 5},
			groups: [][]int{{0, 1}, {3}, {2}, {4, 5}},
		},
		{
			name:   "all separate",
			perm:   []int{3, 1, 2, 0},
			groups: [][]int{{3}, {1, 2}, {0}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := groupConsecutiveAxes(tt.perm)
			if len(got) != len(tt.groups) {
				t.Fatalf("got %d groups, want %d: %v", len(got), len(tt.groups), got)
			}
			for i := range got {
				if len(got[i]) != len(tt.groups[i]) {
					t.Errorf("group %d: got %v, want %v", i, got[i], tt.groups[i])
					continue
				}
				for j := range got[i] {
					if got[i][j] != tt.groups[i][j] {
						t.Errorf("group %d[%d]: got %d, want %d", i, j, got[i][j], tt.groups[i][j])
					}
				}
			}
		})
	}
}

func TestGetProd(t *testing.T) {
	shape := []int64{2, 3, 4, 5}

	// No skip.
	got := getProd(0, 4, shape, map[int]bool{})
	if got != 120 {
		t.Errorf("getProd(0,4) = %d, want 120", got)
	}

	// Skip index 1.
	got = getProd(0, 4, shape, map[int]bool{1: true})
	if got != 40 {
		t.Errorf("getProd(0,4,skip={1}) = %d, want 40", got)
	}

	// Empty range.
	got = getProd(2, 2, shape, map[int]bool{})
	if got != 1 {
		t.Errorf("getProd(2,2) = %d, want 1", got)
	}
}

func TestCollapseConsecutiveReshapes(t *testing.T) {
	// reshape(rank>5) → reshape(rank≤5) should be collapsed into a single reshape.
	b := NewBuilder("test")

	x := b.Input("x", Float32, 24, 512)
	r1 := b.Reshape(x, []int64{1, 1, 1, 24, 1, 512}) // rank 6
	out := b.Reshape(r1, []int64{1, 24, 512})          // rank 3
	b.Output("out", out)

	program := b.Build()
	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// No reshape should have rank > 5.
	for _, op := range block.Operations {
		if op.Type == "reshape" {
			shape := getOpOutputShape(op)
			if len(shape) > 5 {
				t.Errorf("reshape with rank %d > 5 still present, shape=%v", len(shape), shape)
			}
		}
	}

	// Should have exactly one reshape (the collapsed result).
	reshapeCount := 0
	for _, op := range block.Operations {
		if op.Type == "reshape" {
			reshapeCount++
		}
	}
	if reshapeCount != 1 {
		t.Errorf("expected 1 reshape after collapse, got %d", reshapeCount)
	}
}

func TestCollapseReshapeChain(t *testing.T) {
	// reshape(rank 8) → reshape(rank 6) → reshape(rank 3) should collapse all intermediates.
	b := NewBuilder("test")

	x := b.Input("x", Float32, 1024, 576)
	r1 := b.Reshape(x, []int64{1, 1, 1, 1024, 1, 24, 1, 24}) // rank 8
	r2 := b.Reshape(r1, []int64{1, 1, 1, 1024, 576})           // rank 5 — but input is rank 8
	out := b.Reshape(r2, []int64{1024, 576})
	b.Output("out", out)

	program := b.Build()
	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	for _, op := range block.Operations {
		if op.Type == "reshape" {
			shape := getOpOutputShape(op)
			if len(shape) > 5 {
				t.Errorf("reshape with rank %d > 5 still present, shape=%v", len(shape), shape)
			}
		}
	}
}

func TestFuseReshapeTileReshape(t *testing.T) {
	// reshape(rank>5) → tile → reshape(rank≤5) should be fused into
	// reshape(rank≤5) → tile(rank≤5) → reshape(rank≤5).
	b := NewBuilder("test")

	x := b.Input("x", Float32, 1, 24, 512)
	r := b.Reshape(x, []int64{1, 1, 1, 24, 1, 512}) // rank 6
	tiled := b.Tile(r, []int64{2, 3, 4, 1, 5, 1})    // reps on singleton dims
	out := b.Reshape(tiled, []int64{24, 24, 5, 512})  // collapse to rank 4
	b.Output("out", out)

	program := b.Build()
	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// No operation should have rank > 5.
	for _, op := range block.Operations {
		shape := getOpOutputShape(op)
		if len(shape) > 5 {
			t.Errorf("op %s (%s) has rank %d > 5, shape=%v",
				op.Outputs[0].Name, op.Type, len(shape), shape)
		}
	}

	// Should have a tile op (the fused version).
	tileCount := 0
	for _, op := range block.Operations {
		if op.Type == "tile" {
			tileCount++
		}
	}
	if tileCount != 1 {
		t.Errorf("expected 1 tile op, got %d", tileCount)
	}
}

func TestMergeSingletonDims(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int64
		reps      []int64
		wantShape []int64
		wantReps  []int64
	}{
		{
			name:      "merge leading 1-dims",
			shape:     []int64{1, 1, 1, 24, 1, 512},
			reps:      []int64{2, 3, 4, 1, 5, 1},
			wantShape: []int64{1, 24, 1, 512},
			wantReps:  []int64{24, 1, 5, 1},
		},
		{
			name:      "no 1-dims",
			shape:     []int64{24, 512},
			reps:      []int64{2, 3},
			wantShape: []int64{24, 512},
			wantReps:  []int64{2, 3},
		},
		{
			name:      "all 1-dims",
			shape:     []int64{1, 1, 1, 1},
			reps:      []int64{2, 3, 4, 5},
			wantShape: []int64{1},
			wantReps:  []int64{120},
		},
		{
			name:      "alternating",
			shape:     []int64{1, 24, 1, 1, 1, 512},
			reps:      []int64{3, 1, 4, 5, 6, 1},
			wantShape: []int64{1, 24, 1, 512},
			wantReps:  []int64{3, 1, 120, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotShape, gotReps := mergeSingletonDims(tt.shape, tt.reps)
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape: got %v, want %v", gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}
			if len(gotReps) != len(tt.wantReps) {
				t.Fatalf("reps: got %v, want %v", gotReps, tt.wantReps)
			}
			for i := range gotReps {
				if gotReps[i] != tt.wantReps[i] {
					t.Errorf("reps[%d] = %d, want %d", i, gotReps[i], tt.wantReps[i])
				}
			}
		})
	}
}

func TestFindInsertedAxes(t *testing.T) {
	tests := []struct {
		name   string
		src    []int64
		target []int64
		want   []int64
	}{
		{
			name:   "simple insertion",
			src:    []int64{24, 512},
			target: []int64{1, 24, 1, 512},
			want:   []int64{0, 2},
		},
		{
			name:   "many insertions",
			src:    []int64{24, 512},
			target: []int64{1, 1, 1, 24, 1, 512},
			want:   []int64{0, 1, 2, 4},
		},
		{
			name:   "no insertion needed",
			src:    []int64{24, 512},
			target: []int64{24, 512},
			want:   nil, // same rank, no insertion
		},
		{
			name:   "not a pure insertion",
			src:    []int64{24, 512},
			target: []int64{1, 12, 2, 512},
			want:   nil, // 12 doesn't match any source dim
		},
		{
			name:   "src with 1-dims",
			src:    []int64{1, 24, 512},
			target: []int64{1, 1, 24, 1, 512},
			want:   []int64{1, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findInsertedAxes(tt.src, tt.target)
			if tt.want == nil {
				if got != nil {
					t.Errorf("expected nil, got %v", got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("axes[%d] = %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestExpandMultiplePatterns(t *testing.T) {
	// Test that multiple rank-6 patterns in the same graph are all decomposed.
	b := NewBuilder("test")

	x := b.Input("x", Float32, 1, 14, 14, 192)

	// First pattern.
	r1 := b.Reshape(x, []int64{1, 2, 7, 2, 7, 192})
	tr1 := b.Transpose(r1, []int64{0, 1, 3, 2, 4, 5})
	out1 := b.Reshape(tr1, []int64{1, 4, 7, 7, 192})

	// Some intermediate op.
	added := b.Add(out1, out1)

	// Second pattern.
	r2 := b.Reshape(added, []int64{1, 2, 7, 2, 7, 192})
	tr2 := b.Transpose(r2, []int64{0, 1, 3, 2, 4, 5})
	out2 := b.Reshape(tr2, []int64{1, 4, 7, 7, 192})

	b.Output("out", out2)

	program := b.Build()
	block := program.Functions["test"].BlockSpecializations["CoreML7"]

	// No operation should have rank > 5.
	for _, op := range block.Operations {
		if op.Type == "reshape" || op.Type == "transpose" {
			shape := getOpOutputShape(op)
			if len(shape) > 5 {
				t.Errorf("op %s (%s) has rank %d > 5", op.Outputs[0].Name, op.Type, len(shape))
			}
		}
	}
}
