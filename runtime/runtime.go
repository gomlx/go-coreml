// Package runtime provides high-level APIs for compiling and executing
// CoreML models from Go.
//
// Example usage:
//
//	// Build a MIL program
//	b := model.NewBuilder("main")
//	x := b.Input("x", model.Float32, 2, 3)
//	y := b.Relu(x)
//	b.Output("y", y)
//
//	// Compile and load
//	rt := runtime.New()
//	exec, err := rt.Compile(b)
//	if err != nil { ... }
//	defer exec.Close()
//
//	// Execute
//	input := []float32{1, 2, 3, 4, 5, 6}
//	output, err := exec.Run(map[string]interface{}{"x": input})
package runtime

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/gomlx/go-coreml/internal/bridge"
	"github.com/gomlx/go-coreml/model"
)

// Runtime manages CoreML model compilation and execution.
type Runtime struct {
	cacheDir     string
	computeUnits bridge.ComputeUnits
}

// Option configures the runtime.
type Option func(*Runtime)

// WithCacheDir sets the directory for caching compiled models.
func WithCacheDir(dir string) Option {
	return func(r *Runtime) {
		r.cacheDir = dir
	}
}

// WithComputeUnits sets which compute units to use.
func WithComputeUnits(units bridge.ComputeUnits) Option {
	return func(r *Runtime) {
		r.computeUnits = units
	}
}

// New creates a new runtime with the given options.
func New(opts ...Option) *Runtime {
	r := &Runtime{
		cacheDir:     os.TempDir(),
		computeUnits: bridge.ComputeAll,
	}
	for _, opt := range opts {
		opt(r)
	}
	bridge.SetComputeUnits(r.computeUnits)
	return r
}

// Executable represents a compiled CoreML model ready for execution.
type Executable struct {
	model        *bridge.Model
	inputNames   []string
	outputNames  []string
	inputShapes  [][]int64
	outputShapes [][]int64
	outputDTypes []model.DType
	tempDir      string // For cleanup
	mu           sync.Mutex
}

// Compile compiles a MIL program builder into an executable.
func (r *Runtime) Compile(b *model.Builder) (*Executable, error) {
	return r.CompileProgram(b.Build(), b.InputSpecs(), b.OutputSpecs())
}

// CompileProgram compiles a MIL program into an executable.
func (r *Runtime) CompileProgram(program *model.Program, inputs, outputs []model.FeatureSpec) (*Executable, error) {
	// Create temp directory for the model
	tempDir, err := os.MkdirTemp(r.cacheDir, "gocoreml-")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}

	// Convert program to CoreML model
	coremlModel := model.ToModel(program, inputs, outputs, model.DefaultOptions())

	// Save as mlpackage
	packagePath := filepath.Join(tempDir, "model.mlpackage")
	if err := model.SaveMLPackage(coremlModel, packagePath); err != nil {
		os.RemoveAll(tempDir)
		return nil, fmt.Errorf("save mlpackage: %w", err)
	}

	// Compile the model using CoreML's MLModel.compileModel(at:)
	// This works without Xcode - only requires the CoreML framework
	compiledPath, err := bridge.CompileModel(packagePath, tempDir)
	if err != nil {
		os.RemoveAll(tempDir)
		return nil, fmt.Errorf("compile model: %w", err)
	}

	// Load the compiled model
	coremlBridge, err := bridge.LoadModel(compiledPath)
	if err != nil {
		os.RemoveAll(tempDir)
		return nil, fmt.Errorf("load compiled model: %w", err)
	}

	// Extract input/output info
	inputNames := make([]string, len(inputs))
	inputShapes := make([][]int64, len(inputs))
	for i, inp := range inputs {
		inputNames[i] = inp.Name
		inputShapes[i] = inp.Shape
	}

	outputNames := make([]string, len(outputs))
	outputShapes := make([][]int64, len(outputs))
	outputDTypes := make([]model.DType, len(outputs))
	for i, out := range outputs {
		outputNames[i] = out.Name
		outputShapes[i] = out.Shape
		outputDTypes[i] = out.DType
	}

	return &Executable{
		model:        coremlBridge,
		inputNames:   inputNames,
		outputNames:  outputNames,
		inputShapes:  inputShapes,
		outputShapes: outputShapes,
		outputDTypes: outputDTypes,
		tempDir:      tempDir,
	}, nil
}

// Run executes the model with the given inputs.
// Inputs should be a map from input name to data ([]float32, []int32, etc.).
// Returns a map from output name to data.
func (e *Executable) Run(inputs map[string]interface{}) (map[string]interface{}, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Create input tensors
	inputTensors := make([]*bridge.Tensor, len(e.inputNames))
	for i, name := range e.inputNames {
		data, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("missing input: %s", name)
		}

		tensor, err := createTensor(e.inputShapes[i], data)
		if err != nil {
			// Clean up already created tensors
			for j := 0; j < i; j++ {
				inputTensors[j].Close()
			}
			return nil, fmt.Errorf("create input tensor %s: %w", name, err)
		}
		inputTensors[i] = tensor
	}

	// Create output tensors
	outputTensors := make([]*bridge.Tensor, len(e.outputNames))
	for i, shape := range e.outputShapes {
		bridgeDType := modelDTypeToBridge(e.outputDTypes[i])
		tensor, err := bridge.NewTensor(shape, bridgeDType)
		if err != nil {
			// Clean up
			for _, t := range inputTensors {
				t.Close()
			}
			for j := 0; j < i; j++ {
				outputTensors[j].Close()
			}
			return nil, fmt.Errorf("create output tensor: %w", err)
		}
		outputTensors[i] = tensor
	}

	// Run prediction
	err := e.model.Predict(e.inputNames, inputTensors, e.outputNames, outputTensors)

	// Clean up input tensors
	for _, t := range inputTensors {
		t.Close()
	}

	if err != nil {
		for _, t := range outputTensors {
			t.Close()
		}
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// Extract output data
	result := make(map[string]interface{})
	for i, name := range e.outputNames {
		data, err := extractTensorData(outputTensors[i])
		outputTensors[i].Close()
		if err != nil {
			return nil, fmt.Errorf("extract output %s: %w", name, err)
		}
		result[name] = data
	}

	return result, nil
}

// Close releases resources associated with the executable.
func (e *Executable) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.model != nil {
		e.model.Close()
		e.model = nil
	}

	if e.tempDir != "" {
		os.RemoveAll(e.tempDir)
		e.tempDir = ""
	}

	return nil
}

// createTensor creates a bridge.Tensor from Go data.
func createTensor(shape []int64, data interface{}) (*bridge.Tensor, error) {
	// Handle scalar tensors: CoreML requires shape [1] for scalars
	// This matches the handling in serialize.go featureSpecToType
	actualShape := shape
	if len(shape) == 0 {
		actualShape = []int64{1}
	}

	switch d := data.(type) {
	case []float32:
		return bridge.NewTensorWithData(actualShape, bridge.DTypeFloat32, unsafe.Pointer(&d[0]))
	case []float64:
		// Convert to float32
		f32 := make([]float32, len(d))
		for i, v := range d {
			f32[i] = float32(v)
		}
		return bridge.NewTensorWithData(actualShape, bridge.DTypeFloat32, unsafe.Pointer(&f32[0]))
	case []int32:
		return bridge.NewTensorWithData(actualShape, bridge.DTypeInt32, unsafe.Pointer(&d[0]))
	case []int64:
		// Convert to int32
		i32 := make([]int32, len(d))
		for i, v := range d {
			i32[i] = int32(v)
		}
		return bridge.NewTensorWithData(actualShape, bridge.DTypeInt32, unsafe.Pointer(&i32[0]))
	default:
		return nil, fmt.Errorf("unsupported data type: %T", data)
	}
}

// extractTensorData extracts data from a bridge.Tensor.
func extractTensorData(tensor *bridge.Tensor) (interface{}, error) {
	shape := tensor.Shape()
	size := int64(1)
	for _, dim := range shape {
		size *= dim
	}

	ptr := tensor.DataPtr()
	switch tensor.DType() {
	case bridge.DTypeFloat32:
		data := make([]float32, size)
		src := (*[1 << 30]float32)(ptr)[:size:size]
		copy(data, src)
		return data, nil
	case bridge.DTypeInt32:
		data := make([]int32, size)
		src := (*[1 << 30]int32)(ptr)[:size:size]
		copy(data, src)
		return data, nil
	default:
		return nil, fmt.Errorf("unsupported dtype: %v", tensor.DType())
	}
}

// modelDTypeToBridge converts model.DType to bridge.DType.
func modelDTypeToBridge(dtype model.DType) bridge.DType {
	switch dtype {
	case model.Float32:
		return bridge.DTypeFloat32
	case model.Float16:
		return bridge.DTypeFloat16
	case model.Int32:
		return bridge.DTypeInt32
	case model.Bool:
		return bridge.DTypeBool
	default:
		return bridge.DTypeFloat32 // Default fallback
	}
}
