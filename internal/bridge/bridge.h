// bridge.h - C-compatible function declarations for CoreML bridge
// This file exposes CoreML functionality to Go via cgo.

#ifndef COREML_BRIDGE_H
#define COREML_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef void* CoreMLModel;
typedef void* CoreMLTensor;

// Error handling
typedef struct {
    int code;
    const char* message;
} CoreMLError;

// Model loading
CoreMLModel coreml_load_model(const char* path, CoreMLError* error);
void coreml_free_model(CoreMLModel model);

// Model info
int coreml_model_input_count(CoreMLModel model);
int coreml_model_output_count(CoreMLModel model);
const char* coreml_model_input_name(CoreMLModel model, int index);
const char* coreml_model_output_name(CoreMLModel model, int index);

// Tensor creation
CoreMLTensor coreml_tensor_create(int64_t* shape, int rank, int dtype, CoreMLError* error);
CoreMLTensor coreml_tensor_create_with_data(int64_t* shape, int rank, int dtype, void* data, CoreMLError* error);
void coreml_tensor_free(CoreMLTensor tensor);

// Tensor access
int coreml_tensor_rank(CoreMLTensor tensor);
int64_t coreml_tensor_dim(CoreMLTensor tensor, int axis);
int coreml_tensor_dtype(CoreMLTensor tensor);
void* coreml_tensor_data(CoreMLTensor tensor);
int64_t coreml_tensor_size_bytes(CoreMLTensor tensor);

// Model execution
bool coreml_model_predict(CoreMLModel model,
                          const char** input_names, CoreMLTensor* inputs, int num_inputs,
                          const char** output_names, CoreMLTensor* outputs, int num_outputs,
                          CoreMLError* error);

// Compute unit configuration
typedef enum {
    COREML_COMPUTE_ALL = 0,
    COREML_COMPUTE_CPU_ONLY = 1,
    COREML_COMPUTE_CPU_AND_GPU = 2,
    COREML_COMPUTE_CPU_AND_ANE = 3
} CoreMLComputeUnits;

void coreml_set_compute_units(CoreMLComputeUnits units);

// Data types
typedef enum {
    COREML_DTYPE_FLOAT32 = 0,
    COREML_DTYPE_FLOAT16 = 1,
    COREML_DTYPE_INT32 = 2,
    COREML_DTYPE_INT64 = 3,
    COREML_DTYPE_BOOL = 4
} CoreMLDType;

#ifdef __cplusplus
}
#endif

#endif // COREML_BRIDGE_H
