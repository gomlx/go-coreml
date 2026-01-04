// bridge.mm - Objective-C++ implementation of CoreML bridge
// This file wraps CoreML APIs and exposes them via C functions.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "bridge.h"
#include <string.h>

// Global compute units setting
static MLComputeUnits g_computeUnits = MLComputeUnitsAll;

void coreml_set_compute_units(CoreMLComputeUnits units) {
    switch (units) {
        case COREML_COMPUTE_CPU_ONLY:
            g_computeUnits = MLComputeUnitsCPUOnly;
            break;
        case COREML_COMPUTE_CPU_AND_GPU:
            g_computeUnits = MLComputeUnitsCPUAndGPU;
            break;
        case COREML_COMPUTE_CPU_AND_ANE:
            g_computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            break;
        case COREML_COMPUTE_ALL:
        default:
            g_computeUnits = MLComputeUnitsAll;
            break;
    }
}

static void set_error(CoreMLError* error, int code, NSError* nsError) {
    if (error == NULL) return;
    error->code = code;
    if (nsError != nil) {
        error->message = strdup([[nsError localizedDescription] UTF8String]);
    } else {
        error->message = NULL;
    }
}

CoreMLModel coreml_load_model(const char* path, CoreMLError* error) {
    @autoreleasepool {
        NSString* nsPath = [NSString stringWithUTF8String:path];
        NSURL* url = [NSURL fileURLWithPath:nsPath];

        NSError* nsError = nil;

        // Configure model
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = g_computeUnits;

        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:config error:&nsError];
        if (model == nil) {
            set_error(error, 1, nsError);
            return NULL;
        }

        // Return retained model
        return (__bridge_retained void*)model;
    }
}

void coreml_free_model(CoreMLModel model) {
    if (model != NULL) {
        MLModel* m = (__bridge_transfer MLModel*)model;
        (void)m; // ARC will release
    }
}

int coreml_model_input_count(CoreMLModel model) {
    @autoreleasepool {
        MLModel* m = (__bridge MLModel*)model;
        return (int)[[m modelDescription].inputDescriptionsByName count];
    }
}

int coreml_model_output_count(CoreMLModel model) {
    @autoreleasepool {
        MLModel* m = (__bridge MLModel*)model;
        return (int)[[m modelDescription].outputDescriptionsByName count];
    }
}

const char* coreml_model_input_name(CoreMLModel model, int index) {
    @autoreleasepool {
        MLModel* m = (__bridge MLModel*)model;
        NSArray* keys = [[[m modelDescription].inputDescriptionsByName allKeys] sortedArrayUsingSelector:@selector(compare:)];
        if (index < 0 || index >= (int)[keys count]) return NULL;
        return strdup([keys[index] UTF8String]);
    }
}

const char* coreml_model_output_name(CoreMLModel model, int index) {
    @autoreleasepool {
        MLModel* m = (__bridge MLModel*)model;
        NSArray* keys = [[[m modelDescription].outputDescriptionsByName allKeys] sortedArrayUsingSelector:@selector(compare:)];
        if (index < 0 || index >= (int)[keys count]) return NULL;
        return strdup([keys[index] UTF8String]);
    }
}

// Helper to convert dtype enum to MLMultiArrayDataType
static MLMultiArrayDataType dtype_to_ml(int dtype) {
    switch (dtype) {
        case COREML_DTYPE_FLOAT32: return MLMultiArrayDataTypeFloat32;
        case COREML_DTYPE_FLOAT16: return MLMultiArrayDataTypeFloat16;
        case COREML_DTYPE_INT32: return MLMultiArrayDataTypeInt32;
        default: return MLMultiArrayDataTypeFloat32;
    }
}

CoreMLTensor coreml_tensor_create(int64_t* shape, int rank, int dtype, CoreMLError* error) {
    @autoreleasepool {
        NSMutableArray<NSNumber*>* dims = [NSMutableArray arrayWithCapacity:rank];
        for (int i = 0; i < rank; i++) {
            [dims addObject:@(shape[i])];
        }

        NSError* nsError = nil;
        MLMultiArray* array = [[MLMultiArray alloc] initWithShape:dims
                                                         dataType:dtype_to_ml(dtype)
                                                            error:&nsError];
        if (array == nil) {
            set_error(error, 1, nsError);
            return NULL;
        }

        return (__bridge_retained void*)array;
    }
}

CoreMLTensor coreml_tensor_create_with_data(int64_t* shape, int rank, int dtype, void* data, CoreMLError* error) {
    CoreMLTensor tensor = coreml_tensor_create(shape, rank, dtype, error);
    if (tensor == NULL) return NULL;

    @autoreleasepool {
        MLMultiArray* array = (__bridge MLMultiArray*)tensor;
        int64_t total = 1;
        for (int i = 0; i < rank; i++) {
            total *= shape[i];
        }

        // Copy data
        size_t elemSize = (dtype == COREML_DTYPE_FLOAT16) ? 2 : 4;
        memcpy(array.dataPointer, data, total * elemSize);
    }

    return tensor;
}

void coreml_tensor_free(CoreMLTensor tensor) {
    if (tensor != NULL) {
        MLMultiArray* a = (__bridge_transfer MLMultiArray*)tensor;
        (void)a; // ARC will release
    }
}

int coreml_tensor_rank(CoreMLTensor tensor) {
    @autoreleasepool {
        MLMultiArray* a = (__bridge MLMultiArray*)tensor;
        return (int)[a.shape count];
    }
}

int64_t coreml_tensor_dim(CoreMLTensor tensor, int axis) {
    @autoreleasepool {
        MLMultiArray* a = (__bridge MLMultiArray*)tensor;
        if (axis < 0 || axis >= (int)[a.shape count]) return 0;
        return [a.shape[axis] longLongValue];
    }
}

int coreml_tensor_dtype(CoreMLTensor tensor) {
    @autoreleasepool {
        MLMultiArray* a = (__bridge MLMultiArray*)tensor;
        switch (a.dataType) {
            case MLMultiArrayDataTypeFloat32: return COREML_DTYPE_FLOAT32;
            case MLMultiArrayDataTypeFloat16: return COREML_DTYPE_FLOAT16;
            case MLMultiArrayDataTypeInt32: return COREML_DTYPE_INT32;
            default: return COREML_DTYPE_FLOAT32;
        }
    }
}

void* coreml_tensor_data(CoreMLTensor tensor) {
    @autoreleasepool {
        MLMultiArray* a = (__bridge MLMultiArray*)tensor;
        return a.dataPointer;
    }
}

int64_t coreml_tensor_size_bytes(CoreMLTensor tensor) {
    @autoreleasepool {
        MLMultiArray* a = (__bridge MLMultiArray*)tensor;
        int64_t total = 1;
        for (NSNumber* dim in a.shape) {
            total *= [dim longLongValue];
        }
        switch (a.dataType) {
            case MLMultiArrayDataTypeFloat16: return total * 2;
            case MLMultiArrayDataTypeFloat32:
            case MLMultiArrayDataTypeInt32: return total * 4;
            default: return total * 4;
        }
    }
}

bool coreml_model_predict(CoreMLModel model,
                          const char** input_names, CoreMLTensor* inputs, int num_inputs,
                          const char** output_names, CoreMLTensor* outputs, int num_outputs,
                          CoreMLError* error) {
    @autoreleasepool {
        MLModel* m = (__bridge MLModel*)model;

        // Build input dictionary
        NSMutableDictionary<NSString*, MLFeatureValue*>* inputDict = [NSMutableDictionary dictionary];
        for (int i = 0; i < num_inputs; i++) {
            NSString* name = [NSString stringWithUTF8String:input_names[i]];
            MLMultiArray* array = (__bridge MLMultiArray*)inputs[i];
            MLFeatureValue* value = [MLFeatureValue featureValueWithMultiArray:array];
            inputDict[name] = value;
        }

        // Create input provider
        NSError* nsError = nil;
        MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:&nsError];
        if (provider == nil) {
            set_error(error, 1, nsError);
            return false;
        }

        // Run prediction
        id<MLFeatureProvider> result = [m predictionFromFeatures:provider error:&nsError];
        if (result == nil) {
            set_error(error, 2, nsError);
            return false;
        }

        // Extract outputs
        for (int i = 0; i < num_outputs; i++) {
            NSString* name = [NSString stringWithUTF8String:output_names[i]];
            MLFeatureValue* value = [result featureValueForName:name];
            if (value == nil || value.multiArrayValue == nil) {
                set_error(error, 3, nil);
                return false;
            }

            // Copy output data to provided tensor
            MLMultiArray* outArray = (__bridge MLMultiArray*)outputs[i];
            MLMultiArray* resultArray = value.multiArrayValue;

            int64_t total = 1;
            for (NSNumber* dim in resultArray.shape) {
                total *= [dim longLongValue];
            }

            size_t elemSize = (resultArray.dataType == MLMultiArrayDataTypeFloat16) ? 2 : 4;
            memcpy(outArray.dataPointer, resultArray.dataPointer, total * elemSize);
        }

        return true;
    }
}
