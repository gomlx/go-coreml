// Package godarwinml provides Go backends for machine learning on Apple Silicon.
//
// This project provides multiple GoMLX backends targeting Apple hardware:
//
//   - coreml: CoreML backend supporting Apple Neural Engine (ANE), Metal GPU, and CPU
//   - mpsgraph: MPSGraph backend for direct Metal GPU computation
//
// Each backend implements the GoMLX backends.Backend interface and can be used
// interchangeably for inference on Apple Silicon.
//
// # Shared Packages
//
//   - blob: Weight blob storage format
//   - proto: CoreML protobuf specifications
//
// # Requirements
//
//   - macOS 12.0+ (Monterey or later)
//   - Xcode Command Line Tools
//   - Go 1.25+
package godarwinml
