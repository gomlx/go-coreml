# go-coreml

Go bindings to Apple's CoreML framework for high-performance machine learning inference on Apple Silicon.

## Overview

go-coreml provides Go bindings to CoreML, enabling:

- Running ML models on Apple's Neural Engine (ANE)
- Metal GPU acceleration
- Integration with [GoMLX](https://github.com/gomlx/gomlx) as a backend

## Status

**Work in Progress** - This project is in early development.

See [specs/001-initial-plan.md](specs/001-initial-plan.md) for the implementation plan.

## Requirements

- macOS 12.0+ (Monterey or later)
- Xcode Command Line Tools
- Go 1.21+

## Project Structure

```
go-coreml/
├── internal/
│   └── bridge/          # Low-level cgo bindings to CoreML
│       ├── bridge.h     # C-compatible function declarations
│       ├── bridge.mm    # Objective-C++ implementation
│       └── bridge.go    # cgo wrapper
├── model/
│   └── builder.go       # Programmatic model construction (MIL)
├── runtime/
│   ├── model.go         # MLModel wrapper
│   └── tensor.go        # Tensor operations
├── ops/
│   └── mil_ops.go       # MIL operation builders
├── proto/
│   └── coreml/          # Generated Go types from CoreML .proto files
└── specs/
    └── 001-initial-plan.md  # Implementation plan
```

## Development

```bash
# Build
go build ./...

# Test
go test ./...
```

## License

Apache 2.0 - see LICENSE file.
