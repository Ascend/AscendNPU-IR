# Related projects and thanks

This document lists open-source projects and ecosystems closely related to AscendNPU IR and thanks the LLVM/MLIR and other communities.

## [MLIR](https://mlir.llvm.org)

MLIR originates from the LLVM community and provides reusable, extensible compiler infrastructure. AscendNPU IR is built on MLIR. We thank all developers and contributors in the LLVM/MLIR community. AscendNPU IR benefits from MLIR in these ways:

- **Modular design**: Define IR at different abstraction levels for progressive lowering.
- **Reuse of infrastructure**: Parsing, transformation, optimization, and code generation from MLIR.
- **Ecosystem interoperability**: Extend MLIR dialects to interact and convert with other dialects (e.g. TensorFlow, PyTorch IR) and integrate with upper-level frameworks.

## [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)

Triton-Ascend brings Triton programming to Ascend, so Triton code runs efficiently on Ascend hardware. AscendNPU IR serves as the compilation backend for Triton, enabling developers to write high-performance kernels for Ascend NPU with familiar Triton syntax and programming model and lowering the barrier for Python developers.

## [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)

TileLang is a domain-specific language for tensor computation; TileLang-Ascend is its Ascend-oriented version. By using AscendNPU IR as the backend, TileLang-Ascend leverages AscendNPU IR’s Ascend-aware optimizations to generate high-performance Ascend operators.
