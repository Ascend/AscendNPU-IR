# Related Projects and Acknowledgments

This document lists the open-source projects and ecosystem closely related to AscendNPU IR, and acknowledges the LLVM/MLIR and other communities.

## [MLIR](https://mlir.llvm.org)

MLIR originates from the LLVM community and provides a reusable and extensible compiler infrastructure. AscendNPU IR is built on top of MLIR. We express our gratitude to all developers and contributors in the LLVM/MLIR community. AscendNPU IR fully leverages the following advantages of MLIR:

- **Modular design**: Enables the definition of IR at different abstraction levels, facilitating progressive lowering.
- **Infrastructure reuse**: Reuses the parsing, transformation, optimization, and code generation toolchain of MLIR.
- **Ecosystem interoperability**: By extending MLIR dialects, AscendNPU IR can interact with and convert to/from other dialects in the MLIR ecosystem (such as IR exported by PyTorch and TensorFlow), providing a pathway for integrating with upper-level frameworks.

## [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)

**Triton-Ascend** is a Triton compilation framework built for the Ascend platform, enabling Triton code to run efficiently on Ascend hardware. As the compilation backend of Triton, **AscendNPU IR** allows developers to efficiently develop kernels for Ascend NPUs using the familiar Triton syntax and programming model, lowering the barrier for Python developers to develop operators for Ascend.

## [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)

TileLang is a domain-specific language for describing tensor computation, and TileLang-Ascend is its Ascend-oriented version. By using AscendNPU IR as the compilation backend, TileLang-Ascend can leverage the Ascend hardware affinity optimization capabilities of AscendNPU IR to generate high-performance Ascend operators.
